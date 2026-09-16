"""Learn affine actuator/sensor responses, then use them for binding control.

A trusted reference scale and a reset-to-zero calibration opportunity are
assumptions of this experiment. No claim of subjective self-awareness.
"""
from dataclasses import dataclass, asdict
import math

import numpy as np

from research.binding_control import BindingController, Observation


class WindowRegression:
    def __init__(self, noise=.5, window=24, method="bayes"):
        if noise <= 0 or not math.isfinite(noise) or window < 3 or method not in ("bayes", "ols"):
            raise ValueError("Invalid regression configuration")
        self.noise, self.window, self.method = noise, window, method
        self.samples = []

    def update(self, action, value):
        if not all(math.isfinite(v) for v in (action, value)):
            raise ValueError("Finite command and sensor measurement required")
        self.samples.append((float(action), float(value)))
        self.samples = self.samples[-self.window:]

    def fit(self):
        x = np.array([[1., u] for u, _ in self.samples]).reshape(-1, 2)
        y = np.array([v for _, v in self.samples])
        information = x.T@x
        rank = int(np.linalg.matrix_rank(information))
        if self.method == "ols" and len(y) >= 3 and rank == 2:
            mean = np.linalg.solve(information, x.T@y)
            variance = max(1e-6, float(np.sum((y-x@mean)**2)/(len(y)-2)))
            covariance = variance*np.linalg.inv(information)
            return {"mean": mean, "covariance": covariance, "noise_variance": variance,
                    "rank": rank, "alpha": None, "beta": None,
                    "inverse_precision": np.linalg.inv(information)}
        prior_mean = np.array([0., 1.])
        precision = information+.01*np.eye(2)
        inverse = np.linalg.inv(precision)
        mean = np.linalg.solve(precision, x.T@y+.01*prior_mean)
        alpha = 2.+len(y)/2
        beta = self.noise**2+.5*(np.sum((y-x@mean)**2)+.01*np.sum((mean-prior_mean)**2))
        variance = float(beta/(alpha-1))
        return {"mean": mean, "covariance": variance*inverse,
                "noise_variance": variance, "rank": rank, "alpha": alpha,
                "beta": float(beta), "inverse_precision": inverse}

    def forecast(self, action):
        fit = self.fit()
        phi = np.array([1., action])
        mean = float(phi@fit['mean'])
        variance = float(fit['noise_variance']+phi@fit['covariance']@phi)
        alpha = fit['alpha']
        return {"mean": mean, "variance": variance,
                "degrees_of_freedom": None if alpha is None else 2*alpha,
                "scale_squared": variance if alpha is None else variance*(alpha-1)/alpha}

    def choose_probe(self):
        # Account for the sample that will leave the window when this probe enters.
        retained = self.samples[-(self.window-1):]
        x = np.array([[1., u] for u, _ in retained]).reshape(-1, 2)
        inverse = np.linalg.inv(.01*np.eye(2)+x.T@x)
        scores = [math.log1p(float(np.array([1., u])@inverse@np.array([1., u])))
                  for u in (-1., 0., 1.)]
        return (-1., 0., 1.)[int(np.argmax(scores))]

    def snapshot(self):
        return {"noise": self.noise, "window": self.window, "method": self.method,
                "samples": [list(row) for row in self.samples]}

    @classmethod
    def restore(cls, state):
        model = cls(state['noise'], state['window'], state['method'])
        if len(state['samples']) > model.window:
            raise ValueError("Checkpoint exceeds the observation window")
        for action, value in state['samples']:
            model.update(action, value)
        return model


def predictive_nll(prediction, value):
    variance = prediction['scale_squared']
    error = value-prediction['mean']
    df = prediction['degrees_of_freedom']
    if df is None:
        return .5*(math.log(2*math.pi*variance)+error**2/variance)
    return (math.lgamma(df/2)-math.lgamma((df+1)/2)
            +.5*math.log(df*math.pi*variance)
            +(df+1)/2*math.log1p(error**2/(df*variance)))


@dataclass(frozen=True)
class SensorReading:
    reference: float | None
    visual: float

    def __post_init__(self):
        if not math.isfinite(self.visual) or (self.reference is not None and not math.isfinite(self.reference)):
            raise ValueError("Finite sensor measurements required")
        object.__setattr__(self, 'visual', float(self.visual))
        if self.reference is not None:
            object.__setattr__(self, 'reference', float(self.reference))


class AdaptiveBindingAgent:
    def __init__(self, method="bayes", window=24, record=True):
        self.reference = WindowRegression(.15, window, method)
        self.visual = WindowRegression(.5, window, method)
        self.events = []
        self.record = record

    def _record(self, kind, **data):
        if self.record:
            self.events.append({"type": kind, **data})

    def probe(self, execute, *, action=None, learn=True):
        action = self.visual.choose_probe() if action is None else float(action)
        if action not in (-1., 0., 1.):
            raise ValueError("Unsupported calibration command")
        forecasts = {"reference": self.reference.forecast(action),
                     "visual": self.visual.forecast(action)}
        self._record("prediction", action=action, forecasts=forecasts)
        reading = execute(action)
        if not isinstance(reading, SensorReading):
            raise TypeError("The actuator must return only SensorReading")
        self._record("observation", **asdict(reading))
        nll = {"reference": (None if reading.reference is None else
                              predictive_nll(forecasts['reference'], reading.reference)),
               "visual": predictive_nll(forecasts['visual'], reading.visual)}
        if learn:
            if reading.reference is not None:
                self.reference.update(action, reading.reference)
            self.visual.update(action, reading.visual)
        state = self.parameters()
        self._record("revision", learned=learn, parameters=state)
        return {"action": action, "nll": nll, "parameters": state}

    def parameters(self):
        p, v = self.reference.fit(), self.visual.fit()
        gain = float(p['mean'][1])
        scale = float(v['mean'][1]/gain) if abs(gain) >= .05 else None
        return {"gain": gain, "gain_variance": float(p['covariance'][1, 1]),
                "visual_slope": float(v['mean'][1]), "visual_bias": float(v['mean'][0]),
                "visual_scale": scale, "visual_noise": math.sqrt(v['noise_variance']),
                "identified_given_reference_assumption": p['rank'] == v['rank'] == 2,
                "reference_samples": len(self.reference.samples),
                "visual_samples": len(self.visual.samples)}

    def reach(self, observation, *, parameters=None):
        if not isinstance(observation, Observation):
            raise TypeError("Observed fields only")
        state = self.parameters() if parameters is None else parameters
        gain, scale = state['gain'], state['visual_scale']
        visual_available = scale is not None and abs(scale) >= .05
        if visual_available:
            corrected = (observation.visual-state['visual_bias'])/scale
            sigma = max(.001, state['visual_noise']/abs(scale))
            decision = BindingController(.5, visual_sigma=sigma).decide(
                Observation(observation.anchor, corrected, observation.target), report_enabled=False)
            position = decision.estimated_position
            binding = decision.used_binding
        else:
            position, binding = observation.anchor, None
        # Moment-based control; sensor-parameter uncertainty is not integrated.
        denominator = gain**2+state['gain_variance']
        raw = (observation.target-position)*gain/denominator if denominator > 0 else 0.
        command = min(6., max(-6., raw))
        return {"command": command, "estimated_position": position, "binding": binding,
                "saturated": command != raw, "visual_available": visual_available}

    def snapshot(self):
        return {"reference": self.reference.snapshot(), "visual": self.visual.snapshot()}

    @classmethod
    def restore(cls, state, record=True):
        agent = cls(record=record)
        agent.reference = WindowRegression.restore(state['reference'])
        agent.visual = WindowRegression.restore(state['visual'])
        return agent
