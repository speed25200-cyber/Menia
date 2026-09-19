"""A causal-binding estimate used for reaching, separately from a binary report.

Known Gaussian mixture, known unit-gain actuator, independent trials. This is
an experimental controller, not a learned or phenomenal model of selfhood.
"""
from dataclasses import asdict, dataclass
import math

from research.causal_binding import CausalBinding, logistic


@dataclass(frozen=True)
class Observation:
    anchor: float
    visual: float
    target: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Finite observed values required")


@dataclass(frozen=True)
class Decision:
    inferred_binding: float
    used_binding: float
    estimated_position: float
    predicted_error_variance: float
    displacement: float
    report: bool | None


@dataclass(frozen=True)
class BindingController:
    prior: float
    proprio_sigma: float = 1.
    visual_sigma: float = 1.
    source_sigma: float = 4.

    def __post_init__(self):
        CausalBinding(self.prior, self.source_sigma)
        if not all(math.isfinite(v) and v > 0
                   for v in (self.proprio_sigma, self.visual_sigma)):
            raise ValueError("Positive finite sensory noise required")

    def decide(self, observation, *, binding_override=None, report_threshold=0.,
               report_enabled=True):
        if not isinstance(observation, Observation):
            raise TypeError("Only an Observation is accepted, not world state")
        if not math.isfinite(report_threshold):
            raise ValueError("Finite report threshold required")
        vp, vv, vd = (self.proprio_sigma**2, self.visual_sigma**2,
                      self.source_sigma**2)
        residual = observation.visual-observation.anchor
        inferred = CausalBinding(self.prior, self.source_sigma).posterior(
            residual, math.sqrt(vp+vv))
        q = inferred if binding_override is None else binding_override
        if not math.isfinite(q) or not 0 <= q <= 1:
            raise ValueError("Binding intervention must be a probability")
        common_mean = observation.anchor+vp/(vp+vv)*residual
        displaced_mean = observation.anchor+vp/(vp+vv+vd)*residual
        estimate = q*common_mean+(1-q)*displaced_mean
        # Law of total variance, including disagreement between the components.
        variance = (q*vp*vv/(vp+vv)+(1-q)*vp*(vv+vd)/(vp+vv+vd)
                    +q*(1-q)*(common_mean-displaced_mean)**2)
        report = bool(q > logistic(report_threshold)) if report_enabled else None
        return Decision(inferred, q, estimate, variance,
                        observation.target-estimate, report)


def ordinary_conditional_mean(observation, prior, proprio_sigma, visual_sigma,
                              source_sigma):
    """Direct Gaussian regression, with no explicit binding-state variable.

    Evaluated independently of CausalBinding to test algorithmic equivalence.
    Log-density scaling avoids underflow in the mixture normalization.
    """
    vp, vv, vd = proprio_sigma**2, visual_sigma**2, source_sigma**2
    z = observation.visual-observation.anchor
    log_c = math.log(prior)-.5*(math.log(vp+vv)+z*z/(vp+vv))
    log_d = math.log1p(-prior)-.5*(math.log(vp+vv+vd)+z*z/(vp+vv+vd))
    scale = max(log_c, log_d)
    c, d = math.exp(log_c-scale), math.exp(log_d-scale)
    return observation.anchor+z*vp*(c/(vp+vv)+d/(vp+vv+vd))/(c+d)


class BindingControlAgent:
    """Record the estimate and decision before the actuator returns an outcome."""
    def __init__(self, controller):
        self.controller = controller
        self.events = []

    def step(self, observation, execute, **intervention):
        self.events.append({"type": "observation", **asdict(observation)})
        decision = self.controller.decide(observation, **intervention)
        self.events.append({"type": "decision", **asdict(decision)})
        # The evaluator may know x. The controller has already finished deciding.
        outcome = execute(decision.displacement)
        self.events.append({"type": "outcome", "value": outcome})
        return decision, outcome
