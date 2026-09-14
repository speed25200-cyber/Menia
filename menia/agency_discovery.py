"""Experimental active causal attribution; no subjective-consciousness claim.

Six specified generative hypotheses per anonymous channel. Simulator causes are
never passed to the inferer. This is an exact Bayesian reference, not a neural net.
"""
import math
import random


HYPOTHESES = ('action+', 'action-', 'cue+', 'cue-', 'constant+', 'constant-')


def expected(hypothesis, action, cue):
    source = action if hypothesis.startswith('action') else cue if hypothesis.startswith('cue') else 1
    return source if hypothesis.endswith('+') else -source


def entropy(probabilities):
    return -sum(p*math.log(p) for p in probabilities if p > 0)


class AttributionModel:
    def __init__(self, *, noise=.05):
        if not 0 < noise < .5:
            raise ValueError('Noise must lie strictly between zero and one half')
        self.noise = noise
        self.posterior = [[1/6]*6 for _ in range(2)]
        self.attempts = [0, 0]
        self.received = [0, 0]

    def update(self, channel, action, cue, observation):
        if channel not in (0, 1) or action not in (-1, 1) or cue not in (-1, 1):
            raise ValueError('Invalid channel, action or cue')
        if observation not in (-1, 1, None):
            raise ValueError('Invalid observation')
        self.attempts[channel] += 1
        if observation is None:
            return
        self.received[channel] += 1
        weights = [p*((1-self.noise) if expected(h, action, cue) == observation else self.noise)
                   for h, p in zip(HYPOTHESES, self.posterior[channel])]
        total = sum(weights)
        self.posterior[channel] = [w/total for w in weights]

    def information_gain(self, channel, action, cue):
        prior = self.posterior[channel]
        remainder = 0.0
        for outcome in (-1, 1):
            weights = [p*((1-self.noise) if expected(h, action, cue) == outcome else self.noise)
                       for h, p in zip(HYPOTHESES, prior)]
            probability = sum(weights)
            remainder += probability*entropy([w/probability for w in weights])
        reception = (self.received[channel]+1)/(self.attempts[channel]+2)
        return max(0.0, entropy(prior)-remainder)*reception

    def control_probabilities(self):
        return [sum(p[:2]) for p in self.posterior]

    def assessment(self):
        probabilities = self.control_probabilities()
        labels = ['controlled' if p > .95 else 'external' if p < .05 else 'unknown' for p in probabilities]
        unique = labels.index('controlled') if labels.count('controlled') == 1 and labels.count('external') == 1 else None
        return {'control_probabilities': probabilities, 'labels': labels, 'unique_controlled_channel': unique,
                'scope': 'action dependence; not body ownership or subjective selfhood'}


class ProbeAgent:
    def __init__(self, *, seed=0, policy='active', memory=None, episode='agency-discovery'):
        if policy not in ('active', 'random', 'passive'):
            raise ValueError('Unknown exploration policy')
        self.model = AttributionModel()
        self.rng = random.Random(seed)
        self.policy = policy
        self.memory, self.episode = memory, episode
        self.tick = 0
        self.pending = None
        self.trace = []

    def choose(self, cue, *, warmup=False):
        if self.pending is not None:
            raise RuntimeError('Assess the pending probe before choosing another')
        if cue not in (-1, 1):
            raise ValueError('Invalid public cue')
        scores = {(a, c): self.model.information_gain(c, a, cue) for a in (-1, 1) for c in (0, 1)}
        if warmup or self.policy == 'passive':
            action, channel = cue, self.tick % 2
            reason = 'coupled_policy'
        elif self.policy == 'random':
            action, channel = self.rng.choice((-1, 1)), self.rng.choice((0, 1))
            reason = 'uniform_probe'
        else:
            best = max(scores.values())
            choices = [key for key, value in scores.items() if abs(value-best) < 1e-12]
            action, channel = self.rng.choice(choices)
            reason = 'expected_information_gain'
        decision = {'tick': self.tick, 'cue': cue, 'action': action, 'channel': channel,
                    'reason': reason, 'expected_information_gain': scores[action, channel],
                    'before': self.model.assessment()}
        if self.memory is not None:
            decision['event_id'] = self.memory.append(self.episode, self.tick, 'decision',
                                                       'agency_probe', decision)
        self.pending = decision
        return dict(decision)

    def receive(self, observation):
        if self.pending is None:
            raise RuntimeError('No pending probe')
        decision = self.pending
        self.model.update(decision['channel'], decision['action'], decision['cue'], observation)
        record = {'decision': decision, 'observation': observation, 'after': self.model.assessment()}
        if self.memory is not None:
            if observation is not None:
                self.memory.observe(self.episode, self.tick, f"channel-{decision['channel']}",
                                    observation, source='anonymous_channel_sensor')
            self.memory.append(self.episode, self.tick, 'assessment', 'agency_probe', record)
        self.trace.append(record)
        self.pending = None
        self.tick += 1
        return record


class ConfoundedChannels:
    """Evaluation world. Only cue() and read(action, channel) reach the agent."""
    def __init__(self, seed, *, condition='single'):
        if condition not in ('single', 'missing', 'masked', 'none', 'twins', 'twins_missing'):
            raise ValueError('Unknown condition')
        self.rng = random.Random(seed)
        self.condition = condition
        self.selected_body = self.rng.randrange(2)  # evaluator-only physical label
        self.signs = [self.rng.choice((-1, 1)) for _ in range(2)]
        self.causes = ['cue', 'cue']
        if condition in ('single', 'missing', 'masked'):
            self.causes[self.selected_body] = 'action'
        elif condition in ('twins', 'twins_missing'):
            self.causes = ['action', 'action']
        self.current_cue = None

    def cue(self):
        self.current_cue = self.rng.choice((-1, 1))
        return self.current_cue

    def read(self, action, channel):
        if self.current_cue is None or action not in (-1, 1) or channel not in (0, 1):
            raise ValueError('A valid cue, action and channel are required')
        # Draw all channel noise independent of selected channel to couple policies.
        flips = [self.rng.random() < .05 for _ in range(2)]
        misses = [self.rng.random() < .5 for _ in range(2)]
        value = self.signs[channel]*(action if self.causes[channel] == 'action' else self.current_cue)
        self.current_cue = None
        if self.condition == 'masked' and channel == self.selected_body:
            return None
        if self.condition in ('missing', 'twins_missing') and misses[channel]:
            return None
        return -value if flips[channel] else value
