"""User-driven symbolic session around the learned recurrent module.

No autonomous loop, sensor activation, process persistence, or replication.
The workspace is an explicit short trace, not a claimed global-workspace theory
implementation. Predictions are tagged as inference, never as observations.
"""
from collections import deque
from dataclasses import asdict
import json
import numpy as np
from menia.core import Capabilities
from .recurrent import RecurrentMemory
from .reliability import RecallPolicy


class CognitiveSession:
    def __init__(self, model: RecurrentMemory, capacity=32, *, policy: RecallPolicy = None):
        if not isinstance(capacity,int) or not 1 <= capacity <= 128:
            raise ValueError('Invalid episode capacity')
        self.model = model
        if policy is not None:
            policy.check_model(model)
        self.policy = policy
        self.capacity = capacity
        self.capabilities = Capabilities()
        self.state = model.zero()
        self.episodes = deque(maxlen=capacity)
        self.paused = False
        self.sequence = 0
        self.pending = None
        self.observation_age = None

    def observe(self, symbol=None):
        if self.paused:
            raise RuntimeError('Paused by user')
        if symbol is not None and (type(symbol) is not int or not 0 <= symbol < 4):
            raise ValueError('Symbol must be 0, 1, 2, 3 or None')
        if self.policy is not None:
            # An in-place training update invalidates the old calibration too.
            self.policy.check_model(self.model)
        x = np.zeros((1,5))
        if symbol is not None:
            x[0,symbol] = 1
            x[0,4] = 1
        self.state, probabilities, _ = self.model.step(x,self.state)
        p = probabilities[0]
        if symbol is not None:
            self.observation_age = 0
        elif self.observation_age is not None:
            self.observation_age += 1
        if symbol is not None:
            answer, answer_source, reason = symbol, 'observation', None
        else:
            reason = ('no_observation' if self.observation_age is None else
                      'uncalibrated_recall' if self.policy is None else
                      self.policy.reason(p, self.observation_age))
            answer = int(p.argmax()) if reason is None else None
            answer_source = 'recurrent_inference' if reason is None else 'unknown'
        self.sequence += 1
        self.pending = {'sequence':self.sequence, 'probabilities':p.tolist(),
                        'answer_symbol':answer, 'answer_source':answer_source}
        event = {'sequence':self.sequence, 'observed_symbol':symbol,
                 'inferred_last_symbol':answer if answer_source == 'recurrent_inference' else None,
                 'model_candidate':int(p.argmax()), 'class_probabilities':p.tolist(),
                 'answer_symbol':answer, 'answer_source':answer_source,
                 'abstained':answer is None, 'uncertainty_reason':reason,
                 'steps_since_observation':self.observation_age,
                 'scope':'four-symbol recall only', 'source':'user-provided symbolic input'}
        self.episodes.append(event)
        return json.loads(json.dumps(event))

    def assess(self, actual_symbol):
        """Score the already recorded prediction against external ground truth.

        The label is supplied AFTER prediction and never used to update weights.
        Exactly one assessment is allowed per prediction.
        """
        if self.paused:
            raise RuntimeError('Paused by user')
        if self.pending is None:
            raise ValueError('No unassessed prediction')
        if type(actual_symbol) is not int or not 0 <= actual_symbol < 4:
            raise ValueError('Invalid reference label')
        p = np.array(self.pending['probabilities'])
        result = {'sequence':self.pending['sequence'], 'reference':actual_symbol,
                  'correct':int(p.argmax()) == actual_symbol,
                  'answered':self.pending['answer_symbol'] is not None,
                  'answer_source':self.pending['answer_source'],
                  'answer_correct':(self.pending['answer_symbol'] == actual_symbol
                                    if self.pending['answer_symbol'] is not None else None),
                  'brier_multiclass':float(((p-np.eye(4)[actual_symbol])**2).sum())}
        self.pending = None
        self.episodes[-1]['assessment'] = result
        return json.loads(json.dumps(result))

    def context(self):
        if self.paused:
            raise RuntimeError('Paused by user')
        # Copies through JSON prevent clients from mutating the internal trace.
        return json.loads(json.dumps({'capabilities':asdict(self.capabilities),
            'module':'learned symbolic memory; not general self-awareness',
            'recall_limits':{'steps_since_observation':self.observation_age,
                             'calibrated_max_age':self.policy.max_age if self.policy else None,
                             'scope':'synthetic calibration family only'},
            'episodes':list(self.episodes)[-5:]}))

    def save_latest(self, memory, *, authorized=False):
        if self.paused:
            raise RuntimeError('Paused by user')
        if not self.episodes:
            raise ValueError('No episode')
        return memory.remember(json.dumps(self.episodes[-1],separators=(',',':')),
                               source='observation',authorized=authorized)

    def stop(self):
        self.paused = True

    def resume(self, *, user_requested=False):
        if not user_requested:
            raise PermissionError('Only user may resume')
        self.paused = False

    def clear(self):
        self.state = self.model.zero()
        self.episodes.clear()
        self.pending = None
        self.sequence = 0
        self.observation_age = None
        # Clearing never resumes a paused session.
