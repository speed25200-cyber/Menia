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


class CognitiveSession:
    def __init__(self, model: RecurrentMemory, capacity=32):
        if not isinstance(capacity,int) or not 1 <= capacity <= 128:
            raise ValueError('Invalid episode capacity')
        self.model = model
        self.capacity = capacity
        self.capabilities = Capabilities()
        self.state = model.zero()
        self.episodes = deque(maxlen=capacity)
        self.paused = False
        self.sequence = 0
        self.pending = None

    def observe(self, symbol=None):
        if self.paused:
            raise RuntimeError('Paused by user')
        if symbol is not None and (type(symbol) is not int or not 0 <= symbol < 4):
            raise ValueError('Symbol must be 0, 1, 2, 3 or None')
        x = np.zeros((1,5))
        if symbol is not None:
            x[0,symbol] = 1
            x[0,4] = 1
        self.state, probabilities, _ = self.model.step(x,self.state)
        p = probabilities[0]
        self.sequence += 1
        self.pending = {'sequence':self.sequence,'probabilities':p.tolist()}
        event = {'sequence':self.sequence, 'observed_symbol':symbol,
                 'inferred_last_symbol':int(p.argmax()), 'class_probabilities':p.tolist(),
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
        # Clearing never resumes a paused session.
