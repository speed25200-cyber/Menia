"""Experimental Menia agent: source inference -> verification -> memory.

Source beliefs are recorded as inferences; they do not overwrite certified
observations. Weight learning happens in the separate reproducible trainer.
"""
import math
import numpy as np
from .episodic import EpisodeMemory


class SourceAgent:
    def __init__(self, monitor, *, memory=None, episode="source-agent", cost=.25, policy=None):
        if not 0 < cost < .5:
            raise ValueError("Verification cost must be between zero and .5")
        self.monitor = monitor
        self.policy = policy
        self.memory = memory if memory is not None else EpisodeMemory()
        self.episode, self.cost = episode, cost
        self.state = monitor.zero(1)
        self.feedback = (0., 0.)
        self.last_tick = -1
        self.stopped = False

    def stop(self):
        self.stopped = True

    def step(self, packet, verify):
        if self.stopped:
            raise RuntimeError("Agent stopped")
        if set(packet) != {"tick", "content", "strength", "trace", "imagination_intent"}:
            raise ValueError("Unexpected fields in source packet")
        tick = packet["tick"]
        if type(tick) is not int or tick != self.last_tick+1:
            raise ValueError("Non-consecutive source packet")
        if not isinstance(packet["content"], str) or not packet["content"]:
            raise ValueError("Missing content identity")
        if packet["imagination_intent"] not in (0, 1):
            raise ValueError("Invalid simulation intention")
        values = [packet["strength"], packet["trace"], packet["imagination_intent"], *self.feedback]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite source packet")
        self.state, probability = self.monitor.step(np.asarray([values]), self.state)
        q = float(probability[0])
        should_verify = (min(q, 1-q) > self.cost if self.policy is None
                         else bool(self.policy.choose(q, self.cost)))
        prediction = self.memory.append(self.episode, tick, "prediction", "learned_source_monitor",
            {"content": packet["content"], "p_external": q, "features": values,
             "state": self.state[0].tolist(), "verified": False})
        decision_payload = {"prediction_event": prediction, "verify": should_verify, "cost": self.cost,
                            "rule": "verify if min(q,1-q) exceeds verification cost"}
        if self.policy is not None:
            decision_payload.update(rule="minimize learned immediate action cost",
                                    action_costs=self.policy.values(q, self.cost).tolist(),
                                    policy_blind=self.policy.blind)
        decision = self.memory.append(self.episode, tick, "decision", "source_controller", decision_payload)
        self.last_tick = tick
        evidence = None
        if should_verify:
            try:
                external = verify()  # First and only access to current ground truth.
                if type(external) is not bool:
                    raise ValueError("Verifier must return a boolean")
            except Exception:
                self.stopped = True
                self.memory.append(self.episode, tick, "assessment", "source_verifier",
                                   {"decision_event": decision, "verification_failed": True})
                raise
            evidence = self.memory.observe(self.episode, tick, f"source:{tick}", external,
                                           source="source_verifier")
            attributed = external
            confidence = float(external)
            self.feedback = (1., float(external))
        else:
            attributed, confidence = q >= .5, q
            self.feedback = (0., 0.)
        stored = self.memory.append(self.episode, tick, "report", "source_memory",
            {"content": packet["content"], "attributed_external": attributed,
             "p_external": confidence, "verified": evidence is not None,
             "evidence_event": evidence, "decision_event": decision})
        return {"tick": tick, "content": packet["content"], "q_before_verification": q,
                "verified": should_verify, "attributed_external": attributed,
                "prediction_event": prediction, "decision_event": decision,
                "verification_event": evidence, "memory_event": stored}

    def snapshot(self):
        return {"episode": self.episode, "last_tick": self.last_tick,
                "state": self.state.tolist(), "feedback": list(self.feedback),
                "stopped": self.stopped, "monitor": self.monitor.payload(),
                "policy": None if self.policy is None else self.policy.payload()}
