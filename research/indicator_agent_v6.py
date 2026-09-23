"""The indicator agent, version 6 (docs/INDICATOR_AGENT_V6_PROTOCOL.md).

Version 5 with two changes. The monitor judges the position the workspace broadcast when the agent decided by
the consequence its need model predicts for it: a recharge on the charger's square, a meal on the square of an
object in view. When the consequence does not come, the belief was wrong: Pos lowers its belief on that square,
and the revised content competes for the workspace like any other. The attention schema controls binding: a hue
reading is bound only when the schema is confident of where the spotlight landed; otherwise the spotlight comes
back to the still unknown object. Under the constant gain ablation no belief is judged by its consequences.
"""
import numpy as np
from .indicator_agent_v5 import AgentV5

KEEP = 0.02  # belief kept on a square whose predicted consequence did not come
CHARGE_MARGIN = 0.15  # an energy reading this far under the learned charge level means no recharge
BIND = 0.8  # probability of the schema's estimate needed to bind a hue reading


class AgentV6(AgentV5):
    """Version 5 with beliefs judged by their consequences and binding controlled by the attention schema."""

    def __init__(self, params, variant="agent", seed=0, learn=False, phase="adult", epsilon=0.0):
        super().__init__(params, variant, seed, learn, phase, epsilon)
        self.acted_on = None

    def _pos(self, obs, rec):
        super()._pos(obs, rec)
        rec["violation"] = None
        acted, self.acted_on = self.acted_on, None
        if acted is None or self._is("constant_gain") or self.phase == "childhood":
            return
        square, charger, object_seen = acted
        m = self.model
        missed = None
        if square == charger and m.charge is not None:
            if obs["charger"] == charger and obs["energy"] < m.charge - CHARGE_MARGIN:
                missed = "charge"
        elif object_seen and bool(np.any(m.reward)):
            if obs["presence"][square] and square not in obs["onsets"] and obs["reward"] == 0:
                missed = "food"
        if missed is not None:
            self.b[square] *= KEEP
            self.b /= self.b.sum()
            rec["violation"] = missed

    def _schema(self, obs, rec):
        estimate = super()._schema(obs, rec)
        confidence = rec.get("schema_confidence")
        if confidence is not None and confidence < BIND and not self._is("bag"):
            rec["unbound"] = True
            return None
        return estimate

    def _policy(self, obs, rec):
        action = super()._policy(obs, rec)
        if self.phase != "childhood":
            square = int(np.argmax(self.W["pos"]))
            self.acted_on = (square, int(obs["charger"]), bool(self.W["vis"]["presence"][square]))
        return action
