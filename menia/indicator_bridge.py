"""Menia's link to the indicator agent of the Atelier of the senses (version 5).

The agent lives in its own virtual world. What reaches Menia's language model is what the agent's
global workspace holds, each content with its age in steps, and a faithful rendering of the agent's
last decision: the goal its allostatic planner chose and the predicted values that decided it.
Contents that stay inside the modules (the latest position reading, the attention schema's estimate,
the monitor's trust) are not given: in a global workspace architecture, what is broadcast is what
can be reported. Nothing here asserts an experience; the context renders records. Stop and resume
follow Menia's rules: only the user resumes. Results of the agent: docs/INDICATOR_AGENT_V5_RESULTS.md.
"""
import json
from pathlib import Path

LIFE_STEPS = 48


def copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


def goal_text(goal):
    if goal == "charger":
        return "aller me recharger"
    if goal == "stay":
        return "rester sur place"
    if goal == "random":
        return "agir au hasard"
    return f"aller vers l'objet de la case {goal}"


class IndicatorAgentBridge:
    """A version 5 indicator agent, stepped one bounded step at a time, seen through its workspace."""

    def __init__(self, params, *, env_seed=930001000, mode="fixed", agent_seed=0):
        from research.indicator_agent_v5 import AgentV5
        from research.sense_atelier import SenseAtelier
        self.env = SenseAtelier(env_seed, mode, n_objects=3, charger_moves=True)
        self.agent = AgentV5(params, "agent", seed=agent_seed)
        self.obs, _ = self.env.reset()
        self.tick = 0
        self.total_reward = 0.0
        self.stopped = False
        self.last = None

    @classmethod
    def from_artifacts(cls, root="artifacts/indicator-agent-v5", seed=113, **kwargs):
        from research.indicator_agent import Params
        return cls(Params.load(Path(root) / f"params-{seed}.json"), **kwargs)

    def _active(self):
        if self.stopped:
            raise RuntimeError("Stopped by user")

    def stop(self):
        self.stopped = True

    def resume(self, *, user_requested=False):
        if not user_requested:
            raise PermissionError("Only the user may resume")
        self.stopped = False

    def cycle(self, environment=None):
        """One step of the agent in its own world; bounded, never loops. The argument is ignored."""
        self._active()
        if self.tick >= LIFE_STEPS:
            return {"result": {"kind": "done"}, "explanation": self.explain()}
        action, intent, rec = self.agent.step(self.obs)
        self.obs, _ = self.env.step(action, intent)
        self.tick += 1
        self.total_reward += float(self.obs["reward"])
        self.last = rec
        return copy_json({"decision": {"goal": rec["goal"], "plan_values": rec["plan_values"], "writer": rec["writers"][0],
                                       "alarm": rec.get("alarm")},
                          "result": {"kind": "action", "action": int(action), "tick": self.tick},
                          "explanation": self.explain()})

    def workspace(self):
        """The contents of the global workspace, each with the number of steps since it was written."""
        import numpy as np
        W, age = self.agent.W, self.agent.age
        vis = W["vis"]
        return {"position": {"square": int(np.argmax(W["pos"])), "confidence": round(float(np.max(W["pos"])), 3),
                             "age": int(age["pos"])},
                "body": {"body": int(np.argmax(W["body"])), "confidence": round(float(np.max(W["body"])), 3),
                         "age": int(age["body"])},
                "vision": {"present": [x for x in range(len(vis["presence"])) if vis["presence"][x]],
                           "values": {str(x): round(float(v), 3) for x, v in enumerate(vis["values"])
                                      if v is not None and vis["presence"][x]},
                           "age": int(age["vis"])},
                "interoception": {"energy": round(float(W["intero"][0]), 3), "satiety": round(float(W["intero"][1]), 3),
                                  "age": int(age["intero"])}}

    def explain(self):
        """A faithful rendering of the last decision record; it describes what was computed, nothing more."""
        rec = self.last
        if rec is None:
            return "Je n'ai pas encore agi dans cette vie."
        values = ", ".join(f"{goal_text(int(k) if k.isdigit() else k)} {v:.2f}" for k, v in rec["plan_values"].items())
        writer = {"pos": "ma position", "body": "mon corps", "vis": "ma vision", "intero": "mes besoins"}[rec["writers"][0]]
        alarm = " par une alarme" if rec.get("alarm") else ""
        return (f"Je choisis de {goal_text(rec['goal'])} : c'est le but dont mon modèle de mes besoins prévoit le meilleur "
                f"effet sur les seize prochains pas (valeurs prévues : {values}). Le dernier contenu entré dans mon espace "
                f"de travail est {writer}{alarm}.")

    def context(self):
        self._active()
        rec = self.last or {}
        return copy_json({"scope": "agent à indicateurs de l'Atelier des sens, version 5 : contenus de l'espace de travail "
                                   "global et dernière décision ; les états propres aux modules ne sont pas diffusés",
                          "tick": self.tick, "workspace": self.workspace(), "goal": rec.get("goal"),
                          "plan_values": rec.get("plan_values"), "last_writer": (rec.get("writers") or [None])[0],
                          "alarm": rec.get("alarm"), "explanation": self.explain()})
