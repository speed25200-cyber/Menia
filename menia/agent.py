"""Integrated perception -> memory -> prediction -> action -> error feedback.

This executable agent uses online statistical learning and a transparent planner.
Its explanations describe recorded decisions; they do not assert consciousness.
"""
import json
import uuid
from .action_model import ActionModel
from .episodic import EpisodeMemory


def copy_json(value):
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))


class SituatedAgent:
    fields = ("position", "target", "landmark")

    def __init__(self, actions=("a", "b", "c", "d"), *, memory=None, model=None, episode=None):
        self.memory = memory if memory is not None else EpisodeMemory()
        if self.memory.capacity < 2:
            raise ValueError("This planner needs two working-memory slots")
        self.model = model if model is not None else ActionModel(actions)
        if tuple(actions) != self.model.actions:
            raise ValueError("Model and available actions differ")
        self.episode = episode or str(uuid.uuid4())
        self.tick = 0
        self.pending = None
        self.needs_resync = False
        self.stopped = False
        self.attention = {field: {"attempts": 0, "received": 0, "last_event": None} for field in self.fields}
        self.last_decision = None

    def _active(self):
        if self.stopped:
            raise RuntimeError("Stopped by user")

    def stop(self):
        self.stopped = True

    def resume(self, *, user_requested=False):
        if not user_requested:
            raise PermissionError("Only the user may resume")
        self.stopped = False

    def report(self, field, value, *, source):
        self._active()
        if field not in self.fields or not isinstance(source, str) or not source:
            raise ValueError("Unknown field or missing source")
        return self.memory.append(self.episode, self.tick, "report", source,
                                  {"field": field, "value": value, "verified": False})

    def _observation(self, field):
        return self.memory.latest(self.episode, field)

    def attend(self, environment, field):
        self._active()
        if field not in self.fields:
            raise ValueError("Unknown field")
        # A read selects one field. No inspection of environment internals.
        response = environment.read(field)
        monitor = self.attention[field]
        monitor["attempts"] += 1
        received = response is not None
        if received:
            if response.get("field") != field or type(response.get("tick")) is not int or response["tick"] < self.tick:
                raise ValueError("Mismatched or stale sensor receipt")
            value = response.get("value")
            if not isinstance(value, (list, tuple)) or len(value) != 2 or any(type(v) is not int for v in value):
                raise ValueError("Expected two integer coordinates")
            self.tick = response["tick"]
            event_id = self.memory.observe(self.episode, self.tick, field, value, source=response["source"])
            monitor["received"] += 1
            monitor["last_event"] = event_id
            if field == "position" and self.pending is not None:
                self._assess(value, event_id)
            if field == "position":
                self.needs_resync = False
        else:
            event_id = None
        attention_id = self.memory.append(self.episode, self.tick, "attention", "perceptual_controller",
            {"selected": field, "received": received, "observation_event": event_id,
             "unselected": [f for f in self.fields if f != field]})
        return {"kind": "attention", "field": field, "received": received,
                "event_id": attention_id, "observation_event": event_id}

    def _assess(self, position, observation_event):
        pending = self.pending
        delta = [p-q for p, q in zip(position, pending["before"])]
        attributable = (self.tick == pending["after_tick"] == pending["before_tick"]+1
                        and tuple(delta) in self.model.effects)
        if attributable:
            result = self.model.learn(pending["action"], delta, pending["forecast"])
            result["attributable"] = True
        else:
            result = {"attributable": False, "reason": "intervening_change_or_unsupported_effect",
                      "observed_displacement": delta}
        result.update({"prediction_event": pending["prediction_event"],
                       "action_event": pending["action_event"], "observation_event": observation_event})
        self.memory.append(self.episode, self.tick, "assessment", "transition_evaluator", result)
        self.pending = None

    def decide(self):
        self._active()
        if self.needs_resync:
            value = {"kind": "attend", "field": "position", "reason": "reconcile_action", "evidence": []}
        elif self.pending is not None:
            value = {"kind": "attend", "field": "position", "reason": "verify_action",
                     "evidence": [self.pending["prediction_event"]]}
        else:
            value = None
            for field in ("target", "position"):
                event = self._observation(field)
                if event is None or (field == "position" and event["tick"] != self.tick):
                    value = {"kind": "attend", "field": field, "reason": "missing_observation", "evidence": []}
                    break
                if (self.episode, field) not in self.memory.working:
                    value = {"kind": "recall", "field": field, "reason": "evicted_from_working_memory",
                             "evidence": [event["id"]]}
                    break
            if value is None:
                position, target = self._observation("position"), self._observation("target")
                evidence = [position["id"], target["id"]]
                pos, goal = position["payload"]["value"], target["payload"]["value"]
                if pos == goal:
                    value = {"kind": "done", "reason": "observed_at_target", "evidence": evidence}
                else:
                    value = {**self.action_decision(pos, goal), "evidence": evidence}
        event_id = self.memory.append(self.episode, self.tick, "decision", "model_based_planner", value)
        value["event_id"] = event_id
        self.last_decision = copy_json(value)
        return copy_json(value)

    def action_decision(self, position, target):
        least_seen = min(self.model.actions, key=lambda a: len(self.model.samples[a]))
        explore = len(self.model.samples[least_seen]) < 3
        candidates = {a: self.model.expected_distance(a, position, target) for a in self.model.actions}
        action = least_seen if explore else min(candidates, key=candidates.get)
        return {"kind": "act", "action": action,
                "reason": "learn_action_effect" if explore else "predicted_progress",
                "forecast": self.model.forecast(action), "expected_distances": candidates}

    def cycle(self, environment):
        """One bounded decision; never loops or starts a background process."""
        self._active()
        if tuple(environment.actions) != self.model.actions:
            raise ValueError("Environment action vocabulary changed")
        decision = self.decide()
        kind = decision["kind"]
        if kind == "attend":
            result = self.attend(environment, decision["field"])
        elif kind == "recall":
            event = self.memory.recall(self.episode, self.tick, decision["field"])
            result = {"kind": "retrieval", "event_id": event["id"] if event else None}
        elif kind == "act":
            before = self._observation("position")
            prediction_event = self.memory.append(self.episode, self.tick, "prediction", "learned_action_model",
                {**decision["forecast"], "decision_event": decision["event_id"],
                 "before_event": before["id"], "scope": "next_displacement"})
            action = decision["action"]
            # Commit prediction BEFORE dispatch; no ground truth is passed to decide.
            self.needs_resync = True
            try:
                receipt = environment.execute(action)
                if receipt.get("action") != action or receipt.get("executed") is not True or type(receipt.get("tick")) is not int or receipt["tick"] <= self.tick:
                    raise ValueError("Invalid execution receipt; inspect environment before retrying")
            except Exception:
                self.stopped = True
                self.memory.append(self.episode, self.tick, "action", "action_interface",
                    {"action": action, "execution_unknown": True, "prediction_event": prediction_event})
                raise
            action_event = self.memory.append(self.episode, receipt["tick"], "action", "action_interface",
                {**receipt, "prediction_event": prediction_event, "decision_event": decision["event_id"]})
            self.pending = {"action": action, "before": before["payload"]["value"],
                "before_tick": before["tick"], "after_tick": receipt["tick"],
                "forecast": decision["forecast"], "prediction_event": prediction_event, "action_event": action_event}
            self.tick = receipt["tick"]
            self.needs_resync = False
            result = {"kind": "action", "action": action, "event_id": action_event}
        else:
            result = {"kind": "done"}
        return {"decision": decision, "result": result, "explanation": self.explain(decision)}

    def explain(self, decision=None):
        """Deterministic faithful rendering of the pre-action decision record."""
        decision = decision or self.last_decision
        if decision is None:
            return "Je n'ai pas encore pris de décision dans cet épisode."
        reason = decision["reason"]
        if reason == "verify_action":
            text = "Je vérifie ma position pour comparer le résultat de ma commande à ma prédiction."
        elif reason == "reconcile_action":
            text = "Je vérifie ma position après une commande dont le résultat n'a pas été confirmé."
        elif reason == "missing_observation":
            text = f"Je sélectionne {decision['field']} : je n'ai pas d'observation utilisable de cette information."
        elif reason == "evicted_from_working_memory":
            text = f"Je recherche {decision['field']} dans mon histoire : cette information a quitté ma mémoire de travail."
        elif reason == "observed_at_target":
            text = "Ma position observée correspond à la cible observée ; je termine ce déplacement."
        elif reason == "uniform_control":
            text = f"Le contrôle expérimental choisit la commande {decision['action']} au hasard."
        else:
            forecast = decision["forecast"]
            if reason == "learn_action_effect":
                text = f"Je teste la commande {decision['action']} pour apprendre son effet ({forecast['samples']} transitions récentes observées)."
            else:
                text = (f"Je choisis {decision['action']} : mon modèle prévoit le déplacement {forecast['effect']} "
                        f"avec une probabilité de {forecast['confidence']:.0%}. Cette commande minimise la distance attendue à la cible.")
        return text + " Références : " + ", ".join(str(i) for i in decision["evidence"]) + "."

    def context(self):
        self._active()
        knowledge = {field: self.memory.knowledge(self.episode, field, self.tick) for field in self.fields}
        monitoring = {field: {**values,
            "estimated_receipt_probability": (values["received"]+1)/(values["attempts"]+2)}
            for field, values in self.attention.items()}
        return copy_json({"episode": self.episode, "tick": self.tick,
            "scope": "virtual environment; learned action effects and application observation receipts",
            "knowledge": knowledge, "attention": monitoring,
            "action_model": {a: self.model.forecast(a) for a in self.model.actions},
            "recent_errors": self.model.errors[-5:], "pending": self.pending,
            "decision": self.last_decision, "explanation": self.explain(),
            "history": self.memory.recent(self.episode)})

    def state(self):
        return copy_json({"version": 1, "episode": self.episode, "tick": self.tick,
            "model": self.model.state(), "pending": self.pending, "stopped": self.stopped,
            "needs_resync": self.needs_resync,
            "attention": self.attention, "last_decision": self.last_decision})

    @classmethod
    def from_state(cls, state, *, memory):
        if state["version"] != 1:
            raise ValueError("Unsupported agent state")
        model = ActionModel.from_state(state["model"])
        agent = cls(model.actions, memory=memory, model=model, episode=state["episode"])
        agent.tick = state["tick"]
        agent.pending = copy_json(state["pending"])
        agent.needs_resync = state["needs_resync"]
        agent.stopped = state["stopped"]
        agent.attention = copy_json(state["attention"])
        agent.last_decision = copy_json(state["last_decision"])
        if agent.pending:
            for key in ("prediction_event", "action_event"):
                if memory.event(agent.pending[key])["episode"] != agent.episode:
                    raise ValueError("History does not match saved pending action")
        return agent
