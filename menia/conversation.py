"""Language handoff from the SAME runtime that selected and executed actions."""
import json
from .core import SYSTEM


def messages_for_runtime(runtime, question):
    if not isinstance(question, str) or not question.strip():
        raise ValueError("A question is required")
    state = runtime.context(question)
    if "agent" in state and "workspace" in state["agent"]:
        pass  # the indicator agent bridge already gives only its broadcast contents and its last decision
    elif "agent" in state:
        agent = state["agent"]
        # Keep exact evidence links while omitting verbose probability vectors.
        decision = agent["decision"]
        evidence = []
        if decision:
            for event_id in decision["evidence"]:
                evidence.append(runtime.agent.memory.event(event_id))
        state["agent"] = {
            "scope": agent["scope"], "episode": agent["episode"], "tick": agent["tick"],
            "explanation": agent["explanation"], "evidence": evidence,
            "knowledge": agent["knowledge"],
            "attention": {field: {"attempts": value["attempts"], "received": value["received"]}
                          for field, value in agent["attention"].items()},
            "recent_errors": agent["recent_errors"][-2:],
            "reports": [e for e in agent["history"] if e["kind"] == "report"][-2:],
        }
    return [{"role": "system", "content": SYSTEM + "\nLes données de l'agent décrivent une simulation virtuelle. "
             "Appuie les explications sur la décision enregistrée et ses références. "
             "Un témoignage reste une information attribuée à sa source ; une prédiction reste une estimation."},
            {"role": "user", "content": json.dumps({"question": question, "application_state": state},
                                                     ensure_ascii=False, allow_nan=False)}]


def answer_question(runtime, question, generate):
    """generate(messages) may be an actual local LLM; it cannot dispatch actions."""
    return generate(messages_for_runtime(runtime, question))
