"""Interactive Menia with its indicator agent: bounded steps in the Atelier of the senses, the workspace as a journal.

The agent (version 6 by default) lives one life of 48 steps; the user steps it, stops it, and alone may resume it.
Questions in French go to a local language model when one is loaded, with the agent's context: the contents of its
global workspace, each with its age, its last decision, and the journal that states them as explicit conclusions,
without pronouns (docs/MENIA_REPORT_PROTOCOL.md). Nothing here asserts an experience.
"""
import argparse
import json
import shlex
from .conversation import answer_question
from .core import Runtime
from .indicator_bridge import IndicatorAgentBridge, LIFE_STEPS, decision_text, journal

HELP = ("Commandes : /step, /run N (1–48), /journal, /why, /state, /stop, /resume, /quit. "
        "Les questions en français utilisent le modèle de langage local s'il est chargé.")


class IndicatorChatSession:
    def __init__(self, runtime, generate=None):
        self.runtime, self.generate = runtime, generate

    def handle(self, text):
        if not text.strip():
            return HELP
        if not text.startswith("/"):
            if self.generate is None:
                return "Mode structuré actif. /journal donne l'espace de travail de l'agent. " + HELP
            return answer_question(self.runtime, text, self.generate)
        parts = shlex.split(text)
        command, args = parts[0], parts[1:]
        if command == "/help" and not args:
            return HELP
        if command == "/stop" and not args:
            self.runtime.stop()
            return "Arrêt effectué."
        if command == "/resume" and not args:
            self.runtime.resume(user_requested=True)
            return "Reprise demandée par l'utilisateur."
        if command in ("/step", "/run"):
            if (command == "/step" and args) or (command == "/run" and len(args) != 1):
                raise ValueError("Utiliser /step ou /run N")
            count = 1 if command == "/step" else int(args[0])
            if not 1 <= count <= LIFE_STEPS:
                raise ValueError(f"Choisir entre 1 et {LIFE_STEPS} pas")
            lines = []
            for _ in range(count):
                result = self.runtime.step(None)
                if result["result"]["kind"] == "done":
                    lines.append(f"La vie de l'agent est terminée ({LIFE_STEPS} pas).")
                    break
                lines.append(f"Pas {result['result']['tick']} : décision de l'agent, {decision_text(result['decision']['goal'])}.")
            return "\n".join(lines)
        context = self.runtime.context("")["agent"]
        if command == "/journal" and not args:
            return context["journal"] or "L'agent n'a pas encore agi dans cette vie."
        if command == "/why" and not args:
            return self.runtime.agent.explain()
        if command == "/state" and not args:
            return json.dumps(context, ensure_ascii=False, indent=2)
        raise ValueError("Commande inconnue ou arguments invalides. " + HELP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="artifacts/indicator-agent-v6", help="published run of the indicator agent")
    parser.add_argument("--seed", type=int, default=151, help="agent seed of that run")
    parser.add_argument("--life", type=int, default=0, help="test life of set R to live")
    parser.add_argument("--llama-url", help="Local llama.cpp server, for example http://127.0.0.1:8766")
    args = parser.parse_args()
    generate = None
    if args.llama_url:
        from .local_server import LocalServerGenerator
        generate = LocalServerGenerator(args.llama_url)
    bridge = IndicatorAgentBridge.from_artifacts(args.root, args.seed, env_seed=930001 * 1000 + args.life,
                                                 agent_seed=args.seed * 1_000_000 + args.life)
    runtime = Runtime(agent=bridge)
    session = IndicatorChatSession(runtime, generate)
    print(HELP)
    try:
        while True:
            try:
                text = input("Vous > ")
            except EOFError:
                break
            if text.strip() == "/quit":
                break
            try:
                print("Menia > " + session.handle(text))
            except (ValueError, RuntimeError, OSError) as error:
                print("Menia > " + str(error))
    finally:
        runtime.stop()
        runtime.memory.close()


if __name__ == "__main__":
    main()
