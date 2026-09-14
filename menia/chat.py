"""Interactive Menia: bounded virtual actions, typed history and local language."""
import argparse
import json
import shlex
from .agent import SituatedAgent
from .conversation import answer_question
from .core import Runtime
from .environments import VirtualRoom
from .episodic import EpisodeMemory


HELP = ('Commandes : /step, /run N (1–500), /observe position|target|landmark, '
        '/report CHAMP X Y, /why, /state, /history, /stop, /resume, /quit. '
        'Les questions en français utilisent Qwen si le modèle est chargé.')


class ChatSession:
    def __init__(self, runtime, environment, generate=None):
        self.runtime, self.environment, self.generate = runtime, environment, generate

    def handle(self, text):
        if not text.strip():
            return HELP
        if not text.startswith('/'):
            if self.generate is None:
                return 'Mode structuré actif. Utilise /why pour la décision enregistrée. ' + HELP
            return answer_question(self.runtime, text, self.generate)
        parts = shlex.split(text)
        command, args = parts[0], parts[1:]
        if command == '/help' and not args:
            return HELP
        if command == '/stop' and not args:
            self.runtime.stop()
            return 'Arrêt effectué.'
        if command == '/resume' and not args:
            self.runtime.resume(user_requested=True)
            return 'Reprise demandée par l’utilisateur.'
        if command in ('/step', '/run'):
            if (command == '/step' and args) or (command == '/run' and len(args) != 1):
                raise ValueError('Utiliser /step ou /run N')
            count = 1 if command == '/step' else int(args[0])
            if not 1 <= count <= 500:
                raise ValueError('Choisir entre 1 et 500 cycles')
            results = []
            for _ in range(count):
                result = self.runtime.step(self.environment)
                results.append(result['explanation'])
                if result['result']['kind'] == 'done':
                    break
            return '\n'.join(results)
        agent = self.runtime.agent
        if command == '/observe' and len(args) == 1:
            if self.runtime.stopped or not self.runtime.capabilities.memory:
                raise RuntimeError('Agent arrêté ou mémoire désactivée')
            return json.dumps(agent.attend(self.environment, args[0]), ensure_ascii=False)
        if command == '/report' and len(args) == 3:
            if self.runtime.stopped or not self.runtime.capabilities.memory:
                raise RuntimeError('Agent arrêté ou mémoire désactivée')
            event_id = agent.report(args[0], [int(args[1]), int(args[2])], source='user')
            return f'Témoignage enregistré sous la référence {event_id}, sans validation perceptive.'
        if command == '/why' and not args:
            return agent.explain()
        if command in ('/state', '/history') and not args:
            state = self.runtime.context('')['agent']
            return json.dumps(state['history'] if command == '/history' else state,
                              ensure_ascii=False, indent=2)
        raise ValueError('Commande inconnue ou arguments invalides. ' + HELP)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--target', nargs=2, type=int, default=[4, 3])
    storage = parser.add_mutually_exclusive_group()
    storage.add_argument('--journal', default=':memory:', help='Optional SQLite path for episode history only')
    storage.add_argument('--session', help='Directory to save and restore agent, learned effects, history and virtual environment')
    language = parser.add_mutually_exclusive_group()
    language.add_argument('--no-llm', action='store_true', help='Structured mode (the default)')
    language.add_argument('--llama-url', help='Local llama.cpp server, for example http://127.0.0.1:8766')
    language.add_argument('--hf', action='store_true', help='Experimental Hugging Face backend; not validated on this Windows CPU')
    parser.add_argument('--model', default='Qwen/Qwen3-1.7B')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    generate = None
    if args.llama_url:
        from .local_server import LocalServerGenerator
        generate = LocalServerGenerator(args.llama_url)
    elif args.hf:
        from .language import LocalGenerator
        generate = LocalGenerator(args.model, args.revision)
    from .session_store import open_session, save_session
    if args.session:
        runtime, environment = open_session(args.session, seed=args.seed, target=args.target)
        agent = runtime.agent
    else:
        agent = SituatedAgent(memory=EpisodeMemory(args.journal))
        runtime, environment = Runtime(agent=agent), VirtualRoom(args.seed, target=args.target)
    session = ChatSession(runtime, environment, generate)
    print(HELP)
    if runtime.stopped:
        print('Session restaurée à l’arrêt ; /resume permet de continuer.')
    try:
        while True:
            try:
                text = input('Vous > ')
            except EOFError:
                break
            if text.strip() == '/quit':
                break
            try:
                print('Menia > ' + session.handle(text))
            except (ValueError, RuntimeError, OSError) as error:
                print('Menia > ' + str(error))
            finally:
                if args.session:
                    save_session(args.session, runtime, environment)
    finally:
        runtime.stop()
        if args.session:
            save_session(args.session, runtime, environment)
        runtime.memory.close()
        agent.memory.close()


if __name__ == '__main__':
    main()
