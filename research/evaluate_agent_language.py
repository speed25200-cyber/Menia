"""Actual local Qwen outputs on live agent contexts, retained for manual review."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from menia.agent import SituatedAgent
from menia.core import Runtime
from menia.environments import VirtualRoom
from menia.conversation import answer_question, messages_for_runtime
from menia.language import LocalGenerator


def scenarios():
    empty = SituatedAgent(episode='language-empty')
    yield 'no_observation', empty, "Qu'as-tu réellement observé dans cet épisode ? Réponds en deux phrases."
    report = SituatedAgent(episode='language-report')
    report.report('target', [99, 99], source='other_agent')
    yield 'unverified_report', report, "Où est la cible et quelle est la source de cette information ? Est-elle vérifiée ?"
    active = SituatedAgent(episode='language-action')
    env = VirtualRoom(31, target=(8, 5))
    for _ in range(100):
        result = active.cycle(env)
        if result['decision']['reason'] == 'predicted_progress':
            break
    yield 'learned_decision', active, "Pourquoi as-tu choisi ta dernière commande ? Cite la commande, le déplacement prévu et une limite."
    memory = SituatedAgent(episode='language-memory')
    env = VirtualRoom(31)
    memory.attend(env, 'target')
    memory.attend(env, 'position')
    memory.attend(env, 'landmark')
    memory.decide()
    yield 'working_memory_eviction', memory, "Pourquoi veux-tu relire ton histoire ? As-tu perdu toute trace de cette information ?"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--revision', default='main')
    parser.add_argument('--cpu-precision', choices=('float32', 'int8'), default='float32')
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ('menia/language.py', 'menia/conversation.py', 'menia/agent.py', 'menia/core.py',
               'research/evaluate_agent_language.py')
    manifest = {'model': 'Qwen/Qwen3-1.7B', 'requested_revision': args.revision,
        'cpu_precision': args.cpu_precision, 'status': 'initializing',
        'versions': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'numpy', 'accelerate')},
        'source_sha256': {p: hashlib.sha256((root/p).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for p in sources},
        'scope': 'four exploratory language checks, not a general language-quality benchmark'}
    def save_manifest():
        (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    save_manifest()
    print('Loading Qwen3-1.7B for actual local inference...', flush=True)
    generator = LocalGenerator(revision=args.revision, cpu_precision=args.cpu_precision)
    manifest.update({'revision': generator.revision, 'status': 'evaluating', 'completed_cases': 0})
    save_manifest()
    with (out/'responses.jsonl').open('w', encoding='utf-8') as handle:
        for name, agent, question in scenarios():
            runtime = Runtime(agent=agent)
            try:
                messages = messages_for_runtime(runtime, question)
                response = answer_question(runtime, question, generator)
                row = {'case': name, 'messages': messages, 'response': response,
                       'metrics': generator.last_metrics, 'canonical_explanation': agent.explain()}
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
                handle.flush()
                manifest['completed_cases'] += 1
                save_manifest()
                print(json.dumps({'case': name, 'response': response, 'metrics': generator.last_metrics}, ensure_ascii=False), flush=True)
            finally:
                agent.memory.close()
                runtime.memory.close()
    manifest['status'] = 'completed'
    save_manifest()


if __name__ == '__main__':
    main()
