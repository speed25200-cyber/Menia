"""Actual local Qwen outputs on live agent contexts, retained for manual review."""
import argparse
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
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    print('Loading Qwen3-1.7B for actual local inference...', flush=True)
    generator = LocalGenerator(revision=args.revision)
    (out/'manifest.json').write_text(json.dumps({'model': generator.model_id, 'revision': generator.revision,
        'scope': 'four exploratory language checks, not a general language-quality benchmark'}, indent=2)+'\n', encoding='utf-8')
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
                print(json.dumps({'case': name, 'response': response, 'metrics': generator.last_metrics}, ensure_ascii=False), flush=True)
            finally:
                agent.memory.close()
                runtime.memory.close()


if __name__ == '__main__':
    main()
