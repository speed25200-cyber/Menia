"""Same four live-agent language checks, using a local llama.cpp server."""
import argparse
import hashlib
import json
from pathlib import Path
from menia.core import Runtime
from menia.conversation import answer_question, messages_for_runtime
from menia.local_server import LocalServerGenerator
from research.evaluate_agent_language import scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    generator = LocalServerGenerator(args.url)
    root = Path(__file__).resolve().parents[1]
    sources = ('menia/local_server.py', 'menia/conversation.py', 'menia/agent.py', 'menia/core.py',
               'research/evaluate_agent_language.py', 'research/evaluate_agent_server.py')
    manifest = {'backend': 'llama.cpp', 'model': generator.model_id, 'status': 'evaluating',
        'completed_cases': 0, 'temperature': 0, 'seed': 17, 'max_tokens': 192,
        'source_sha256': {p: hashlib.sha256((root/p).read_text(encoding='utf-8').encode('utf-8')).hexdigest() for p in sources},
        'scope': 'four exploratory language checks; outputs require content review'}
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
                manifest['completed_cases'] += 1
            finally:
                runtime.memory.close()
                agent.memory.close()
    manifest['status'] = 'completed'
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
