"""Exploratory held-out evaluation specified in INTEGRATED_AGENT_PROTOCOL.md."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from menia.agent import SituatedAgent
from menia.action_model import ActionModel
from menia.environments import VirtualRoom

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ('menia/agent.py', 'menia/action_model.py', 'menia/episodic.py',
           'menia/environments.py', 'menia/core.py', 'menia/conversation.py',
           'research/evaluate_agent.py', 'docs/INTEGRATED_AGENT_PROTOCOL.md')


class UniformAgent(SituatedAgent):
    def __init__(self, *args, seed, **kwargs):
        super().__init__(*args, **kwargs)
        self.policy_rng = random.Random(seed+10000)

    def action_decision(self, position, target):
        action = self.policy_rng.choice(self.model.actions)
        return {'kind': 'act', 'action': action, 'reason': 'uniform_control',
                'forecast': self.model.forecast(action)}


def fingerprints():
    return {p: hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode('utf-8')).hexdigest()
            for p in SOURCES}


def warmup(seed):
    agent = SituatedAgent(episode=f'train-{seed}')
    env = VirtualRoom(seed, target=(100, 100))
    try:
        for _ in range(100):
            agent.cycle(env)
            if len(agent.model.errors) == 12:
                if not all(len(samples) == 3 for samples in agent.model.samples.values()):
                    raise AssertionError('Unbalanced warmup')
                return agent.model.state()
        raise AssertionError('Warmup did not finish')
    finally:
        agent.memory.close()


def trial(seed, target, trained, variant, *, reverse, dropout):
    model = ActionModel.from_state(trained)
    model.errors = []
    model.learning_enabled = variant != 'frozen'
    agent_type = UniformAgent if variant == 'uniform' else SituatedAgent
    kwargs = {'seed': seed} if variant == 'uniform' else {}
    agent = agent_type(model=model, episode=f'eval-{seed}-{variant}-{reverse}-{dropout}', **kwargs)
    env = VirtualRoom(seed, target=target, dropout=dropout)
    if reverse:
        env.reverse_controls()
    try:
        done = False
        for index in range(160):
            result = agent.cycle(env)
            if result['result']['kind'] == 'done':
                done = True
                break
        # Reaching the target without observing it by the budget is insufficient.
        if done and env.position != env.target:
            raise AssertionError('False completion')
        errors = agent.model.errors
        return {'seed': seed, 'target': list(target), 'variant': variant,
                'reverse': reverse, 'dropout': dropout, 'success': done,
                'cycles': index+1, 'actions': len(env.moves), 'transitions': len(errors),
                'forecast_errors': sum(not e['correct'] for e in errors),
                'model_revisions': sum(e['regime_reset'] for e in errors),
                'mean_brier': sum(e['brier'] for e in errors)/len(errors) if errors else None,
                'final_position': list(env.position)}
    finally:
        agent.memory.close()


def evaluate():
    targets = ((7, 5), (-6, 4), (5, -7), (-4, -6))
    training = {str(seed): warmup(seed) for seed in range(101, 121)}
    rows = []
    for index, seed in enumerate(range(101, 121)):
        for reverse, dropout in ((False, 0.0), (True, 0.0), (False, 0.3), (True, 0.3)):
            for variant in ('adaptive', 'frozen', 'uniform'):
                rows.append(trial(seed, targets[index % len(targets)], training[str(seed)], variant,
                                  reverse=reverse, dropout=dropout))
    summary = []
    for reverse, dropout in ((False, 0.0), (True, 0.0), (False, 0.3), (True, 0.3)):
        for variant in ('adaptive', 'frozen', 'uniform'):
            group = [r for r in rows if (r['reverse'], r['dropout'], r['variant']) == (reverse, dropout, variant)]
            summary.append({'variant': variant, 'reverse': reverse, 'dropout': dropout,
                'n': len(group), 'successes': sum(r['success'] for r in group),
                'mean_cycles_all_trials': sum(r['cycles'] for r in group)/len(group),
                'mean_actions_all_trials': sum(r['actions'] for r in group)/len(group)})
    return {'schema': 1, 'scope': 'exploratory virtual-room functional learning; not consciousness',
            'source_sha256': fingerprints(), 'python': sys.version.split()[0],
            'training': training, 'trials': rows, 'summary': summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New output directory')
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(out)
    report = evaluate()
    out.mkdir(parents=True)
    (out/'report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
