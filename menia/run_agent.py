"""Run the integrated agent on CPU and save a reviewable trace and learned model."""
import argparse
import json
from pathlib import Path
from .agent import SituatedAgent
from .core import Runtime
from .environments import VirtualRoom
from .episodic import EpisodeMemory
from .conversation import messages_for_runtime


def run(out, *, seed=17, steps=100, target=(4, 3), reverse_after=None, dropout=0.0, slip=0.0):
    if type(steps) is not int or steps < 1:
        raise ValueError("Positive bounded step count required")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    memory = EpisodeMemory(out/'history.sqlite')
    agent = SituatedAgent(memory=memory, episode=f'virtual-room-{seed}')
    runtime = Runtime(agent=agent)
    env = VirtualRoom(seed, target=target, dropout=dropout, slip=slip)
    trace = []
    reversed_controls = False
    try:
        for index in range(steps):
            if reverse_after is not None and len(agent.model.errors) >= reverse_after and not reversed_controls:
                env.reverse_controls()
                reversed_controls = True
            result = runtime.step(env)
            trace.append(result)
            if result['result']['kind'] == 'done':
                break
        context = runtime.context('Explique ta dernière décision et ce que tu as appris.')
        summary = {"seed": seed, "cycles": len(trace), "actions": len(env.moves),
            "goal_reached": env.position == env.target, "final_position": list(env.position),
            "target": list(env.target), "observed_transitions": len(agent.model.errors),
            "control_reversal": reversed_controls, "dropout": dropout, "slip": slip,
            "mean_forecast_brier": (sum(e['brier'] for e in agent.model.errors)/len(agent.model.errors)
                                    if agent.model.errors else None),
            "explanation": agent.explain(), "scope": "virtual-room online learning; no subjective-consciousness inference"}
        events = memory.recent(agent.episode, limit=1000)
        # A bounded run can exceed the UI window; export the complete ordered log.
        ids = memory.db.execute('SELECT id FROM events WHERE episode=? ORDER BY id', (agent.episode,)).fetchall()
        events = [memory.event(row[0]) for row in ids]
        artifacts = {'summary.json': summary, 'trace.json': trace, 'events.json': events,
                     'agent-state.json': agent.state(), 'context.json': context,
                     'language-messages.json': messages_for_runtime(runtime, 'Explique ta dernière décision et ses limites.')}
        for filename, value in artifacts.items():
            (out/filename).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        return summary
    finally:
        runtime.memory.close()
        memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='New output directory; existing directories are preserved')
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--target', nargs=2, type=int, default=[4, 3])
    parser.add_argument('--reverse-after', type=int)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--slip', type=float, default=0.0)
    args = parser.parse_args()
    print(json.dumps(run(args.out, seed=args.seed, steps=args.steps, target=args.target,
                         reverse_after=args.reverse_after, dropout=args.dropout, slip=args.slip),
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
