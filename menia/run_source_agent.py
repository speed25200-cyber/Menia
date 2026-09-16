"""Run Menia's experimental learned source-monitor loop on a synthetic stream."""
import argparse
import json
from pathlib import Path
from .episodic import EpisodeMemory
from .source_agent import SourceAgent
from .source_environment import SourceWorld, CONDITIONS
from .source_monitor import SourceMonitor
from .source_policy import SourcePolicy


def run(checkpoint, out, *, seed=61000, steps=48, condition="standard", policy=None, cost=.25):
    if type(steps) is not int or not 1 <= steps <= 10000:
        raise ValueError("Step count must be between 1 and 10000")
    model = SourceMonitor.load(checkpoint)
    controller = None if policy is None else SourcePolicy.load(policy)
    if not 0 < cost < .5:
        raise ValueError("Verification cost must be between zero and .5")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    memory = EpisodeMemory(out/"history.sqlite")
    agent = SourceAgent(model, memory=memory, episode=f"source-{seed}", policy=controller, cost=cost)
    world = SourceWorld(seed, length=steps, condition=condition)
    try:
        trace = [agent.step(world.next_packet(), world.verify) for _ in range(steps)]
        ids = memory.db.execute("SELECT id FROM events WHERE episode=? ORDER BY id", (agent.episode,)).fetchall()
        events = [memory.event(row[0]) for row in ids]
        summary = {"steps": steps, "seed": seed, "condition": condition, "kind": model.kind,
                   "controller": "analytic" if controller is None else "learned", "cost": cost,
                   "verified": sum(row["verified"] for row in trace),
                   "scope": "synthetic source inference; not evidence of subjective consciousness"}
        for name, value in (("trace", trace), ("events", events), ("agent-state", agent.snapshot()), ("summary", summary)):
            (out/f"{name}.json").write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
                                           encoding="utf-8", newline="\n")
        return summary
    finally:
        memory.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=61000)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--condition", choices=CONDITIONS, default="standard")
    parser.add_argument("--policy", help="Learned action-cost policy checkpoint")
    parser.add_argument("--cost", type=float, default=.25)
    args = parser.parse_args()
    print(json.dumps(run(args.checkpoint, args.out, seed=args.seed, steps=args.steps,
                         condition=args.condition, policy=args.policy, cost=args.cost), indent=2))


if __name__ == "__main__":
    main()
