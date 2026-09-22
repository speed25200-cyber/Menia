"""Exploratory, not pre-registered: are the AE-1 thresholds reachable, and at what cost in return?

The version 3 agents keep everything they learned except the choice of goal, replaced by a fixed need rule:
recharge when the energy read in the workspace is below 0.35, else go to the best known good object, else stay.
Output: artifacts/indicator-agent-v3/rule-arbitration.json.
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from .sense_atelier import SenseAtelier, LIFE
from .indicator_agent import Params
from .indicator_agent_v2 import AgentV3
from .indicator_experiment import SETS, SEEDS_V3, Tally, tally_life, summarize, life_seed

THRESHOLD = 0.35


class RuleAgent(AgentV3):
    def _greedy(self, phi, candidate):
        if float(self.W["intero"][0]) < THRESHOLD:
            return "charger"
        return candidate if candidate is not None else "stay"


def run(seed, root="artifacts/indicator-agent-v3", lives=200):
    params = Params.load(Path(root) / f"params-{seed}.json")
    out = {}
    for set_index, (name, (base, mode)) in enumerate(SETS.items()):
        tally = Tally()
        for i in range(lives):
            env = SenseAtelier(life_seed(base, i), mode, n_objects=3)
            obs, truth = env.reset()
            agent = RuleAgent(params, "agent", seed=seed * 1_000_000 + set_index * 10_000 + i)
            steps, rewards = [], []
            for _ in range(LIFE):
                action, intent, rec = agent.step(obs)
                steps.append((rec, truth))
                obs, truth = env.step(action, intent)
                rewards.append(obs["reward"])
            tally_life(tally, {"steps": steps, "rewards": rewards, "faints": env.faints,
                               "energy_faints": env.energy_faints}, name)
        out[name] = summarize(tally)
    learned = json.loads((Path(root) / f"report-{seed}.json").read_text())["evaluation"]["agent"]
    keys = ("return", "energy_faints", "faints", "charge_when_low", "object_when_high", "choice")
    return seed, {"rule": {k: out["R"].get(k) for k in keys}, "learned": {k: learned["R"].get(k) for k in keys}}


def main():
    with ProcessPoolExecutor(3) as pool:
        results = dict(pool.map(run, SEEDS_V3))
    summary = {who: {k: float(np.mean([results[s][who][k] for s in SEEDS_V3])) for k in results[SEEDS_V3[0]][who]}
               for who in ("rule", "learned")}
    out = {"status": "exploratoire, non pré-enregistré", "threshold": THRESHOLD, "per_seed": results, "summary": summary}
    Path("artifacts/indicator-agent-v3/rule-arbitration.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
