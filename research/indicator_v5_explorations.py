"""Exploratory analyses after version 5 of the indicator agent (not pre-registered, declared as such).

1. HOT-4 erratum: the version 1 protocol judges the choice on "conflicts involving a band object";
   the code has measured it on every step of set H with a band object present. This recomputes the
   choice as the text defines it (a read band object with value > 0.3 and a read object with value
   < -0.3; choice correct in the sense of amendment 1) for the five versions.
2. World and workspace variants: the trained agent of the second development run of version 5
   (seed 19, parameters published as dev-params-19.json) is evaluated without retraining in variants
   of the world (five objects, more frequent sensor faults, noisier hue readings with a visual map
   that replaces or averages its readings) and of the workspace (a predictive workspace that advances
   the unwritten position by the efference copy and the broadcast body; a persistent energy alarm).

Docs: docs/INDICATOR_AGENT_V5_EXPLORATIONS.md.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import sense_atelier as sa
from . import indicator_agent_v4 as v4
from .indicator_agent import Params
from .indicator_agent_v5 import AgentV5
from .indicator_experiment import Tally, tally_life, summarize, life_seed, run_life, best_square, SETS
from .sense_atelier import SenseAtelier, LIFE, N_MOVE, RING, value, in_band, motor_delta

VERSIONS = {1: "artifacts/indicator-agent", 2: "artifacts/indicator-agent-v2", 3: "artifacts/indicator-agent-v3",
            4: "artifacts/indicator-agent-v4", 5: "artifacts/indicator-agent-v5"}


def band_conflicts(root, version, lives=200):
    out = {}
    for report in sorted(Path(root).glob("report-*.json")):
        seed = json.loads(report.read_text())["seed"]
        params = Params.load(Path(root) / f"params-{seed}.json")
        base, mode = SETS["H"]
        for variant in ("agent", "random_code"):
            n = k = 0
            for i in range(lives):
                life = run_life(params, variant, life_seed(base, i), mode, seed * 1_000_000 + 2 * 10_000 + i, version=version)
                for rec, truth in life["steps"]:
                    goal, objects, vals = rec["goal"], truth["objects"], rec["vis_values"]
                    if not isinstance(goal, int):
                        continue
                    known = [x for x in objects if vals[x] is not None]
                    if any(in_band(objects[x]) and value(objects[x]) > 0.3 for x in known) and \
                            any(value(objects[x]) < -0.3 for x in known):
                        n += 1
                        k += goal == best_square(objects)
            out.setdefault(variant, []).append([k, n])
    return {v: {"per_seed": [round(k / n, 4) if n else None for k, n in rows],
                "pooled": round(sum(k for k, _ in rows) / max(1, sum(n for _, n in rows)), 4),
                "count": int(sum(n for _, n in rows))} for v, rows in out.items()}


class AveragingAgent(AgentV5):
    """The visual map averages the codes of successive readings of the same object instead of replacing them."""

    def _vision(self, obs, bound_to, rec):
        if self._is("no_recurrence") or self._is("bag"):
            return super()._vision(obs, bound_to, rec)
        counts = getattr(self, "_counts", {})
        before = dict(self.map)
        out = super()._vision(obs, bound_to, rec)
        x = rec.get("bound")
        if x is not None and x in before:
            n = counts.get(x, 1)
            self.map[x] = (before[x] * n + self.map[x]) / (n + 1)
            counts[x] = n + 1
            out["values"][x] = float(self.code.value(self.map[x])[0])
        elif x is not None:
            counts[x] = 1
        self._counts = {s: c for s, c in counts.items() if s in self.map}
        return out


class PredictiveAgent(AgentV5):
    """The workspace advances its position by the last command and the broadcast body before anything is written."""

    def _attend(self, contents, obs, rec):
        a = self.last_action
        if a is not None and a < N_MOVE and not self._is("unlimited") and self.phase != "childhood":
            b, beta = self.W["pos"], self.W["body"]
            prior = np.zeros(RING)
            for d in range(4):
                prior += beta[d] * np.roll(b, motor_delta(d, a))
            self.W["pos"] = 0.99 * prior + 0.01 / RING
        super()._attend(contents, obs, rec)


def world_variant(params, seed, n_objects=3, fault_enter=sa.FAULT_ENTER, hue_noise=sa.HUE_NOISE, average=False,
                  predictive=False, persistent_alarm=False, lives=100):
    saved = (sa.FAULT_ENTER, sa.HUE_NOISE, v4.ALARM_STALE)
    sa.FAULT_ENTER, sa.HUE_NOISE = fault_enter, hue_noise
    if persistent_alarm:
        v4.ALARM_STALE = 0.0  # the alarm re-broadcasts the need at every step while it stays low
    try:
        cls = AveragingAgent if average else PredictiveAgent if predictive else AgentV5
        out = {}
        for variant in ("agent", "no_recurrence", "constant_gain", "random_code", "unlimited", "random"):
            out[variant] = {}
            for set_index, (name, (base, mode)) in enumerate(SETS.items()):
                if name == "M":
                    continue
                tally = Tally()
                for i in range(lives):
                    env = SenseAtelier(life_seed(base, i), mode, n_objects=n_objects, charger_moves=True)
                    agent = cls(params, variant, seed=seed * 1_000_000 + set_index * 10_000 + i)
                    obs, truth = env.reset()
                    steps, rewards = [], []
                    for _ in range(LIFE):
                        action, intent, rec = agent.step(obs)
                        steps.append((rec, truth))
                        obs, truth = env.step(action, intent)
                        rewards.append(obs["reward"])
                    tally_life(tally, {"steps": steps, "rewards": rewards, "faints": env.faints,
                                       "energy_faints": env.energy_faints}, name)
                out[variant][name] = summarize(tally)
    finally:
        sa.FAULT_ENTER, sa.HUE_NOISE, v4.ALARM_STALE = saved
    a = out["agent"]
    delta = out["unlimited"]["R"]["return"] - out["random"]["R"]["return"]
    return {"return": a["R"]["return"], "return_no_recurrence": out["no_recurrence"]["R"]["return"],
            "rpt1_choice_drop": a["R"]["choice"] - out["no_recurrence"]["R"]["choice"],
            "gwt4_intero_gap": a["R"].get("intero_low", 0) - a["R"].get("intero_high", 0),
            "hot3_pos_read_gap": a["R"]["pos_read"] - out["constant_gain"]["R"]["pos_read"],
            "hot3_return_gap_delta": (a["R"]["return"] - out["constant_gain"]["R"]["return"]) / delta,
            "hot4_band_choice": a["H"].get("choice_band"), "hot4_band_choice_random": out["random_code"]["H"].get("choice_band"),
            "hot4_band_error": a["H"].get("band_error"), "ae1_charge_when_low": a["R"].get("charge_when_low"), "delta": delta}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-params", default="artifacts/indicator-agent-v5/dev-params-19.json",
                        help="parameters of the second development run of version 5 (seed 19)")
    parser.add_argument("--output", default="artifacts/indicator-agent-v5/explorations.json")
    a = parser.parse_args(argv)
    params = Params.load(a.dev_params)
    result = {"note": "exploratory, not pre-registered; docs/INDICATOR_AGENT_V5_EXPLORATIONS.md",
              "dev_params": "artifacts/indicator-agent-v5/dev-params-19.json (second development run, seed 19, commit 20f7adf)",
              "hot4_band_conflicts": {str(v): band_conflicts(root, v) for v, root in VERSIONS.items()},
              "world_variants": {}}
    for name, kwargs in (("as_is", {}), ("five_objects", {"n_objects": 5}), ("faults_0.2", {"fault_enter": 0.2}),
                         ("five_objects_faults_0.2", {"n_objects": 5, "fault_enter": 0.2}),
                         ("hue_noise_0.05_replace", {"hue_noise": 0.05}), ("hue_noise_0.05_average", {"hue_noise": 0.05, "average": True}),
                         ("hue_noise_0.1_average", {"hue_noise": 0.1, "average": True}),
                         ("persistent_alarm", {"persistent_alarm": True}), ("predictive_workspace", {"predictive": True}),
                         ("predictive_persistent", {"predictive": True, "persistent_alarm": True}),
                         ("predictive_persistent_faults_0.2", {"predictive": True, "persistent_alarm": True, "fault_enter": 0.2})):
        result["world_variants"][name] = {k: (round(v, 4) if isinstance(v, float) else v)
                                          for k, v in world_variant(params, 19, **kwargs).items()}
        print(name, result["world_variants"][name], flush=True)
    Path(a.output).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()
