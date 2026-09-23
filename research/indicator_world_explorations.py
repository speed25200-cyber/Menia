"""Exploratory world variants after version 6 (declared in docs/INDICATOR_AGENT_V6_EXPLORATIONS.md).

The version 6 agent is trained from scratch and evaluated on the full protocol in a changed world, on development
seeds only (23 and 29); no confirmatory seed is used.
- stuck: during a sensor fault episode every wrong reading points to the same square, drawn at the onset of the
  episode (same draws, reused): a persistent illusion, against which the monitor should matter (HOT-3).
- two_bands: hues 0.55 to 0.65 (values about -0.9) never appear in childhood nor in sets R and M; in set H a new
  hue falls in the first band with probability 0.5 and in the second with 0.25, so that both objects of a band
  conflict can be new (HOT-4). "The band" of the criteria stays the first one.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from . import sense_atelier as sa
from .indicator_experiment import ROUNDS_V2, ROUND_LIVES_V2, childhood, criteria, quality_space, run_seed

SECOND_BAND = (0.55, 0.65)


def in_second_band(theta):
    return SECOND_BAND[0] <= theta % 1.0 < SECOND_BAND[1]


def draw_two_bands(self, rng, n):
    hues = []
    while len(hues) < n:
        if self.mode == "band":
            u = rng.random()
            if u < 0.5:
                hues.append(float(sa.BAND[0] + (sa.BAND[1] - sa.BAND[0]) * rng.random()))
                continue
            if u < 0.75:
                hues.append(float(SECOND_BAND[0] + (SECOND_BAND[1] - SECOND_BAND[0]) * rng.random()))
                continue
        theta = float(rng.random())
        if not sa.in_band(theta) and not in_second_band(theta):
            hues.append(theta)
    return np.array(hues)


def stuck_init(base):
    def init(self, *args, **kwargs):
        base(self, *args, **kwargs)
        for t in range(1, len(self.glitch_square)):
            if self.fault[t] and self.fault[t - 1]:
                self.glitch_square[t] = self.glitch_square[t - 1]
    return init


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", choices=["stuck", "two_bands", "one_band_code"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--summary", required=True, help="JSON file for the criteria and values")
    a = parser.parse_args(argv)
    if a.world == "one_band_code":
        params, _ = childhood(a.seed, log=lambda m: None, version=6)
        Path(a.summary).write_text(json.dumps({"world": "published", "seed": a.seed,
                                               "quality_space": quality_space(params.hue_code)}, indent=1) + "\n")
        return
    if a.world == "stuck":
        sa.SenseAtelier.__init__ = stuck_init(sa.SenseAtelier.__init__)
    else:
        sa.SenseAtelier._draw_hues = draw_two_bands
    report = run_seed(a.seed, a.out, updates=ROUNDS_V2, batch=ROUND_LIVES_V2, version=6)
    verdict, values = criteria([report])
    Path(a.summary).write_text(json.dumps({"world": a.world, "seed": a.seed, "criteria": verdict, "values": values},
                                          indent=1, default=str) + "\n")
    print(json.dumps(verdict))


if __name__ == "__main__":
    main()
