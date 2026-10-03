"""The second pilot of the curiosity test (docs/LLM_CURIOSITY_PROTOCOL.md, amendment 2), checked from its lives.

The pilot's sweep is recomputed from the published random lives of each candidate rate with the functions the Mac used
(research/curiosity_world.py: calibrate, domains_valid, interference) and compared with the published sweep. Numpy only.
"""
import argparse
import json
from pathlib import Path
from . import curiosity_world as CW
from .curiosity_mlx import RATES  # numpy only at import: mlx is imported inside its functions


def sweep(folder):
    out = []
    for lr in RATES:
        path = Path(folder) / f"pilot-random-{lr:g}.jsonl"
        if not path.exists():
            break
        lives = CW.read_lives(path)
        cal = CW.calibrate(lives)
        record = dict(lr=lr, calibration=cal, domains=CW.domains_valid(lives, cal["temperature"]),
                      interference=CW.interference(lives))
        checks = record["domains"]["checks"]
        record["passes"] = bool(checks["mots_learned"] and checks["base_learned"] and checks["suites_flat"]
                                and record["interference"]["small"])
        out.append(record)
        if record["passes"]:
            break
    return json.loads(json.dumps(out))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verdicts",))
    parser.add_argument("--out", default="artifacts/llm-curiosity/pilot-3/curiosity")
    a = parser.parse_args(argv)
    result = sweep(a.out)
    published = json.loads((Path(a.out) / "pilot-sweep.json").read_text())
    differs = result != published
    stopped = json.loads((Path(a.out) / "pilot.json").read_text()).get("stopped")
    if not any(r["passes"] for r in result) and stopped != "no rate passes the pilot":
        raise SystemExit("no rate passes, yet the pilot did not stop")
    print("differs:", "yes" if differs else "none", "| stopped:", stopped)
    if differs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
