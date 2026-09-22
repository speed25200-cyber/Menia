"""Actual Qwen backend; replay changes routing policies, never Qwen weights."""
import argparse
import json
from pathlib import Path

from research.activation_monitor_gpu import ActivationBackend, sample_with_states
from research.replay_controller import analyze, collect, export_controller, source_hash


class ReplayBackend(ActivationBackend):
    def __init__(self):
        super().__init__()
        self.metadata["replaySourceHash"] = source_hash()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    # A completed journal requires no GPU/model reload.
    if args.resume and args.journal.exists() and analyze(args.journal)["complete"]:
        export_controller(args.journal, args.journal.with_suffix('.policy.json'))
        print("Complete journal already present; no generation replayed.")
        return
    backend = ReplayBackend()
    try:
        collect(args.journal, backend, resume=args.resume)
    finally:
        if args.journal.exists():
            result = analyze(args.journal)
            args.journal.with_suffix('.summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
            if result['updates'] == ['round1','round2','round3']:
                export_controller(args.journal, args.journal.with_suffix('.policy.json'))


if __name__ == '__main__':
    main()
