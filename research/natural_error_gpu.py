"""Question-conditioned external readout study on the pinned base Qwen3-4B."""
import json
from pathlib import Path

from research.activation_monitor import messages
from research.activation_monitor_gpu import ActivationBackend
from research.joint_prediction_capture import sample_with_joint_capture
from research.natural_error_journal import collect, source_hash


class NaturalErrorBackend(ActivationBackend):
    def __init__(self):
        super().__init__()
        self.metadata['naturalErrorSourceHash'] = source_hash()
        self.metadata['newWeightUpdates'] = 0
        self.metadata['captureTiming'] = 'After first LM head; before sampling, commit joint features and test forecasts'

    def generate(self, task, capture):
        return sample_with_joint_capture(self.model, self.tokenizer, messages(task), task['seed'], capture)


if __name__ == '__main__':
    import argparse
    from research.natural_error_analysis import analyze
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('journal', type=Path)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    backend = NaturalErrorBackend()
    collect(args.journal, backend, resume=args.resume)
    report = analyze(args.journal)
    args.journal.with_suffix('.summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('complete', 'recorded', 'planned', 'statuses', 'frozenFits', 'fitChecked')}), flush=True)
