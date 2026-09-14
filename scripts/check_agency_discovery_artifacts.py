"""Recompute both causal-attribution experiments, including the stronger control."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.evaluate_agency_discovery import evaluate as pilot
from research.evaluate_agency_fixed_probe import evaluate as fixed_control
from scripts.check_agent_artifacts import compare


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    for folder, evaluate in [('agency-discovery', pilot), ('agency-discovery-fixed-control', fixed_control)]:
        expected = json.loads((root/'artifacts'/folder/'report.json').read_text(encoding='utf-8'))
        compare(evaluate(), expected)
        print(folder + ': all trials, source fingerprints and summaries reproduced')
