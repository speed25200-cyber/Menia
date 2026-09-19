"""Recompute saved source-monitor evaluations without retraining or downloads."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from menia.source_monitor import SourceMonitor
from menia.source_environment import CONDITIONS, streams
from research.evaluate_source_monitor import full_evaluation, baselines, training_features


def main():
    directory = ROOT/"artifacts/learned-source-monitor"
    report = json.loads((directory/"report.json").read_text(encoding="utf-8"))
    assert report["status"] == "completed"
    for filename, digest in report["source_sha256"].items():
        assert hashlib.sha256((ROOT/filename).read_text(encoding="utf-8").encode()).hexdigest() == digest, filename
    identities = {(m["kind"], m["seed"]) for m in report["models"]}
    assert identities == {(kind, seed) for kind in ("recurrent", "reset", "window") for seed in (11, 23, 37)}
    assert len(report["models"]) == 9
    for entry in report["models"]:
        path = directory/entry["checkpoint"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["checkpoint_sha256"]
        assert len(entry["training_losses"]) == 300 and np.isfinite(entry["training_losses"]).all()
        model = SourceMonitor.load(path)
        assert model.kind == entry["kind"]
        recomputed = full_evaluation(model)
        for expected, actual in zip(entry["evaluation"], recomputed, strict=True):
            assert expected["condition"] == actual["condition"] and expected["seed"] == actual["seed"]
            assert set(expected["metrics"]) == set(actual["metrics"])
            for key, value in actual["metrics"].items():
                assert np.isclose(value, expected["metrics"][key], rtol=0, atol=1e-8), (entry["checkpoint"], key)
    for i, condition in enumerate(CONDITIONS):
        for name, values in baselines(condition, 51000+i*1000).items():
            for key, value in values.items():
                assert np.isclose(value, report["baselines"][condition][name][key], rtol=0, atol=1e-8)
    frames, truth = streams(41000, batch=512)
    mask = np.random.default_rng(41001).random(truth.shape) < .5
    features, targets = training_features(frames, truth, mask)
    assert int(mask.sum()) == report["training"]["verified_targets"]
    assert np.all(targets[~mask] == 0) and np.all(features[0, :, 3:] == 0)
    demo = directory/"demo"
    trace = json.loads((demo/"trace.json").read_text())
    events = json.loads((demo/"events.json").read_text())
    snapshot = json.loads((demo/"agent-state.json").read_text())
    by_id = {row["id"]: row for row in events}
    model = SourceMonitor.load(directory/"recurrent-11.json")
    frames, truth = streams(61000, batch=1, length=48)
    state, feedback = model.zero(1), np.zeros((1, 2))
    assert len(trace) == 48 and snapshot["monitor"] == model.payload()
    for t, row in enumerate(trace):
        state, q = model.step(np.column_stack((frames[t], feedback)), state)
        verify = min(q[0], 1-q[0]) > .25
        assert np.isclose(q[0], row["q_before_verification"], rtol=0, atol=1e-10)
        assert row["verified"] == verify
        assert row["prediction_event"] < row["decision_event"] < row["memory_event"]
        memory = by_id[row["memory_event"]]
        assert memory["kind"] == "report" and memory["payload"]["verified"] == verify
        if verify:
            assert row["decision_event"] < row["verification_event"] < row["memory_event"]
            evidence = by_id[row["verification_event"]]
            assert evidence["kind"] == "observation"
            assert evidence["payload"]["value"] == bool(truth[t, 0])
            assert row["attributed_external"] == bool(truth[t, 0])
            feedback = np.array([[1., truth[t, 0]]])
        else:
            assert row["verification_event"] is None
            assert row["attributed_external"] == (q[0] >= .5)
            feedback = np.zeros((1, 2))
    np.testing.assert_allclose(snapshot["state"], state, atol=1e-10, rtol=0)
    print("Verified 9 checkpoints, 45 evaluations, 15 references, target masking and the 48-step Menia journal")


if __name__ == "__main__":
    main()
