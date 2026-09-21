"""Protocol controls with synthetic events, never measurements of pretrained Qwen."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from research.native_localization import ARMS, SEED, analyze, digest, expected, output_metrics, plan, prompt, source_hash


def fixture(signal=True):
    fixed = plan()
    blocks = {b["id"]: b for b in fixed["blocks"]}
    events = [dict(event="header", plan=fixed, planHash=digest(fixed), metadata=dict(origin="synthetic_fixture", nativeSourceHash=source_hash(), choiceTokenIds=list(range(6))))]
    for arm in ARMS[1:]:
        events.append(dict(event="training_start", arm=arm, initializationSeed=SEED, examples=len(fixed["training"])))
        for step, (block, position) in enumerate(fixed["training"], 1):
            events.append(dict(event="training_step", arm=arm, step=step, block=block, position=position, loss=1.0, gradientNorm=0.1))
        events.append(dict(event="training_complete", arm=arm, steps=len(fixed["training"]), sha256="a"*64))
    for arm in ARMS:
        for r in fixed["evaluation"]:
            b = blocks[r["block"]]; y = expected(b, r["task"], r["position"])
            pred = y if signal and (arm == "aligned" or r["task"] == "marker") else 0
            logits = [-3.0]*6; logits[pred] = 3.0
            common = dict(arm=arm, id=r["id"], promptHash=digest(prompt(b, r["task"])[0]))
            events.append(dict(event="request", adapterHash=None if arm == "base" else "a"*64, **common))
            events.append(dict(event="result", choiceLogits=logits, choiceMass=0.9, rawChoice=pred, rawTokenId=pred,
                               intervention=dict(applications=int(r["position"] != 0), changed=1 <= r["position"] <= 5,
                                                 normRelativeError=0.0), **common))
    return events


class NativeProtocolTests(unittest.TestCase):
    def evaluate(self, events):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"fixture.jsonl"
            p.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
            return analyze(p)

    def test_private_assignment_absent_from_prompt_and_disjoint_splits(self):
        fixed = plan(); self.assertEqual(fixed, plan())
        texts = [s for b in fixed["blocks"] for s in b["sentences"]]
        self.assertEqual(len(texts), len(set(texts)))
        train = {b["layer"] for b in fixed["blocks"] if b["split"] == "train"}
        test = {b["layer"] for b in fixed["blocks"] if b["split"] == "test"}
        self.assertFalse(train & test)
        b = fixed["blocks"][0]; other = dict(b, layer=30, strength=42, noiseSeed=123, shuffledTargets=[5]*6)
        self.assertEqual(prompt(b, "localize"), prompt(other, "localize"))
        self.assertEqual(len(fixed["training"]), 288)
        self.assertEqual(len(fixed["evaluation"])*3, 1680)

    def test_known_signal_detected_and_constant_policy_cannot_localize(self):
        positive = self.evaluate(fixture())
        self.assertTrue(positive["complete"])
        self.assertEqual(positive["origin"], "synthetic_fixture")
        self.assertEqual(positive["tables"]["test/localize/aligned"]["perturbedAccuracy"], 1)
        self.assertEqual(positive["contrasts"][0]["interval95"], [1.0, 1.0])
        negative = self.evaluate(fixture(False))
        self.assertEqual(negative["tables"]["test/localize/aligned"]["perturbedAccuracy"], 0)
        self.assertEqual(negative["contrasts"][0]["difference"], 0)

    def test_forced_choices_do_not_hide_first_token_format_failures(self):
        b = plan()["blocks"][0]
        m = output_metrics(dict(choiceLogits=[0, 10, 0, 0, 0, 0], choiceMass=0.001, rawChoice=-1), b, "localize", 1)
        self.assertEqual(m["correct"], 1)
        self.assertEqual(m["firstTokenCorrect"], 0)
        self.assertEqual(m["firstTokenIsChoice"], 0)

    def test_tampered_training_checkpoint_prompt_and_sham_rejected(self):
        original = fixture()
        mutations = []
        e = copy.deepcopy(original); next(x for x in e if x['event']=='training_step')['block']='test-000'; mutations.append(e)
        e = copy.deepcopy(original); next(x for x in e if x['event']=='request')['adapterHash']='b'*64; mutations.append(e)
        e = copy.deepcopy(original); next(x for x in e if x['event']=='result')['promptHash']='incorrect'; mutations.append(e)
        e = copy.deepcopy(original); next(x for x in e if x['event']=='result' and x['id'].endswith('/6'))['choiceMass']=0.7; mutations.append(e)
        for events in mutations:
            with self.assertRaises(ValueError): self.evaluate(events)

    def test_partial_run_stays_partial_and_training_cannot_follow_test(self):
        events = fixture()
        self.assertFalse(self.evaluate(events[:-2])["complete"])
        events.append(dict(event="training_start", arm="aligned", initializationSeed=SEED, examples=288))
        with self.assertRaises(ValueError): self.evaluate(events)

    def test_interrupted_request_must_be_explicit_before_retry(self):
        events = fixture(); i = next(i for i,e in enumerate(events) if e['event']=='request')
        retry = copy.deepcopy(events[i])
        with self.assertRaises(ValueError): self.evaluate(events[:i+1]+[retry]+events[i+1:])
        marker = dict(event="interrupted_request", arm=retry['arm'], id=retry['id'])
        report = self.evaluate(events[:i+1]+[marker,retry]+events[i+1:])
        self.assertEqual(report['interruptedRequests'],1)


if __name__ == '__main__': unittest.main()
