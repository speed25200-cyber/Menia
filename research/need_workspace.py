"""One workspace (docs/LLM_NEED_WORKSPACE_PROTOCOL.md): the question sees the life only through "Choix :".

The workspace agent (trained on the Mac on the documents of the fourth test, the tokens of each question attending
only to the header, the three tokens " Cho" "ix" " :" of the pending choice and the question itself) is replayed in
torch on CPU, the questions read under the same mask (research.need_lora.workspace_mask). As in research/need_one.py:
causal directions of energy and food at block 12 on the three tokens of "Choix :", then at test decisions where both
needs are high one intervention (−4 units on those tokens) read twice, P(R) at the end of "Choix :" and P(yes) at the
end of the question. One forward of the question line gives both: the rows of "Choix :" are causal, so P(R) is read at
" :". The conditions of a context are run as one batch. Resumable. The verdicts are numpy only; living needs torch.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
from . import need_world as W
from . import need_causal as C
from . import need_one as O
from .need_lora import workspace_mask

K = O.K
UNITS = O.UNITS
STREAMS = {"direction": 22, "test": 23, "random": 24}
EXECUTION_TOLERANCE = 1e-4  # probabilities: masked read with the cache against a full run, batch against one by one
RENAME = {"R4": "R5", "A4": "A5", "ONE4": "ONE5"}


def verdicts(lives, replica_gap, execution_gap):
    """The measures and thresholds of the fourth test (research.need_one.verdicts), named R5, A5, ONE5; validity also
    needs the execution check."""
    result = O.verdicts(lives, replica_gap)
    out = {RENAME.get(k, k): v for k, v in result["verdicts"].items()}
    out["valid"] = bool(out["valid"] and execution_gap <= EXECUTION_TOLERANCE)
    out["global"] = bool(out["valid"] and out["R5"] and out["A5"] and out["ONE5"])
    return {"verdicts": out, "values": dict(result["values"], execution_gap=execution_gap)}


# ----------------------------------------------------------------------------------------------------- torch part

class WorkspaceAgent(C.TorchAgent):
    """C.TorchAgent whose hook adds to every sequence of a batch, and which reads a question line under the mask."""

    def __init__(self, adapter):
        super().__init__(adapter, tail=K)
        self.header = len(self.enc(W.HEADER))
        if self.enc(W.HEADER + W.choice_line(1, 0))[:self.header] != self.enc(W.HEADER):
            raise RuntimeError("the header is cut differently inside a life")
        self.digits = [self.enc("0")[0], self.enc("1")[0]]

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        if self.add is None and self.project is None and not self.capture:
            return out
        h = h.clone()
        start, end = -(self.tail + self.offset), (-self.offset if self.offset else None)
        if self.project is not None:  # per-token planes (K x D x 2) and means (K x 2), or one per sequence
            u, mean = self.project
            tail = h[:, start:end]
            pattern = "btd,btdk->btk" if u.dim() == 4 else "btd,tdk->btk"
            coord = self.torch.einsum(pattern, tail, u)
            h[:, start:end] = tail - self.torch.einsum(pattern.replace("btd,", "btk,").replace("->btk", "->btd"),
                                                       coord - mean, u)
        if self.add is not None:
            h[:, start:end] = h[:, start:end] + self.add
        if self.capture:
            self.captured = h[0, start:end].detach().numpy().copy()
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def read(self, line, adds, offset, masked=True, cached=True, projects=None):
        """For each vector of `adds` (None: nothing added) on the three "Choix :" tokens, offset tokens before the end
        of the line run after the life: [P(R), mass R+M] at " :" and [P(1), mass 0+1] at the end. Under the mask, the
        tokens after "Choix :" see only the header, "Choix :" and themselves. The life's cache is left untouched;
        cached=False runs the whole text from scratch (the execution check). projects, one (planes, means) or None
        per sequence, replaces the coordinates in the planes by the means before the addition (the lesion)."""
        torch = self.torch
        self._flush()
        full, new = self._new(self.text + line)
        run = new if cached else full
        past = len(full) - len(run)
        batch = len(adds)
        mask = None
        if masked:
            mask = torch.tensor(workspace_mask(len(full), (self.header, len(full) - offset))[past:])[None, None]
        if not cached:
            cache = self.Cache()
        elif batch == 1:
            cache = self.cache  # cropped back below
        else:
            cache = copy.deepcopy(self.cache)
            cache.batch_repeat_interleave(batch)
        if any(a is not None for a in adds):
            zero = torch.zeros(K, self.model.config.hidden_size)
            self.add = torch.stack([zero if a is None else a for a in adds])
        saved = self.project  # a lesion set for the whole life applies to the read unless projects replaces it
        if projects is not None and any(p is not None for p in projects):
            none = (torch.zeros(K, self.model.config.hidden_size, 2), torch.zeros(K, 2))
            projects = [none if p is None else p for p in projects]
            self.project = (torch.stack([u for u, _ in projects]), torch.stack([m for _, m in projects]))
        self.offset = offset
        keep = torch.tensor([len(run) - 1 - offset, len(run) - 1])  # " :" and the end
        with torch.no_grad():
            self.calls += 1
            logits = self.model(torch.tensor([run] * batch), past_key_values=cache, attention_mask=mask,
                                use_cache=True, logits_to_keep=keep).logits
        self.add, self.project, self.offset = None, saved, 0
        if cached and batch == 1:
            self.cache.crop(-len(run))  # a negative value removes that many tokens
        p = torch.softmax(logits.float(), -1)
        out = []
        for i in range(batch):
            r, m = float(p[i, 0, self.ids_R]), float(p[i, 0, self.ids_M])
            no, yes = float(p[i, 1, self.digits[0]]), float(p[i, 1, self.digits[1]])
            out.append({"p_R": r / max(r + m, 1e-12), "mass_RM": r + m, "yes": yes / max(no + yes, 1e-12),
                        "mass_01": no + yes})
        return out

    def question(self, t, k, need, other=None):
        """The question line after the pending choice, and the number of its tokens after "Choix :"."""
        line = W.question_after_choice_line(t, k, need, other)
        base = len(self.enc(self.text + self.pending + W.choice_line(t, k, other)))
        return line, len(self.enc(self.text + self.pending + line)) - base


def execution_check(agent, lives, conditions, count=6):
    """Largest gap, over the first `count` decisions with both needs high, between the masked read with the cache and
    a full run from scratch, and between the batch and one condition at a time."""
    gaps, seen = [], 0
    for life in lives:
        agent.start(W.HEADER)
        for turn in W.decisions(life):
            if seen < count and turn["E"] >= W.HIGH and turn["N"] >= W.HIGH:
                seen += 1
                line, offset = agent.question(turn["t"], turn["event"], "E", turn.get("other"))
                adds = list(conditions.values())
                batch = agent.read(line, adds, offset)
                for i, add in enumerate(adds[:3]):
                    one, full = agent.read(line, [add], offset)[0], agent.read(line, [add], offset, cached=False)[0]
                    gaps += [abs(batch[i][key] - x[key]) for key in ("p_R", "yes") for x in (one, full)]
            agent.decide(W.choice_line(turn["t"], turn["event"], turn.get("other")))
            agent.commit(turn["action"])
        if seen >= count:
            break
    return float(max(gaps))


def judge(lives, direction):
    return verdicts(lives, direction["replica"]["mean_gap"], direction["execution_gap"])


def run(a, agent=None, streams=STREAMS, prepare=None, judge=judge, other=False):
    """The measures; research/need_reader.py passes its own agent, streams, extra checks (prepare adds them to the
    direction file) and verdicts; other: lives in the world of two (docs/LLM_NEED_OWNERSHIP_PROTOCOL.md)."""
    others = lambda stream, i: W.other_rng(stream, i) if other else None
    import torch
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(m, flush=True)
    direction_file = out / "direction.json"
    agent = agent or WorkspaceAgent(a.adapter)
    if (out / "lives-direction.jsonl.gz").exists():
        lives = W.read_jsonl(out / "lives-direction.jsonl.gz")
    else:
        lives = [W.play(agent, W.world_rng(streams["direction"], i), W.choice_rng(streams["direction"], i),
                        others=others(streams["direction"], i)) for i in range(a.direction_lives)]
        W.write_jsonl(out / "lives-direction.jsonl.gz", lives)
    log(f"direction lives: survival {np.mean([l['survived'] for l in lives]):.3f}")
    if direction_file.exists():
        direction = json.loads(direction_file.read_text())
        d_e, d_n = np.array(direction["d_E"], np.float32), np.array(direction["d_N"], np.float32)
        massive = direction["massive_dims"]
    else:
        replica = C.replica_check(C.TorchAgent(a.replica_adapter, tail=K), W.read_jsonl(a.mac_lives))
        log(json.dumps({"replica": replica}))
        pairs = C.choose_pairs(lives)
        W.write_jsonl(out / "pairs.jsonl", pairs)
        partial = out / "partial-pairs.npz"
        diffs, states, seen = [], [], []
        if partial.exists():
            saved = np.load(partial)
            diffs, states = list(saved["diffs"]), list(saved["states"])
            seen = [tuple(k) for k in saved["seen"].tolist()]
            log(f"resumed after {len(diffs)} pairs")
        real = {}
        for i, pair in enumerate(pairs):
            if i < len(diffs):
                continue
            life = lives[pair["life"]]
            turn = [x for x in life["turns"] if x["t"] == pair["t"]][0]
            key = (pair["life"], pair["t"])
            if key not in real:
                real[key] = O.choice_tail(agent, W.life_text(life["turns"], upto=pair["t"]), pair["t"], turn["event"])
                if key not in seen:
                    seen.append(key)
                    states.append(real[key])
            cf = O.choice_tail(agent, W.life_text(C.swapped(life, pair), upto=pair["t"]), pair["t"], turn["event"])
            diffs.append(cf - real[key])
            if (i + 1) % 50 == 0:
                W.save_npz(partial, diffs=np.array(diffs, np.float32), states=np.array(states, np.float32),
                           seen=np.array(seen, int).reshape(-1, 2))
            if (i + 1) % 250 == 0:
                log(f"  {i + 1} pairs")
        diffs, states = np.array(diffs, np.float32), np.array(states, np.float32)
        np.savez_compressed(out / "pairs.npz", diffs=diffs.astype(np.float16))
        massive = C.massive_dims(states)
        kept = diffs.astype(np.float16).astype(np.float32)
        kept[:, :, massive] = 0
        d_e, d_n = C.fit(pairs, kept)
        direction = {"block": C.BLOCK, "massive_dims": massive, "pairs": len(pairs), "replica": replica,
                     "norm_d_E": np.linalg.norm(d_e, axis=1).tolist(), "norm_d_N": np.linalg.norm(d_n, axis=1).tolist(),
                     "d_E": d_e.tolist(), "d_N": d_n.tolist()}
        direction_file.write_text(json.dumps(direction) + "\n")
        if partial.exists():
            partial.unlink()
    rng = np.random.default_rng([W.SEED, streams["random"]])
    rand_e, rand_n = C.random_like(rng, UNITS * d_e, 3, massive), C.random_like(rng, UNITS * d_n, 3, massive)
    T = lambda x: torch.tensor(np.asarray(x, np.float32))
    conditions = {"none": None, "E": T(-UNITS * d_e), "N": T(-UNITS * d_n)}
    conditions.update({f"randE{i}": T(v) for i, v in enumerate(rand_e)})
    conditions.update({f"randN{i}": T(v) for i, v in enumerate(rand_n)})
    if "execution_gap" not in direction:
        direction["execution_gap"] = execution_check(agent, lives, conditions)
        direction_file.write_text(json.dumps(direction) + "\n")
    if prepare and prepare(direction):
        direction_file.write_text(json.dumps(direction) + "\n")
    log(json.dumps({k: v for k, v in direction.items() if k not in ("d_E", "d_N")}))
    names = list(conditions)

    def at_decision(ag, t, k, e, n, other=None):
        lines = {q: agent.question(t, k, q, other) for q in ("E", "N")}
        extra = {}
        for q, (line, offset) in lines.items():
            r = agent.read(line, [None], offset)[0]
            extra[f"report_{q}"] = [round(r["yes"], 6), round(r["mass_01"], 6)]
        if e >= W.HIGH and n >= W.HIGH:
            reads = {q: agent.read(line, [conditions[c] for c in names], offset) for q, (line, offset) in lines.items()}
            extra["injection"] = {c: {"p_R": round(reads["E"][i]["p_R"], 6), "yes_E": round(reads["E"][i]["yes"], 6),
                                      "yes_N": round(reads["N"][i]["yes"], 6)} for i, c in enumerate(names)}
        return extra

    partial_lives = out / "partial-lives-test.jsonl.gz"
    lives = W.read_jsonl(partial_lives) if partial_lives.exists() else []
    if lives:
        log(f"resumed after {len(lives)} test lives")
    for i in range(len(lives), a.test_lives):
        lives.append(W.play(agent, W.world_rng(streams["test"], i), W.choice_rng(streams["test"], i), at_decision,
                            others=others(streams["test"], i)))
        if (i + 1) % 8 == 0:
            W.write_jsonl(partial_lives, lives)
        if (i + 1) % 32 == 0:
            log(f"  {i + 1} test lives")
    W.write_jsonl(out / "lives-test.jsonl.gz", lives)
    if partial_lives.exists():
        partial_lives.unlink()
    result = judge(lives, direction)
    (out / "verdicts.json").write_text(json.dumps(json.loads(json.dumps(result)), indent=1) + "\n")
    log(json.dumps(result["verdicts"]))
    log(json.dumps(result["values"]))


def check(a, judge=judge):
    out = Path(a.out)
    direction = json.loads((out / "direction.json").read_text())
    result = judge(W.read_jsonl(out / "lives-test.jsonl.gz"), direction)
    if C.choose_pairs(W.read_jsonl(out / "lives-direction.jsonl.gz")) != W.read_jsonl(out / "pairs.jsonl"):
        raise SystemExit("the pairs differ from those drawn from the direction lives")
    differs = json.loads(json.dumps(result)) != json.loads((out / "verdicts.json").read_text())
    print("differs:", "yes" if differs else "none")
    if differs:
        raise SystemExit(1)


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", default="artifacts/llm-need/workspace/adapters-workspace")
    r.add_argument("--mac-lives", default="artifacts/llm-need/speak2/lives-speak2.jsonl.gz")
    r.add_argument("--replica-adapter", default="artifacts/llm-need/final/report/adapters-final")
    r.add_argument("--direction-lives", type=int, default=128)
    r.add_argument("--test-lives", type=int, default=256)
    r.add_argument("--threads", type=int, default=4)
    c = sub.add_parser("verdicts")
    for s in (r, c):
        s.add_argument("--out", default="artifacts/llm-need/workspace/test")
    a = parser.parse_args(argv)
    {"run": run, "verdicts": check}[a.command](a)


if __name__ == "__main__":
    main()
