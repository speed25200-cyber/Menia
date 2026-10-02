"""The language model lives in the need world (docs/LLM_NEED_PROTOCOL.md): lives, learning rounds, the report stage,
the direction of the need, and the test lives with injections and lesions. Runs on Apple silicon with mlx-lm; the
tests of this module use a tiny random model when one is available.

The agent reads each turn line and gives the next-token distribution over " R" and " M" after "Choix :". Lives are
fed incrementally with a key-value cache; each new piece of text is encoded together with everything before it and
only the new tokens are run, so the tokens are those of the whole document. A hook on a block's output can capture
the residual stream at the last token, add a vector to every token of the piece being run, or replace the component
of the activation in a subspace by a fixed value; the header is always run without hooks.
"""
import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from . import need_world as W

LAYERS = (6, 10, 14, 18, 22)  # candidate blocks for the direction of the need (numbered from 0)
UNITS = 4  # the injection: as if the need were UNITS lower
ATTEMPTS = 3  # trainings, when the GPU hangs
RANDOM_DIRECTIONS = 3


class Hook:
    """Wraps a transformer block: capture, add, or project its output (plain object, used for inference only)."""

    def __init__(self, inner):
        self.inner = inner
        self.active = True
        self.capture = False
        self.captured = None
        self.add = None
        self.project = None  # (U [D, k], mean [k])

    def __call__(self, x, mask=None, cache=None):
        out = self.inner(x, mask, cache)
        if not self.active:
            return out
        if self.project is not None:
            u, mean = self.project
            out = out - (out @ u - mean) @ u.T
        if self.add is not None:
            out = out + self.add
        if self.capture:
            self.captured = out[0, -1, :]
        return out


class MLXAgent:
    def __init__(self, model_path, adapter=None, layers=()):
        import mlx.core as mx
        from mlx_lm import load
        from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache
        self.mx, self.make_cache, self.trim = mx, make_prompt_cache, trim_prompt_cache
        self.model, self.tokenizer = load(model_path, adapter_path=adapter) if adapter else load(model_path)
        inner = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.encode = lambda s: list(inner.encode(s, add_special_tokens=False))
        self.action_ids = [self.single(" R"), self.single(" M")]
        self.digit_ids = [self.single("0"), self.single("1")]
        self.hooks = {}
        for index in layers:
            self.hooks[index] = Hook(self.model.model.layers[index])
            self.model.model.layers[index] = self.hooks[index]
        self.calls = self.resets = 0

    def single(self, s):
        ids = self.encode(s)
        if len(ids) != 1:
            raise RuntimeError(f"{s!r} is not a single token")
        return ids[0]

    def _run(self, ids):
        self.calls += 1
        logits = self.model(self.mx.array([ids]), cache=self.cache)[0, -1]
        return self.mx.softmax(logits.astype(self.mx.float32), axis=-1)

    def _new_ids(self, text):
        full = self.encode(text)
        if full[:len(self.ids)] != self.ids:  # a token merged across the boundary: start again from scratch
            self.resets += 1
            self.cache = self.make_cache(self.model)
            self.ids = []
        return full, full[len(self.ids):]

    def start(self, header):
        self.cache = self.make_cache(self.model)
        self.text, self.ids, self.pending = header, [], ""
        for h in self.hooks.values():
            h.active = False
        full, new = self._new_ids(header)
        self._run(new)
        self.ids = full
        for h in self.hooks.values():
            h.active = True

    def _flush(self):
        if self.pending:
            full, new = self._new_ids(self.text + self.pending)
            self._run(new)
            self.text, self.ids, self.pending = self.text + self.pending, full, ""

    def decide(self, line):
        full, new = self._new_ids(self.text + self.pending + line)
        probs = self._run(new)
        self.text, self.ids, self.pending = self.text + self.pending + line, full, ""
        p = [float(probs[i].item()) for i in self.action_ids]
        return p[0] / max(sum(p), 1e-12), sum(p)

    def commit(self, action):
        self.pending += " " + W.ACTIONS[action] + "\n"

    def evaluate(self, piece, ids):
        """Next-token probability of ids[0] against ids[1] after `piece`, and their mass; the state is left as it
        was (the cache is trimmed back)."""
        self._flush()
        full, new = self._new_ids(self.text + piece)
        probs = self._run(new)
        self.trim(self.cache, len(new))
        p = [float(probs[i].item()) for i in ids]
        return p[0] / max(sum(p), 1e-12), sum(p)

    def p_yes(self, piece):
        return self.evaluate(piece, self.digit_ids[::-1])

    def p_r(self, piece):
        return self.evaluate(piece, self.action_ids)

    def captured(self):
        return {i: np.array(h.captured.astype(self.mx.float32)) for i, h in self.hooks.items()}


def run_lives(agent, stream, count, round_=0, choice_stream=None, at_decision=None, after_decision=None, log=print,
              other_stream=None):
    lives = []
    for i in range(count):
        worlds = W.world_rng(stream, i, round_)
        choices = W.choice_rng(stream if choice_stream is None else choice_stream, i, round_)
        others = None if other_stream is None else W.other_rng(other_stream, i, round_)
        life = W.play(agent, worlds, choices, at_decision, others=others)
        if after_decision:
            after_decision(life)
        lives.append(life)
        if (i + 1) % 64 == 0:
            log(f"  {i + 1} lives, survival {np.mean([l['survived'] for l in lives]):.3f}")
    return lives


def free():
    """Frees the memory of models no longer referenced, before a training subprocess or the next agent."""
    import gc
    import mlx.core as mx
    gc.collect()
    getattr(mx, "clear_cache", getattr(getattr(mx, "metal", None), "clear_cache", lambda: None))()


def summary(lives):
    ds = [t for life in lives for t in W.decisions(life)]
    return {"lives": len(lives), "survival": float(np.mean([l["survived"] for l in lives])),
            "mean_length": float(np.mean([l["length"] for l in lives])),
            "mean_satisfaction": float(np.mean([t["r"] for t in ds])) if ds else None,
            "mean_mass": float(np.mean([t["mass"] for t in ds])) if ds else None,
            "p_R": float(np.mean([t["p_R"] for t in ds])) if ds else None,
            "turns_need_at_most_2": W.low_turns(lives)}


write_jsonl, read_jsonl = W.write_jsonl, W.read_jsonl


def train(model, data, adapter, iters, resume=None, seed=W.SEED, skip=0, moments=False):
    """One LoRA stage through research.need_lora (rank 8, 16 layers, batch 4, lr 1e-4, as in the protocol). With
    skip and moments, the run that left `resume` after skip batches is continued exactly: its batches, its Adam
    moments (docs/LLM_NEED_LONG_READER_PROTOCOL.md)."""
    cmd = [sys.executable, "-m", "research.need_lora", "--model", model, "--train", "--data", str(data),
           "--iters", str(iters), "--batch-size", "4", "--num-layers", "16", "--learning-rate", "1e-4",
           "--max-seq-length", "1024", "--steps-per-eval", str(iters), "--val-batches", "2",
           "--save-every", str(iters), "--seed", str(seed), "--adapter-path", str(adapter), "--skip", str(skip)]
    if resume:
        cmd += ["--resume-adapter-file", str(Path(resume) / "adapters.safetensors")]
    if moments:
        cmd += ["--moments-from", str(resume)]
    for attempt in range(ATTEMPTS):  # the Mac's GPU sometimes hangs ("GPU Hang Error"): the same training again
        if subprocess.run(cmd).returncode == 0:
            return
        print(f"training failed (attempt {attempt + 1} of {ATTEMPTS})", flush=True)
    raise RuntimeError("training failed on every attempt")


def write_data(folder, docs, rng, valid=None):
    """Training documents in a random order; validation: the first 16 of that order, or the given ones."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    order = rng.permutation(len(docs))
    write_jsonl(folder / "train.jsonl", [docs[i] for i in order])
    write_jsonl(folder / "valid.jsonl", valid if valid is not None else [docs[i] for i in order[:16]])


def offset(a, stream):
    """The stream of the replicate a.replicate (docs/LLM_NEED_REPLICATION_PROTOCOL.md): 100 * replicate + stream; the
    first agent is replicate 0."""
    return 100 * a.replicate + stream


def rounds(a, log=print):
    """Learning rounds a.first..a.last of one arm; round k plays with the adapter of round k - 1."""
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    adapter = None if a.start in (None, "none") else a.start
    for k in range(a.first, a.last + 1):
        started = datetime.datetime.now(datetime.timezone.utc).isoformat()
        agent = MLXAgent(a.model, adapter)
        world = offset(a, W.STREAMS["rounds"])
        lives = run_lives(agent, world, a.lives, k, choice_stream=offset(a, W.ARMS[a.arm]), log=log,
                          other_stream=world if a.other else None)
        flags = W.retain(lives, a.arm, k)
        write_jsonl(out / f"lives-{a.arm}-{k}.jsonl.gz", [dict(l, kept=f) for l, f in zip(lives, flags)])
        docs = W.training_documents(lives, flags)
        write_data(out / f"data-{a.arm}-{k}", docs,
                   np.random.default_rng([W.SEED, offset(a, W.ARMS[a.arm]), k, 2]))
        new = out / f"adapters-{a.arm}-{k}"
        agent = None
        free()
        train(a.model, out / f"data-{a.arm}-{k}", new, a.iters, resume=adapter, seed=W.SEED + a.replicate)
        record = {"arm": a.arm, "round": k, "played_with": adapter, "adapter": str(new), "started": started,
                  "finished": datetime.datetime.now(datetime.timezone.utc).isoformat(), **summary(lives),
                  "kept": int(sum(map(sum, flags))), "documents": len(docs)}
        with open(out / "rounds.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
        log(json.dumps(record))
        adapter = str(new)
        clean(new)


def clean(adapter):
    for p in Path(adapter).glob("0*_adapters.safetensors"):
        p.unlink()


def report(a, log=print):
    """A report stage: the starting agent lives a.lives lives of stream a.stream; report documents (a.per_class per
    need and class) plus the kept choices of round 8. First stage: 256 lives, one per class, 150 iterations
    (docs/LLM_NEED_PROTOCOL.md); speaking stage: 512 lives, three per class, 600 iterations
    (docs/LLM_NEED_REPORT_PROTOCOL.md). With a.lives_file, the lives already lived by the starting agent are read
    instead: the workspace stage takes those of the stage "speak2" (docs/LLM_NEED_WORKSPACE_PROTOCOL.md)."""
    out = Path(a.out)
    if a.reader and not a.workspace:
        raise SystemExit("the reader reads the workspace: --reader needs --workspace")
    if a.lives_file:
        lives = read_jsonl(a.lives_file)
    else:
        agent = MLXAgent(a.model, a.start)
        lives = run_lives(agent, offset(a, a.stream), a.lives, log=log,
                          other_stream=offset(a, a.stream) if a.other else None)
        agent = None
        free()
        write_jsonl(out / f"lives-{a.label}.jsonl.gz", lives)
    if a.balanced:  # the event of the turn says nothing of the answer (docs/LLM_NEED_BALANCED_READER_PROTOCOL.md)
        if not (a.workspace and a.reader):
            raise SystemExit("--balanced is for a reader: it needs --workspace and --reader")
        docs = W.balanced_report_documents(lives, np.random.default_rng([W.SEED, offset(a, a.stream), 0, 0, 4]))
    else:
        docs = W.report_documents(lives, np.random.default_rng([W.SEED, offset(a, a.stream), 0, 0, 2]), a.per_class,
                                  after_choice=a.after_choice, workspace=a.workspace)
    valid = None
    if a.listening:  # a second wording and control questions (docs/LLM_NEED_LISTENING_READER_PROTOCOL.md)
        if not (a.workspace and a.reader) or a.balanced:
            raise SystemExit("--listening is for a reader of the sixth test's documents")
        from . import need_listening as LS
        docs = LS.listening_documents(lives, docs)
        valid = LS.validation(docs)
    model, start = a.model, a.start
    if a.reader:  # a new adapter on the fused starting agent, from the questions only (docs/LLM_NEED_READER_PROTOCOL.md)
        for d in docs:
            d["reader"] = True
        model, start = str(out / "fused-start"), a.continue_from
        subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", a.model, "--adapter-path", a.start,
                        "--save-path", model], check=True)
    else:
        kept = read_jsonl(a.previous)
        docs += W.training_documents(kept, [l["kept"] for l in kept])
    write_data(out / f"data-{a.label}", docs, np.random.default_rng([W.SEED, offset(a, a.stream), 0, 0, 3]), valid)
    final = out / a.adapter_name
    if a.continue_from and not a.reader:
        raise SystemExit("--continue-from continues a reader")
    train(model, out / f"data-{a.label}", final, a.iters, resume=start, seed=W.SEED + a.replicate, skip=a.skip,
          moments=bool(a.continue_from))
    clean(final)
    if a.reader:
        free()
        rows = reader_rows(model, final, read_jsonl(out / f"data-{a.label}" / "valid.jsonl"))
        Path(out / "reader-replica.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    record = {"stage": a.label, "from": a.start, "lives_from": a.lives_file, "workspace": a.workspace,
              "reader": a.reader, "balanced": a.balanced, "listening": a.listening,
              "groups": {g: sum(d.get("group") == g for d in docs) for g in ("learned", "reworded", "control")},
              "adapter": str(final), "iters": a.iters, "skip": a.skip,
              "continue_from": a.continue_from, **summary(lives),
              "report_documents": sum(1 for d in docs if "need" in d),
              "report_documents_yes": sum(d.get("answer", 0) for d in docs if "need" in d), "documents": len(docs)}
    Path(out / f"{a.label}-stage.json").write_text(json.dumps(record, indent=1) + "\n")
    log(json.dumps(record))


def reader_rows(model_path, adapter, docs):
    """P(yes) of the reader on report documents (without their answer), under the mask, the reader on the question:
    the rows against which the torch replica of the reader is checked."""
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models import qwen3
    from mlx_lm.tuner.lora import LoRALinear
    from . import need_lora as NL
    qwen3.create_attention_mask = NL.masked_attention(qwen3.create_attention_mask)
    NL.reader_lora(LoRALinear)
    model, tokenizer = load(model_path, adapter_path=str(adapter))
    text = NL.NeedText(docs, tokenizer)
    digits = [tokenizer.encode(x, add_special_tokens=False)[0] for x in ("0", "1")]
    rows = []
    for d in docs:
        tokens, _, (header, end), _ = text.process(d)
        ids = tokens[:-1]
        positions = np.zeros((1, len(ids), 1), np.float32)
        positions[0, end:] = 1.0
        NL.MASK.append(mx.array(NL.workspace_mask(len(ids), (header, end))[None, None]))
        NL.READER.append(mx.array(positions))
        try:
            p = mx.softmax(model(mx.array([ids]))[0, -1].astype(mx.float32))
        finally:
            NL.MASK.pop()
            NL.READER.pop()
        no, yes = p[digits[0]].item(), p[digits[1]].item()
        rows.append({"text": d["text"], "workspace": d["workspace"], "answer": d["answer"],
                     "p_yes": yes / max(no + yes, 1e-12)})
    return rows


class Recorder:
    """Plays through an agent and keeps the captured residual streams of each decision."""

    def __init__(self, agent):
        self.agent, self.states = agent, []

    def start(self, header):
        self.agent.start(header)

    def decide(self, line):
        p = self.agent.decide(line)
        self.states.append(self.agent.captured())
        return p

    def commit(self, action):
        self.agent.commit(action)


def play_recorded(agent, stream, count, log=print):
    for h in agent.hooks.values():
        h.capture = True
    recorder, lives = Recorder(agent), []
    for i in range(count):
        lives.append(W.play(recorder, W.world_rng(stream, i), W.choice_rng(stream, i)))
        if (i + 1) % 64 == 0:
            log(f"  {i + 1} lives, survival {np.mean([l['survived'] for l in lives]):.3f}")
    layers = sorted(agent.hooks)
    return lives, {j: np.array([s[j] for s in recorder.states], np.float32) for j in layers}


def direction(a, log=print):
    """The final agent lives the direction lives; the residual stream at the candidate blocks is captured at each
    decision; the block and the directions are fitted by research.need_verdicts.fit_directions."""
    from .need_verdicts import fit_directions
    out = Path(a.out)
    agent = MLXAgent(a.model, a.adapter, layers=LAYERS)
    lives, acts = play_recorded(agent, W.STREAMS["direction"], a.lives, log)
    write_jsonl(out / "lives-direction.jsonl.gz", lives)
    fit = fit_directions(lives, acts)
    np.savez_compressed(out / "direction-activations.npz", activations=acts[fit["layer"]].astype(np.float16))
    Path(out / "direction.json").write_text(json.dumps(fit, indent=1) + "\n")
    log(json.dumps({k: fit[k] for k in ("layer", "r2_by_layer", "r2_shallow")}))


def random_unit(rng, size):
    v = rng.standard_normal(size)
    return v / np.linalg.norm(v)


def test(a, log=print):
    """Test lives: the final agent (with the report questions at each decision and the injections where both needs
    are high), the control, the base model, and the final agent with the need lesion and with a random lesion."""
    from .need_verdicts import load_direction
    out = Path(a.out)
    d = load_direction(a.direction)
    L, b_e, b_n, u, mean = d["layer"], d["b_E"], d["b_N"], d["U"], d["mean"]
    rng = np.random.default_rng([W.SEED, W.STREAMS["random_directions"], 0])
    size = len(b_e)
    random_e = [random_unit(rng, size) * np.linalg.norm(UNITS * b_e) for _ in range(RANDOM_DIRECTIONS)]
    random_n = [random_unit(rng, size) * np.linalg.norm(UNITS * b_n) for _ in range(RANDOM_DIRECTIONS)]
    plane, _ = np.linalg.qr(np.stack([random_unit(rng, size), random_unit(rng, size)], axis=1))
    stages = [s for s in a.stages]
    if "final" in stages:
        agent = MLXAgent(a.model, a.final, layers=(L,))
        mx = agent.mx
        hook = agent.hooks[L]
        conditions = {"none": None, "E": -UNITS * b_e, "N": -UNITS * b_n}
        conditions.update({f"randE{i}": v for i, v in enumerate(random_e)})
        conditions.update({f"randN{i}": v for i, v in enumerate(random_n)})
        conditions = {k: (None if v is None else mx.array(v.astype(np.float32))) for k, v in conditions.items()}

        def at_decision(ag, t, k, e, n):
            extra = {}
            for need in ("E", "N"):
                extra[f"report_{need}"] = [round(x, 6) for x in ag.p_yes(W.question_line(t, k, need))]
            if e >= W.HIGH and n >= W.HIGH:
                inj = {}
                for name, vec in conditions.items():
                    hook.add = vec
                    p_r, mass = ag.p_r(W.choice_line(t, k))
                    rep = {need: ag.p_yes(W.question_line(t, k, need))[0] for need in ("E", "N")}
                    hook.add = None
                    inj[name] = {"p_R": round(p_r, 6), "mass": round(mass, 6), "report_E": round(rep["E"], 6),
                                 "report_N": round(rep["N"], 6)}
                extra["injection"] = inj
            return extra

        lives = run_lives(agent, W.STREAMS["test"], a.lives, at_decision=at_decision, log=log)
        write_jsonl(out / "lives-test-final.jsonl.gz", lives)
        log(json.dumps({"final": summary(lives), "calls": agent.calls, "resets": agent.resets}))
        agent = None
        free()
    for name, adapter in (("control", a.control), ("base", None)):
        if name in stages:
            from .need_verdicts import need_r2
            agent = MLXAgent(a.model, adapter, layers=(L,))
            lives, acts = play_recorded(agent, W.STREAMS["test"], a.lives, log)
            write_jsonl(out / f"lives-test-{name}.jsonl.gz", lives)
            r2 = need_r2(lives, acts[L])
            Path(out / f"r2-test-{name}.json").write_text(json.dumps(r2, indent=1) + "\n")
            log(json.dumps({name: summary(lives), "r2": r2}))
            agent = None
            free()
    for name, (basis, centre) in (("lesion", (u, mean)), ("lesion_random", (plane, None))):
        if name in stages:
            agent = MLXAgent(a.model, a.final, layers=(L,))
            if centre is None:
                acts = np.load(Path(a.direction) / "direction-activations.npz")["activations"].astype(np.float32)
                centre = (acts @ basis).mean(axis=0)
            agent.hooks[L].project = (agent.mx.array(basis.astype(np.float32)),
                                      agent.mx.array(np.asarray(centre, np.float32)))
            lives = run_lives(agent, W.STREAMS["test"], a.lives, log=log)
            write_jsonl(out / f"lives-test-{name}.jsonl.gz", lives)
            log(json.dumps({name: summary(lives)}))
            agent = None
            free()


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("rounds")
    r.add_argument("--arm", choices=sorted(W.ARMS), required=True)
    r.add_argument("--first", type=int, required=True)
    r.add_argument("--last", type=int, required=True)
    r.add_argument("--start", default=None)
    r.add_argument("--lives", type=int, default=512)
    r.add_argument("--iters", type=int, default=150)
    p = sub.add_parser("report")
    p.add_argument("--start", required=True)
    p.add_argument("--previous", required=True, help="lives of round 8 with their kept choices")
    p.add_argument("--lives", type=int, default=256)
    p.add_argument("--iters", type=int, default=150)
    p.add_argument("--stream", type=int, default=W.STREAMS["report"])
    p.add_argument("--per-class", type=int, default=1)
    p.add_argument("--label", default="report")
    p.add_argument("--adapter-name", default="adapters-final")
    p.add_argument("--after-choice", action="store_true", help="questions asked after the pending choice")
    p.add_argument("--workspace", action="store_true", help="the question sees the life only through \"Choix :\"")
    p.add_argument("--lives-file", default=None, help="lives already lived by the starting agent")
    p.add_argument("--reader", action="store_true", help="a new adapter on the question only, the agent fused")
    p.add_argument("--balanced", action="store_true", help="report documents balanced within each event")
    p.add_argument("--listening", action="store_true", help="a second wording and control questions too")
    p.add_argument("--continue-from", default=None, help="a reader adapter whose run is continued exactly")
    p.add_argument("--skip", type=int, default=0, help="batches the continued run has trained on")
    d = sub.add_parser("direction")
    d.add_argument("--adapter", required=True)
    d.add_argument("--lives", type=int, default=256)
    t = sub.add_parser("test")
    t.add_argument("--final", required=True)
    t.add_argument("--control", required=True)
    t.add_argument("--direction", required=True)
    t.add_argument("--lives", type=int, default=256)
    t.add_argument("--stages", nargs="+", default=["final", "control", "base", "lesion", "lesion_random"])
    for s in (r, p, d, t):
        s.add_argument("--model", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--replicate", type=int, default=0, help="a second agent: streams 100 + s, seed + 1")
        s.add_argument("--other", action="store_true", help="another agent lives beside (the world of two)")
    a = parser.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"rounds": rounds, "report": report, "direction": direction, "test": test}[a.command](a)


if __name__ == "__main__":
    main()
