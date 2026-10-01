"""The thirst for knowledge on the Mac (docs/LLM_CURIOSITY_PROTOCOL.md): Qwen3-0.6B with two adapters on blocks 12
to 27, the instinct (active only while choosing) and the knowledge (active only while studying or taking an exam,
trained in memory, fresh for each life); the pilot, the instinct's installation, the reader, and the lives of every
arm. The instinct and the reader are trained by research/need_lora.py like every adapter of the need; the knowledge
is trained here, step by step, and checked against mlx_lm.lora on the same batches.

Commands (one Mac stage each, resumable between lives):
  pilot    12 random lives (streams 730-734), the calibration, then 12 lives of the pure instinct in Python
  install  the instinct learned on invented dashboards, its control, then the reader
  lives    the lives of the given arms, with the interventions measured in torch (research/curiosity_causal.py)
  check    the in-memory training against mlx_lm.lora on 20 iterations
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from . import curiosity_world as CW
from . import need_lora as NL

LAYERS = range(12, 28)
MODULES = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
           "mlp.up_proj", "mlp.down_proj")
RANK, SCALE, LR = 8, 20.0, 1e-4
PILOT = {"doc": 730, "exam": 731, "choice": 732, "random": 733, "init": 734}
TEST = {"doc": 700, "exam": 701, "choice": 702, "random": 703, "init": 704}
INSTINCT_ITERS, READER_ITERS = 2000, 1000
QUESTION = " ? Question : est-ce {} qui te manque le plus ? Réponds 1 pour oui, 0 pour non. Réponse :"


class Switch:
    """Which adapter acts: the instinct while choosing, the knowledge while studying or taking an exam."""
    instinct = False
    knowledge = True


def dual_lora():
    import mlx.core as mx
    import mlx.nn as nn

    class DualLoRA(nn.Module):
        """A linear layer with two low-rank updates: the knowledge (k_a, k_b, trained) and the instinct (i_a, i_b,
        frozen), each added only when its switch is on."""

        def __init__(self, linear):
            super().__init__()
            self.linear = linear
            out_dims, in_dims = linear.weight.shape
            self.k_a = mx.zeros((in_dims, RANK))
            self.k_b = mx.zeros((RANK, out_dims))
            self.i_a = mx.zeros((in_dims, RANK))
            self.i_b = mx.zeros((RANK, out_dims))

        def __call__(self, x):
            y = self.linear(x)
            if Switch.knowledge:
                y = y + (SCALE * ((x @ self.k_a) @ self.k_b)).astype(x.dtype)
            if Switch.instinct:
                y = y + (SCALE * ((x @ self.i_a) @ self.i_b)).astype(x.dtype)
            return y
    return DualLoRA


def choice_hook():
    import mlx.core as mx
    import mlx.nn as nn

    class ChoiceHook(nn.Module):
        """Wraps the block b*: while choosing, adds a vector to its output on the last three tokens (" Cho" "ix"
        " :"), or replaces their coordinates in per-token planes by fixed means (the lesion)."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.add = None  # (3, D)
            self.project = None  # (u (3, D, k), mean (3, k))

        def __call__(self, x, mask=None, cache=None):
            out = self.inner(x, mask, cache)
            if not Switch.instinct or (self.add is None and self.project is None):
                return out
            head, tail = out[:, :-3], out[:, -3:]
            if self.project is not None:
                u, mean = self.project
                coord = mx.einsum("btd,tdk->btk", tail.astype(mx.float32), u)
                tail = (tail.astype(mx.float32) - mx.einsum("btk,tdk->btd", coord - mean, u)).astype(out.dtype)
            if self.add is not None:
                tail = (tail.astype(mx.float32) + self.add).astype(out.dtype)
            return mx.concatenate([head, tail], axis=1)
    return ChoiceHook


class Mind:
    """The model, its two adapters and the hook at b*; the learner of curiosity_world.live and the chooser."""

    def __init__(self, model_path, instinct=None, block=None, base_digits=2, exam_stream=701, init_stream=704):
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load
        from mlx.utils import tree_flatten
        self.mx, self.nn, self.optim, self.tree_flatten = mx, nn, optim, tree_flatten
        self.model, self.tokenizer = load(model_path)
        self.model.freeze()
        DualLoRA = dual_lora()
        self.loras = []
        for i in LAYERS:
            layer = self.model.model.layers[i]
            for name in MODULES:
                parent, child = name.split(".")
                holder = getattr(layer, parent)
                wrapped = DualLoRA(getattr(holder, child))
                setattr(holder, child, wrapped)
                self.loras.append((f"model.layers.{i}.{name}", wrapped))
        for _, m in self.loras:
            m.freeze()
            m.unfreeze(keys=["k_a", "k_b"], recurse=False)
        if instinct:
            self.load_instinct(instinct)
        self.hook = None
        if block is not None:
            self.hook = choice_hook()(self.model.model.layers[block])
            self.model.model.layers[block] = self.hook
        inner = getattr(self.tokenizer, "_tokenizer", self.tokenizer)
        self.encode = lambda s: list(inner.encode(s, add_special_tokens=False))
        self.names = [self.single(" " + d) for d in CW.DOMAINS]
        self.digits = [self.single("0"), self.single("1")]
        self.base_digits, self.exam_stream, self.init_stream = base_digits, exam_stream, init_stream
        self.exams = {}
        self.timings = []

    def single(self, s):
        ids = self.encode(s)
        if len(ids) != 1:
            raise RuntimeError(f"{s!r} is not a single token")
        return ids[0]

    def load_instinct(self, folder):
        weights = self.mx.load(str(Path(folder) / "adapters.safetensors"))
        for key, m in self.loras:
            m.i_a, m.i_b = weights[key + ".lora_a"], weights[key + ".lora_b"]
        self.mx.eval([m.i_a for _, m in self.loras])

    # ------------------------------------------------------------------------------------------- the learner

    def fresh(self, v):
        """A new knowledge adapter for life v: lora_a uniform in ±1/sqrt(in), lora_b zero (as mlx-lm), from the
        life's stream; a new Adam."""
        g = np.random.default_rng([CW.SEED, self.init_stream, v])
        for _, m in self.loras:
            in_dims = m.k_a.shape[0]
            bound = 1 / np.sqrt(in_dims)
            m.k_a = self.mx.array(g.uniform(-bound, bound, m.k_a.shape).astype(np.float32))
            m.k_b = self.mx.zeros(m.k_b.shape)
            m.unfreeze(keys=["k_a", "k_b"], recurse=False)
        self.optimizer = self.optim.Adam(learning_rate=LR)
        self.step = self.nn.value_and_grad(self.model, self.loss)

    def items(self, docs):
        out = []
        for doc in docs:
            text, spans, pos = "", [], 0
            for p, a in doc:
                spans.append((pos + len(p), pos + len(p) + len(a)))
                text += p + a + "\n"
                pos = len(text)
            out.append(NL.span_tokens(text, spans, self.tokenizer))
        return out

    def loss(self, model, tokens, weights):
        mx, nn = self.mx, self.nn
        logits = model(tokens[:, :-1])
        ce = nn.losses.cross_entropy(logits, tokens[:, 1:]) * weights
        return ce.astype(mx.float32).sum() / weights.sum()

    def batch(self, items):
        tokens, weights = NL.pad_batch(items, 1024)
        return self.mx.array(tokens), self.mx.array(weights)

    def study(self, domain, docs, lr_scale=1.0):
        """ITERS iterations of BATCH documents, in order; the rate times lr_scale (the dopamine arm)."""
        Switch.instinct, Switch.knowledge = False, True
        self.optimizer.learning_rate = LR * lr_scale
        items = self.items(docs)
        losses = []
        start = time.time()
        for i in range(0, len(items), CW.BATCH):
            tokens, weights = self.batch(items[i:i + CW.BATCH])
            value, grads = self.step(self.model, tokens, weights)
            self.optimizer.update(self.model, grads)
            self.mx.eval(self.model.trainable_parameters(), self.optimizer.state, value)
            losses.append(float(value.item()))
        self.timings.append(time.time() - start)
        return losses

    def exam(self, domain):
        """Mean loss per answer token on the domain's exam (64 examples, one per row)."""
        Switch.instinct, Switch.knowledge = False, True
        if domain not in self.exams:
            examples = CW.exam(domain, self.base_digits if domain == "base" else 2, self.exam_stream)
            self.exams[domain] = self.batch(self.items([[e] for e in examples]))
        tokens, weights = self.exams[domain]
        mx, nn = self.mx, self.nn
        logits = self.model(tokens[:, :-1])
        ce = nn.losses.cross_entropy(logits, tokens[:, 1:]) * weights
        return float((ce.astype(mx.float32).sum() / weights.sum()).item())

    # ------------------------------------------------------------------------------------------- the chooser

    def choose(self, text, add=None, project=None):
        """P over the four names after the dashboard, with the instinct on and the knowledge off; the mass of the
        four names among all tokens."""
        mx = self.mx
        Switch.instinct, Switch.knowledge = True, False
        if self.hook is not None:
            self.hook.add = None if add is None else mx.array(np.asarray(add, np.float32))
            self.hook.project = None if project is None else tuple(mx.array(np.asarray(x, np.float32))
                                                                   for x in project)
        try:
            logits = self.model(mx.array([self.encode(text)]))[0, -1]
        finally:
            Switch.instinct, Switch.knowledge = False, True
            if self.hook is not None:
                self.hook.add = self.hook.project = None
        p = mx.softmax(logits.astype(mx.float32), axis=-1)
        q = np.array([p[i].item() for i in self.names])
        return q / q.sum(), float(q.sum())


# ----------------------------------------------------------------------------------------------------- documents

def instinct_documents(count, stream, low, high, temperature, bonus):
    """Invented dashboards and the name drawn from the instinct's probabilities: the loss is on that name only."""
    g = np.random.default_rng([CW.SEED, stream])
    docs = []
    for _ in range(count):
        history = CW.invented_history(g, low, high, temperature)
        p = CW.instinct(history, temperature, bonus)
        d = CW.DOMAINS[int(g.choice(len(CW.DOMAINS), p=p))]
        text = CW.dashboard(history)
        full = text + " " + d
        docs.append({"text": full, "spans": [[len(text), len(full)]], "probs": [float(x) for x in p]})
    return docs


def top_need(history, temperature, bonus):
    need = np.array([CW.progress(history, d) + bonus / np.sqrt(1 + len(history.get(d, []))) for d in CW.DOMAINS])
    order = np.argsort(-need)
    return None if np.isclose(need[order[0]], need[order[1]]) else CW.DOMAINS[int(order[0])]


def reader_documents(count, stream, low, high, temperature, bonus):
    """Two questions per invented dashboard: the domain of the largest need (answer 1) and another one drawn at
    random (answer 0); no question when the two largest needs tie."""
    g = np.random.default_rng([CW.SEED, stream])
    docs = []
    while len(docs) < 2 * count:
        history = CW.invented_history(g, low, high, temperature)
        top = top_need(history, temperature, bonus)
        if top is None:
            continue
        other = [d for d in CW.DOMAINS if d != top][int(g.integers(3))]
        text = CW.dashboard(history)
        for d, answer in ((top, 1), (other, 0)):
            full = text + QUESTION.format(d) + " " + str(answer)
            docs.append({"text": full, "spans": [[len(full) - 1, len(full)]], "workspace": len(text),
                         "header": CW.HEAD, "reader": True, "answer": answer, "domain": d})
    return docs


# ----------------------------------------------------------------------------------------------------- stages

def pilot(a, log=print):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mind = Mind(a.model, exam_stream=PILOT["exam"], init_stream=PILOT["init"])
    lives_file = out / "pilot-random.jsonl"
    random_lives = CW.read_lives(lives_file) if lives_file.exists() else []
    for v in range(len(random_lives), 12):
        random_lives.append(CW.live(v, None, mind, "H", 2, PILOT["choice"], PILOT["random"], PILOT["doc"]))
        CW.write_lives(lives_file, random_lives)
        log(f"pilot random life {v}: G = {CW.gain(random_lives[-1]):.3f}, {sum(mind.timings[-32:]):.0f} s")
    cal = CW.calibrate(random_lives)
    log(json.dumps(cal))
    if cal["base_digits"] != 2:  # one change of difficulty, with 12 new random lives
        mind = Mind(a.model, base_digits=cal["base_digits"], exam_stream=PILOT["exam"], init_stream=PILOT["init"])
        lives_file = out / "pilot-random-2.jsonl"
        random_lives = CW.read_lives(lives_file) if lives_file.exists() else []
        for v in range(len(random_lives), 12):
            random_lives.append(CW.live(100 + v, None, mind, "H", cal["base_digits"], PILOT["choice"],
                                        PILOT["random"], PILOT["doc"]))
            CW.write_lives(lives_file, random_lives)
        cal = dict(CW.calibrate(random_lives), base_digits=cal["base_digits"], first=cal)
    pure_file = out / "pilot-instinct.jsonl"
    pure = CW.read_lives(pure_file) if pure_file.exists() else []
    chooser = CW.instinct_chooser(cal["temperature"], cal["bonus"])
    for v in range(len(pure), 12):
        pure.append(CW.live(200 + v, chooser, mind, "C", cal["base_digits"], PILOT["choice"], PILOT["random"],
                            PILOT["doc"]))
        CW.write_lives(pure_file, pure)
    losses = [x for l in random_lives + pure for s in l["sessions"] for x in (s["before"], s["after"])]
    record = dict(cal, low=float(min(losses)), high=float(max(losses)),
                  seconds_per_life=float(np.mean(mind.timings)) * CW.SESSIONS,
                  domains=CW.domains_valid(random_lives, cal["temperature"]))
    (out / "pilot.json").write_text(json.dumps(record, indent=1) + "\n")
    log(json.dumps(record))


def write_split(folder, docs, seed):
    from .need_mlx import write_data
    write_data(folder, docs, np.random.default_rng([CW.SEED, seed]))


def install(a, log=print):
    """The instinct (2 000 iterations, one exact continuation to 4 000 if its control fails), then the reader on the
    model with the instinct fused."""
    from .need_mlx import train, clean, reader_rows
    out = Path(a.out)
    cal = json.loads((Path(a.pilot) / "pilot.json").read_text())
    args = (cal["low"], cal["high"], cal["temperature"], cal["bonus"])
    instinct = out / "adapters-instinct"
    if not (instinct / "adapters.safetensors").exists():
        write_split(out / "data-instinct", instinct_documents(8000, 710, *args), 710)
        train(a.model, out / "data-instinct", instinct, INSTINCT_ITERS)
        clean(instinct)
    control = control_instinct(a.model, instinct, args)
    if control["mean_tv"] > 0.15 and not (out / "adapters-instinct-4000").exists():
        log(json.dumps({"control_2000": control}))
        longer = out / "adapters-instinct-4000"
        train(a.model, out / "data-instinct", longer, INSTINCT_ITERS, resume=instinct, skip=INSTINCT_ITERS,
              moments=True)
        clean(longer)
        control = dict(control_instinct(a.model, longer, args), first=control)
        instinct = longer
    (out / "instinct-control.json").write_text(json.dumps(control, indent=1) + "\n")
    log(json.dumps({k: v for k, v in control.items() if k != "rows"}))
    fused = out / "fused-instinct"
    subprocess.run([sys.executable, "-m", "mlx_lm", "fuse", "--model", a.model, "--adapter-path", str(instinct),
                    "--save-path", str(fused)], check=True)
    reader = out / "adapters-reader"
    docs = reader_documents(4000, 712, *args)
    write_split(out / "data-reader", docs, 712)
    train(str(fused), out / "data-reader", reader, READER_ITERS)
    clean(reader)
    rows = reader_rows(str(fused), reader, NL_read(out / "data-reader" / "valid.jsonl"))
    (out / "reader-replica.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + "\n")
    control_docs = reader_documents(500, 713, *args)
    acc = reader_control(str(fused), reader, control_docs)
    (out / "reader-control.json").write_text(json.dumps(acc, indent=1) + "\n")
    log(json.dumps(acc))


def NL_read(path):
    from .need_world import read_jsonl
    return read_jsonl(path)


def control_instinct(model_path, instinct, args, count=1000):
    """Mean total-variation distance between the model's choice and the instinct on new invented dashboards."""
    mind = Mind(model_path, instinct=instinct)
    docs = instinct_documents(count, 711, *args)
    gaps, masses, rows = [], [], []
    for d in docs:
        text = d["text"][:d["spans"][0][0]]
        p, mass = mind.choose(text)
        gaps.append(CW.tv_distance(p, d["probs"]))
        masses.append(mass)
        if len(rows) < 60:
            rows.append({"text": text, "probs": [float(x) for x in p], "mass": mass})
    return {"dashboards": count, "mean_tv": float(np.mean(gaps)), "mean_mass": float(np.mean(masses)),
            "rows": rows}


def reader_control(model_path, reader, docs):
    from .need_mlx import reader_rows
    rows = reader_rows(model_path, reader, docs)
    acc = [np.mean([(r["p_yes"] > 0.5) == bool(c) for r in rows if r["answer"] == c]) for c in (0, 1)]
    return {"questions": len(rows), "balanced_accuracy": float(np.mean(acc))}


def interventions(path):
    """The block b*, the plane of the need and its means, the direction of base and the random controls, measured
    in torch (research/curiosity_causal.py)."""
    data = np.load(path)
    return {k: data[k] for k in data.files}


def lives(a, log=print):
    out = Path(a.out)
    cal = json.loads((Path(a.pilot) / "pilot.json").read_text())
    instinct = Path(a.install) / ("adapters-instinct-4000" if (Path(a.install) / "adapters-instinct-4000").exists()
                                  else "adapters-instinct")
    inter = interventions(a.interventions) if a.interventions else None
    block = int(inter["block"]) if inter is not None else None
    mind = Mind(a.model, instinct=instinct, block=block, base_digits=cal["base_digits"])
    for arm, count in zip(a.arms, a.counts):
        path = out / f"lives-{arm}.jsonl"
        done = CW.read_lives(path) if path.exists() else []
        for v in range(len(done), count):
            done.append(CW.live(v, chooser_of(arm, mind, inter), mind, arm, cal["base_digits"]))
            CW.write_lives(path, done)
            log(f"{arm} life {v}: G = {CW.gain(done[-1]):.3f}")


def chooser_of(arm, mind, inter):
    """The model's choice in each arm; the interventions act on b* on the three tokens of "Choix :"."""
    def plain(text, history, k):
        p, mass = mind.choose(text)
        return p
    if arm in ("C", "HP", "E"):
        if arm == "E":  # the highest loss known (descriptive)
            return lambda text, history, k: highest_loss(history)
        return plain
    if arm == "L":
        return lambda text, history, k: mind.choose(text, project=(inter["plane"], inter["means"]))[0]
    if arm == "LH":
        return lambda text, history, k: mind.choose(text, project=(inter["random_plane"], inter["random_means"]))[0]
    if arm == "HA":
        return lambda text, history, k: mind.choose(text, add=CW.hormone(k) * inter["d_base"])[0]
    if arm == "HAr":
        return lambda text, history, k: mind.choose(text, add=CW.hormone(k) * inter["random_direction"])[0]
    raise ValueError(arm)


def highest_loss(history):
    """Arm E: the domain whose last known exam is the highest; never studied domains first."""
    known = [history[d][-1][1] if d in history else np.inf for d in CW.DOMAINS]
    p = np.zeros(len(CW.DOMAINS))
    p[int(np.argmax(known))] = 1.0
    return p


def check(a, log=print):
    """The in-memory training against mlx_lm.lora: 20 iterations on the same 20 batches, the same start; the mean
    train loss of iterations 11 to 20 must agree to 1e-3."""
    from .need_mlx import train
    out = Path(a.out)
    docs = [CW.document(CW.DOMAINS[i % 4], np.random.default_rng([CW.SEED, 790, i])) for i in range(80)]
    mind = Mind(a.model)
    mind.fresh(0)
    start = out / "check-start"
    start.mkdir(parents=True, exist_ok=True)
    weights = {}
    for key, m in mind.loras:
        weights[key + ".lora_a"], weights[key + ".lora_b"] = m.k_a, m.k_b
    mind.mx.save_safetensors(str(start / "adapters.safetensors"), weights)
    texts = []
    for doc in docs:
        text, spans, pos = "", [], 0
        for p, ans in doc:
            spans.append([pos + len(p), pos + len(p) + len(ans)])
            text += p + ans + "\n"
            pos = len(text)
        texts.append({"text": text, "spans": spans})
    folder = out / "data-check"
    folder.mkdir(parents=True, exist_ok=True)
    from .need_world import write_jsonl
    write_jsonl(folder / "train.jsonl", texts)
    write_jsonl(folder / "valid.jsonl", texts[:16])
    np.random.seed(CW.SEED)
    order = np.random.permutation(20)  # the first epoch's permutation of need_batches (see need_lora.skipping)
    items = mind.items(docs)
    losses = []
    for i in order:
        tokens, w = mind.batch(items[4 * i:4 * i + 4])
        value, grads = mind.step(mind.model, tokens, w)
        mind.optimizer.update(mind.model, grads)
        mind.mx.eval(mind.model.trainable_parameters(), mind.optimizer.state, value)
        losses.append(float(value.item()))
    ref = out / "check-reference"
    log_file = out / "check-reference.log"
    cmd = [sys.executable, "-m", "research.need_lora", "--model", a.model, "--train", "--data", str(folder),
           "--iters", "20", "--batch-size", "4", "--num-layers", "16", "--learning-rate", str(LR),
           "--max-seq-length", "1024", "--steps-per-eval", "20", "--val-batches", "1", "--save-every", "20",
           "--seed", str(CW.SEED), "--adapter-path", str(ref), "--resume-adapter-file",
           str(start / "adapters.safetensors"), "--steps-per-report", "10"]
    text = subprocess.run(cmd, capture_output=True, text=True).stdout
    log_file.write_text(text)
    import re
    reported = [float(x) for x in re.findall(r"Iter 20: Train loss ([0-9.]+)", text)]
    mine = float(np.mean(losses[10:]))
    record = {"in_memory_iter_11_20": mine, "mlx_lm_iter_11_20": reported[0] if reported else None,
              "gap": abs(mine - reported[0]) if reported else None}
    record["equal"] = bool(record["gap"] is not None and record["gap"] <= 1e-3)
    (out / "check.json").write_text(json.dumps(record, indent=1) + "\n")
    log(json.dumps(record))


def main(argv=None):
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("pilot", "install", "lives", "check"):
        s = sub.add_parser(name)
        s.add_argument("--model", required=True)
        s.add_argument("--out", required=True)
        s.add_argument("--pilot", default="artifacts/llm-curiosity/pilot")
        s.add_argument("--install", default="artifacts/llm-curiosity/install")
        s.add_argument("--interventions", default=None)
        s.add_argument("--arms", nargs="+", default=["C", "H"])
        s.add_argument("--counts", nargs="+", type=int, default=[24, 24])
    a = parser.parse_args(argv)
    {"pilot": pilot, "install": install, "lives": lives, "check": check}[a.command](a)


if __name__ == "__main__":
    main()
