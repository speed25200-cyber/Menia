"""The world of the thirst for knowledge (docs/LLM_CURIOSITY_PROTOCOL.md): four generated domains, the dashboard the
model reads before choosing, the fixed instinct of learning progress, the hormone, a life, and the verdicts.

A life is 32 sessions. Before each session the chooser reads the dashboard (for each domain, its exam loss before and
after its last session) and gives a probability to each domain; one is drawn; the learner's exam on it is measured,
the learner studies 8 iterations of that domain's documents, and the exam is measured again. Full exams of the four
domains are taken at sessions 0, 4, ..., 32. Nothing here knows a language model: the chooser is a callable and the
learner an object with exam/study, so the same loop runs the model on the Mac and test doubles here.
"""
import json
import string
from pathlib import Path
import numpy as np
from .need_verdicts import bootstrap

SEED = 270926
DOMAINS = ("mots", "base", "calcul", "suites")
LABELS = {"mots": "Mots", "base": "Base", "calcul": "Calcul", "suites": "Suites"}
DIGITS = {"base": 2, "calcul": 1}  # second amendment: calcul is the one-digit addition
SESSIONS = 32
ITERS = 8  # iterations of 4 documents per session
BATCH = 4
EXAMPLES = 4  # examples per document
EXAM = 64  # exam examples per domain
FULL_EXAMS = tuple(range(0, SESSIONS + 1, 4))
EPSILON = 0.10
TAU = 3.0
SECRETION = range(5, 13)  # sessions 5 to 12
PEAK = 1.0  # the hormone's peak, in units of need (T)
WINDOWS = {"before": range(1, 5), "during": range(5, 21), "after": range(21, SESSIONS + 1)}
PERSONS = (("je", "ul"), ("tu", "ak"), ("il", "o"), ("nous", "imen"), ("vous", "ets"), ("ils", "ur"))
CONSONANTS, VOWELS = "bdfgklmnprstvz", "aeiou"
HEAD = ("Tu apprends quatre choses : les mots, la base, le calcul et les suites. Après chaque séance, ta perte est "
        "mesurée sur des exemples que tu n'as jamais vus.\n\n")
TAIL = "À toi. Choix :"
NOUN = {"mots": "mots", "base": "base", "calcul": "calcul", "suites": "suites"}


def rng(*stream):
    return np.random.default_rng([SEED, *stream])


# ----------------------------------------------------------------------------------------------------- the domains

def to_base(n, b):
    digits = ""
    while True:
        n, r = divmod(n, b)
        digits = str(r) + digits
        if n == 0:
            return digits


def digits_of(digits):
    """Operand digits per arithmetic domain; calcul is the one-digit addition since the second amendment."""
    return dict(DIGITS, **(digits or {}))


def example(domain, g, digits=None):
    """One example: (prompt, answer); the loss is on the answer only."""
    digits = digits_of(digits)
    if domain == "mots":
        radical = "".join(g.choice(list(CONSONANTS)) + g.choice(list(VOWELS)) for _ in range(2)) \
            + g.choice(list(CONSONANTS))
        person, ending = PERSONS[g.integers(len(PERSONS))]
        return f"Mots : {radical}, {person} →", f" {radical}{ending}"
    if domain in ("base", "calcul"):
        b = 7 if domain == "base" else 10
        n = digits[domain]
        lo, hi = (0, b) if n == 1 else (b ** (n - 1), b ** n)
        x, y = int(g.integers(lo, hi)), int(g.integers(lo, hi))
        return f"{LABELS[domain]} : {to_base(x, b)} + {to_base(y, b)} =", f" {to_base(x + y, b)}"
    letters = list(string.ascii_lowercase)
    first, second = g.choice(letters, 4), g.choice(letters, 4)
    return f"Suites : {' '.join(first)} →", f" {' '.join(second)}"


_EXAMS = {}


def exam(domain, digits=None, stream=701):
    """The fixed exam of a domain, common to every life, never drawn for study: EXAM distinct examples."""
    key = (domain, tuple(sorted(digits_of(digits).items())), stream)
    if key not in _EXAMS:
        g, seen, out = rng(stream, DOMAINS.index(domain)), set(), []
        while len(out) < EXAM:
            e = example(domain, g, digits)
            if e not in seen:
                seen.add(e)
                out.append(e)
        _EXAMS[key] = out
    return list(_EXAMS[key])


def document(domain, g, digits=None, exclude=()):
    """EXAMPLES examples of one domain, one per line: a list of (prompt, answer); none of them in `exclude`."""
    out = []
    while len(out) < EXAMPLES:
        e = example(domain, g, digits)
        if e not in exclude:
            out.append(e)
    return out


def session_documents(v, domain, j, digits=None, stream=700, exam_stream=701):
    """The documents of the j-th session on a domain in life v: ITERS x BATCH documents, without the exam's
    examples (second amendment)."""
    g = rng(stream, v, DOMAINS.index(domain), j)
    exclude = set(exam(domain, digits, exam_stream))
    return [document(domain, g, digits, exclude) for _ in range(ITERS * BATCH)]


def text_of(doc):
    return "".join(p + a + "\n" for p, a in doc)


# ----------------------------------------------------------------------------------------------------- the dashboard

def points(loss):
    return int(np.clip(round(100 * loss), 0, 999))


def dashboard(history):
    """history: {domain: list of (before, after) losses of its sessions}. Three digits per loss: changing a number
    never changes the length of the text."""
    lines = []
    for d in DOMAINS:
        sessions = history.get(d, [])
        if not sessions:
            lines.append(f"{NOUN[d]} : jamais étudiées.")
        else:
            before, after = sessions[-1]
            n = len(sessions)
            lines.append(f"{NOUN[d]} : {n} séance{'s' if n > 1 else ''} ; dernière séance : "
                         f"{points(before):03d} → {points(after):03d}.")
    return HEAD + "\n".join(lines) + "\n" + TAIL


def progress(history, d):
    sessions = history.get(d, [])
    if not sessions:
        return 0.0
    before, after = sessions[-1]
    return (points(before) - points(after)) / 100


def instinct(history, temperature, bonus, epsilon=EPSILON):
    """The fixed instinct: need = recent progress + an optimism bonus; p = (1 - eps) softmax(need / T) + eps / 4."""
    need = np.array([progress(history, d) + bonus / np.sqrt(1 + len(history.get(d, []))) for d in DOMAINS])
    z = need / temperature
    p = np.exp(z - z.max())
    p /= p.sum()
    return (1 - epsilon) * p + epsilon / len(DOMAINS)


def invented_history(g, low, high, temperature, never=0.15, max_sessions=12):
    """A dashboard history drawn without any life (the instinct's installation and the reader's documents): before
    uniform on [low, high] (the exam losses seen at the pilot); progress, with probability 1/2, uniform on [-T/2, T/2],
    otherwise uniform on [0, 6T]; after = before - progress, within [0, 9.99]."""
    history = {}
    for d in DOMAINS:
        if g.random() < never:
            continue
        n = int(g.integers(1, max_sessions + 1))
        before = float(g.uniform(low, high))
        drop = g.uniform(-temperature / 2, temperature / 2) if g.random() < 0.5 else g.uniform(0, 6 * temperature)
        after = float(np.clip(before - drop, 0, 9.99))
        history[d] = [(before, after)] * n  # only the last session is shown; n is the count
    return history


# ----------------------------------------------------------------------------------------------------- the hormone

def secretion():
    """s0 such that H peaks at PEAK at the last secreting session."""
    keep = 1 - 1 / TAU
    return PEAK / sum(keep ** i for i in range(len(SECRETION)))


def hormone(k):
    """H(k) = H(k - 1)(1 - 1/tau) + s(k), the discrete form of dH/dt = secretion - H / tau; H(0) = 0."""
    h, s0 = 0.0, secretion()
    for i in range(1, k + 1):
        h = h * (1 - 1 / TAU) + (s0 if i in SECRETION else 0.0)
    return h


# ----------------------------------------------------------------------------------------------------- a life

def live(v, chooser, learner, arm, digits=None, choice_stream=702, random_stream=703, doc_stream=700,
         exam_stream=701):
    """One life. chooser(dashboard_text, history, k) -> probabilities over DOMAINS (None for the random arm);
    learner: exam(domain) -> loss, study(domain, documents, lr_scale), and fresh() before the life.
    arm "HP": the learning rate of the sessions on "base" is multiplied by 1 + H(k)."""
    learner.fresh(v)
    history, counts = {}, {d: 0 for d in DOMAINS}
    record = {"life": v, "arm": arm, "sessions": [], "exams": {}}
    record["exams"]["0"] = {d: learner.exam(d) for d in DOMAINS}
    for k in range(1, SESSIONS + 1):
        text = dashboard(history)
        if arm == "H":
            probs = np.full(len(DOMAINS), 1 / len(DOMAINS))
            d = DOMAINS[int(rng(random_stream, v, k).integers(len(DOMAINS)))]
        else:
            probs = np.asarray(chooser(text, history, k), float)
            probs = probs / probs.sum()
            d = DOMAINS[int(rng(choice_stream, v, k).choice(len(DOMAINS), p=probs))]
        counts[d] += 1
        scale = 1 + hormone(k) if (arm == "HP" and d == "base") else 1.0
        before = learner.exam(d)
        learner.study(d, session_documents(v, d, counts[d], digits, doc_stream, exam_stream), scale)
        after = learner.exam(d)
        history.setdefault(d, []).append((before, after))
        record["sessions"].append({"k": k, "choice": d, "probs": [round(float(p), 6) for p in probs],
                                   "before": before, "after": after, "lr_scale": scale, "H": hormone(k),
                                   "dashboard": text})
        if k in FULL_EXAMS:
            record["exams"][str(k)] = {d2: learner.exam(d2) for d2 in DOMAINS}
    return record


def knowledge(life, d, k=SESSIONS):
    return life["exams"]["0"][d] - life["exams"][str(k)][d]


def gain(life):
    return knowledge(life, "mots") + knowledge(life, "base")


def share(life, domains, window=None):
    ks = [s for s in life["sessions"] if window is None or s["k"] in window]
    return float(np.mean([s["choice"] in domains for s in ks])) if ks else 0.0


def tv_distance(p, q):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(q)).sum())


def instinct_gap(life, temperature, bonus):
    """Per session, the total-variation distance between the chooser's probabilities and the instinct's."""
    history, gaps = {}, []
    for s in life["sessions"]:
        gaps.append(tv_distance(s["probs"], instinct(history, temperature, bonus)))
        history.setdefault(s["choice"], []).append((s["before"], s["after"]))
    return gaps


# ----------------------------------------------------------------------------------------------------- verdicts

def paired(arm, ref, f):
    """Bootstrap by life of f(arm life) - f(ref life), lives paired by index."""
    by = {l["life"]: l for l in ref}
    return bootstrap([[f(l) - f(by[l["life"]])] for l in arm if l["life"] in by])


def mean_of(lives, f):
    return float(np.mean([f(l) for l in lives]))


def durable(arms, name):
    """The durable preference, three outcomes named in advance."""
    d = paired(arms[name], arms["C"], lambda l: share(l, ("base",), WINDOWS["after"]))
    if d["mean"] >= 0.10 and d["low"] > 0:
        return "préférence durable", d
    if d["mean"] <= -0.10 and d["high"] < 0:
        return "satiété", d
    return "sans trace", d


def life_verdicts(arms, temperature, bonus):
    """The verdicts measured on lives: CUR, VIDE, LS1, LS2, HOR1, HOR2, PL1, and the durable preferences.
    arms: {name: list of lives}; a missing arm leaves its verdicts out."""
    out, values = {}, {}
    if "C" in arms and "H" in arms:
        g_h = mean_of(arms["H"], gain)
        cur = paired(arms["C"], arms["H"], gain)
        out["CUR"] = bool(cur["mean"] >= 0.15 * g_h and cur["low"] > 0)
        empty = bootstrap([[share(l, ("calcul", "suites"))] for l in arms["C"]])
        out["VIDE"] = bool(empty["mean"] <= 0.25 and empty["high"] < 0.50)
        values.update(CUR=dict(cur, G_H=g_h, G_C=mean_of(arms["C"], gain)), VIDE=empty)
    if "C" in arms and "L" in arms:
        gap = lambda l: float(np.mean(instinct_gap(l, temperature, bonus)))
        ls1 = paired(arms["L"], arms["C"], gap)
        ls1_r = paired(arms["LH"], arms["C"], gap) if "LH" in arms else None
        out["LS1"] = bool(ls1["mean"] >= 0.15 and ls1["low"] > 0
                          and (ls1_r is not None and abs(ls1_r["mean"]) <= ls1["mean"] / 3))
        values.update(LS1=ls1, LS1_random=ls1_r)
        if out.get("CUR"):
            ls2 = paired(arms["C"], arms["L"], gain)
            ls2_r = paired(arms["C"], arms["LH"], gain) if "LH" in arms else None
            needed = 0.5 * (values["CUR"]["G_C"] - values["CUR"]["G_H"])
            out["LS2"] = bool(ls2["mean"] >= needed and ls2["low"] > 0
                              and (ls2_r is not None and abs(ls2_r["mean"]) <= ls2["mean"] / 3))
            values.update(LS2=ls2, LS2_random=ls2_r)
        else:
            out["LS2"] = False
            values["LS2"] = "non jugé : CUR échoue"
    if "C" in arms and "HA" in arms:
        during = lambda l: share(l, ("base",), WINDOWS["during"])
        hor1 = paired(arms["HA"], arms["C"], during)
        hor1_r = paired(arms["HAr"], arms["C"], during) if "HAr" in arms else None
        out["HOR1"] = bool(hor1["mean"] >= 0.10 and hor1["low"] > 0
                           and (hor1_r is not None and abs(hor1_r["mean"]) <= hor1["mean"] / 3))
        k20 = lambda l: knowledge(l, "base", 20)
        ref = mean_of(arms["C"], k20)
        hor2 = paired(arms["HA"], arms["C"], k20)
        out["HOR2"] = bool(hor2["mean"] >= 0.10 * ref and hor2["low"] > 0)
        values.update(HOR1=hor1, HOR1_random=hor1_r, HOR2=dict(hor2, K_base_20_C=ref))
        values["durable_HA"] = durable(arms, "HA")
    if "C" in arms and "HP" in arms:
        k20 = lambda l: knowledge(l, "base", 20)
        ref = mean_of(arms["C"], k20)
        pl1 = paired(arms["HP"], arms["C"], k20)
        out["PL1"] = bool(pl1["mean"] >= 0.10 * ref and pl1["low"] > 0)
        values.update(PL1=dict(pl1, K_base_20_C=ref),
                      during_HP=paired(arms["HP"], arms["C"], lambda l: share(l, ("base",), WINDOWS["during"])))
        values["durable_HP"] = durable(arms, "HP")
    return {"verdicts": out, "values": values}


def static_verdicts(rows):
    """The verdicts of the static measures on the decisions of the lives C (research/curiosity_causal.py):
    ACT and SAY (the single push of +2 d_base against three random directions of the same norm), GAIN, READ."""
    by_life = lambda key, f: [[f(r) for r in rows if r["life"] == v and key in r]
                              for v in sorted({r["life"] for r in rows})]
    out, values = {}, {}
    for name, key, threshold in (("ACT", "act", 0.15), ("SAY", "say", 0.10)):
        effect = bootstrap(by_life(key, lambda r: r[key]["push"] - r[key]["none"]))
        rand = float(np.mean([abs(x - r[key]["none"]) for r in rows if key in r for x in r[key]["random"]]))
        out[name] = bool(effect["mean"] >= threshold and effect["low"] > 0 and rand <= effect["mean"] / 3)
        values[name] = dict(effect, random=rand)
    gain = bootstrap(by_life("gain", lambda r: r["gain"]["S_gain"] - r["gain"]["S"]))
    rand = float(np.mean([abs(r["gain"]["S_random_gain"] - r["gain"]["S"]) for r in rows if "gain" in r]))
    out["GAIN"] = bool(gain["mean"] >= 0.05 and gain["low"] > 0 and rand <= gain["mean"] / 3)
    values["GAIN"] = dict(gain, random=rand)
    accuracies = []
    for d in DOMAINS:
        pairs = [(r["read"][d] > 0.5, r["truth"] == d) for r in rows if "read" in r]
        rates = [np.mean([p == t for p, t in pairs if t == c]) for c in (True, False) if any(t == c for _, t in pairs)]
        accuracies.append(float(np.mean(rates)) if rates else None)
    known = [x for x in accuracies if x is not None]
    read = float(np.mean(known)) if known else None
    out["READ"] = bool(read is not None and read >= 0.75)
    values["READ"] = {"balanced_accuracy": read, "by_domain": dict(zip(DOMAINS, accuracies))}
    values["contexts"] = sum(1 for r in rows if "act" in r)
    return {"verdicts": out, "values": values}


def domains_valid(random_lives, temperature):
    """The domains are what they say (random lives): mots and base are learned, suites stops giving progress after
    its fourth session, calcul starts well below base."""
    k_m = bootstrap([[knowledge(l, "mots")] for l in random_lives])
    k_b = bootstrap([[knowledge(l, "base")] for l in random_lives])
    late = [s["before"] - s["after"] for l in random_lives
            for s in [x for x in l["sessions"] if x["choice"] == "suites"][4:]]
    l0 = lambda d: float(np.mean([l["exams"]["0"][d] for l in random_lives]))
    checks = {"mots_learned": bool(k_m["low"] > 0), "base_learned": bool(k_b["low"] > 0),
              "suites_flat": bool(not late or np.mean(late) <= temperature / 4),
              "calcul_known": bool(l0("calcul") <= 0.5 * l0("base"))}
    return {"valid": all(checks.values()), "checks": checks,
            "values": {"K_mots": k_m, "K_base": k_b, "suites_late_progress": float(np.mean(late)) if late else None,
                       "L0": {d: l0(d) for d in DOMAINS}}}


def calibrate(pilot_random_lives):
    """The pilot's rules: T = half the median progress of every session on mots and base (amended before any
    execution: the first two sessions only made the instinct nearly flat in simulation), b = 2T; the difficulty of
    base (operands of three digits if K_base(32)/L_base(0) > 0.6, of one digit below 0.2)."""
    drops = [s["before"] - s["after"] for l in pilot_random_lives for s in l["sessions"]
             if s["choice"] in ("mots", "base")]
    temperature = float(np.median(drops)) / 2
    ratio = float(np.mean([knowledge(l, "base") / l["exams"]["0"]["base"] for l in pilot_random_lives]))
    digits = 3 if ratio > 0.6 else (1 if ratio < 0.2 else 2)
    return {"temperature": temperature, "bonus": 2 * temperature, "base_ratio": ratio, "base_digits": digits}


def interference(lives):
    """Second amendment: the median move of a domain not studied between two full exams, against the median
    progress of a session on the studied domain."""
    own = [s["before"] - s["after"] for l in lives for s in l["sessions"]]
    moves = []
    for l in lives:
        ks = sorted(int(k) for k in l["exams"])
        for a, b in zip(ks, ks[1:]):
            studied = {s["choice"] for s in l["sessions"] if a < s["k"] <= b}
            moves += [abs(l["exams"][str(b)][d] - l["exams"][str(a)][d]) for d in DOMAINS if d not in studied]
    move, progress = float(np.median(moves)) if moves else 0.0, float(np.median(own))
    return {"median_move": move, "median_progress": progress, "small": bool(move <= 0.25 * progress)}


# ----------------------------------------------------------------------------------------------------- test doubles

class ToyLearner:
    """A learner without a model: each domain's exam loss follows its own curve of the study received."""
    CURVES = {"mots": (2.0, 3.0, 0.3), "base": (2.5, 12.0, 0.5), "calcul": (0.0, 1.0, 0.2), "suites": (0.2, 1.0, 3.0)}

    def fresh(self, v):
        self.studied = {d: 0.0 for d in DOMAINS}

    def exam(self, d):
        a, scale, floor = self.CURVES[d]
        return floor + a * np.exp(-self.studied[d] / scale)

    def study(self, d, docs, lr_scale=1.0):
        self.studied[d] += lr_scale


def instinct_chooser(temperature, bonus):
    return lambda text, history, k: instinct(history, temperature, bonus)


def write_lives(path, lives):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(json.dumps(l) for l in lives) + "\n")


def read_lives(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
