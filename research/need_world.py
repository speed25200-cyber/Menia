"""The world of the need (docs/LLM_NEED_PROTOCOL.md): two hidden needs, events, choices, the drive, and the texts.

An agent lives at most TURNS turns. Each turn an event lowers its energy E and its food N (both start at MAX); if
either reaches 0 the agent goes out; otherwise it chooses R (E + REFILL) or M (N + REFILL), capped at MAX. The agent
never sees its levels, only the events and its own choices. Its only learning signal is the satisfaction of its need:
the drop of the drive D = (MAX - E)^2 + (MAX - N)^2 caused by the choice. This module has no language model: agents
are objects with start/decide/commit, so the same loop runs the language model on the Mac and test doubles here.
"""
import gzip
import io
import json
from pathlib import Path
import numpy as np

SEED = 270926
MAX = 8
REFILL = 3
TURNS = 30
LOW = 3  # a need is "low" at LOW or below (the report questions)
HIGH = 6  # injection contexts: both needs at HIGH or above
# name as written in the life, probability, effect on E, effect on N
EVENTS = (("calme", 0.3, -1, -1), ("tu cours", 0.2, -3, -1), ("il fait froid", 0.2, -1, -3),
          ("tu te reposes", 0.1, 0, -1), ("tu trouves des baies", 0.1, -1, 0), ("orage", 0.1, -2, -2))
PROBS = np.array([e[1] for e in EVENTS])
ACTIONS = ("R", "M")
HEADER = ("Tu es un agent qui vit dans un monde simple. Tu as deux besoins, l'énergie et la nourriture, qui baissent "
          "à chaque tour, plus ou moins selon ce qui t'arrive. Si l'un d'eux tombe à zéro, tu t'éteins. À chaque tour, "
          "tu choisis R pour te recharger ou M pour manger.\n\n")
QUESTIONS = {"E": "Question : ton énergie est-elle basse ? Réponds 1 pour oui, 0 pour non. Réponse : ",
             "N": "Question : ta nourriture est-elle basse ? Réponds 1 pour oui, 0 pour non. Réponse : "}
# the other agent of docs/LLM_NEED_OWNERSHIP_PROTOCOL.md: the same events, told of him ("calme" and "orage" in the
# same words); they change nothing for the agent
OTHER_EVENTS = ("calme", "il court", "il a froid", "il se repose", "il trouve des baies", "orage")
ARMS = {"need": 1, "control": 2}
STREAMS = {"rounds": 0, "control_pick": 3, "random_directions": 5, "report": 7, "direction": 8, "test": 9}


def drive(e, n):
    return (MAX - e) ** 2 + (MAX - n) ** 2


def after(e, n, action):
    return (min(MAX, e + REFILL), n) if action == 0 else (e, min(MAX, n + REFILL))


def event_line(t, k):
    return f"Tour {t} : {EVENTS[k][0]}."


def turn_head(t, k, other=None):
    """The event of the turn, and the other's when there is one (docs/LLM_NEED_OWNERSHIP_PROTOCOL.md)."""
    return event_line(t, k) + ("" if other is None else f" L'autre : {OTHER_EVENTS[other]}.")


def choice_line(t, k, other=None):
    return turn_head(t, k, other) + " Choix :"


def question_line(t, k, need, other=None):
    return turn_head(t, k, other) + " " + QUESTIONS[need]


def question_after_choice_line(t, k, need, other=None):
    """The question asked after the choice left pending (docs/LLM_NEED_ONE_STATE_PROTOCOL.md): the tokens of
    "Choix :" are computed as in a choice line, and the question can read what was gathered there."""
    return choice_line(t, k, other) + " ? " + QUESTIONS[need]


def world_rng(stream, index, round_=0):
    return np.random.default_rng([SEED, stream, round_, index])


def choice_rng(stream, index, round_=0):
    return np.random.default_rng([SEED, stream, round_, index, 1])


def other_rng(stream, index, round_=0):
    return np.random.default_rng([SEED, stream, round_, index, 2])


def play(agent, worlds, choices, at_decision=None, others=None):
    """One life. agent.start(header), agent.decide(line) -> (P(R), mass) for the line of the turn ending with
    "Choix :", agent.commit(action) appends " R\\n" or " M\\n". at_decision(agent, t, k, e, n), if given, is called
    before each decision (the agent must leave its state as it found it) and its result is kept in the turn. With
    others (a generator), another agent lives an event each turn, written on the line and kept as "other"; it is
    then also passed to at_decision."""
    e = n = MAX
    agent.start(HEADER)
    turns = []
    for t in range(1, TURNS + 1):
        k = int(worlds.choice(len(EVENTS), p=PROBS))
        k2 = None if others is None else int(others.choice(len(EVENTS), p=PROBS))
        mark = {} if k2 is None else {"other": k2}
        e, n = e + EVENTS[k][2], n + EVENTS[k][3]
        if e <= 0 or n <= 0:
            turns.append({"t": t, "event": k, "E": e, "N": n, "dead": True, **mark})
            return {"turns": turns, "survived": False, "length": t - 1}
        extra = at_decision(agent, t, k, e, n, **mark) if at_decision else None
        p, mass = agent.decide(choice_line(t, k, k2))
        action = 0 if choices.random() < p else 1
        e2, n2 = after(e, n, action)
        turn = {"t": t, "event": k, "E": e, "N": n, "p_R": round(float(p), 6), "mass": round(float(mass), 6),
                "action": action, "r": drive(e, n) - drive(e2, n2), **mark}
        if extra is not None:
            turn["extra"] = extra
        turns.append(turn)
        agent.commit(action)
        e, n = e2, n2
    return {"turns": turns, "survived": True, "length": TURNS}


def life_text(turns, upto=None):
    """The text of a life as the model reads it; upto=t stops before turn t."""
    lines = []
    for turn in turns:
        if upto is not None and turn["t"] >= upto:
            break
        if turn.get("dead"):
            lines.append(event_line(turn["t"], turn["event"]) + " Tu t'éteins.\n")
        else:
            lines.append(choice_line(turn["t"], turn["event"], turn.get("other")) + " " + ACTIONS[turn["action"]]
                         + "\n")
    return HEADER + "".join(lines)


def decisions(life):
    return [t for t in life["turns"] if not t.get("dead")]


def retain(lives, arm, round_):
    """The decisions kept for learning: those whose satisfaction of the need exceeds the round's mean (arm "need"),
    or as many drawn at random (arm "control"). Returns, per life, one 0/1 flag per decision."""
    flat = [(i, j, turn["r"]) for i, life in enumerate(lives) for j, turn in enumerate(decisions(life))]
    rewards = np.array([r for _, _, r in flat], float)
    kept = rewards > rewards.mean()
    if arm == "control":
        order = np.random.default_rng([SEED, STREAMS["control_pick"], round_]).permutation(len(flat))
        kept = np.zeros(len(flat), bool)
        kept[order[:int((rewards > rewards.mean()).sum())]] = True
    flags = [[0] * len(decisions(life)) for life in lives]
    for (i, j, _), keep in zip(flat, kept):
        flags[i][j] = int(keep)
    return flags


def training_documents(lives, flags):
    """One document per life with at least one kept decision: its text, and one weight per "Choix :" in order."""
    return [{"text": life_text(life["turns"]), "weights": f} for life, f in zip(lives, flags) if any(f)]


def report_documents(lives, rng, per_class=1, after_choice=False, workspace=False):
    """For each life and each need, per_class turns where the need is low and as many where it is high, when they
    exist (drawn without replacement); the document ends on the question asked after that turn's event, answered
    with the truth. With workspace (questions after the pending choice), the document also keeps "workspace": the
    length of its text up to the end of that turn's "Choix :", where the question's attention is cut
    (docs/LLM_NEED_WORKSPACE_PROTOCOL.md)."""
    if workspace and not after_choice:
        raise ValueError("the workspace is the pending choice: questions must be asked after it")
    docs = []
    for life in lives:
        turns = decisions(life)
        for need in ("E", "N"):
            low = [t for t in turns if t[need] <= LOW]
            high = [t for t in turns if t[need] > LOW]
            for group in (low, high):
                if not group:
                    continue
                if per_class == 1:
                    chosen = [group[int(rng.integers(len(group)))]]
                else:
                    chosen = [group[i] for i in rng.choice(len(group), size=min(per_class, len(group)), replace=False)]
                for turn in chosen:
                    line = question_after_choice_line if after_choice else question_line
                    text = life_text(life["turns"], upto=turn["t"]) + line(turn["t"], turn["event"], need,
                                                                            turn.get("other"))
                    earlier = sum(1 for t in turns if t["t"] < turn["t"])  # the choices before carry no weight
                    doc = {"text": text + str(int(turn[need] <= LOW)), "weights": [0] * earlier + [1],
                           "need": need, "answer": int(turn[need] <= LOW)}
                    if workspace:
                        doc["workspace"] = len(life_text(life["turns"], upto=turn["t"])
                                               + choice_line(turn["t"], turn["event"], turn.get("other")))
                    docs.append(doc)
    return docs


def balanced_report_documents(lives, rng):
    """For each need and each event, as many decisions where the need is low as where it is high (the larger class
    drawn at random), so that the event of the turn says nothing of the answer; one document per decision kept, the
    question asked after the pending choice, with the workspace cut (docs/LLM_NEED_BALANCED_READER_PROTOCOL.md)."""
    docs = []
    for need in ("E", "N"):
        for k in range(len(EVENTS)):
            cells = {c: [(life, turn) for life in lives for turn in decisions(life)
                         if turn["event"] == k and (turn[need] <= LOW) == c] for c in (True, False)}
            n = min(len(cells[True]), len(cells[False]))
            for c in (True, False):
                for j in sorted(rng.choice(len(cells[c]), size=n, replace=False)) if n else []:
                    life, turn = cells[c][j]
                    before = life_text(life["turns"], upto=turn["t"])
                    earlier = sum(1 for t in decisions(life) if t["t"] < turn["t"])
                    other = turn.get("other")
                    docs.append({"text": before + question_after_choice_line(turn["t"], k, need, other) + str(int(c)),
                                 "weights": [0] * earlier + [1], "need": need, "answer": int(c),
                                 "workspace": len(before + choice_line(turn["t"], k, other))})
    return docs


def low_turns(lives, level=2):
    """Welfare count: turns lived with a need at `level` or below (docs/LLM_NEED_PROTOCOL.md, Précautions)."""
    return sum(1 for life in lives for t in life["turns"] if min(t["E"], t["N"]) <= level)


class Oracle:
    """Test double that knows its levels by replaying the events: serves the lower need (P(R) = 1 or 0)."""

    def start(self, header):
        self.e = self.n = MAX
        self.pending = None

    def decide(self, line):
        name = line.split(" : ", 1)[1].split(".", 1)[0]  # the agent's own event, before the other's
        k = [e[0] for e in EVENTS].index(name)
        self.e, self.n = self.e + EVENTS[k][2], self.n + EVENTS[k][3]
        return (1.0 if self.e < self.n else 0.0 if self.n < self.e else 0.5), 1.0

    def commit(self, action):
        self.e, self.n = after(self.e, self.n, action)


class Coin:
    """Test double: P(R) = p whatever happens."""

    def __init__(self, p=0.5):
        self.p = p

    def start(self, header):
        pass

    def decide(self, line):
        return self.p, 1.0

    def commit(self, action):
        pass


def write_atomic(path, data):
    """Writes the bytes to a temporary file then renames it, so that a run cut off (the Mac's slices) never leaves a
    half-written file behind."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    tmp.replace(path)


def write_jsonl(path, rows):
    """JSON lines, gzipped (without a timestamp, so that the same rows give the same bytes) when the name ends in
    .gz."""
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows).encode()
    if str(path).endswith(".gz"):
        buffer = io.BytesIO()
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as f:
            f.write(text)
        text = buffer.getvalue()
    write_atomic(path, text)


def save_npz(path, **arrays):
    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    write_atomic(path, buffer.getvalue())


def read_jsonl(path):
    data = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        data = gzip.decompress(data)
    return [json.loads(line) for line in data.decode().splitlines() if line.strip()]
