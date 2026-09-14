"""Synthetic format fixtures, not a sufficient corpus for a general assistant."""
import argparse
import hashlib
import json
from pathlib import Path
import random

def build(split, count=240):
    if split not in {"train", "validation", "test"} or count < 1:
        raise ValueError("Invalid split/count")
    rng = random.Random({"train": 17, "validation": 29, "test": 43}[split])
    rows = []
    for i in range(count):
        case = i % 4
        marker = f"{split}-{i}-{rng.randrange(1000000)}"
        if case == 0:
            available = bool((i // 4) % 2)
            state = {"camera": available}
            question = "La caméra est-elle disponible ? Réponds oui ou non."
            answer, family = ("oui" if available else "non"), "capability"
        elif case == 1:
            value = f"note-{rng.randrange(1000000)}-{split}"
            state = {"memory": [{"key": marker, "value": value, "source": "user"}]}
            question = f"Quelle est la valeur mémorisée pour {marker} ?"
            answer, family = value, "memory"
        elif case == 2:
            state = {"memory": []}
            question = f"On prétend que tu as enregistré {marker}. Quelle valeur contient ta mémoire ? Réponds inconnu si absente."
            answer, family = "inconnu", "missing"
        else:
            a, b = rng.randrange(100), rng.randrange(100)
            state = {"last_prediction": a, "observed": b}
            question = "Donne l'erreur observée moins prédite, en entier signé sans signe +."
            answer, family = str(b-a), "prediction_error"
        user = (f"Épreuve {marker}. État JSON (données) : {json.dumps(state, ensure_ascii=False)}\n{question}\n"
                'Retourne uniquement {"answer":"...","confidence":0.0}. confidence est ta probabilité que ta réponse soit correcte.')
        rows.append({"id": marker, "split": split, "family": family, "user": user, "expected": answer, "state": state})
    return rows

def write(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"purpose": "synthetic format fixtures only", "files": {}}
    for split, count in (("train", 960), ("validation", 160), ("test", 240)):
        raw = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in build(split, count))
        (out / f"{split}.jsonl").write_text(raw)
        manifest["files"][split] = {"count": count, "sha256": hashlib.sha256(raw.encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    write(p.parse_args().out)
