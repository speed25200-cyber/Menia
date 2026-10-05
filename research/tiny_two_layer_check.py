"""Exploration (not pre-registered, after reading test 29 and the first results of test 31): with 2 layers under the
"route" mask, can the carried tokens of turn j hold anything of the events before j? At layer 0 they see, of past turns,
only the embeddings of "Choix" and of actions, so what later turns read of them (their input to layer 1) should not
change when an earlier event changes. Checked on the test 31 agents of seed 0 ("route", and "free" for contrast), on 64
random sequences; the change of the decision two turns later is given for comparison."""
import json
from pathlib import Path
import numpy as np
from . import tiny_survival as TS
from . import tiny_survival_pilot as SP

ROOT = "artifacts/tiny-survival-ac"


def run(torch, net, seen, ids):
    x = net.tok(ids) + net.pos(torch.arange(ids.shape[1]))[None]
    hidden = [x]
    m = ~seen[:ids.shape[1], :ids.shape[1]]
    for layer in net.layers.layers:
        x = layer(x, src_mask=m)
        hidden.append(x)
    return hidden


def main():
    import torch
    torch.set_num_threads(1)
    rng = np.random.default_rng(12345)
    out = {}
    for arm in ("route", "free"):
        net = SP.model(0)
        net.load_state_dict(torch.load(f"{ROOT}/checkpoints/{arm}-0.pt", weights_only=False)["net"])
        net.eval()
        seen = torch.tensor(TS.mask(arm))
        carried, decision = [], []
        with torch.no_grad():
            for _ in range(64):
                events = rng.integers(0, 6, TS.W.TURNS)
                actions = rng.integers(0, 2, TS.W.TURNS)

                def ids_of(ev):
                    seq = [TS.BOS]
                    for e, a in zip(ev, actions):
                        seq += [TS.EVENT0 + int(e), TS.CHOICE, TS.R if a == 0 else TS.M]
                    return torch.tensor([seq])
                j = int(rng.integers(5, 25))
                changed = events.copy()
                k = int(rng.integers(0, j - 1))  # an event strictly before turn j (0-based turns)
                changed[k] = (changed[k] + 1 + rng.integers(0, 5)) % 6
                a, b = run(torch, net, seen, ids_of(events)), run(torch, net, seen, ids_of(changed))
                pos = [1 + 3 * j + 1, 1 + 3 * j + 2]  # "Choix" and the action of turn j
                carried.append(float((a[1][0, pos] - b[1][0, pos]).abs().max()))
                t = min(j + 2, TS.W.TURNS - 1)
                decision.append(float((a[2][0, 1 + 3 * t + 1] - b[2][0, 1 + 3 * t + 1]).abs().max()))
        out[arm] = {"carried_state_of_turn_j_max_change_when_an_earlier_event_changes": max(carried),
                    "decision_at_turn_j_plus_2_mean_change": float(np.mean(decision))}
    text = json.dumps(out, indent=1)
    print(text)
    path = Path(ROOT) / "exploration" / "two-layer-check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n")


if __name__ == "__main__":
    main()
