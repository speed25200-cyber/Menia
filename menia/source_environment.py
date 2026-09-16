"""Synthetic source ambiguity. Hidden truth never appears in a public packet."""
import numpy as np


CONDITIONS = ("standard", "strong_imagery", "trace_removed", "persistent", "ambiguous")


def streams(seed, *, batch=128, length=48, condition="standard"):
    if condition not in CONDITIONS or batch < 1 or length < 1:
        raise ValueError("Invalid stream request")
    rng = np.random.default_rng(seed)
    gain = rng.uniform(.8, 1.2, batch)
    imaginary_gain = rng.uniform(.8, 1.2, batch)
    if condition == "strong_imagery":
        imaginary_gain[:] = 1.8
    channel = rng.integers(2, size=batch)
    frames = np.empty((length, batch, 3))
    truth = np.empty((length, batch))
    for t in range(length):
        channel = np.where(rng.random(batch) < (.02 if condition == "persistent" else .08),
                           1-channel, channel)
        external = rng.random(batch) < (.15+.7*channel)
        intention = rng.integers(2, size=batch)
        strength = external*gain+intention*imaginary_gain+rng.normal(0, .8, batch)
        trace = (2*channel-1)*.6+rng.normal(0, 1, batch)
        if condition == "trace_removed":
            trace[:] = 0
        if condition == "ambiguous":
            # Fresh independent source after all observable values are generated.
            external = rng.random(batch) < .5
        frames[t] = np.column_stack((strength, trace, intention))
        truth[t] = external
    return frames, truth


class SourceWorld:
    def __init__(self, seed=51000, *, length=48, condition="standard"):
        frames, truth = streams(seed, batch=1, length=length, condition=condition)
        self._frames, self._truth = frames[:, 0], truth[:, 0]
        self.index = -1
        self._checked = False

    def next_packet(self):
        if self.index+1 >= len(self._frames):
            raise StopIteration
        self.index += 1
        self._checked = False
        values = self._frames[self.index]
        return {"tick": self.index, "content": f"item-{self.index % 4}",
                "strength": float(values[0]), "trace": float(values[1]),
                "imagination_intent": int(values[2])}

    def verify(self):
        if self.index < 0 or self._checked:
            raise RuntimeError("Verification unavailable or already consumed")
        self._checked = True
        return bool(self._truth[self.index])
