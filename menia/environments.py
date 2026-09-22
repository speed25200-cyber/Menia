"""A limited-observation virtual environment for the integrated agent.

The agent sees only read(field) responses and execution receipts. Evaluators may
inspect state separately. Actions have arbitrary names and a hidden permutation.
"""
import random


class VirtualRoom:
    fields = ("position", "target", "landmark")
    actions = ("a", "b", "c", "d")
    effects = ((1, 0), (-1, 0), (0, 1), (0, -1))

    def __init__(self, seed=17, *, position=(0, 0), target=(4, 3), dropout=0.0, slip=0.0):
        if not 0 <= dropout <= 1 or not 0 <= slip <= 1:
            raise ValueError("Invalid noise probability")
        self.rng = random.Random(seed)
        permutation = list(self.effects)
        self.rng.shuffle(permutation)
        self.mapping = dict(zip(self.actions, permutation))
        self.position = tuple(position)
        self.target = tuple(target)
        self.dropout = dropout
        self.slip = slip
        self.tick = 0
        self.reads = []
        self.moves = []

    def read(self, field):
        if field not in self.fields:
            raise ValueError("Unknown perceptual field")
        self.reads.append((self.tick, field))
        if self.rng.random() < self.dropout:
            return None
        values = {"position": self.position, "target": self.target, "landmark": (2, 2)}
        return {"field": field, "value": list(values[field]), "tick": self.tick,
                "source": "virtual_room_sensor"}

    def execute(self, action):
        if action not in self.mapping:
            raise ValueError("Unknown action")
        effect = self.mapping[action]
        if self.rng.random() < self.slip:
            effect = self.rng.choice(self.effects)
        self.position = tuple(p+d for p, d in zip(self.position, effect))
        self.tick += 1
        self.moves.append(action)
        return {"action": action, "tick": self.tick, "executed": True}

    def reverse_controls(self):
        self.mapping = {action: (-effect[0], -effect[1]) for action, effect in self.mapping.items()}

    def relocate(self, position):
        self.position = tuple(position)
        self.tick += 1
