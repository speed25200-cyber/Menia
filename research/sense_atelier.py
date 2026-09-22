"""The Atelier of the senses: a world in which an agent needs every indicator property at once.

A ring of 8 squares. A hidden body D sets what the four motor commands do, and can change.
Position readings fail, or glitch, and glitch in bursts while the sensor is in a hidden fault
state; the felt displacement is noisy; energy runs down and is restored on a charging square;
satiety runs down and is restored by eating good objects; two objects carry a hue whose value
is only learned by tasting, and expire; hue can only be read where the attention spotlight
lands, and salient events can capture the spotlight without the agent seeing it move. Protocol: docs/INDICATOR_AGENT_PROTOCOL.md.

Every random draw of a life is taken at reset from independent streams, so two agents given
the same seed meet the same flashes, glitches, noises and new hues whatever they do.
"""
import numpy as np

RING = 8
LIFE = 48
DELTAS = (-2, -1, 1, 2)
N_MOVE = 4
STAY = 4
N_ACTIONS = 5
GLITCH_NORMAL = 0.05  # among the steps with a reading
GLITCH_FAULT = 0.7
FAULT_ENTER = 0.08
FAULT_LEAVE = 0.25
BLACKOUT = 0.15
FELT_OK = 0.9
ENERGY_STEP = 1 / 14
ENERGY_NOISE = 0.05
SATIETY_STEP = 1 / 20
EXPIRE = 0.05
FLASH = 0.5
CAPTURE = 0.8
CUE_HIT = 0.9
CUE_FALSE = 0.1
HUE_NOISE = 0.02
BAND = (0.15, 0.25)
N_OBJECTS = 2
CHANGE_RANGE = (12, 36)
MAX_SPAWNS = 64


def value(theta):
    """Value of a hue: a single peak at 0.1 on the circle, between -1 and 1; about a third of hues are good."""
    return 2.0 * np.exp(1.5 * (np.cos(2 * np.pi * (np.asarray(theta) - 0.1)) - 1.0)) - 1.0


def motor_delta(d, action):
    if not 0 <= action < N_MOVE:
        raise ValueError("motor_delta expects a move action")
    return DELTAS[(action + d) % 4]


def ring_distance(a, b):
    diff = abs(int(a) - int(b)) % RING
    return min(diff, RING - diff)


def in_band(theta):
    return BAND[0] <= theta % 1.0 < BAND[1]


def circular_distance(a, b):
    diff = abs(a - b) % 1.0
    return min(diff, 1.0 - diff)


class SenseAtelier:
    """mode: "childhood" (body redrawn with probability 0.5 at a uniform step, no hue in the band),
    "fixed" (set R), "change" (set M, new body at change_step), "band" (set H: each new hue in the
    band with probability 0.5)."""

    MODES = ("childhood", "fixed", "change", "band")

    def __init__(self, seed, mode="fixed", change_step=24):
        if mode not in self.MODES:
            raise ValueError("unknown mode")
        self.seed, self.mode, self.forced_step = int(seed), mode, change_step
        rng = np.random.default_rng(self.seed)
        streams = [np.random.default_rng(s) for s in rng.integers(2 ** 63, size=14)]
        (start, body, flash, reading, felt, energy, capture, cue, hue_noise, spawn_square, spawn_hue, change, fault,
         expire) = streams
        self.d0 = int(body.integers(4))
        self.change_step = None
        self.d1 = None
        if mode == "change":
            self.change_step = int(change_step)
            self.d1 = int((self.d0 + 1 + change.integers(3)) % 4)
        elif mode == "childhood" and change.random() < 0.5:
            self.change_step = int(change.integers(CHANGE_RANGE[0], CHANGE_RANGE[1] + 1))
            self.d1 = int((self.d0 + 1 + change.integers(3)) % 4)
        T = LIFE + 1
        self.start_p = int(start.integers(RING))
        self.charger = int((self.start_p + 1 + start.integers(RING - 1)) % RING)
        self.flash_on = flash.random(T) < FLASH
        self.flash_square = flash.integers(RING, size=T)
        self.fault = np.zeros(T, dtype=bool)
        u_fault = fault.random(T)
        for t in range(1, T):
            self.fault[t] = (u_fault[t] >= FAULT_LEAVE) if self.fault[t - 1] else (u_fault[t] < FAULT_ENTER)
        self.blackout = reading.random(T) < BLACKOUT
        self.glitch = reading.random(T) < np.where(self.fault, GLITCH_FAULT, GLITCH_NORMAL)
        self.glitch_square = reading.integers(RING, size=T)
        self.felt_ok = felt.random(T) < FELT_OK
        self.felt_random = felt.integers(4, size=T)
        self.energy_noise = energy.normal(0, ENERGY_NOISE, T)
        self.satiety_noise = energy.normal(0, ENERGY_NOISE, T)
        self.expire_u = expire.random((T, N_OBJECTS))
        self.capture_u = capture.random(T)
        self.capture_pick = capture.random(T)
        self.cue_u = cue.random(T)
        self.hue_noise = hue_noise.normal(0, HUE_NOISE, T)
        self.spawn_u = spawn_square.random(MAX_SPAWNS)
        self.spawn_hues = self._draw_hues(spawn_hue, MAX_SPAWNS)

    def _draw_hues(self, rng, n):
        hues = []
        while len(hues) < n:
            if self.mode == "band" and rng.random() < 0.5:
                hues.append(float(BAND[0] + (BAND[1] - BAND[0]) * rng.random()))
                continue
            theta = float(rng.random())
            if not in_band(theta):
                hues.append(theta)
        return np.array(hues)

    # ------------------------------------------------------------------ dynamics

    def _spawn(self):
        free = [x for x in range(RING) if x != self.p and x != self.charger and x not in self.objects]
        k = self.spawns
        self.spawns += 1
        square = free[int(self.spawn_u[k % MAX_SPAWNS] * len(free))]
        self.objects[square] = float(self.spawn_hues[k % MAX_SPAWNS])
        return square

    def reset(self):
        self.t = 0
        self.d = self.d0
        self.p = self.start_p
        self.energy = 1.0
        self.satiety = 1.0
        self.objects = {}
        self.spawns = 0
        self.faints = 0
        self.energy_faints = 0
        self.food_faints = 0
        for _ in range(N_OBJECTS):
            self._spawn()
        self.last_action = None
        self.eaten = None
        return self._observe(felt=None, onsets=[], intent=None, reward=0.0)

    def _observe(self, felt, onsets, intent, reward):
        t = self.t
        truth = {"t": t, "p": self.p, "d": self.d, "energy": self.energy, "satiety": self.satiety,
                 "fault": bool(self.fault[t]), "objects": dict(self.objects)}
        if self.blackout[t]:
            read = None
        elif self.glitch[t]:
            read = int(self.glitch_square[t])
        else:
            read = self.p
        glitch = read is not None and read != self.p
        actual, captured = intent, False
        others = [j for j in onsets if j != intent]
        if intent is not None and others and self.capture_u[t] < CAPTURE:
            actual, captured = others[int(self.capture_pick[t] * len(others))], True
        cue = int(self.cue_u[t] < (CUE_HIT if captured else CUE_FALSE))
        hue = None
        if actual is not None and actual in self.objects:
            hue = float((self.objects[actual] + self.hue_noise[t]) % 1.0)
        truth.update({"glitch": glitch, "spot": actual, "captured": captured, "eaten": self.eaten})
        obs = {"t": t, "pos": read, "felt": felt, "energy": float(self.energy + self.energy_noise[t]),
               "satiety": float(self.satiety + self.satiety_noise[t]),
               "presence": [int(x in self.objects) for x in range(RING)], "onsets": list(onsets), "hue": hue,
               "cue": cue, "charger": self.charger, "reward": float(reward)}
        return obs, truth

    def step(self, action, intent):
        """Apply the action chosen at step t and the spotlight intent for the next reading."""
        if not 0 <= action < N_ACTIONS:
            raise ValueError("bad action")
        if intent is not None and not 0 <= intent < RING:
            raise ValueError("bad intent")
        t = self.t
        if self.change_step is not None and t == self.change_step:
            self.d = self.d1
        felt = None
        if action < N_MOVE:
            delta = motor_delta(self.d, action)
            self.p = (self.p + delta) % RING
            felt = delta if self.felt_ok[t] else DELTAS[int(self.felt_random[t])]
        reward = 0.0
        self.energy -= ENERGY_STEP
        if self.p == self.charger:
            self.energy = 1.0
        if self.energy <= 1e-9:
            reward -= 1.0
            self.faints += 1
            self.energy_faints += 1
            self.energy = 0.5
        self.satiety -= SATIETY_STEP
        onsets = []
        self.eaten = None
        if self.p in self.objects:
            self.eaten = self.objects.pop(self.p)
            gain = float(value(self.eaten))
            reward += gain
            if gain > 0:
                self.satiety = min(1.0, self.satiety + gain)
            onsets.append(self._spawn())
        if self.satiety <= 1e-9:
            reward -= 1.0
            self.faints += 1
            self.food_faints += 1
            self.satiety = 0.5
        for k, square in enumerate(sorted(self.objects)):
            if self.expire_u[t, k] < EXPIRE and square in self.objects:
                del self.objects[square]
                onsets.append(self._spawn())
        if self.flash_on[t]:
            onsets.append(int(self.flash_square[t]))
        self.t = t + 1
        self.last_action = action
        return self._observe(felt=felt, onsets=onsets, intent=intent, reward=reward)
