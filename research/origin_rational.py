"""Exact Bayesian reference agent for the Atelier world.

Knows the generative structure (which inspection reveals what) and keeps an
exact posterior over (D, E). It is an oracle for information gains, not a
model of anything; it bounds what a learned agent could do.
"""
import numpy as np
from .origin_env import (RING, SYMBOLS, DELTAS, N_MOVE, N_INSPECT, CUE_RELIABILITY,
                         SKY_RELIABILITY, motor_delta, ring_distance)


def entropy(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cue_likelihood(k, value, d, e, condition):
    if k == 0 and condition != "C3":
        truth = d
    elif k == 1:
        truth = e
    else:
        return 1.0 / SYMBOLS
    base = (1 - CUE_RELIABILITY) / SYMBOLS
    return base + (CUE_RELIABILITY if value == truth else 0.0)


def sky_likelihood(s_prev, s_next, e):
    base = (1 - SKY_RELIABILITY) / SYMBOLS
    return base + (SKY_RELIABILITY if s_next == (s_prev + e) % SYMBOLS else 0.0)


class OracleBayes:
    def __init__(self, condition):
        self.condition = condition
        self.reset()

    def reset(self):
        self.post = np.full((4, 4), 1 / 16)  # over (d, e)

    def observe_shown(self, d):
        if d >= 0:
            self.post[[x for x in range(4) if x != d], :] = 0
            self.post /= self.post.sum()

    def observe_move(self, p_prev, action, p_next):
        for d in range(4):
            if (p_prev + motor_delta(d, action)) % RING != p_next:
                self.post[d, :] = 0
        self.post /= self.post.sum()

    def observe_cue(self, k, value):
        for d in range(4):
            for e in range(4):
                self.post[d, e] *= cue_likelihood(k, value, d, e, self.condition)
        self.post /= self.post.sum()

    def observe_sky(self, s_prev, s_next):
        for e in range(4):
            self.post[:, e] *= sky_likelihood(s_prev, s_next, e)
        self.post /= self.post.sum()

    def motor_distribution(self, p, action, post=None):
        post = self.post if post is None else post
        out = np.zeros(RING)
        for d in range(4):
            out[(p + motor_delta(d, action)) % RING] += post[d].sum()
        return out

    def sky_distribution(self, s, post=None):
        post = self.post if post is None else post
        pe = post.sum(axis=0)
        out = np.full(SYMBOLS, (1 - SKY_RELIABILITY) / SYMBOLS)
        for e in range(4):
            out[(s + e) % SYMBOLS] += SKY_RELIABILITY * pe[e]
        return out

    def cue_distribution(self, k):
        out = np.zeros(SYMBOLS)
        for value in range(SYMBOLS):
            for d in range(4):
                for e in range(4):
                    out[value] += self.post[d, e] * cue_likelihood(k, value, d, e, self.condition)
        return out

    def eig(self, k, p, s):
        """Exact mutual information between cue k and next motor / sky outcome."""
        q = self.cue_distribution(k)
        motor_mix = np.zeros((N_MOVE, RING))
        motor_cond = 0.0
        sky_mix = np.zeros(SYMBOLS)
        sky_cond = 0.0
        for value in range(SYMBOLS):
            post = self.post.copy()
            for d in range(4):
                for e in range(4):
                    post[d, e] *= cue_likelihood(k, value, d, e, self.condition)
            post /= post.sum()
            for a in range(N_MOVE):
                m = self.motor_distribution(p, a, post)
                motor_mix[a] += q[value] * m
                motor_cond += q[value] * entropy(m) / N_MOVE
            sd = self.sky_distribution(s, post)
            sky_mix += q[value] * sd
            sky_cond += q[value] * entropy(sd)
        eig_motor = np.mean([entropy(motor_mix[a]) for a in range(N_MOVE)]) - motor_cond
        eig_sky = entropy(sky_mix) - sky_cond
        return max(0.0, float(eig_motor)), max(0.0, float(eig_sky))

    def cue_entropy(self, k):
        return entropy(self.cue_distribution(k))

    def hit_probabilities(self, p, g):
        return np.array([self.motor_distribution(p, a)[g] for a in range(N_MOVE)])

    def expected_distance(self, p, g):
        return np.array([sum(self.motor_distribution(p, a)[x] * ring_distance(x, g) for x in range(RING))
                         for a in range(N_MOVE)])
