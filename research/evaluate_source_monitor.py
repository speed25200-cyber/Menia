"""Closed-loop source inference and isolated causal probes; synthetic task only."""
import numpy as np
from menia.source_environment import CONDITIONS, streams


def training_features(frames, truth, verified):
    """Only PREVIOUS verified feedback is available before the current action."""
    visible = np.where(verified, truth, 0.)
    x = np.zeros((*frames.shape[:2], 5))
    x[:, :, :3] = frames
    x[1:, :, 3] = verified[:-1]
    x[1:, :, 4] = visible[:-1]
    return x, visible


def decision(q, y, *, override_verify=None):
    verify = np.minimum(q, 1-q) > .25 if override_verify is None else override_verify
    stored = np.where(verify, y, q >= .5)
    error = stored != y
    return verify, stored, error.astype(float)+.25*verify


def evaluate(model, condition, *, seed, batch=128, length=48):
    frames, truth = streams(seed, batch=batch, length=length, condition=condition)
    state, feedback = model.zero(batch), np.zeros((batch, 2))
    totals = {k: 0. for k in ("decision_loss", "verification_rate", "brier", "memory_error_rate",
        "false_external_memory_rate", "overconfident_error_rate", "probe_verification_changed",
        "probe_memory_changed", "probe_loss_delta", "report_only_verification_changed",
        "report_only_memory_changed")}
    for frame, y in zip(frames, truth):
        state, q = model.step(np.column_stack((frame, feedback)), state)
        verify, stored, loss = decision(q, y)
        feedback = np.column_stack((verify, np.where(verify, y, 0.)))
        totals["decision_loss"] += loss.sum()
        totals["verification_rate"] += verify.sum()
        totals["brier"] += ((q-y)**2).sum()
        totals["memory_error_rate"] += (stored != y).sum()
        totals["false_external_memory_rate"] += ((stored == 1) & (y == 0)).sum()
        totals["overconfident_error_rate"] += (((q < .1) | (q > .9)) & ((q >= .5) != y)).sum()
        # All members share the same symbolic content at a given timestep.
        # Donor q is a real output, but matching the content does not ensure the
        # entire donor state is on the recipient's conditional distribution.
        q_donor = np.roll(q, 1)
        v_probe, m_probe, l_probe = decision(q_donor, y)
        totals["probe_verification_changed"] += (v_probe != verify).sum()
        totals["probe_memory_changed"] += (m_probe != stored).sum()
        totals["probe_loss_delta"] += (l_probe-loss).sum()
        # A report-only replacement leaves control on q by definition.
        vr, mr, _ = decision(q, y)
        totals["report_only_verification_changed"] += (vr != verify).sum()
        totals["report_only_memory_changed"] += (mr != stored).sum()
    return {"steps": batch*length, **{k: float(v/(batch*length)) for k, v in totals.items()}}


def baselines(condition, seed, *, batch=128, length=48):
    _, truth = streams(seed, batch=batch, length=length, condition=condition)
    rng = np.random.default_rng(seed+800_000)
    out = {}
    for name, verified in (("always_verify", np.ones_like(truth, dtype=bool)),
                           ("never_verify", np.zeros_like(truth, dtype=bool)),
                           ("random_half", rng.random(truth.shape) < .5)):
        _, stored, loss = decision(np.full_like(truth, .5), truth, override_verify=verified)
        out[name] = {"decision_loss": float(loss.mean()), "verification_rate": float(verified.mean()),
                     "memory_error_rate": float((stored != truth).mean())}
    return out


def full_evaluation(model):
    return [{"condition": c, "seed": 51000+i*1000,
             "metrics": evaluate(model, c, seed=51000+i*1000)} for i, c in enumerate(CONDITIONS)]
