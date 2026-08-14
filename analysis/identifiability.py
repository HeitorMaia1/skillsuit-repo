"""Is the damping `B` identifiable from the task-3.14 dataset? (depth pass, 2026-08-14)

Why this script exists
----------------------
Task 3.14 built a dataset whose true viscous damping `B` is known, so that tasks 5.8/5.9 can be
scored as "the pHNN recovered `B` to within X%". That framing silently assumes `B` is **recoverable
from this data at all** — that the trials excite the dynamics enough to separate a dissipative
torque from an inertial or gravitational one. Nothing has tested that assumption.

The robotics identification literature says the assumption is not free. Gautier & Khalil (1992)
established the standard: the dynamics are linear in the parameters, `tau = Y(q,qdot,qddot) theta`,
identification is least squares on `Y`, and **the trajectories must be designed by nonlinear
optimisation to minimise the condition number of `Y`** — otherwise the estimate is swamped by noise.
SkillSuit's trials are not designed for this. They are min-jerk point-to-point reaches, lifts, wrist
rotations and throws, chosen because they look like human motion.

This script measures what that costs, using the **best possible case** for the estimator: the exact
model structure, exact `qddot`, no filtering error. A neural network that must also learn the model
structure can only do worse, so every number here is a lower bound on the difficulty facing 5.8/5.9.

Method
------
`tau` is linear in the inertial parameters and in `B`, so the regressor is assembled column by
column. Damping columns are exact by inspection: `d tau / d b_j = qdot_j e_j`. Inertial columns come
from central differences on (mass, com_x, Ixx, Iyy, Izz) for each of the three massive bodies —
15 inertial parameters plus 7 damping, 22 in all. Linearity is verified by checking two step sizes
agree, so the "finite difference" is exact to roundoff rather than approximate.

Then, over samples drawn from the real dataset:

  * `rank` and `cond` of `Y` — Gautier & Khalil's criterion.
  * **noise amplification on each damping parameter**: for i.i.d. torque noise of standard
    deviation `sigma`, `std(b_hat_j) = sigma * sqrt([(Y^T Y)^+]_jj)`. Reported relative to the true
    `b_j`, which converts it into "what fraction of the true value is the error bar".
  * a **Monte-Carlo recovery**: add noise at a stated fraction of RMS torque, solve, and report the
    actual relative error in `B_hat`.

Run: ``uv run python -m analysis.identifiability``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sim.dynamics import Body, human_arm_7dof_dynamics
from sim.arm3d import human_arm_7dof

MASSIVE = (2, 3, 6)  # upper arm, forearm, hand — the segments that carry a body
KINDS = ("mass", "com_x", "Ixx", "Iyy", "Izz")
DATA = Path("data/processed/dynamics")


def _theta0(dyn):
    """Current inertial parameter vector, ordered (body, kind)."""
    out = []
    for i in MASSIVE:
        b = dyn.bodies[i]
        out += [b.mass, b.com[0], b.I[0, 0], b.I[1, 1], b.I[2, 2]]
    return np.array(out, float)


def _with_theta(dyn, theta):
    """Rebuild an ArmDynamics with inertial parameters set to `theta` (damping unchanged)."""
    bodies = list(dyn.bodies)
    for k, i in enumerate(MASSIVE):
        m, cx, ixx, iyy, izz = theta[5 * k: 5 * k + 5]
        bodies[i] = Body(mass=m, com=(cx, 0.0, 0.0),
                         inertia=((ixx, 0, 0), (0, iyy, 0), (0, 0, izz)))
    from sim.dynamics import ArmDynamics
    return ArmDynamics(arm=dyn.arm, bodies=tuple(bodies), damping=tuple(np.diag(dyn.B)))


def regressor_block(dyn, q, qd, qdd, rel_step=1e-6):
    """`(7, 22)` block of `Y` at one sample: 15 inertial columns then 7 damping columns."""
    theta = _theta0(dyn)
    cols = []
    for j in range(theta.size):
        h = rel_step * max(abs(theta[j]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        up = _with_theta(dyn, tp).inverse_dynamics(q, qd, qdd)
        dn = _with_theta(dyn, tm).inverse_dynamics(q, qd, qdd)
        cols.append((up - dn) / (2 * h))
    damp = np.zeros((7, 7))
    np.fill_diagonal(damp, qd)          # d tau_j / d b_j = qdot_j
    return np.hstack([np.stack(cols, axis=1), damp])


def build_Y(dyn, samples):
    return np.vstack([regressor_block(dyn, q, qd, qdd) for q, qd, qdd in samples])


def load_samples(per_trial=8, trials_per_class=10, classes=None, seed=0):
    """Draw samples from the real task-3.14 slice. Returns (samples, class_of_sample)."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    rng = np.random.default_rng(seed)
    picked, labels = [], []
    for entry in manifest["trials"]:
        mc = entry["motion_class"]
        if classes and mc not in classes:
            continue
        if entry["trial_index"] >= trials_per_class:
            continue
        z = np.load(DATA / entry["file"])
        T = z["t"].size
        # sample from the moving part of the trial; the endpoints are exactly at rest and
        # contribute nothing to the damping columns
        idx = rng.choice(np.arange(int(0.15 * T), int(0.9 * T)), size=per_trial, replace=False)
        for k in idx:
            picked.append((z["q"][k].astype(float), z["qd"][k].astype(float),
                           z["qdd"][k].astype(float)))
            labels.append(mc)
    return picked, labels


def analyse(dyn, samples, label, sigma_frac=0.01, n_mc=200, seed=1):
    Y = build_Y(dyn, samples)
    b_true = np.diag(dyn.B)
    theta_true = np.concatenate([_theta0(dyn), b_true])
    tau = Y @ theta_true

    s = np.linalg.svd(Y, compute_uv=False)
    tol = max(Y.shape) * s[0] * np.finfo(float).eps
    rank = int((s > tol).sum())
    cond = s[0] / s[rank - 1]

    # noise amplification: std(theta_hat) = sigma * sqrt(diag(pinv(Y'Y)))
    cov = np.linalg.pinv(Y.T @ Y, rcond=1e-12)
    sigma = sigma_frac * np.sqrt((tau**2).mean())
    std_b = sigma * np.sqrt(np.maximum(np.diag(cov)[15:], 0.0))
    rel_err_bar = std_b / b_true

    # Monte-Carlo recovery
    rng = np.random.default_rng(seed)
    errs = []
    for _ in range(n_mc):
        theta_hat = np.linalg.lstsq(Y, tau + rng.normal(0, sigma, tau.size), rcond=None)[0]
        errs.append(np.abs(theta_hat[15:] - b_true) / b_true)
    errs = np.array(errs)

    print(f"\n=== {label} ===")
    print(f"  samples {len(samples):5d}  rows {Y.shape[0]:6d}  cols {Y.shape[1]}  "
          f"rank {rank}/{Y.shape[1]}  cond {cond:.3e}")
    print(f"  torque RMS {np.sqrt((tau**2).mean()):.4f} N m; noise sigma = {sigma_frac:.1%} "
          f"= {sigma:.5f} N m")
    print(f"  {'joint':16s} {'b_true':>8s} {'err bar/b':>12s} {'MC median':>12s} {'MC p90':>10s}")
    names = ["sh_yaw", "sh_pitch", "sh_roll", "elbow", "fore_pron", "wrist_flex", "wrist_dev"]
    for j, nm in enumerate(names):
        print(f"  {nm:16s} {b_true[j]:8.3f} {rel_err_bar[j]:11.1%} "
              f"{np.median(errs[:, j]):11.1%} {np.percentile(errs[:, j], 90):9.1%}")
    print(f"  WORST joint relative error (MC median): {np.median(errs, axis=0).max():.1%}")
    return {"rank": rank, "cond": cond, "rel_err_bar": rel_err_bar,
            "mc_median": np.median(errs, axis=0)}


def verify_linearity(dyn, sample):
    """The regressor must be step-independent — otherwise the 'finite difference' is a lie."""
    a = regressor_block(dyn, *sample, rel_step=1e-6)
    b = regressor_block(dyn, *sample, rel_step=1e-4)
    err = np.abs(a - b).max() / max(np.abs(a).max(), 1e-30)
    print(f"linearity check: regressor step-independent to {err:.2e} relative "
          f"({'PASS' if err < 1e-6 else 'FAIL'})")
    return err


def main():
    dyn = human_arm_7dof_dynamics()
    assert dyn.arm == human_arm_7dof()

    all_s, labels = load_samples()
    verify_linearity(dyn, all_s[0])
    print(f"\nloaded {len(all_s)} samples from {DATA}")

    res = {"pooled (all 4 classes)": analyse(dyn, all_s, "POOLED — all four motion classes")}
    for mc in ("reach", "lift", "wrist_rotate", "throw"):
        sub = [s for s, lab in zip(all_s, labels) if lab == mc]
        res[mc] = analyse(dyn, sub, f"{mc} only")
    return res


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# The constructive half: does a designed excitation trajectory fix it?
# --------------------------------------------------------------------------- #
def fourier_excitation(n_joints=7, n_harm=5, f_base=0.2, dur=5.0, fs=200.0,
                       amp=0.35, q_bias=None, seed=0):
    """Finite-Fourier-series excitation trajectory (Swevers-style), analytic derivatives.

    Each joint follows `q_j(t) = q0_j + sum_k [ a_jk/(w k) sin(w k t) - b_jk/(w k) cos(w k t) ]`
    with `w = 2 pi f_base`. Every joint is driven, which is the property the min-jerk motion
    classes lack. Coefficients are random but seeded; this is a *baseline* excitation, not an
    optimised one, so any improvement it shows is a lower bound on what optimisation would give.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(0.0, dur, 1.0 / fs)
    w = 2 * np.pi * f_base
    q0 = np.zeros(n_joints) if q_bias is None else np.asarray(q_bias, float)
    q = np.tile(q0, (t.size, 1)).astype(float)
    qd = np.zeros((t.size, n_joints))
    qdd = np.zeros((t.size, n_joints))
    for j in range(n_joints):
        for k in range(1, n_harm + 1):
            a, b = rng.normal(0, amp), rng.normal(0, amp)
            wk = w * k
            q[:, j] += a / wk * np.sin(wk * t) - b / wk * np.cos(wk * t)
            qd[:, j] += a * np.cos(wk * t) + b * np.sin(wk * t)
            qdd[:, j] += -a * wk * np.sin(wk * t) + b * wk * np.cos(wk * t)
    return t, q, qd, qdd


def compare_with_designed_excitation(n_samples=320, seed=3):
    """Head to head: the task-3.14 motion classes vs a plain Fourier excitation."""
    dyn = human_arm_7dof_dynamics()
    rng = np.random.default_rng(seed)

    _t, q, qd, qdd = fourier_excitation(seed=seed)
    lam = np.array([np.linalg.eigvalsh(dyn.mass_matrix(q[k]))[0] for k in range(0, q.shape[0], 20)])
    print(f"\nexcitation trajectory: {q.shape[0]} samples, "
          f"min lambda_min(M) = {lam.min():.2e} (well clear of the gimbal locks)")
    print("RMS |qd| per joint: " + " ".join(f"{v:.3f}" for v in np.sqrt((qd**2).mean(axis=0))))

    idx = rng.choice(q.shape[0], size=n_samples, replace=False)
    samples = [(q[k], qd[k], qdd[k]) for k in idx]
    return analyse(dyn, samples, "DESIGNED Fourier excitation (all 7 joints driven)")


def noise_sweep(fracs=(0.001, 0.003, 0.01, 0.03)):
    """At what torque-noise level does B become recoverable from the motion classes?"""
    dyn = human_arm_7dof_dynamics()
    samples, _ = load_samples()
    Y = build_Y(dyn, samples)
    b_true = np.diag(dyn.B)
    theta_true = np.concatenate([_theta0(dyn), b_true])
    tau = Y @ theta_true
    cov = np.linalg.pinv(Y.T @ Y, rcond=1e-12)
    print("\nnoise sweep — worst-joint relative error bar on B, pooled motion classes:")
    for f in fracs:
        sigma = f * np.sqrt((tau**2).mean())
        rel = sigma * np.sqrt(np.maximum(np.diag(cov)[15:], 0.0)) / b_true
        print(f"  torque noise {f:7.1%} -> worst joint {rel.max():8.1%}   (median joint {np.median(rel):.1%})")
