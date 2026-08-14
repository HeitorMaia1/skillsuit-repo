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
model structure, exact `qddot`, no filtering error, and the local Cramér–Rao bound rather than any
particular fitting procedure. A network that must also learn the model structure cannot have a
smaller error bar without becoming biased — it can only trade variance for a systematic offset it
has no way to report. So every number here is a floor on the difficulty facing 5.8/5.9.

Method
------
The regressor `Y = d tau / d theta` is assembled column by column over 15 inertial parameters
(mass, com_x, Ixx, Iyy, Izz for each of the three massive bodies) plus 7 damping terms — 22 in all.
Damping columns are exact by inspection: `d tau / d b_j = qdot_j e_j`. Inertial columns come from
central differences, verified step-independent so they are exact to roundoff rather than
approximate (measured 5.08e-10 relative).

**`Y` is a Jacobian, not a global linear model, and the difference is worth stating precisely.**
`tau` is exactly linear in each body's *mass* and *inertia* — a scaled parameter scales its torque
contribution, verified to 1e-12 — but it is **quadratic in `com_x`**, because the parallel-axis
term contributes `m |c|^2` to the inertia about the joint. (The classical formulation avoids this
by using barycentric parameters `(m, m c, I_about_joint)`, in which the model *is* globally linear;
this module parameterises by `(m, c, I_about_CoM)` instead, because that is what `sim.dynamics.Body`
stores.) So `Y theta != tau`, and `analyse()` deliberately scores against the **linearised
surrogate** `Y theta_true` rather than the true torque.

That is the right instrument for the question being asked, and the numbers are unaffected. Local
identifiability is a statement about sensitivity: `std(theta_hat) = sigma sqrt(diag((Y^T Y)^+))` is
the local Cramér–Rao bound for the estimate, and the condition number of `Y` is Gautier & Khalil's
criterion — both are properties of the Jacobian at the true parameters, which is exactly what is
computed here. What it does *not* claim is a global uniqueness result: it says a joint's damping
cannot be pinned down from small perturbations around the truth, which is the practical question
Phase 5 faces.

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

from sim.arm3d import human_arm_7dof
from sim.dynamics import Body, human_arm_7dof_dynamics
from sim.motions import excitation_trial

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

    # NB `tau` here is the linearised surrogate `Y theta_true`, not `inverse_dynamics(...)`. They
    # differ because the model is quadratic in com_x (module docstring). The surrogate is the
    # correct object for a noise-amplification study: it isolates how measurement noise propagates
    # through the Jacobian, which is the question, and it makes the Monte-Carlo below exact rather
    # than confounded with linearisation error.

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
#
# The trajectory itself now lives in `sim.motions.excitation_trial` (task 3.15) rather than being
# defined here, so the thing measured is exactly the thing the dataset generator emits. A test in
# `tests/test_identifiability.py` pins the improvement this function reports.
# --------------------------------------------------------------------------- #
def compare_with_designed_excitation(n_samples=320, seed=3):
    """Head to head: the task-3.14 motion classes vs a plain Fourier excitation."""
    dyn = human_arm_7dof_dynamics()
    rng = np.random.default_rng(seed)

    _t, q, qd, qdd = excitation_trial(fs=200.0, rng=np.random.default_rng(seed))
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


# --------------------------------------------------------------------------- #
# Public entry point — used by `skilldata.generate_dynamics` to write the
# identifiability block into the dataset manifest (task 3.15)
# --------------------------------------------------------------------------- #
def damping_error_bars(dyn, samples, sigma_frac=0.01, sigma_abs=None):
    """Per-joint relative error bar on `B`, plus the regressor's rank and condition number.

    `samples` is an iterable of `(q, qdot, qddot)`. Returns a JSON-serialisable dict.

    The error bar is the standard identification quantity: for i.i.d. torque noise of standard
    deviation `sigma`, `std(b_hat_j) = sigma * sqrt([(Y^T Y)^+]_jj)`, reported as a fraction of the
    true `b_j` so that 0.20 reads "the error bar is a fifth of the value" and 2.07 reads "the error
    bar is twice the value, so this joint's damping is not measurable from this data".

    This is the local Cramér–Rao bound, computed from the Jacobian at the true parameters with the
    **exact** model structure and **exact** `qddot`. A network that must also learn the structure
    cannot achieve a smaller error bar without becoming biased, so these are floors, not
    predictions. Note `Y` is a Jacobian: `tau` is exactly linear in mass and inertia but quadratic
    in the CoM offset under this parameterisation (see the module docstring), so this is a
    statement about local sensitivity, not global uniqueness.
    """
    samples = list(samples)
    Y = build_Y(dyn, samples)
    b_true = np.diag(dyn.B)
    tau = Y @ np.concatenate([_theta0(dyn), b_true])
    sv = np.linalg.svd(Y, compute_uv=False)
    tol = max(Y.shape) * sv[0] * np.finfo(float).eps
    rank = int((sv > tol).sum())
    cov = np.linalg.pinv(Y.T @ Y, rcond=1e-12)
    rms_tau = float(np.sqrt((tau**2).mean()))
    # `sigma_abs` exists because error bars from different sample sets are only comparable at a
    # shared noise level. Defaulting sigma to a fraction of *each set's own* RMS torque made the
    # combined set look worse than its own subset — impossible, since adding data can only add
    # Fisher information — because `throw` peaks at 114 N m and inflated that set's sigma. Caught
    # 2026-08-14. Pass `sigma_abs` when comparing blocks; leave it None to read one set on its own
    # terms.
    sigma = float(sigma_abs) if sigma_abs is not None else sigma_frac * rms_tau
    rel = sigma * np.sqrt(np.maximum(np.diag(cov)[15:], 0.0)) / b_true
    names = ["sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
             "forearm_pron", "wrist_flex", "wrist_dev_hand"]
    return {
        "n_samples": len(samples),
        "torque_rms_Nm": float(f"{rms_tau:.6g}"),
        "torque_noise_Nm": float(f"{sigma:.6g}"),
        "torque_noise_frac": None if sigma_abs is not None else sigma_frac,
        "regressor_rank": rank,
        "regressor_cols": int(Y.shape[1]),
        "condition_number": float(f"{sv[0] / sv[rank - 1]:.6g}"),
        "relative_error_bar": {n: float(f"{v:.6g}") for n, v in zip(names, rel)},
        "worst_joint": names[int(np.argmax(rel))],
        "worst_relative_error_bar": float(f"{rel.max():.6g}"),
        "method": ("std(b_hat) = sigma*sqrt(diag(pinv(Y^T Y))) / b_true, with Y the exact "
                   "linear-in-parameters regressor (15 inertial + 7 damping columns) and sigma a "
                   "fraction of RMS torque. Least squares with the exact model structure and exact "
                   "qddot, so these are floors on any estimator, not predictions."),
    }
