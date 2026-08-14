"""Why natural motion cannot identify the arm — and whether the fix is humanly performable.

(depth pass, 2026-08-14, after task 3.15)

Why this script exists
----------------------
Task 3.15 fixed the identifiability failure the previous depth pass measured, by adding an
`excite` motion class: a finite-Fourier-series trajectory driving all seven joints. It worked —
worst-joint damping error bar 174.8% -> 30.2%. But the manifest justifies the class with a single
sentence that nothing has tested:

    "It is an instrument, not a motion anyone performs."

That sentence is load-bearing for the whole project. SkillSuit is a **wearable that records
humans**. If the trajectory that rescues identifiability is one no human can produce, then the
rescue exists only in simulation, Track B hardware inherits the original problem untouched, and
every Phase-5 number about recovering `B` describes a dataset that cannot be collected. If instead
the trajectory *is* humanly performable, the manifest is wrong in the project's favour and the
excitation belongs in the capture protocol, not just in the simulator.

Nobody had checked which. This script checks, along the three axes that could make a trajectory
humanly impossible, and finds that the first two are not the binding constraint and the third is.

The three axes
--------------
1. **Torque.** Does the motion demand joint torques beyond human capability? Benchmarked against
   Fleisig, Andrews, Dillman & Escamilla (1995), *Kinetics of Baseball Pitching with Implications
   About Injury Mechanisms*, Am J Sports Med 23(2):233-239, doi:10.1177/036354659502300218, which
   measured 26 highly skilled healthy adult pitchers and reports **67 N m of shoulder internal
   rotation torque, 64 N m of elbow varus torque** and, after ball release, **97 N m of shoulder
   horizontal abduction torque**. Those are near the ceiling of what an adult arm produces in any
   common movement, so a simulated motion demanding more is not a motion a person performs.

2. **Range of motion.** Does it exceed the joint's travel? Benchmarked against Zwerus, Willigenburg,
   Scholtes, Somford, Eygendaal & van den Bekerom (2017), *Normative values and affecting factors
   for the elbow range of motion*, Shoulder & Elbow 11(3):215-224,
   doi:10.1177/1758573217728711, n=352 healthy adults, goniometer, **active** ROM of the dominant
   arm: **flexion 146 deg, extension -2 deg, pronation 80 deg, supination 87 deg**. Those two joints
   (elbow, forearm) are the ones this arm model maps one-to-one onto a measured human joint, so they
   are the ones scored. Task 3.16 later added the shoulder and wrist from the AAOS reference
   standard, so the table now scores six of seven joints; `sh_yaw` stays unsourced because mapping
   a frontal-plane abduction limit onto an azimuth axis means inventing a convention. Limits and
   their provenance live in `sim.limits`, which is the single table this module reads.

3. **Coordination.** Can a person move seven joints *independently*? This is the axis the other two
   miss, and it is the one the motor-control literature is about. Sanger (2000), *Human Arm
   Movements Described by a Low-Dimensional Superposition of Principal Components*,
   J Neurosci 20(3):1066-1072, doi:10.1523/jneurosci.20-03-01066.2000, reports that smooth human arm
   trajectories "have very low dimension and ... converge toward a linear superposition of the first
   few principal components", and attributes the low dimensionality to "combined properties of the
   internal controller and the musculoskeletal system" — that is, to the mover, not to the task.

   So the measurable question is: **how many dimensions does each motion class actually use, and is
   the identifiability failure the same fact as the dimensionality deficit?**

What is measured
----------------
For each class, over the real 500-trial task-3.15 dataset:

* peak and 99th-percentile |tau| per joint, against the Fleisig envelope;
* joint excursion in degrees, against the Zwerus elbow/forearm ROM;
* peak |qdot| per joint;
* the **effective dimensionality** of the joint-velocity covariance, by two measures that disagree
  in an informative way:
    - `participation ratio` PR = (sum lambda_i)^2 / sum lambda_i^2, a standard effective-rank
      statistic, continuous in [1, 7];
    - `n_pc_90`, the number of principal components needed to reach 90% of variance, which is the
      number the motor-synergy literature usually quotes.
  Both are computed on the **raw** velocity covariance (what the dynamics actually sees, and what
  the regressor is built from) and on the **correlation** matrix (each joint standardised, which
  removes the 43x velocity spread and asks the purely geometric question of how many directions are
  used at all). The two answers differ and the difference is the finding.
* per-class regressor condition number and rank, so dimensionality and identifiability can be put
  in the same table.

Run as ``uv run python -m analysis.synergy``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from sim.dynamics import human_arm_7dof_dynamics
from sim.limits import JOINT_LIMITS, JOINT_ORDER, UNSOURCED

from .identifiability import build_Y

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "dynamics"

JOINTS = list(JOINT_ORDER)

# Fleisig et al. (1995), doi:10.1177/036354659502300218, 26 highly skilled adult pitchers.
# The three torque magnitudes quoted verbatim in that paper's abstract. Used as a single scalar
# ceiling rather than per-joint, because the paper's joint conventions do not map one-to-one onto
# this model's 7 axes and inventing a mapping would be worse than admitting the coarseness.
FLEISIG_ELBOW_VARUS_NM = 64.0
FLEISIG_SHOULDER_IR_NM = 67.0
FLEISIG_SHOULDER_HABD_NM = 97.0
FLEISIG_CEILING_NM = FLEISIG_SHOULDER_HABD_NM  # the largest value the paper reports

# Range-of-motion limits now live in `sim.limits`, which is the single sourced table and carries
# the tier and citation of every number. When this module was written only the elbow and forearm had
# a limit at all; task 3.16 added the shoulder and wrist from the AAOS reference standard, leaving
# `sh_yaw` unsourced because mapping a frontal-plane abduction limit onto an azimuth axis would mean
# inventing a convention. The violation counts printed below are therefore still a LOWER BOUND.
ROM_DEG = {name: (math.degrees(lim.lo), math.degrees(lim.hi))
           for name, lim in JOINT_LIMITS.items() if lim is not None}
ROM_UNVERIFIED = list(UNSOURCED)


def load_class(motion_class, max_trials=100):
    """Concatenate q, qd, tau over every trial of one class. Returns dict of (N, 7) arrays."""
    manifest = json.loads((DATA / "manifest.json").read_text())
    q, qd, tau = [], [], []
    n = 0
    for entry in manifest["trials"]:
        if entry["motion_class"] != motion_class or entry["trial_index"] >= max_trials:
            continue
        z = np.load(DATA / entry["file"])
        q.append(z["q"].astype(float))
        qd.append(z["qd"].astype(float))
        tau.append(z["tau"].astype(float))
        n += 1
    return {"q": np.vstack(q), "qd": np.vstack(qd), "tau": np.vstack(tau), "n_trials": n}


def participation_ratio(eigvals):
    """(sum l)^2 / sum l^2 — the effective number of directions carrying the variance.

    Equals 1 when one eigenvalue dominates and n when all n are equal, and unlike a
    variance-threshold count it is continuous, so it does not jump on an arbitrary cutoff.
    """
    lam = np.maximum(np.asarray(eigvals, dtype=float), 0.0)
    s = lam.sum()
    return float(s * s / (lam**2).sum()) if s > 0 else 0.0


def n_pc_for(eigvals, frac=0.90):
    lam = np.sort(np.maximum(np.asarray(eigvals, dtype=float), 0.0))[::-1]
    if lam.sum() <= 0:
        return 0
    return int(np.searchsorted(np.cumsum(lam) / lam.sum(), frac) + 1)


def dimensionality(qd):
    """Effective dimensionality of a joint-velocity cloud, raw and standardised.

    Raw covariance is what the dynamics and the regressor actually see. The correlation matrix
    standardises every joint to unit variance first, which discards the 43x speed spread between
    joints and asks the separate, purely geometric question: how many independent *directions* does
    this motion use at all? A class can score low on raw and high on correlation — that means it
    does move every joint, just not comparably hard.
    """
    x = qd - qd.mean(axis=0)
    cov = (x.T @ x) / max(len(x) - 1, 1)
    lam_raw = np.linalg.eigvalsh(cov)

    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    live = sd > 1e-12  # a joint that never moves has no direction to contribute
    if live.sum() >= 2:
        corr = cov[np.ix_(live, live)] / np.outer(sd[live], sd[live])
        lam_corr = np.linalg.eigvalsh(corr)
    else:
        lam_corr = np.array([1.0] * int(live.sum()))

    return {
        "pr_raw": participation_ratio(lam_raw),
        "n_pc_90_raw": n_pc_for(lam_raw, 0.90),
        "pr_corr": participation_ratio(lam_corr),
        "n_pc_90_corr": n_pc_for(lam_corr, 0.90),
        "n_joints_moving": int(live.sum()),
    }


def sample_for_regressor(motion_class, per_trial=8, trials=10, seed=0):
    manifest = json.loads((DATA / "manifest.json").read_text())
    rng = np.random.default_rng(seed)
    out = []
    for entry in manifest["trials"]:
        if entry["motion_class"] != motion_class or entry["trial_index"] >= trials:
            continue
        z = np.load(DATA / entry["file"])
        T = z["t"].size
        idx = rng.choice(np.arange(int(0.15 * T), int(0.9 * T)), size=per_trial, replace=False)
        for k in idx:
            out.append((z["q"][k].astype(float), z["qd"][k].astype(float),
                        z["qdd"][k].astype(float)))
    return out


def conditioning(dyn, motion_class):
    Y = build_Y(dyn, sample_for_regressor(motion_class))
    s = np.linalg.svd(Y, compute_uv=False)
    tol = max(Y.shape) * s[0] * np.finfo(float).eps
    rank = int((s > tol).sum())
    return {"rank": rank, "cols": Y.shape[1], "cond": float(s[0] / s[rank - 1])}


def main():
    dyn = human_arm_7dof_dynamics()
    classes = ["reach", "lift", "wrist_rotate", "throw", "excite"]

    print("=" * 96)
    print("1. TORQUE — is the motion within human capability?")
    print(f"   Ceiling: Fleisig 1995 elite pitchers — elbow varus {FLEISIG_ELBOW_VARUS_NM} N m, "
          f"shoulder IR {FLEISIG_SHOULDER_IR_NM} N m, shoulder horiz. abd. "
          f"{FLEISIG_SHOULDER_HABD_NM} N m")
    print("=" * 96)
    print(f"{'class':14s} {'peak|tau|':>10s} {'p99|tau|':>10s} {'worst joint':>12s} "
          f"{'vs 97 N m':>10s}")
    data = {}
    for mc in classes:
        d = load_class(mc)
        data[mc] = d
        a = np.abs(d["tau"])
        peak, p99 = a.max(), np.percentile(a, 99)
        worst = JOINTS[int(a.max(axis=0).argmax())]
        flag = "OVER" if peak > FLEISIG_CEILING_NM else "ok"
        print(f"{mc:14s} {peak:10.2f} {p99:10.2f} {worst:>12s} {flag:>10s}")

    print()
    print("=" * 96)
    print("2. RANGE OF MOTION — does it exceed the joint's travel?")
    print("   Ceiling: sim.limits — tier A (Zwerus 2017, n=352) for elbow and forearm, tier B "
          "(AAOS) for shoulder and wrist.")
    print(f"   Not scored: {', '.join(ROM_UNVERIFIED)} — so every count below is a LOWER BOUND.")
    print("=" * 96)
    print(f"{'class':14s} {'joint':16s} {'min deg':>9s} {'max deg':>9s} {'limit':>16s} {'':>6s}")
    for mc in classes:
        for j, nm in enumerate(JOINTS):
            if nm not in ROM_DEG:
                continue
            deg = np.rad2deg(data[mc]["q"][:, j])
            lo, hi = ROM_DEG[nm]
            bad = deg.min() < lo - 1e-9 or deg.max() > hi + 1e-9
            print(f"{mc:14s} {nm:16s} {deg.min():9.1f} {deg.max():9.1f} "
                  f"{f'[{lo:.0f}, {hi:.0f}]':>16s} {'OVER' if bad else 'ok':>6s}")

    print()
    print("=" * 96)
    print("3. COORDINATION — how many dimensions does the motion use?")
    print("   Sanger 2000: human arm trajectories 'have very low dimension'. 7 joints available.")
    print("=" * 96)
    print(f"{'class':14s} {'joints':>7s} {'PR raw':>8s} {'PC90 raw':>9s} {'PR corr':>8s} "
          f"{'PC90 corr':>10s} {'rank':>7s} {'cond':>10s}")
    rows = {}
    for mc in classes:
        dim = dimensionality(data[mc]["qd"])
        cnd = conditioning(dyn, mc)
        rows[mc] = {**dim, **cnd}
        print(f"{mc:14s} {dim['n_joints_moving']:7d} {dim['pr_raw']:8.2f} "
              f"{dim['n_pc_90_raw']:9d} {dim['pr_corr']:8.2f} {dim['n_pc_90_corr']:10d} "
              f"{cnd['rank']:4d}/{cnd['cols']:<2d} {cnd['cond']:10.2e}")

    print()
    print("   peak |qdot| per joint, deg/s")
    print(f"   {'class':14s} " + " ".join(f"{nm:>15s}" for nm in JOINTS))
    for mc in classes:
        v = np.rad2deg(np.abs(data[mc]["qd"]).max(axis=0))
        print(f"   {mc:14s} " + " ".join(f"{x:15.1f}" for x in v))

    print()
    print("=" * 96)
    print("Does dimensionality predict identifiability?")
    print("=" * 96)
    pr = np.array([rows[mc]["pr_raw"] for mc in classes])
    lc = np.log10(np.array([rows[mc]["cond"] for mc in classes]))
    rk = np.array([rows[mc]["rank"] for mc in classes], dtype=float)
    print(f"  Pearson r(participation ratio, log10 cond) = {np.corrcoef(pr, lc)[0, 1]:+.3f}"
          f"   (n={len(classes)} classes)")
    print(f"  Pearson r(participation ratio, rank)       = {np.corrcoef(pr, rk)[0, 1]:+.3f}")
    print(f"  Pearson r(joints moving,       rank)       = "
          f"{np.corrcoef([rows[mc]['n_joints_moving'] for mc in classes], rk)[0, 1]:+.3f}")
    return rows


if __name__ == "__main__":
    main()
