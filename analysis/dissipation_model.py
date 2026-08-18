"""Is the *form* of the damping identifiable, not just its value? (depth pass, 2026-08-14)

Why this script exists
----------------------
Task 3.15 reduced the **variance** of estimating `B` — the worst joint's error bar fell from
174.8% to 30.2% by adding an excitation motion class. Every one of those numbers assumes the
dissipation model is right: `tau_d = -B qdot`, `B` diagonal and constant. The dataset's own
manifest states that assumption as fact, and `sim.dynamics.human_arm_7dof_dynamics` says in its
docstring that the number "only has to be recorded, not correct" — which is true of the *value*
and silently untrue of the *structure*.

If the structure is wrong, an error bar is the wrong thing to have shrunk. A biased estimator with
a tight error bar reports a wrong number confidently, which is worse than reporting a wide one.

So this script asks two questions the 3.15 work did not:

  1. **Bias under misspecification.** If the true dissipation carries a dry-friction (Coulomb)
     component alongside the viscous one, and the estimator fits viscous-only, how far off is
     `b_hat`? Compare that bias against the 30.2% noise error bar 3.15 achieved. If bias >> error
     bar, then 3.15 sharpened the wrong quantity.

  2. **Discriminability.** Can this dataset tell the two apart at all? Fit the extended model
     `tau_d = -B qdot - F_c sign(qdot)` and read off the error bar on `F_c`. A dataset that cannot
     resolve `F_c` cannot falsify the viscous assumption, and a pHNN trained on it is being scored
     on a model choice the data never tested.

  3. **The sampling window.** `analysis.identifiability.load_samples` draws from 15%-90% of each
     trial, on the stated grounds that the endpoints are at rest and "contribute nothing to the
     damping columns". That is right for a *viscous* column and exactly wrong for discriminating
     viscous from Coulomb, because the two differ most at low speed. So the window is re-run wide.

Everything below is exact and cheap: the extended model is still linear in `(b, f)`, so the bias is
the closed form `bias = (Y_v^+ Y_c) f` and no simulation is needed. `Y` inertial columns are reused
from `analysis.identifiability`, which already verifies them step-independent.

Run: ``uv run python -m analysis.dissipation_model``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.identifiability import DATA, _theta0, build_Y
from sim.dynamics import human_arm_7dof_dynamics

JOINTS = ["sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
          "forearm_pron", "wrist_flex", "wrist_dev_hand"]

# Smoothing velocity for the Coulomb term. tanh(qdot/V_EPS) is the standard smooth stand-in for
# sign(qdot) (Armstrong-Helouvry, Dupont & Canudas de Wit 1994, doi:10.1016/0005-1098(94)90209-7).
# 0.05 rad/s is ~2% of the fastest joint's RMS speed in this dataset, so the smoothing is narrow
# relative to the motion but wide enough that the column is differentiable.
V_EPS = 0.05


def _damping_columns(qd_stack):
    """`(7N, 7)` viscous and `(7N, 7)` Coulomb blocks for a stack of joint velocities."""
    n = qd_stack.shape[0]
    visc = np.zeros((7 * n, 7))
    coul = np.zeros((7 * n, 7))
    for k in range(n):
        np.fill_diagonal(visc[7 * k: 7 * k + 7], qd_stack[k])
        np.fill_diagonal(coul[7 * k: 7 * k + 7], np.tanh(qd_stack[k] / V_EPS))
    return visc, coul


def load_samples_window(per_trial=8, trials_per_class=10, classes=None, seed=0,
                        lo=0.15, hi=0.90):
    """Like `identifiability.load_samples` but with the trial window exposed.

    `lo`/`hi` are fractions of trial duration. The shipped analysis uses (0.15, 0.90); (0.0, 1.0)
    keeps the near-rest endpoints, which is where viscous and Coulomb friction differ most.
    """
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
        span = np.arange(int(lo * T), max(int(hi * T), int(lo * T) + per_trial + 1))
        idx = rng.choice(span, size=per_trial, replace=False)
        for k in idx:
            picked.append((z["q"][k].astype(float), z["qd"][k].astype(float),
                           z["qdd"][k].astype(float)))
            labels.append(mc)
    return picked, labels


def analyse_form(dyn, samples, label, sigma_abs, coulomb_share=0.5, verbose=True):
    """Bias from misspecification, and the error bar on the Coulomb term.

    `coulomb_share` sets the true `F_c` so that, on these samples, the Coulomb term dissipates
    `share/(1-share)` times as much power as the viscous term, per joint. 0.5 is an even split.
    """
    Y_inert = build_Y(dyn, samples)[:, :15]
    qd = np.array([s[1] for s in samples])
    visc, coul = _damping_columns(qd)

    b_true = np.diag(dyn.B)
    # f_j chosen so mean Coulomb power f_j E|qdot_j| equals `share/(1-share)` x viscous power
    # b_j E[qdot_j^2]. Guard joints that barely move.
    e_abs = np.abs(qd).mean(axis=0)
    e_sq = (qd ** 2).mean(axis=0)
    ratio = coulomb_share / max(1.0 - coulomb_share, 1e-9)
    f_true = np.where(e_abs > 1e-9, ratio * b_true * e_sq / np.maximum(e_abs, 1e-9), 0.0)

    Y_v = np.hstack([Y_inert, visc])                # what the estimator fits
    Y_f = np.hstack([Y_inert, visc, coul])          # what generated the data
    theta_v = np.concatenate([_theta0(dyn), b_true])
    theta_f = np.concatenate([_theta0(dyn), b_true, f_true])
    tau = Y_f @ theta_f

    # --- 1. bias of the viscous-only fit against Coulomb-contaminated torque (noise-free) ---
    fit = np.linalg.lstsq(Y_v, tau, rcond=None)[0]
    bias_rel = np.abs(fit[15:22] - b_true) / b_true

    # --- 2. noise error bar on the viscous-only fit, at the manifest's shared sigma ---
    cov_v = np.linalg.pinv(Y_v.T @ Y_v, rcond=1e-12)
    bar_v = sigma_abs * np.sqrt(np.maximum(np.diag(cov_v)[15:22], 0.0)) / b_true

    # --- 3. the extended fit: can the data resolve F_c at all? ---
    sv_f = np.linalg.svd(Y_f, compute_uv=False)
    tol = max(Y_f.shape) * sv_f[0] * np.finfo(float).eps
    rank_f = int((sv_f > tol).sum())
    cond_f = sv_f[0] / sv_f[rank_f - 1]
    cov_f = np.linalg.pinv(Y_f.T @ Y_f, rcond=1e-12)
    bar_b_ext = sigma_abs * np.sqrt(np.maximum(np.diag(cov_f)[15:22], 0.0)) / b_true
    bar_f_ext = np.where(f_true > 0,
                         sigma_abs * np.sqrt(np.maximum(np.diag(cov_f)[22:29], 0.0))
                         / np.maximum(f_true, 1e-30), np.inf)

    # --- 4. how collinear are the two damping columns, per joint? ---
    collin = []
    for j in range(7):
        a, c = visc[:, j], coul[:, j]
        na, nc = np.linalg.norm(a), np.linalg.norm(c)
        collin.append(abs(a @ c) / (na * nc) if na > 1e-12 and nc > 1e-12 else np.nan)
    collin = np.array(collin)

    if verbose:
        print(f"\n=== {label} ===")
        print(f"  samples {len(samples)}  sigma {sigma_abs:.5f} N m  "
              f"Coulomb share {coulomb_share:.0%}  ext rank {rank_f}/{Y_f.shape[1]}  "
              f"ext cond {cond_f:.3e}")
        print(f"  {'joint':15s} {'bias(B)':>9s} {'errbar(B)':>10s} {'bias/bar':>9s} "
              f"{'errbar(Fc)':>11s} {'cos(v,c)':>9s}")
        for j, nm in enumerate(JOINTS):
            r = bias_rel[j] / bar_v[j] if bar_v[j] > 0 else np.inf
            print(f"  {nm:15s} {bias_rel[j]:8.1%} {bar_v[j]:9.1%} {r:8.1f}x "
                  f"{bar_f_ext[j]:10.1%} {collin[j]:8.4f}")
        print(f"  worst bias {bias_rel.max():.1%} on {JOINTS[int(np.argmax(bias_rel))]}; "
              f"worst Fc error bar {np.nanmax(bar_f_ext[np.isfinite(bar_f_ext)]):.1%}")
    return {"bias_rel": bias_rel, "bar_v": bar_v, "bar_b_ext": bar_b_ext,
            "bar_f_ext": bar_f_ext, "collinearity": collin,
            "cond_ext": cond_f, "rank_ext": rank_f}


def main():
    manifest = json.loads((DATA / "manifest.json").read_text())
    damping = tuple(manifest["true_damping"]["B_diagonal_Nms_per_rad"])
    sigma = float(manifest["identifiability"]["shared_torque_noise_Nm"])
    dyn = human_arm_7dof_dynamics(damping=damping)
    nat = ["reach", "lift", "wrist_rotate", "throw"]

    print(f"true B      = {list(damping)}")
    print(f"shared sigma = {sigma:.6f} N m (from the manifest, so numbers compare to 3.15's)")

    out = {}
    s_nat, _ = load_samples_window(classes=nat)
    s_exc, _ = load_samples_window(classes=["excite"], trials_per_class=40)
    out["naturalistic (15-90% window)"] = analyse_form(
        dyn, s_nat, "NATURALISTIC classes, shipped 15-90% window", sigma)
    out["excite (15-90% window)"] = analyse_form(
        dyn, s_exc, "EXCITE class, shipped 15-90% window", sigma)
    out["combined (15-90% window)"] = analyse_form(
        dyn, s_nat + s_exc, "COMBINED, shipped 15-90% window", sigma)

    # Test 3: does keeping the near-rest endpoints help discriminate the two friction forms?
    w_nat, _ = load_samples_window(classes=nat, lo=0.0, hi=1.0)
    w_exc, _ = load_samples_window(classes=["excite"], trials_per_class=40, lo=0.0, hi=1.0)
    out["combined (full 0-100% window)"] = analyse_form(
        dyn, w_nat + w_exc, "COMBINED, full 0-100% window (endpoints kept)", sigma)

    # Test 1b: bias is exactly linear in the Coulomb share, so sweeping it is free.
    print("\nbias on B from fitting viscous-only, vs how much of the dissipation is dry friction")
    print(f"  {'share':>7s} {'worst bias':>11s} {'median bias':>12s}")
    for share in (0.1, 0.25, 0.5, 0.75):
        r = analyse_form(dyn, s_nat + s_exc, "", sigma, coulomb_share=share, verbose=False)
        print(f"  {share:6.0%} {r['bias_rel'].max():10.1%} {np.median(r['bias_rel']):11.1%}")
    return out


if __name__ == "__main__":
    main()
