"""Task 3.14 — a labelled dataset slice that carries dynamics ground truth.

Why this exists
---------------
The SkillData v1 dataset (task 3.9) is kinematic: joint angles, IMU streams, retargeted robot
joints. It has `q(t)` but no momentum, no forces and — the part that matters — **no dissipation
in the generating process**. Task 5.9 wants to claim that a pHNN's learned dissipation
concentrates in prep and settle and is near zero during the active phase. Fitted to v1, that
claim is untestable: the true damping is identically zero everywhere, so a plausible-looking
learned curve would be an artefact of the min-jerk trajectory generator rather than a result.

This module regenerates a subset of those same trials and labels each sample with the rigid-body
dynamics from `sim.dynamics`: generalized momentum `p = M(q) qdot`, total energy
`H = 1/2 qdot^T M qdot + U(q)`, the actuation torque `tau` from inverse dynamics, and the true
dissipated power `qdot^T B qdot` — with the **damping matrix `B` recorded in the manifest**. Tasks
5.8 and 5.9 can then be scored as "the network recovered `B` to within X%" instead of "the curve
looks like the story".

What this does NOT do
---------------------
It does not touch SkillData v1. That schema is a locked contract (`schema_version` const
`skilldata-v1`) and adding dynamics fields to it is a v2 question, not a task-3.14 question — the
case is filed in `PROPOSALS.md` (2026-08-14). This writes a **sidecar** under
`data/processed/dynamics/` with its own manifest and its own file format.

Index alignment — read this before joining the two datasets
-----------------------------------------------------------
Trials here are drawn by **replaying the exact random stream of `generate_synthetic`** for the
same `--align-to-n` and keeping the first `--per-class` trials of each motion class. With the
default alignment (`--align-to-n 1000`, matching the shipped v1 dataset) a trial labelled
`(motion_class, trial_index)` here is **the identical motion** to the SkillData v1 record with the
same `(motion_class, trial_index)`, so the two can be joined on that pair. Change `--align-to-n`
or `--seed` and that guarantee is gone, which is why both are recorded in the manifest.

Format
------
One `.npz` per trial, float32, plus `manifest.json`. Per-sample arrays, all `(T, 7)` unless noted:

    t                 (T,)   time [s]
    q, qd, qdd        joint angle / velocity / acceleration [rad, rad/s, rad/s^2]
    p                 generalized momentum M(q) qdot [kg m^2 / s]
    tau               inverse-dynamics torque M qddot + C qdot + g + B qdot [N m]
    H, T_kin, U       (T,)   total / kinetic / potential energy [J]
    power_dissipated  (T,)   qdot^T B qdot — the TRUE dissipation rate [W]
    lambda_min        (T,)   smallest eigenvalue of M(q) [kg m^2]

`lambda_min` is stored deliberately. The 7-DOF arm has two gimbal-lock configurations where `M`
loses rank — the wrist at `q[5] = pi/2`, and, more awkwardly, the **shoulder at `q[1] = pi/2`,
which is the arm's own hanging rest pose** (see `tests/test_dynamics.py`). Near either of them
`H = 1/2 p^T M^-1 p` is ill-conditioned. Phase 5 should filter or weight on this column rather
than discover the problem as a training instability.

Run:  ``uv run python -m skilldata.generate_dynamics --out data/processed/dynamics``
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from sim.dynamics import DE_LEVA_MALE, human_arm_7dof_dynamics
from sim.motions import MOTION_CLASSES, generate_trial

from .generate_synthetic import SUBJECT_ID, _class_counts

FORMAT_VERSION = "skilldata-dynamics-v1"

# Per-joint viscous damping [N m s / rad] for the 7-DOF arm, proximal -> distal. Larger at the
# shoulder than the wrist, which is the right ordering physically, but the magnitudes are a
# deliberate, uncalibrated choice: the point of this dataset is that B is *known*, not that it is
# the physiological value. Whatever is here is what 5.9 is scored against, so it goes in the
# manifest.
DEFAULT_DAMPING = (0.080, 0.080, 0.050, 0.050, 0.020, 0.020, 0.020)


def _aligned_trials(per_class: int, align_to_n: int, seed: int, fs: float):
    """Replay `generate_synthetic`'s draw order and yield the first `per_class` of each class.

    Yields `(motion_class, trial_index, t, q, qd, qdd)`. Consuming the full stream (not just the
    kept trials) is what preserves index alignment with the v1 dataset — the RNG must advance
    exactly as it did there.
    """
    rng = np.random.default_rng(seed)
    counts = _class_counts(align_to_n, MOTION_CLASSES)
    for mc, count in zip(MOTION_CLASSES, counts):
        if per_class > count:
            raise ValueError(
                f"--per-class {per_class} exceeds the {count} trials class {mc!r} has under "
                f"--align-to-n {align_to_n}; raise the alignment or lower per-class"
            )
        for local_idx in range(count):
            t, qs, qds, qdds = generate_trial(mc, fs=fs, rng=rng)  # always draw, to stay aligned
            if local_idx < per_class:
                yield mc, local_idx, t, qs, qds, qdds


def generate_dynamics_dataset(out, *, per_class=100, align_to_n=1000, seed=0, fs=500.0,
                              damping=DEFAULT_DAMPING, body_mass=75.0, decimate=1, verbose=False):
    """Write the labelled dynamics slice into `out`. Returns the manifest dict."""
    counts = _class_counts(align_to_n, MOTION_CLASSES)
    if per_class > min(counts):  # validate before creating anything on disk
        raise ValueError(
            f"--per-class {per_class} exceeds the {min(counts)} trials the smallest class has "
            f"under --align-to-n {align_to_n}; raise the alignment or lower per-class"
        )
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    dyn = human_arm_7dof_dynamics(body_mass=body_mass, damping=damping)

    per_class_stats = {mc: {"n_trials": 0, "n_samples": 0, "peak_abs_tau": 0.0,
                            "peak_power_dissipated": 0.0, "energy_dissipated_J": 0.0,
                            "min_lambda_min": float("inf")} for mc in MOTION_CLASSES}
    trials, n_done = [], 0

    for mc, idx, t, q, qd, qdd in _aligned_trials(per_class, align_to_n, seed, fs):
        if decimate > 1:
            t, q, qd, qdd = t[::decimate], q[::decimate], qd[::decimate], qdd[::decimate]
        lab = dyn.label_trajectory(q, qd, qdd)

        dt = float(t[1] - t[0])
        e_diss = float(np.trapezoid(lab["power_dissipated"], dx=dt))
        name = f"{mc}_{idx:04d}.npz"
        np.savez_compressed(
            out / name,
            t=t.astype(np.float32), q=q.astype(np.float32), qd=qd.astype(np.float32),
            qdd=qdd.astype(np.float32),
            **{k: v.astype(np.float32) for k, v in lab.items()},
        )

        s = per_class_stats[mc]
        s["n_trials"] += 1
        s["n_samples"] += int(t.size)
        s["peak_abs_tau"] = max(s["peak_abs_tau"], float(np.abs(lab["tau"]).max()))
        s["peak_power_dissipated"] = max(s["peak_power_dissipated"],
                                         float(lab["power_dissipated"].max()))
        s["energy_dissipated_J"] += e_diss
        s["min_lambda_min"] = min(s["min_lambda_min"], float(lab["lambda_min"].min()))

        trials.append({
            "file": name, "motion_class": mc, "trial_index": idx,
            "n_samples": int(t.size), "duration_s": round(float(t[-1] + dt), 6),
            "peak_abs_tau_Nm": round(float(np.abs(lab["tau"]).max()), 6),
            "energy_dissipated_J": round(e_diss, 6),
            "min_lambda_min": float(f"{lab['lambda_min'].min():.6g}"),
        })
        n_done += 1
        if verbose and n_done % 25 == 0:
            print(f"  ... {n_done} trials", flush=True)

    for s in per_class_stats.values():
        s["energy_dissipated_J"] = round(s["energy_dissipated_J"], 6)
        if s["n_trials"]:
            s["mean_energy_dissipated_J"] = round(s["energy_dissipated_J"] / s["n_trials"], 6)

    manifest = {
        "format_version": FORMAT_VERSION,
        "purpose": (
            "Dynamics ground truth for tasks 5.8/5.9. SkillData v1 is kinematic and has no "
            "dissipation in its generating process, so a pHNN's learned damping could not be "
            "scored against anything. Here the damping matrix B is known and recorded below, so "
            "5.9 becomes 'the network recovered B to within X%'."
        ),
        "generator": {
            "module": "skilldata.generate_dynamics",
            "seed": seed,
            "sample_rate_hz": fs / decimate,
            "native_sample_rate_hz": fs,
            "decimate": decimate,
            "per_class": per_class,
            "align_to_n": align_to_n,
            "arm": "human_arm_7dof",
            "body_mass_kg": body_mass,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
            "subject_id": SUBJECT_ID,
        },
        # ---- the reason this dataset exists ---------------------------------
        "true_damping": {
            "B_diagonal_Nms_per_rad": list(map(float, np.diag(dyn.B))),
            "joint_order": ["sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
                            "forearm_pron", "wrist_flex", "wrist_dev_hand"],
            "model": "viscous joint damping, tau_d = -B qdot, B diagonal and constant in time",
            "note": (
                "This is the quantity tasks 5.8/5.9 must recover. It is CONSTANT — the true "
                "dissipation rate qdot^T B qdot varies over a trial only because qdot does, "
                "peaking in the active phase and vanishing at the endpoints where the arm is at "
                "rest. Task 5.9 predicts learned dissipation concentrated in prep/settle; against "
                "this data that prediction is falsifiable, and note that the naive expectation "
                "runs the other way."
            ),
        },
        "anthropometry": {
            "source": ("de Leva, P. (1996), J. Biomech. 29(9):1223-1230, "
                       "doi:10.1016/0021-9290(95)00178-6 — adjusted Zatsiorsky-Seluyanov "
                       "parameters, male column"),
            "relative_mass_com_gyration": {k: list(v[:2]) + [list(v[2])]
                                           for k, v in DE_LEVA_MALE.items()},
            "total_limb_mass_kg": round(float(sum(b.mass for b in dyn.bodies)), 6),
        },
        "fields": {
            "t": "(T,) time [s]",
            "q/qd/qdd": "(T,7) joint angle / velocity / acceleration [rad, rad/s, rad/s^2]",
            "p": "(T,7) generalized momentum M(q) qdot [kg m^2/s]",
            "tau": "(T,7) inverse-dynamics torque M qddot + C qdot + g + B qdot [N m]",
            "H/T_kin/U": "(T,) total / kinetic / potential energy [J]",
            "power_dissipated": "(T,) qdot^T B qdot, the TRUE dissipation rate [W]",
            "lambda_min": "(T,) smallest eigenvalue of M(q) [kg m^2]",
        },
        "conditioning_warning": (
            "human_arm_7dof has two gimbal-lock configurations where M loses rank: the wrist at "
            "q[5]=pi/2, and the shoulder at q[1]=pi/2 — which is the arm's own hanging rest pose. "
            "H = 1/2 p^T M^-1 p is ill-conditioned near either. Filter or weight on the "
            "lambda_min column rather than meeting this as a training instability."
        ),
        "index_alignment": (
            f"Trials replay generate_synthetic's RNG stream at seed {seed} with "
            f"align_to_n={align_to_n}, keeping the first {per_class} of each class. A trial "
            f"(motion_class, trial_index) here is the SAME motion as the SkillData v1 record with "
            f"the same pair, so the datasets join on it. This holds only for these exact seed and "
            f"align_to_n values."
        ),
        "schema_note": (
            "SkillData v1 is NOT modified. Its schema_version const 'skilldata-v1' is a locked "
            "contract; folding these fields into it is a v2 decision, routed to PROPOSALS.md "
            "(2026-08-14), not taken here."
        ),
        "n_trials": len(trials),
        "motion_classes": list(MOTION_CLASSES),
        "per_class": per_class_stats,
        "trials": trials,
    }
    with (out / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _print_summary(manifest):
    g = manifest["generator"]
    print(f"dynamics ground-truth slice -> {manifest['n_trials']} trials "
          f"({g['per_class']}/class) @ {g['sample_rate_hz']:.0f} Hz, seed {g['seed']}")
    print(f"  true B (diag) = {[round(b, 4) for b in manifest['true_damping']['B_diagonal_Nms_per_rad']]}")
    print(f"  limb mass     = {manifest['anthropometry']['total_limb_mass_kg']} kg (de Leva 1996 male)")
    print(f"  {'class':14s} {'trials':>6s} {'peak|tau|':>10s} {'peak P_diss':>12s} "
          f"{'mean E_diss':>12s} {'min lam(M)':>11s}")
    for mc in manifest["motion_classes"]:
        s = manifest["per_class"][mc]
        if not s["n_trials"]:
            continue
        print(f"  {mc:14s} {s['n_trials']:6d} {s['peak_abs_tau']:9.3f}N {s['peak_power_dissipated']:11.3f}W "
              f"{s.get('mean_energy_dissipated_J', 0.0):11.4f}J {s['min_lambda_min']:11.2e}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Generate the dynamics ground-truth slice (task 3.14). Sidecar to SkillData v1.")
    ap.add_argument("--out", default="data/processed/dynamics", help="output directory")
    ap.add_argument("--per-class", type=int, default=100, help="trials kept per motion class")
    ap.add_argument("--align-to-n", type=int, default=1000,
                    help="base-trial count to align indices with (must match the v1 dataset)")
    ap.add_argument("--seed", type=int, default=0, help="master RNG seed (must match the v1 dataset)")
    ap.add_argument("--fs", type=float, default=500.0, help="native sample rate (Hz)")
    ap.add_argument("--decimate", type=int, default=1, help="keep every Nth sample (1 = full rate)")
    ap.add_argument("--body-mass", type=float, default=75.0, help="subject body mass (kg)")
    ap.add_argument("--damping", type=float, nargs=7, default=list(DEFAULT_DAMPING),
                    help="the 7 diagonal entries of B [N m s/rad] — recorded in the manifest")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    manifest = generate_dynamics_dataset(
        a.out, per_class=a.per_class, align_to_n=a.align_to_n, seed=a.seed, fs=a.fs,
        damping=tuple(a.damping), body_mass=a.body_mass, decimate=a.decimate, verbose=a.verbose)
    _print_summary(manifest)
    return manifest


if __name__ == "__main__":
    main()
