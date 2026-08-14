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

from analysis.identifiability import damping_error_bars
from sim.dynamics import DE_LEVA_MALE, human_arm_7dof_dynamics
from sim.limits import JOINT_ORDER, feasibility, limits_manifest
from sim.motions import ALL_CLASSES, EXCITATION_CLASS, MOTION_CLASSES, generate_trial

from .generate_synthetic import SUBJECT_ID, _class_counts

FORMAT_VERSION = "skilldata-dynamics-v1"

# Per-joint viscous damping [N m s / rad] for the 7-DOF arm, proximal -> distal. Larger at the
# shoulder than the wrist, which is the right ordering physically, but the magnitudes are a
# deliberate, uncalibrated choice: the point of this dataset is that B is *known*, not that it is
# the physiological value. Whatever is here is what 5.9 is scored against, so it goes in the
# manifest.
DEFAULT_DAMPING = (0.080, 0.080, 0.050, 0.050, 0.020, 0.020, 0.020)

# Excitation trials are drawn from their own RNG stream so their count can change without
# perturbing the naturalistic draw order that SkillData v1 alignment depends on.
EXCITE_SEED_OFFSET = 10_000
# How many samples feed the identifiability block written into the manifest.
# The cap is applied **per class**, not to a single pooled bucket. Pooling was the first attempt and
# it was wrong: trials are generated in class order, so a pooled cap of 320 filled entirely from the
# first 40 `reach` trials and the block labelled "naturalistic_classes" was silently measuring
# `reach` alone (rank 18/22 rather than 22/22). Caught 2026-08-14 by the rank not matching the depth
# pass. The regressor is 22 columns wide, so 80 samples x 7 rows per class is ample.
ID_SAMPLES_PER_TRIAL = 8
ID_SAMPLES_PER_CLASS = 80
# The excitation block gets the same *total* budget as the pooled naturalistic block (4 x 80), so
# the two condition numbers and error bars are computed from the same number of samples and the
# comparison between them is apples to apples. Judging excite on a quarter of the samples made it
# look four times worse than it is — error bars scale as 1/sqrt(N).
ID_SAMPLE_BUDGET = {"__excite__": ID_SAMPLES_PER_CLASS * 4}


def _id_budget(motion_class):
    return ID_SAMPLE_BUDGET.get("__excite__" if motion_class == EXCITATION_CLASS else "",
                                ID_SAMPLES_PER_CLASS)


def _as_stored(*arrays):
    """Round trajectory samples through float32, the precision this dataset is written at.

    The manifest must describe **the dataset as shipped**, not the generator's internal state, and
    at this conditioning the difference is measurable. Found 2026-08-14 (task 3.16): the same 320
    naturalistic samples give cond(Y) = 2.7128e11 from float64 memory and 2.5329e11 after a float32
    round trip -- a 6.6% gap, because the smallest singular value (2.09e-8 vs 2.24e-8) sits exactly
    where float32 storage noise lives.

    That is why the shipped 3.15 manifest said 2.53e11 while a fresh in-line run said 2.71e11: the
    3.15 block had been refreshed through `--identifiability-only`, which reloads from disk, and the
    two code paths silently disagreed about the same dataset. Nothing else moved -- the damping
    error bars agree to 1.4e-8 relative -- so this changes no conclusion, but a number that depends
    on which function computed it is a number nobody can reproduce.

    Casting here makes the in-line path agree with the reload path, and picks the honest one: a
    consumer of this dataset reads float32, so float32 is the conditioning they actually face.
    """
    return tuple(a.astype(np.float32).astype(float) for a in arrays)


def _feasible_block(dyn, id_feasible, sigma):
    """Identifiability of `B` from **only the trials a human could perform** (task 3.16).

    This is the number Phase 5 actually has to live with. The 3.16 gate says a training run may not
    score a recovered `B` on infeasible trials, and two of the four naturalistic classes are
    infeasible -- so the set Phase 5 may train on is not the set the other blocks describe. Without
    this, Phase 5 would plan against `combined` and then discover the real conditioning after the
    fact.

    Returns ``None`` when nothing is feasible, which would itself be worth knowing.
    """
    samples = [smp for mc in ALL_CLASSES for smp in id_feasible[mc]]
    if not samples:
        return None
    blk = damping_error_bars(dyn, samples, sigma_abs=sigma)
    blk["what_this_is"] = (
        "Damping identifiability using ONLY trials whose human_feasible is true -- the subset the "
        "Phase-5 gate allows. Compare against 'combined': the difference is what excluding "
        "throw and wrist_rotate costs."
    )
    blk["classes_contributing"] = {mc: len(id_feasible[mc]) for mc in ALL_CLASSES
                                   if id_feasible[mc]}
    return blk


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


def _excitation_trials(n_trials, seed, fs):
    """Yield `(class, index, t, q, qd, qdd)` for the excitation class (task 3.15).

    Uses a **separate RNG stream** (`seed + EXCITE_SEED_OFFSET`) rather than the aligned one, so
    adding, removing or changing the number of excitation trials cannot perturb the naturalistic
    trials' draw order and therefore cannot break index alignment with SkillData v1.
    """
    rng = np.random.default_rng(seed + EXCITE_SEED_OFFSET)
    for idx in range(n_trials):
        t, q, qd, qdd = generate_trial(EXCITATION_CLASS, fs=fs, rng=rng)
        yield EXCITATION_CLASS, idx, t, q, qd, qdd


def generate_dynamics_dataset(out, *, per_class=100, align_to_n=1000, seed=0, fs=500.0,
                              damping=DEFAULT_DAMPING, body_mass=75.0, decimate=1,
                              excite_trials=100, identifiability=True, verbose=False):
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
                            "min_lambda_min": float("inf"), "n_human_feasible": 0,
                            "worst_rom_excess_deg": 0.0,
                            "rom_violating_joints": {}} for mc in ALL_CLASSES}
    trials, n_done = [], 0
    id_samples = {mc: [] for mc in ALL_CLASSES}
    # Samples drawn only from trials a human could perform. This is the set Phase 5 is actually
    # allowed to train on under the 3.16 gate, so the manifest has to say how identifiable B is
    # from it -- otherwise Phase 5 discovers the answer the hard way, after the fact.
    id_feasible = {mc: [] for mc in ALL_CLASSES}

    from itertools import chain
    sources = chain(_aligned_trials(per_class, align_to_n, seed, fs),
                    _excitation_trials(excite_trials, seed, fs))
    for mc, idx, t, q, qd, qdd in sources:
        if decimate > 1:
            t, q, qd, qdd = t[::decimate], q[::decimate], qd[::decimate], qdd[::decimate]
        lab = dyn.label_trajectory(q, qd, qdd)
        feas = feasibility(q)                       # task 3.16 — before sampling, which reads it
        excess_deg = np.rad2deg(feas["excess_rad"])

        want_all = len(id_samples[mc]) < _id_budget(mc)
        want_feasible = feas["human_feasible"] and len(id_feasible[mc]) < _id_budget(mc)
        if identifiability and (want_all or want_feasible):
            # a handful of samples per trial is plenty; the regressor is 22 columns wide.
            # Sampled through float32 ON PURPOSE — see `_as_stored`.
            lo, hi = int(0.15 * t.size), int(0.9 * t.size)
            for k in np.linspace(lo, hi - 1, ID_SAMPLES_PER_TRIAL, dtype=int):
                smp = _as_stored(q[k], qd[k], qdd[k])
                if want_all:
                    id_samples[mc].append(smp)
                if want_feasible:
                    id_feasible[mc].append(smp)

        dt = float(t[1] - t[0])
        e_diss = float(np.trapezoid(lab["power_dissipated"], dx=dt))
        name = f"{mc}_{idx:04d}.npz"
        np.savez_compressed(
            out / name,
            t=t.astype(np.float32), q=q.astype(np.float32), qd=qd.astype(np.float32),
            qdd=qdd.astype(np.float32),
            human_feasible=np.asarray(feas["human_feasible"]),
            rom_excess_deg=excess_deg.astype(np.float32),
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
        s["n_human_feasible"] += int(feas["human_feasible"])
        s["worst_rom_excess_deg"] = max(s["worst_rom_excess_deg"], float(excess_deg.max()))
        for j, jn in enumerate(JOINT_ORDER):
            if excess_deg[j] > 0:
                v = s["rom_violating_joints"].setdefault(jn, {"n_trials": 0, "worst_excess_deg": 0.0})
                v["n_trials"] += 1
                v["worst_excess_deg"] = max(v["worst_excess_deg"], round(float(excess_deg[j]), 3))

        trials.append({
            "file": name, "motion_class": mc, "trial_index": idx,
            "n_samples": int(t.size), "duration_s": round(float(t[-1] + dt), 6),
            "peak_abs_tau_Nm": round(float(np.abs(lab["tau"]).max()), 6),
            "energy_dissipated_J": round(e_diss, 6),
            "min_lambda_min": float(f"{lab['lambda_min'].min():.6g}"),
            "human_feasible": feas["human_feasible"],
            "worst_rom_excess_deg": round(float(excess_deg.max()), 3),
            "worst_rom_joint": feas["worst_joint"],
        })
        n_done += 1
        if verbose and n_done % 25 == 0:
            print(f"  ... {n_done} trials", flush=True)

    for s in per_class_stats.values():
        s["energy_dissipated_J"] = round(s["energy_dissipated_J"], 6)
        if s["n_trials"]:
            s["mean_energy_dissipated_J"] = round(s["energy_dissipated_J"] / s["n_trials"], 6)

    id_block = None
    if identifiability:
        if verbose:
            print("  computing identifiability block...", flush=True)
        _nat = [smp for mc in MOTION_CLASSES for smp in id_samples[mc]]
        # one noise level for all three blocks, so they are comparable
        _sigma = float(f"{0.01 * _rms_torque(dyn, _nat):.6g}")
        id_block = {
            "what_this_is": (
                "Which of this dataset's own parameters are actually recoverable from it. A "
                "dataset that carries a 'true B' without saying whether B is measurable from its "
                "own trials is a trap: tasks 5.8/5.9 would report a recovered value for joints "
                "the data contains no information about. Added by task 3.15."
            ),
            "naturalistic_classes": damping_error_bars(dyn, _nat, sigma_abs=_sigma),
            EXCITATION_CLASS: (damping_error_bars(dyn, id_samples[EXCITATION_CLASS],
                                                  sigma_abs=_sigma)
                               if id_samples[EXCITATION_CLASS] else None),
            "combined": damping_error_bars(
                dyn, _nat + id_samples[EXCITATION_CLASS], sigma_abs=_sigma),
            "feasible_only": _feasible_block(dyn, id_feasible, _sigma),
            "shared_torque_noise_Nm": _sigma,
            "sampled_per_class": {mc: len(id_samples[mc]) for mc in ALL_CLASSES},
            "sampled_per_class_feasible": {mc: len(id_feasible[mc]) for mc in ALL_CLASSES},
            "precision_note": (
                "Computed from trajectories rounded to float32, the precision this dataset is "
                "stored at, so the block describes the data as shipped rather than the generator's "
                "float64 internals. This matters only for condition_number: the same naturalistic "
                "samples give 2.7128e11 in float64 and 2.5329e11 through float32, a 6.6% gap, "
                "because the smallest singular value sits where float32 noise lives. The error "
                "bars are unaffected (agreement 1.4e-8 relative). Quote the error bars; treat "
                "condition_number as an order of magnitude. Fixed in task 3.16 -- before that the "
                "in-line and --identifiability-only paths disagreed on the same dataset."
            ),
            "reading": (
                "relative_error_bar is std(b_hat_j)/b_true_j at the stated torque noise. Below "
                "~0.2 the joint's damping is measurable; at 1.0 the error bar equals the value; "
                "above 1.0 the estimate is noise. Filter Phase-5 claims on this. "
                "Use the 'combined' block: the excitation class rescues the two joints the "
                "naturalistic classes cannot see at all, but wrist_rotate and throw drive the "
                "distal joints harder than a balanced excitation does, so neither set dominates "
                "the other joint-by-joint and the union beats both."
            ),
        }

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
            "excite_trials": excite_trials,
            "excite_seed": seed + EXCITE_SEED_OFFSET,
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
        "identifiability": id_block,
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
        "excitation_class": {
            "name": EXCITATION_CLASS,
            "n_trials": excite_trials,
            "why": (
                "A finite-Fourier-series trajectory driving all 7 joints (task 3.15). It is an "
                "instrument rather than a naturalistic motion, and it is deliberately NOT a member "
                "of motion_classes: those four are naturalistic and their per-class statistics "
                "would be contaminated by it, and MOTION_CLASSES also drives the RNG draw order "
                "that SkillData v1 index alignment depends on. Drawn from a separate RNG stream "
                "(seed + 10000) for the same reason."
            ),
            "feasibility_correction": (
                "Task 3.15 justified this class with the sentence 'it is an instrument, not a "
                "motion anyone performs'. The depth pass tested that and found it true of the "
                "SHIPPED trajectory for an unintended reason: q0 defaulted to zero, which is the "
                "end of the elbow's travel, so the oscillation hyperextended the elbow to -66.2 "
                "deg against a -2 deg limit in 10 of 10 trials. Task 3.16 centres each joint in "
                "sim.limits.usable_interval instead. The class is now anatomically performable, "
                "which matters because a wearable has to be able to record it. What remains true "
                "is only that nobody performs it SPONTANEOUSLY -- it is a protocol you would ask "
                "a subject to follow, not a movement they would make on their own."
            ),
            "caveat": (
                "Unlike the four naturalistic classes this trajectory does NOT start and end at "
                "rest, so momentum and kinetic energy are non-zero at both endpoints. Anything "
                "asserting rest endpoints must exclude it."
            ),
        },
        "human_feasibility": {
            **limits_manifest(),
            "n_trials_feasible": sum(1 for tr in trials if tr["human_feasible"]),
            "n_trials": len(trials),
            "per_class": {
                mc: {"n_trials": per_class_stats[mc]["n_trials"],
                     "n_human_feasible": per_class_stats[mc]["n_human_feasible"],
                     "worst_rom_excess_deg": round(per_class_stats[mc]["worst_rom_excess_deg"], 3),
                     "rom_violating_joints": per_class_stats[mc]["rom_violating_joints"]}
                for mc in ALL_CLASSES
            },
            "gate": (
                "This is the Phase-5 gate agreed on 2026-08-14: no pHNN training run scores a "
                "recovered B on trials whose human_feasible is false. A network that recovers "
                "damping from motions nobody can perform has demonstrated something about this "
                "simulator's conditioning, not about human movement, and needs no wearable."
            ),
            "known_infeasible_classes": (
                "throw and wrist_rotate fail and are NOT fixed here. Their amplitude ranges live "
                "in sim.motions._SPECS, and changing _SPECS changes the RNG draw order, which "
                "breaks this dataset's index alignment with SkillData v1 and invalidates the "
                "Phase 4 fusion results measured on it. Routed to Heitor in PROPOSALS.md "
                "(2026-08-14); the standing recommendation is to flag rather than regenerate and "
                "fold the real fix into the pending SkillData v2 decision. The excite class WAS "
                "fixed, because it is generated from a separate RNG stream and breaks nothing."
            ),
        },
        "per_class": per_class_stats,
        "trials": trials,
    }
    with (out / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _rms_torque(dyn, samples):
    """RMS of the linearised torque over `samples` — the reference level for the noise sigma."""
    from analysis.identifiability import _theta0, build_Y

    theta = np.concatenate([_theta0(dyn), np.diag(dyn.B)])
    return float(np.sqrt(((build_Y(dyn, samples) @ theta) ** 2).mean()))


def recompute_identifiability(out, *, body_mass=None, damping=None):
    """Recompute the manifest's identifiability block from an already-written slice.

    The block is a property of the trajectories, not of the labelling, so it can be refreshed
    without paying for another full pass of inverse dynamics. `damping` and `body_mass` default to
    whatever the existing manifest recorded, so the recomputation describes the dataset on disk
    rather than a different arm.
    """
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    damping = tuple(manifest["true_damping"]["B_diagonal_Nms_per_rad"]) if damping is None else damping
    body_mass = manifest["generator"]["body_mass_kg"] if body_mass is None else body_mass
    dyn = human_arm_7dof_dynamics(body_mass=body_mass, damping=damping)

    buckets = {mc: [] for mc in ALL_CLASSES}
    feasible = {mc: [] for mc in ALL_CLASSES}
    for entry in manifest["trials"]:
        mc = entry["motion_class"]
        ok = entry.get("human_feasible", True)
        if len(buckets[mc]) >= _id_budget(mc) and not (ok and len(feasible[mc]) < _id_budget(mc)):
            continue
        z = np.load(out / entry["file"])
        T = z["t"].size
        lo, hi = int(0.15 * T), int(0.9 * T)
        for k in np.linspace(lo, hi - 1, ID_SAMPLES_PER_TRIAL, dtype=int):
            smp = (z["q"][k].astype(float), z["qd"][k].astype(float), z["qdd"][k].astype(float))
            if len(buckets[mc]) < _id_budget(mc):
                buckets[mc].append(smp)
            if ok and len(feasible[mc]) < _id_budget(mc):
                feasible[mc].append(smp)

    natural = [smp for mc in MOTION_CLASSES for smp in buckets[mc]]
    sigma = float(f"{0.01 * _rms_torque(dyn, natural):.6g}")
    manifest["identifiability"] = {
        "what_this_is": manifest.get("identifiability", {}).get("what_this_is") or (
            "Which of this dataset's own parameters are actually recoverable from it."),
        "naturalistic_classes": damping_error_bars(dyn, natural, sigma_abs=sigma),
        EXCITATION_CLASS: (damping_error_bars(dyn, buckets[EXCITATION_CLASS], sigma_abs=sigma)
                           if buckets[EXCITATION_CLASS] else None),
        "combined": damping_error_bars(
            dyn, natural + buckets[EXCITATION_CLASS], sigma_abs=sigma),
        "feasible_only": _feasible_block(dyn, feasible, sigma),
        "shared_torque_noise_Nm": sigma,
        "sampled_per_class": {mc: len(buckets[mc]) for mc in ALL_CLASSES},
        "sampled_per_class_feasible": {mc: len(feasible[mc]) for mc in ALL_CLASSES},
        "precision_note": manifest.get("identifiability", {}).get("precision_note", ""),
        "reading": manifest.get("identifiability", {}).get("reading") or (
            "relative_error_bar is std(b_hat_j)/b_true_j at the stated torque noise."),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def recompute_feasibility(out):
    """Re-score `human_feasible` for every trial against the CURRENT `sim.limits`, in place.

    Exists because the limits table is expected to change: the shoulder and wrist are on a tier-B
    secondary source, and replacing them with Aizawa et al. (2013) is a queued task. Re-deriving the
    flag from the stored joint angles takes seconds, where regenerating the dataset takes ~13
    minutes and would also mean re-running the identifiability block for no reason. The trajectories
    do not change — only the yardstick does.

    Rewrites the per-trial flags, the per-class summary and the `human_feasibility` block, and
    leaves everything else in the manifest untouched.
    """
    out = Path(out)
    manifest = json.loads((out / "manifest.json").read_text())
    per_class = {mc: {"n_trials": 0, "n_human_feasible": 0, "worst_rom_excess_deg": 0.0,
                      "rom_violating_joints": {}} for mc in ALL_CLASSES}

    for entry in manifest["trials"]:
        z = dict(np.load(out / entry["file"]))
        feas = feasibility(z["q"].astype(float))
        excess_deg = np.rad2deg(feas["excess_rad"])
        z["human_feasible"] = np.asarray(feas["human_feasible"])
        z["rom_excess_deg"] = excess_deg.astype(np.float32)
        np.savez_compressed(out / entry["file"], **z)

        entry["human_feasible"] = feas["human_feasible"]
        entry["worst_rom_excess_deg"] = round(float(excess_deg.max()), 3)
        entry["worst_rom_joint"] = feas["worst_joint"]

        b = per_class[entry["motion_class"]]
        b["n_trials"] += 1
        b["n_human_feasible"] += int(feas["human_feasible"])
        b["worst_rom_excess_deg"] = max(b["worst_rom_excess_deg"], float(excess_deg.max()))
        for j, jn in enumerate(JOINT_ORDER):
            if excess_deg[j] > 0:
                v = b["rom_violating_joints"].setdefault(jn, {"n_trials": 0,
                                                              "worst_excess_deg": 0.0})
                v["n_trials"] += 1
                v["worst_excess_deg"] = max(v["worst_excess_deg"], round(float(excess_deg[j]), 3))

    for b in per_class.values():
        b["worst_rom_excess_deg"] = round(b["worst_rom_excess_deg"], 3)

    old = manifest.get("human_feasibility", {})
    manifest["human_feasibility"] = {
        **limits_manifest(),
        "n_trials_feasible": sum(1 for tr in manifest["trials"] if tr["human_feasible"]),
        "n_trials": len(manifest["trials"]),
        "per_class": per_class,
        "gate": old.get("gate", ""),
        "known_infeasible_classes": old.get("known_infeasible_classes", ""),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _print_summary(manifest):
    g = manifest["generator"]
    print(f"dynamics ground-truth slice -> {manifest['n_trials']} trials "
          f"({g['per_class']}/class) @ {g['sample_rate_hz']:.0f} Hz, seed {g['seed']}")
    print(f"  true B (diag) = {[round(b, 4) for b in manifest['true_damping']['B_diagonal_Nms_per_rad']]}")
    print(f"  limb mass     = {manifest['anthropometry']['total_limb_mass_kg']} kg (de Leva 1996 male)")
    print(f"  {'class':14s} {'trials':>6s} {'peak|tau|':>10s} {'peak P_diss':>12s} "
          f"{'mean E_diss':>12s} {'min lam(M)':>11s}")
    classes = list(manifest["motion_classes"])
    if manifest.get("excitation_class"):
        classes.append(manifest["excitation_class"]["name"])
    for mc in classes:
        s = manifest["per_class"][mc]  # noqa: F841 -- consumed by the print below
        if not s["n_trials"]:
            continue
        print(f"  {mc:14s} {s['n_trials']:6d} {s['peak_abs_tau']:9.3f}N {s['peak_power_dissipated']:11.3f}W "
              f"{s.get('mean_energy_dissipated_J', 0.0):11.4f}J {s['min_lambda_min']:11.2e}")

    hf = manifest.get("human_feasibility")
    if hf:
        print(f"\n  human feasibility — {hf['n_trials_feasible']}/{hf['n_trials']} trials are "
              f"motions a person could perform")
        print(f"  scored on {hf['n_joints_scored']}/7 joints; "
              f"unsourced (not scored): {', '.join(hf['unsourced_joints'])} "
              f"-> this is a LOWER BOUND on violations")
        print(f"  {'class':14s} {'feasible':>10s} {'worst excess':>13s}  offending joints")
        for mc in classes:
            b = hf["per_class"][mc]
            if not b["n_trials"]:
                continue
            bad = ", ".join(f"{j} ({v['n_trials']}x, +{v['worst_excess_deg']:.0f} deg)"
                            for j, v in b["rom_violating_joints"].items()) or "-"
            print(f"  {mc:14s} {b['n_human_feasible']:5d}/{b['n_trials']:<4d} "
                  f"{b['worst_rom_excess_deg']:10.1f} deg  {bad}")

    idb = manifest.get("identifiability")
    if idb:
        print("\n  damping identifiability — relative error bar on B at 1% torque noise")
        print(f"  {'trajectories':16s} {'cond(Y)':>10s} {'rank':>6s} {'worst joint':>16s} {'worst':>8s}")
        for key, lbl in (("naturalistic_classes", "naturalistic"), (EXCITATION_CLASS, "excite"),
                         ("combined", "combined"), ("feasible_only", "FEASIBLE ONLY")):
            blk = idb.get(key)
            if not blk:
                continue
            print(f"  {lbl:16s} {blk['condition_number']:10.2e} "
                  f"{blk['regressor_rank']:3d}/{blk['regressor_cols']:<2d} "
                  f"{blk['worst_joint']:>16s} {blk['worst_relative_error_bar']:7.1%}")


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
    ap.add_argument("--excite-trials", type=int, default=100,
                    help="finite-Fourier excitation trials appended to the naturalistic ones (3.15)")
    ap.add_argument("--no-identifiability", action="store_true",
                    help="skip the manifest's identifiability block (it costs ~30 s)")
    ap.add_argument("--identifiability-only", action="store_true",
                    help="recompute only the manifest's identifiability block from an existing slice")
    ap.add_argument("--feasibility-only", action="store_true",
                    help="re-score human_feasible against the current sim.limits, in place (3.16). "
                         "Seconds rather than the ~13 min a full regeneration costs — use this when "
                         "a joint limit gets a better source, since the trajectories do not change")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)

    if a.feasibility_only:
        manifest = recompute_feasibility(a.out)
        _print_summary(manifest)
        return manifest
    if a.identifiability_only:
        manifest = recompute_identifiability(a.out)
        _print_summary(manifest)
        return manifest
    manifest = generate_dynamics_dataset(
        a.out, per_class=a.per_class, align_to_n=a.align_to_n, seed=a.seed, fs=a.fs,
        damping=tuple(a.damping), body_mass=a.body_mass, decimate=a.decimate,
        excite_trials=a.excite_trials, identifiability=not a.no_identifiability,
        verbose=a.verbose)
    _print_summary(manifest)
    return manifest


if __name__ == "__main__":
    main()
