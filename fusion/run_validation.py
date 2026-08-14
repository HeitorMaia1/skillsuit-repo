"""fusion.run_validation — score every orientation filter against the synthetic dataset
(task 4.5), write the numbers to JSON, and render the paper figures (feeding task 4.6).

Run::

    uv run python -m fusion.run_validation --data data/processed --figures paper/figures

(this is exactly what the Makefile's ``fusion`` target invokes), or ``--limit-per-class 10``
for a fast pass. Results land in ``<figures>/fusion_validation.json`` and the figures in
``<figures>/fusion_*.pdf``; ``notebooks/01_fusion_validation.ipynb`` (task 4.6) reads that JSON
rather than recomputing, so the notebook executes in seconds instead of minutes.

How an orientation estimate is scored
=====================================

Every **noisy** record is run through each filter, per sensor (S2/S4/S5), and compared
sample-by-sample against the *analytic* ground-truth orientation — rebuilt from that record's
own ``segment_kinematics.joint_angles_rad`` via ``sim.arm3d.Arm3D.state()``, which is the same
closed-form rotation the simulator used to generate the IMU readings in the first place.
Ground truth is never "a filter run on the clean stream": that would grade a filter against
itself, and the clean records store gyro/accel streams rather than orientations anyway.
**Two** error metrics are reported for every run, and the difference between them carries most
of the interpretation:

``total``   the angle of the relative rotation between estimate and truth
            (``quat_angle_error_deg``) — representation-independent, no Euler-angle wrapping
            and no quaternion double-cover ambiguity. This is the number work10/work11 quote.
``tilt``    the angle between the *predicted* and *true* gravity directions. This is the part
            of the orientation a gyro+accel pair can actually observe; the remainder is
            heading, which neither filter can measure (the reference ICM-42688-P, decision D2,
            has no magnetometer) and which is therefore free to drift.

Carrying both is not decoration. A filter can have near-perfect tilt and a large ``total``,
which means its error is entirely in the unobservable channel — a completely different fault
from one that cannot track gravity, and the two call for opposite fixes.

What is compared, and why the control columns are not optional
==============================================================

Task 4.5 asks for "both filters". Reporting only those two would have hidden the actual
finding of task 4.3 (see ``WORK/work11.md``), so three controls run alongside them:

``madgwick``      the task 4.1 filter, ``beta=0.1``. The baseline.
``ekf``           the task 4.3 filter with its shipped defaults.
``gyro_only``     Madgwick with ``beta=0`` — pure gyro integration, no accelerometer at all.
                  **This is the column that changes the interpretation.** It isolates how much
                  of each filter's error is drift the accelerometer should have removed, versus
                  error the accelerometer *introduced* (it measures gravity plus the sensor's
                  own linear acceleration, and during motion the second term is large).
``ekf_no_bias``   the EKF with its gyro-bias state disabled. Isolates what the bias state —
                  the entire justification for building an EKF — is actually contributing.
``ekf_static_r``  the EKF with the dynamic-acceleration term removed from its measurement
                  noise. Isolates the mechanism that turns out to be carrying the result.

Each is also run under two initialisations, because the choice is load-bearing and work10/11
only used the first:

``truth``   seeded with the exact true initial orientation. Comparable with work10/work11, but
            generous to any gyro-leaning configuration — it hands the filter for free the one
            thing an accelerometer is unambiguously good at.
``accel``   seeded from the first accelerometer sample: tilt taken from the measured gravity
            direction, heading left at zero (gravity carries no heading information, so no
            honest initialiser can do better without a magnetometer). This is what a real
            capture session gets. **Read this column's ``tilt`` metric, not its ``total``**:
            several motion classes start with a non-zero heading (``reach`` opens with the
            elbow at 0.2 rad about the vertical, ``wrist_rotate`` at 0.6 rad), so ``total``
            here is dominated by a constant, permanent, and strictly unknowable heading offset
            that is identical for every filter and says nothing about any of them.

Beyond the by-class RMS the task asks for, three extras that Phase 4 needs before it can be
called closed: a **per-phase** breakdown (``prep``/``active``/``settle``, from each record's own
``phase_labels``) since the dynamic-acceleration story predicts the damage is concentrated in
``active``; a **long-horizon** run (a real trial followed by a stationary hold), since every
dataset trial is 1.2-2.5 s and gyro bias needs longer than that to matter; and a **noise
sensitivity** sweep that re-corrupts clean records at scaled sensor-noise levels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from sim.arm3d import human_arm_7dof
from sim.sensor import SensorModel

from .ekf import OrientationEKF
from .madgwick import MadgwickFilter, predicted_gravity_direction, quat_angle_error_deg

# Sensor -> the segment whose frame orientation the sensor rides (sim.arm3d.human_arm_7dof).
SENSOR_SEGMENT = {"S2": "sh_roll_upper", "S4": "elbow_fore", "S5": "wrist_dev_hand"}
JOINT_ORDER = ("sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
               "forearm_pron", "wrist_flex", "wrist_dev_hand")
CLASSES = ("reach", "lift", "wrist_rotate", "throw")
PHASES = ("prep", "active", "settle")

# The D2 sensor-model parameters the dataset was generated with (task 3.9). Kept here so the
# noise-sensitivity sweep can scale *the same* model rather than an approximation of it.
D2_NOISE = dict(gyro_noise_dps=0.1, accel_noise_g=0.01,
                initial_gyro_bias_dps=0.5, gyro_bias_drift_dps_per_sqrt_s=0.02)

FILTERS = {
    "madgwick": ("Madgwick (β=0.1)", lambda fs: MadgwickFilter(sample_rate_hz=fs, beta=0.1)),
    "ekf": ("EKF (bias state)", lambda fs: OrientationEKF(sample_rate_hz=fs)),
    "gyro_only": ("gyro-only control", lambda fs: MadgwickFilter(sample_rate_hz=fs, beta=0.0)),
    "ekf_no_bias": ("EKF, bias state off",
                    lambda fs: OrientationEKF(sample_rate_hz=fs, p0_bias_dps=0.0,
                                              bias_rw_dps_sqrt_s=0.0)),
    "ekf_static_r": ("EKF, static R",
                     lambda fs: OrientationEKF(sample_rate_hz=fs, accel_dynamic_gain=0.0)),
}
MAIN_TWO = ("madgwick", "ekf")

_ARM = human_arm_7dof()


# --------------------------------------------------------------------------- #
# Ground truth and initialisation
# --------------------------------------------------------------------------- #
def rotmat_to_quat_wxyz(c_mat):
    """``(..., 3, 3)`` rotation matrices -> ``(..., 4)`` quaternions in ``(w, x, y, z)`` order.

    scipy returns ``(x, y, z, w)``; this reorders to the convention both filters use. Generic
    linear-algebra plumbing, not a reimplementation of anything under test.
    """
    xyzw = Rotation.from_matrix(c_mat).as_quat()
    return np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)


def ground_truth(rec):
    """Per-sensor ``(T, 4)`` true orientation quaternions for one record.

    Rebuilt analytically from the record's own joint angles, not from any filter. Velocities
    and accelerations are irrelevant to ``frames`` so they are passed as zeros.
    """
    ja = rec["segment_kinematics"]["joint_angles_rad"]
    qs = np.column_stack([np.asarray(ja[j], float) for j in JOINT_ORDER])
    zeros = np.zeros_like(qs)
    frames = _ARM.state(qs, zeros, zeros)["frames"]
    return {sid: rotmat_to_quat_wxyz(frames[seg]) for sid, seg in SENSOR_SEGMENT.items()}


def q0_from_accel(accel_sample):
    """Initial orientation from one accelerometer reading: tilt from gravity, heading zero.

    The minimal rotation carrying the measured body-frame "up" direction onto the earth
    ``(0,0,1)``. Heading is left at zero and cannot be otherwise: a gyro+accel pair carries no
    absolute heading information, so any initialiser that claimed one would be inventing it.
    """
    a = np.asarray(accel_sample, float)
    n = np.linalg.norm(a)
    if n < 1e-9:
        return (1.0, 0.0, 0.0, 0.0)
    a = a / n
    up = np.array([0.0, 0.0, 1.0])
    axis = np.cross(a, up)
    s, c = np.linalg.norm(axis), float(np.dot(a, up))
    if s < 1e-9:                       # already aligned, or antiparallel
        return (1.0, 0.0, 0.0, 0.0) if c > 0 else (0.0, 1.0, 0.0, 0.0)
    axis = axis / s
    half = np.arctan2(s, c) / 2.0
    return tuple(np.array([np.cos(half), *(axis * np.sin(half))]))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def gravity_dirs(q):
    """Batched ``predicted_gravity_direction``: ``(T, 4)`` quaternions -> ``(T, 3)`` directions.

    Same closed form as ``fusion.madgwick.predicted_gravity_direction`` (whose docstring
    carries the derivation), vectorised because this runs over millions of samples.
    """
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2 * (x * z - w * y), 2 * (w * x + y * z),
                     w * w - x * x - y * y + z * z], axis=-1)


def tilt_error_deg(q_est, q_true):
    """Angle between the estimated and true gravity directions, in degrees.

    The observable part of the orientation error for a gyro+accel sensor pair. Whatever is in
    ``quat_angle_error_deg`` but not in here is heading, which nothing in this pipeline can
    measure.

    Computed as ``atan2(|a x b|, a . b)`` rather than ``arccos(a . b)``. The two agree
    mathematically, but ``arccos`` is ill-conditioned exactly where most of these samples live:
    near zero error the dot product is ``1 - eps``, and float rounding on ``eps`` becomes an
    error of ``sqrt(eps)`` in the angle — a floor of ~1e-6 degrees that a converged filter
    would sit on. The ``atan2`` form is well-conditioned at both ends and resolves small angles
    to full precision, which matters because "how close to zero does tilt actually get" is a
    question this module is asked.
    """
    a = gravity_dirs(q_est)
    b = gravity_dirs(q_true)
    return np.degrees(np.arctan2(np.linalg.norm(np.cross(a, b), axis=-1),
                                 np.sum(a * b, axis=-1)))


class _Acc:
    """Running accumulator for both metrics; RMS without holding 2.9 M errors in memory."""

    __slots__ = ("sq", "abs", "n", "max", "sq_t", "abs_t", "max_t")

    def __init__(self):
        self.sq = self.abs = self.max = 0.0
        self.sq_t = self.abs_t = self.max_t = 0.0
        self.n = 0

    def add(self, err, tilt):
        self.sq += float(np.sum(err**2))
        self.abs += float(np.sum(err))
        self.sq_t += float(np.sum(tilt**2))
        self.abs_t += float(np.sum(tilt))
        self.n += int(err.size)
        if err.size:
            self.max = max(self.max, float(err.max()))
            self.max_t = max(self.max_t, float(tilt.max()))

    def summary(self):
        if not self.n:
            return None
        return {"rms": (self.sq / self.n) ** 0.5, "mean": self.abs / self.n, "max": self.max,
                "tilt_rms": (self.sq_t / self.n) ** 0.5, "tilt_mean": self.abs_t / self.n,
                "tilt_max": self.max_t, "n": self.n}


def evaluate(files, filter_key, *, init="truth", progress=None):
    """Score one filter over ``files`` (noisy records). Returns a nested result dict.

    ``init`` is ``"truth"`` (seed with the exact initial orientation) or ``"accel"`` (seed from
    the first accelerometer sample). Aggregates by class, by class x phase, by class x sensor,
    and keeps one RMS per trial per class so the notebook can draw distributions rather than
    just means.
    """
    label, make = FILTERS[filter_key]
    by_class = {c: _Acc() for c in CLASSES}
    by_phase = {c: {p: _Acc() for p in PHASES} for c in CLASSES}
    by_sensor = {c: {s: _Acc() for s in SENSOR_SEGMENT} for c in CLASSES}
    trial_rms = {c: [] for c in CLASSES}
    trial_tilt_rms = {c: [] for c in CLASSES}
    t0 = time.time()

    for i, path in enumerate(files):
        rec = json.loads(path.read_text())
        mc = rec["session"]["motion_class"]
        fs = rec["session"]["sample_rate_hz"]
        truth = ground_truth(rec)
        phase = np.asarray(rec["phase_labels"])
        trial_sq = trial_sq_t = 0.0
        trial_n = 0

        for sid, stream in rec["imu_streams"].items():
            gyro = np.asarray(stream["angular_velocity_dps"], float)
            accel = np.asarray(stream["linear_accel_g"], float)
            q0 = tuple(truth[sid][0]) if init == "truth" else q0_from_accel(accel[0])
            est = make(fs).run(gyro, accel, q0=q0)
            err = quat_angle_error_deg(est, truth[sid])
            tilt = tilt_error_deg(est, truth[sid])

            by_class[mc].add(err, tilt)
            by_sensor[mc][sid].add(err, tilt)
            for p in PHASES:
                sel = phase == p
                if sel.any():
                    by_phase[mc][p].add(err[sel], tilt[sel])
            trial_sq += float(np.sum(err**2))
            trial_sq_t += float(np.sum(tilt**2))
            trial_n += err.size

        if trial_n:
            trial_rms[mc].append((trial_sq / trial_n) ** 0.5)
            trial_tilt_rms[mc].append((trial_sq_t / trial_n) ** 0.5)
        if progress and (i + 1) % progress == 0:
            print(f"    {label} [{init}]: {i + 1}/{len(files)}", flush=True)

    total = _Acc()
    for attr in ("sq", "abs", "sq_t", "abs_t"):
        setattr(total, attr, sum(getattr(a, attr) for a in by_class.values()))
    total.n = sum(a.n for a in by_class.values())
    total.max = max((a.max for a in by_class.values()), default=0.0)
    total.max_t = max((a.max_t for a in by_class.values()), default=0.0)

    return {
        "label": label, "init": init, "seconds": round(time.time() - t0, 1),
        "overall": total.summary(),
        "by_class": {c: a.summary() for c, a in by_class.items() if a.n},
        "by_phase": {c: {p: a.summary() for p, a in d.items() if a.n}
                     for c, d in by_phase.items()},
        "by_sensor": {c: {s: a.summary() for s, a in d.items() if a.n}
                      for c, d in by_sensor.items()},
        "trial_rms": {c: v for c, v in trial_rms.items() if v},
        "trial_tilt_rms": {c: v for c, v in trial_tilt_rms.items() if v},
    }


# --------------------------------------------------------------------------- #
# Convergence traces (one representative trial per class, for the notebook)
# --------------------------------------------------------------------------- #
def convergence_traces(files, *, sensor="S4", stride=5):
    """Per-sample error vs time for the first trial of each class, for the main two filters.

    Subsampled by ``stride`` purely to keep the JSON small; the RMS numbers elsewhere are
    computed on every sample.
    """
    out = {}
    seen = set()
    for path in files:
        rec = json.loads(path.read_text())
        mc = rec["session"]["motion_class"]
        if mc in seen:
            continue
        seen.add(mc)
        fs = rec["session"]["sample_rate_hz"]
        truth = ground_truth(rec)[sensor]
        stream = rec["imu_streams"][sensor]
        gyro = np.asarray(stream["angular_velocity_dps"], float)
        accel = np.asarray(stream["linear_accel_g"], float)
        t = np.arange(rec["session"]["n_samples"]) / fs
        entry = {"t": t[::stride].tolist(), "sensor": sensor,
                 "phase": np.asarray(rec["phase_labels"])[::stride].tolist()}
        for key in MAIN_TWO:
            est = FILTERS[key][1](fs).run(gyro, accel, q0=tuple(truth[0]))
            entry[key] = quat_angle_error_deg(est, truth)[::stride].tolist()
            entry[key + "_tilt"] = tilt_error_deg(est, truth)[::stride].tolist()
        out[mc] = entry
        if len(seen) == len(CLASSES):
            break
    return out


# --------------------------------------------------------------------------- #
# Long horizon: a real trial, then a stationary hold
# --------------------------------------------------------------------------- #
def long_horizon(path, *, hold_s=60.0, sensor="S4", seed=7, stride=50):
    """Run a real trial, then keep the filters running through ``hold_s`` of stationary data.

    The hold is generated by the *same* ``SensorModel`` the dataset used, holding the trial's
    final orientation — so the gyro keeps reporting its 0.5 deg/s turn-on bias plus the
    random walk, and the accelerometer keeps reporting the (now constant) gravity direction.
    This is the regime the dataset's 1.2-2.5 s trials cannot probe: bias needs time to matter.
    """
    rec = json.loads(path.read_text())
    fs = rec["session"]["sample_rate_hz"]
    truth = ground_truth(rec)[sensor]
    stream = rec["imu_streams"][sensor]
    gyro = np.asarray(stream["angular_velocity_dps"], float)
    accel = np.asarray(stream["linear_accel_g"], float)

    n_hold = int(hold_s * fs)
    g_body = predicted_gravity_direction(truth[-1])
    model = SensorModel(seed=seed, **D2_NOISE)
    gyro_h, accel_h, _ = model.corrupt_stream(
        np.zeros((n_hold, 3)), np.tile(g_body, (n_hold, 1)), 1.0 / fs)

    gyro_all = np.vstack([gyro, gyro_h])
    accel_all = np.vstack([accel, accel_h])
    truth_all = np.vstack([truth, np.tile(truth[-1], (n_hold, 1))])
    t = np.arange(truth_all.shape[0]) / fs
    n_trial = gyro.shape[0]

    out = {"motion_class": rec["session"]["motion_class"], "sensor": sensor,
           "hold_s": hold_s, "trial_end_s": n_trial / fs, "t": t[::stride].tolist(),
           "filters": {}}
    for key in ("madgwick", "gyro_only", "ekf", "ekf_no_bias"):
        est = FILTERS[key][1](fs).run(gyro_all, accel_all, q0=tuple(truth[0]))
        err = quat_angle_error_deg(est, truth_all)
        tilt = tilt_error_deg(est, truth_all)

        def at(seconds, arr=err, n_trial=n_trial, fs=fs):
            return float(arr[min(n_trial + int(seconds * fs), arr.size - 1)])

        out["filters"][key] = {
            "label": FILTERS[key][0],
            "trace": err[::stride].tolist(),
            "tilt_trace": tilt[::stride].tolist(),
            "at_trial_end": float(err[n_trial - 1]),
            "at_10s": at(10), "at_30s": at(30), "final": float(err[-1]),
            "tilt_at_trial_end": float(tilt[n_trial - 1]),
            "tilt_at_10s": at(10, tilt), "tilt_at_30s": at(30, tilt),
            "tilt_final": float(tilt[-1]),
        }
    return out


# --------------------------------------------------------------------------- #
# Noise sensitivity: re-corrupt the clean records at scaled noise levels
# --------------------------------------------------------------------------- #
def noise_sensitivity(clean_files, *, scales=(0.25, 0.5, 1.0, 2.0, 4.0), seed=11):
    """RMS vs sensor-noise level, re-corrupting clean records with a scaled D2 model.

    Every D2 parameter (gyro white noise, accel white noise, turn-on bias, bias random walk)
    is scaled by the same factor, so ``1.0`` reproduces the shipped dataset's noise level and
    the curve reads as "how does each filter degrade as the sensor gets worse".
    """
    out = {"scales": list(scales), "filters": {k: {c: [] for c in CLASSES} for k in MAIN_TWO}}
    for scale in scales:
        acc = {k: {c: _Acc() for c in CLASSES} for k in MAIN_TWO}
        for j, path in enumerate(clean_files):
            rec = json.loads(path.read_text())
            mc = rec["session"]["motion_class"]
            fs = rec["session"]["sample_rate_hz"]
            truth = ground_truth(rec)
            model = SensorModel(seed=seed + j, **{k: v * scale for k, v in D2_NOISE.items()})
            for sid, stream in rec["imu_streams"].items():
                gyro, accel, _ = model.corrupt_stream(
                    np.asarray(stream["angular_velocity_dps"], float),
                    np.asarray(stream["linear_accel_g"], float), 1.0 / fs)
                for key in MAIN_TWO:
                    est = FILTERS[key][1](fs).run(gyro, accel, q0=tuple(truth[sid][0]))
                    acc[key][mc].add(quat_angle_error_deg(est, truth[sid]),
                                     tilt_error_deg(est, truth[sid]))
        for key in MAIN_TWO:
            for c in CLASSES:
                s = acc[key][c].summary()
                out["filters"][key][c].append(s["rms"] if s else None)
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _plt():
    """matplotlib on the Agg backend — importing here keeps the module headless-importable."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


COLOR = {"madgwick": "#C44E52", "ekf": "#4C72B0", "gyro_only": "#8C8C8C",
         "ekf_no_bias": "#55A868", "ekf_static_r": "#CCB974"}


def make_figures(results, outdir):
    """Render every ``fusion_*.pdf`` the paper needs. Returns the list of paths written."""
    plt = _plt()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    runs = results["runs"]["truth"]

    def save(fig, name):
        path = outdir / name
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        written.append(path)

    # 1 — per-trial RMS distribution, Madgwick vs EKF, by motion class
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    width = 0.36
    for k, key in enumerate(MAIN_TWO):
        data = [runs[key]["trial_rms"].get(c, [0.0]) for c in CLASSES]
        pos = np.arange(len(CLASSES)) + (k - 0.5) * width
        bp = ax.boxplot(data, positions=pos, widths=width * 0.85, patch_artist=True,
                        showfliers=False, medianprops=dict(color="black"))
        for box in bp["boxes"]:
            box.set_facecolor(COLOR[key])
            box.set_alpha(0.85)
        ax.plot([], [], color=COLOR[key], lw=6, alpha=0.85, label=runs[key]["label"])
    ax.set_xticks(np.arange(len(CLASSES)))
    ax.set_xticklabels(CLASSES)
    ax.set_ylabel("per-trial RMS orientation error (°)")
    ax.set_title("Orientation error by motion class (1000 trials, 3 sensors each)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fusion_rms_by_class.pdf")

    # 2 — per-phase breakdown: where in the motion the error actually lives
    fig, ax = plt.subplots(1, len(CLASSES), figsize=(12, 3.4), sharey=True)
    x = np.arange(len(PHASES))
    for a, c in zip(ax, CLASSES):
        for k, key in enumerate(MAIN_TWO):
            vals = [runs[key]["by_phase"][c].get(p, {}).get("rms", 0.0) for p in PHASES]
            a.bar(x + (k - 0.5) * 0.38, vals, 0.36, color=COLOR[key], alpha=0.9,
                  label=runs[key]["label"] if c == CLASSES[0] else None)
        a.set_xticks(x)
        a.set_xticklabels(PHASES, rotation=20)
        a.set_title(c)
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("RMS error (°)")
    ax[0].legend(fontsize=8)
    fig.suptitle("Orientation error by motion phase — the damage is concentrated in `active`")
    save(fig, "fusion_rms_by_phase.pdf")

    # 3 — convergence traces
    conv = results["convergence"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 6.0))
    for a, c in zip(ax.ravel(), CLASSES):
        if c not in conv:
            continue
        e = conv[c]
        t = np.asarray(e["t"])
        for key in MAIN_TWO:
            a.plot(t, e[key], color=COLOR[key], lw=1.1, label=runs[key]["label"])
        ph = np.asarray(e["phase"])
        act = ph == "active"
        if act.any():
            a.axvspan(t[act].min(), t[act].max(), color="#000000", alpha=0.07)
        a.set_title(f"{c}  (sensor {e['sensor']}, active phase shaded)", fontsize=10)
        a.set_xlabel("time (s)")
        a.set_ylabel("error (°)")
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    save(fig, "fusion_convergence.pdf")

    # 4 — ablations: what is actually producing the EKF's advantage
    fig, ax = plt.subplots(figsize=(9, 4.0))
    keys = ["madgwick", "gyro_only", "ekf", "ekf_no_bias", "ekf_static_r"]
    x = np.arange(len(CLASSES))
    w = 0.16
    for k, key in enumerate(keys):
        vals = [runs[key]["by_class"][c]["rms"] for c in CLASSES]
        ax.bar(x + (k - 2) * w, vals, w * 0.9, color=COLOR[key], label=runs[key]["label"])
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES)
    ax.set_yscale("log")
    ax.set_ylabel("RMS error (°, log scale)")
    ax.set_title("Ablations — the bias state is not what wins; the dynamic-acceleration term is")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fusion_ablation.pdf")

    # 5 — long horizon
    lh = results["long_horizon"]
    fig, ax = plt.subplots(figsize=(9, 4.0))
    t = np.asarray(lh["t"])
    for key, d in lh["filters"].items():
        ax.plot(t, d["trace"], color=COLOR[key], lw=1.2, label=d["label"])
    ax.axvline(lh["trial_end_s"], color="black", ls="--", lw=1,
               label=f"trial ends ({lh['trial_end_s']:.1f} s)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("orientation error (°)")
    ax.set_yscale("log")
    ax.set_title(f"Long horizon — one {lh['motion_class']} trial then a "
                 f"{lh['hold_s']:.0f} s stationary hold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    save(fig, "fusion_long_horizon.pdf")

    # 6 — noise sensitivity
    ns = results["noise_sensitivity"]
    fig, ax = plt.subplots(1, len(CLASSES), figsize=(12, 3.2), sharey=True)
    for a, c in zip(ax, CLASSES):
        for key in MAIN_TWO:
            a.plot(ns["scales"], ns["filters"][key][c], "o-", color=COLOR[key],
                   label=runs[key]["label"] if c == CLASSES[0] else None)
        a.set_xscale("log")
        a.set_yscale("log")
        a.set_xlabel("sensor-noise scale (×D2)")
        a.set_title(c)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("RMS error (°)")
    ax[0].legend(fontsize=8)
    fig.suptitle("Noise sensitivity — every D2 noise parameter scaled together")
    save(fig, "fusion_noise_sensitivity.pdf")

    # 7 — which channel the error lives in, and how initialisation changes the picture
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    x = np.arange(len(CLASSES))
    for a, init in zip(ax, ("truth", "accel")):
        r_init = results["runs"][init]
        for k, key in enumerate(MAIN_TWO):
            tot = [r_init[key]["by_class"][c]["rms"] for c in CLASSES]
            tilt = [r_init[key]["by_class"][c]["tilt_rms"] for c in CLASSES]
            pos = x + (k - 0.5) * 0.38
            a.bar(pos, tot, 0.36, color=COLOR[key], alpha=0.35,
                  label=f"{r_init[key]['label']} — total" if init == "truth" else None)
            a.bar(pos, tilt, 0.36, color=COLOR[key],
                  label=f"{r_init[key]['label']} — tilt only" if init == "truth" else None)
        a.set_xticks(x)
        a.set_xticklabels(CLASSES, rotation=15)
        a.set_yscale("log")
        a.set_title(f"init = {init}")
        a.grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("RMS error (°, log scale)")
    ax[0].legend(fontsize=7)
    fig.suptitle("Observable vs unobservable: solid = tilt (measurable), "
                 "faded = total (tilt + heading drift)")
    save(fig, "fusion_tilt_vs_heading.pdf")

    return written


# --------------------------------------------------------------------------- #
# Report + CLI
# --------------------------------------------------------------------------- #
def print_table(results):
    """The by-class RMS table task 4.5 asks for, in work10/work11's format.

    Printed twice per initialisation: total orientation error, then the tilt-only component.
    With no magnetometer the difference between the two *is* the heading error, so printing
    only the first would hide which channel a filter is failing in.
    """
    for init, runs in results["runs"].items():
        for metric, key_rms in (("total orientation error", "rms"), ("tilt only", "tilt_rms")):
            print(f"\n=== initialisation: {init}  ·  {metric} " + "=" * 30)
            head = f"{'filter':24s} {'overall':>8s} " + " ".join(f"{c:>13s}" for c in CLASSES)
            print(head)
            print("-" * len(head))
            for r in runs.values():
                row = " ".join(f"{r['by_class'][c][key_rms]:13.3f}" for c in CLASSES)
                print(f"{r['label']:24s} {r['overall'][key_rms]:8.3f} {row}")


def run(data_dir, figures_dir, *, limit_per_class=0, hold_s=60.0, noise_trials=5,
        out_path=None, figures=True):
    """Full task-4.5 validation. Returns the results dict (also written to ``out_path``)."""
    data_dir = Path(data_dir)
    noisy = sorted(data_dir.glob("S001_*_noisy_*.json"))
    clean = sorted(data_dir.glob("S001_*_clean_*.json"))
    if not noisy:
        raise FileNotFoundError(
            f"no noisy SkillData records in {data_dir} — run `make synthetic` first")
    if limit_per_class:
        noisy = [p for c in CLASSES
                 for p in [q for q in noisy if f"_{c}_noisy_" in q.name][:limit_per_class]]
    noise_files = [p for c in CLASSES
                   for p in [q for q in clean if f"_{c}_clean_" in q.name][:noise_trials]]

    results = {
        "dataset": {"dir": str(data_dir), "n_noisy_records": len(noisy),
                    "limit_per_class": limit_per_class or None},
        "filters": {k: v[0] for k, v in FILTERS.items()},
        "runs": {},
    }
    for init in ("truth", "accel"):
        results["runs"][init] = {}
        for key in FILTERS:
            print(f"  scoring {FILTERS[key][0]} [{init}] over {len(noisy)} records...",
                  flush=True)
            results["runs"][init][key] = evaluate(noisy, key, init=init)

    print("  convergence traces...", flush=True)
    results["convergence"] = convergence_traces(noisy)
    print("  long-horizon run...", flush=True)
    lift = [p for p in noisy if "_lift_noisy_" in p.name]
    results["long_horizon"] = long_horizon(lift[0] if lift else noisy[0], hold_s=hold_s)
    print(f"  noise sensitivity ({len(noise_files)} clean records)...", flush=True)
    results["noise_sensitivity"] = noise_sensitivity(noise_files)

    if out_path is None:
        out_path = Path(figures_dir) / "fusion_validation.json"
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=1))
    results["_out_path"] = str(out_path)

    if figures:
        written = make_figures(results, figures_dir)
        print(f"  figures: {', '.join(p.name for p in written)}", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Score every orientation filter against the synthetic dataset (task 4.5).")
    ap.add_argument("--data", default="data/processed", help="SkillData record directory")
    ap.add_argument("--figures", default="paper/figures", help="where fusion_*.pdf are written")
    ap.add_argument("--out", default=None, help="results JSON (default <figures>/fusion_validation.json)")
    ap.add_argument("--limit-per-class", type=int, default=0,
                    help="score only the first N records per class (0 = all)")
    ap.add_argument("--hold-s", type=float, default=60.0, help="long-horizon stationary hold")
    ap.add_argument("--noise-trials", type=int, default=5,
                    help="clean records per class in the noise-sensitivity sweep")
    ap.add_argument("--no-figures", action="store_true", help="skip figure rendering")
    args = ap.parse_args(argv)

    t0 = time.time()
    results = run(args.data, args.figures, limit_per_class=args.limit_per_class,
                  hold_s=args.hold_s, noise_trials=args.noise_trials,
                  out_path=args.out, figures=not args.no_figures)
    print_table(results)
    print(f"\nresults -> {results['_out_path']}   ({time.time() - t0:.0f} s total)")


if __name__ == "__main__":
    main()
