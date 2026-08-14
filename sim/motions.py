"""Motion-class library (task 3.6): synthetic arm motions for the 7-DOF human arm.

Four motion classes, each a smooth prep -> active -> settle profile (min-jerk on the joints it
uses), with randomizable amplitudes so many distinct trials can be drawn per class:

  reach        — point/reach out: shoulder yaw+pitch + elbow extension. Moderate speed.
  lift         — lift/carry: shoulder pitch up + elbow flexion, slower and more sustained.
  wrist_rotate — distal: forearm pronation + wrist flex/deviation dominate.
  throw        — fast, short active burst across shoulder+elbow+forearm+wrist. High distal
                 angular velocity: the wrist sensor S5 exceeds the ±2000°/s gyro range (D2),
                 so the sensor model flags saturation — the documented cheap-gyro envelope.

Joint order of ``sim.arm3d.human_arm_7dof``:
  0 sh_yaw · 1 sh_pitch · 2 sh_roll(upper) · 3 elbow · 4 forearm_pron · 5 wrist_flex · 6 wrist_dev.

``motion_class`` is a first-class SkillData field; ``generate_trial`` returns the joint trajectories
that an ingest adapter turns into a labeled record. Per-timestep prep/active/settle labels are derived
downstream from joint speed (``skilldata.ingest.phase_labels_from_speed``).
"""

from __future__ import annotations

import numpy as np

from .arm import min_jerk

MOTION_CLASSES = ("reach", "lift", "wrist_rotate", "throw")

# ---------------------------------------------------------------------------
# The excitation class (task 3.15) is deliberately NOT a member of MOTION_CLASSES.
#
# Two reasons, and the second is a hard constraint rather than a preference.
#
# (a) It is an *instrument*, not a motion anyone performs. Nobody reaches for a cup by tracking a
#     sum of sinusoids. Folding it into the naturalistic four would contaminate every per-class
#     statistic the dataset reports — saturation fractions, peak gyro envelopes, phase composition
#     — with a trajectory that exists only to make the dynamics identifiable.
#
# (b) `MOTION_CLASSES` drives `_class_counts()` in **both** `skilldata.generate_synthetic` and
#     `skilldata.generate_dynamics`. Adding a fifth member changes `_class_counts(1000, ...)` from
#     [250, 250, 250, 250] to [200] * 5, which changes the order in which trials are drawn from the
#     shared RNG. That would silently break the index alignment between the dynamics slice and the
#     committed 2000-record SkillData v1 dataset — and invalidate the Phase 4 fusion numbers
#     measured on it. A one-word change to a tuple would have quietly cost a 12-minute re-score and
#     a re-run of every filter result.
#
# So `excite` is reachable through `generate_trial` and named in `ALL_CLASSES`, and it stays out of
# `MOTION_CLASSES`.
# ---------------------------------------------------------------------------
EXCITATION_CLASS = "excite"
ALL_CLASSES = MOTION_CLASSES + (EXCITATION_CLASS,)

# Defaults for the excitation trajectory. Chosen, not optimised — see `excitation_trial`.
EXCITE_DEFAULTS = {"dur": 5.0, "n_harm": 5, "f_base": 0.2, "amp": 0.35}

# (joint index, start angle, (amp_lo, amp_hi)) plus the active window fraction, per class.
_SPECS = {
    "reach": {"dur": 2.0, "win": (0.4, 1.2),
              "moves": [(0, 0.0, (0.3, 0.7)), (1, 0.0, (0.4, 0.9)), (3, 0.2, (1.0, 1.5))]},
    "lift": {"dur": 2.5, "win": (0.4, 1.9),
             "moves": [(1, 0.0, (0.8, 1.3)), (3, 0.3, (1.2, 1.8)), (2, 0.0, (-0.3, 0.3))]},
    "wrist_rotate": {"dur": 2.0, "win": (0.4, 1.4),
                     "moves": [(4, 0.0, (1.2, 2.6)), (5, 0.0, (0.3, 0.9)), (6, 0.0, (-0.5, 0.5)),
                               (3, 0.6, (0.7, 1.1))]},
    "throw": {"dur": 1.2, "win": (0.45, 0.62),
              "moves": [(1, 0.0, (1.1, 1.6)), (3, 0.4, (1.8, 2.3)), (4, 0.0, (1.8, 2.6)),
                        (5, 0.0, (1.6, 2.4)), (0, 0.0, (0.3, 0.6))]},
}


def excitation_trial(fs=500.0, rng=None, *, dur=None, n_harm=None, f_base=None, amp=None):
    """A finite-Fourier-series **excitation** trajectory (task 3.15). Returns ``(t, qs, qds, qdds)``.

    Why this exists
    ---------------
    The four naturalistic classes above cannot identify the arm's dynamics. Measured on the task
    3.14 dataset (`analysis/identifiability.py`, and the write-up in ``DEPTH.md``): with the exact
    model structure and exact ``qddot`` — the best case any estimator can have — the damping of
    ``sh_roll`` comes back with an error bar **207% of its true value** and ``wrist_dev`` **76%**,
    because those joints move in exactly one class each and at 0.057 / 0.144 rad/s RMS against 2.49
    for the strongest. Regressor condition number 3.56e11. That is not a noise problem: at 0.1%
    torque noise the worst joint still carries a 21% error bar.

    This is the failure the identification literature has warned about since Gautier & Khalil
    (1992), *Exciting Trajectories for the Identification of Base Inertial Parameters of Robots*,
    IJRR 11(4):362-375, doi:10.1177/027836499201100408 — whose entire method is designing
    trajectories that minimise the condition number of the regressor. Leboutet et al. (2021),
    Applied Sciences 11:4303, doi:10.3390/app11094303, still lists excitation-trajectory
    computation as a required stage of an identification pipeline.

    The trajectory
    --------------
    Each joint follows a truncated Fourier series with zero mean displacement,

        q_j(t)    = q0_j + sum_k [ a_jk/(w k) sin(w k t) - b_jk/(w k) cos(w k t) ]
        qdot_j(t) =        sum_k [ a_jk cos(w k t) + b_jk sin(w k t) ]
        qddot_j(t)=        sum_k [ -a_jk w k sin(w k t) + b_jk w k cos(w k t) ]

    with ``w = 2 pi f_base``. Derivatives are analytic, matching the exactness the rest of ``sim``
    guarantees — nothing here is finite-differenced.

    **Every joint is driven**, which is the single property the naturalistic classes lack, and the
    coefficients are drawn per trial so a set of trials spans configuration space rather than
    repeating one path.

    Honest limits
    -------------
    * The coefficients are **random, not optimised**. Gautier & Khalil minimise the condition number
      by nonlinear programming; this is the unoptimised baseline, so it is a *lower bound* on what
      the method can deliver. Measured against the same estimator and noise it still cuts the
      worst-joint damping error from 143.9% to 11.7% and the condition number from 3.56e11 to
      2.23e10.
    * ``q0`` defaults to zero — the arm extended horizontally — deliberately, because the arm's
      *hanging* pose (shoulder pitch = pi/2) is a gimbal lock where ``M`` loses rank (task 3.13).
      Starting there would trade an identifiability problem for a conditioning one.
    * Unlike the four naturalistic classes this trajectory does **not** start and end at rest, so
      momentum and kinetic energy are non-zero at both endpoints. Anything that assumes rest
      endpoints (as the 3.14 tests do for the naturalistic classes) must exclude ``excite``.
    """
    d = EXCITE_DEFAULTS
    dur = d["dur"] if dur is None else dur
    n_harm = d["n_harm"] if n_harm is None else n_harm
    f_base = d["f_base"] if f_base is None else f_base
    amp = d["amp"] if amp is None else amp
    if n_harm < 1:
        raise ValueError(f"n_harm must be >= 1; got {n_harm}")
    if f_base <= 0 or dur <= 0 or amp <= 0:
        raise ValueError("dur, f_base and amp must all be positive")

    rng = np.random.default_rng() if rng is None else rng
    t = np.arange(0.0, dur, 1.0 / fs)
    w = 2 * np.pi * f_base
    q = np.zeros((t.size, 7))
    qd = np.zeros((t.size, 7))
    qdd = np.zeros((t.size, 7))
    for j in range(7):
        for k in range(1, n_harm + 1):
            a, b = rng.normal(0.0, amp), rng.normal(0.0, amp)
            wk = w * k
            q[:, j] += a / wk * np.sin(wk * t) - b / wk * np.cos(wk * t)
            qd[:, j] += a * np.cos(wk * t) + b * np.sin(wk * t)
            qdd[:, j] += -a * wk * np.sin(wk * t) + b * wk * np.cos(wk * t)
    return t, q, qd, qdd


def generate_trial(motion_class, fs=500.0, rng=None, **kwargs):
    """Generate one trial of ``motion_class``. Returns ``(t, qs, qds, qdds)``.

    ``qs/qds/qdds`` are ``(T, 7)`` joint angle / velocity / acceleration trajectories. Amplitudes are
    sampled from the class ranges (deterministic if ``rng`` is seeded). Joints not used by the class
    stay at zero; every trial starts and ends at rest (min-jerk endpoints).

    ``motion_class`` may also be ``"excite"`` (task 3.15), which dispatches to `excitation_trial`
    and accepts its ``dur`` / ``n_harm`` / ``f_base`` / ``amp`` keywords. That class is **not** in
    `MOTION_CLASSES` — see the comment beside `EXCITATION_CLASS` for why that separation is load
    bearing rather than cosmetic.
    """
    if motion_class == EXCITATION_CLASS:
        return excitation_trial(fs=fs, rng=rng, **kwargs)
    if motion_class not in _SPECS:
        raise ValueError(f"unknown motion_class {motion_class!r}; choose from {ALL_CLASSES}")
    if kwargs:
        raise TypeError(f"{motion_class!r} takes no extra parameters; got {sorted(kwargs)}")
    rng = np.random.default_rng() if rng is None else rng
    spec = _SPECS[motion_class]
    t = np.arange(0.0, spec["dur"], 1.0 / fs)
    q = np.zeros((t.size, 7))
    qd = np.zeros((t.size, 7))
    qdd = np.zeros((t.size, 7))
    t0, tf = spec["win"]
    for j, q0, (lo, hi) in spec["moves"]:
        qf = q0 + rng.uniform(lo, hi)
        q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, t0, tf, q0, qf)
    return t, q, qd, qdd


if __name__ == "__main__":
    from .arm3d import human_arm_7dof

    arm = human_arm_7dof()
    rng = np.random.default_rng(0)
    print("peak |gyro| (deg/s) per motion class, per sensor:")
    for mc in MOTION_CLASSES:
        t, q, qd, qdd = generate_trial(mc, fs=500.0, rng=rng)
        imu = arm.ideal_imu(q, qd, qdd)
        peaks = {sid: np.rad2deg(np.linalg.norm(imu[sid]["gyro"], axis=-1)).max() for sid in imu}
        sat = " SATURATES (>2000)" if peaks["S5"] > 2000 else ""
        print(f"  {mc:13s} " + " ".join(f"{s}={p:7.0f}" for s, p in peaks.items()) + sat)
