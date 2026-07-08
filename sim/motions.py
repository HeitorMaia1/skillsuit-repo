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


def generate_trial(motion_class, fs=500.0, rng=None):
    """Generate one trial of ``motion_class``. Returns ``(t, qs, qds, qdds)``.

    ``qs/qds/qdds`` are ``(T, 7)`` joint angle / velocity / acceleration trajectories. Amplitudes are
    sampled from the class ranges (deterministic if ``rng`` is seeded). Joints not used by the class
    stay at zero; every trial starts and ends at rest (min-jerk endpoints).
    """
    if motion_class not in _SPECS:
        raise ValueError(f"unknown motion_class {motion_class!r}; choose from {MOTION_CLASSES}")
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
