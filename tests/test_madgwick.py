"""Tests for the Madgwick orientation filter (task 4.2): the three cases the action plan
names — stationary gravity alignment, 90 deg/axis rotation recovery, and noiseless synthetic
convergence to <0.5 deg error.

Ground truth for the synthetic-convergence case comes from ``sim.arm3d``'s analytic frame
orientation (``Arm3D.state()['frames']``), not from re-running the filter on clean data —
that would just be checking the filter against itself. ``scipy.spatial.transform.Rotation``
converts that ground-truth rotation *matrix* to a quaternion for comparison; this is generic
linear algebra plumbing (scipy is an existing project dependency), not a port of the filter
under test.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from fusion.madgwick import (
    MadgwickFilter,
    predicted_gravity_direction,
    quat_angle_error_deg,
    quat_from_axis_angle,
)
from sim.arm import G, min_jerk
from sim.arm3d import human_arm_7dof

FS = 500.0


def _rotmat_to_quat_wxyz(C):
    """(..., 3, 3) rotation matrices -> (..., 4) quaternions in this module's (w,x,y,z) order.

    scipy returns (x, y, z, w); reordered to match ``MadgwickFilter``'s convention.
    """
    xyzw = Rotation.from_matrix(C).as_quat()
    return np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)


# --------------------------------------------------------------------------- #
# Case 1 — stationary: converges to a gravity-aligned quaternion
# --------------------------------------------------------------------------- #
def test_stationary_converges_to_gravity_aligned_quaternion():
    """Held still (gyro=0) but the initial estimate is tilted 25 deg off; pure accelerometer
    correction must pull the predicted gravity direction onto the true (0,0,1) reading.

    Starts off-axis about y (roll/pitch), which accel-only correction *can* observe — see the
    module docstring's yaw-unobservability note for why an off-axis-about-z start would not
    converge by this mechanism.
    """
    theta0 = np.deg2rad(25.0)
    q_start = tuple(quat_from_axis_angle((0.0, 1.0, 0.0), theta0))
    f = MadgwickFilter(sample_rate_hz=FS, beta=0.5, q0=q_start)

    n = 2000  # 4 s at 500 Hz — ample time for a 25 deg correction at beta=0.5
    gyro = np.zeros((n, 3))
    accel = np.tile([0.0, 0.0, 1.0], (n, 1))  # true stationary reading: gravity 'up' on z
    qs = f.run(gyro, accel)

    pred = predicted_gravity_direction(qs[-1])
    assert np.allclose(pred, [0.0, 0.0, 1.0], atol=1e-3)


def test_stationary_already_aligned_stays_aligned():
    """Starting already gravity-aligned with zero gyro input, the estimate does not drift."""
    f = MadgwickFilter(sample_rate_hz=FS, beta=0.1)
    n = 1000
    gyro = np.zeros((n, 3))
    accel = np.tile([0.0, 0.0, 1.0], (n, 1))
    qs = f.run(gyro, accel)
    assert quat_angle_error_deg(qs[-1], (1.0, 0.0, 0.0, 0.0)) < 0.1


# --------------------------------------------------------------------------- #
# Case 2 — a 90 deg rotation about each axis is recovered
# --------------------------------------------------------------------------- #
def _rotate_about_axis_trial(axis, total_angle_rad=np.pi / 2, duration_s=1.0, fs=FS):
    """Exact analytic gyro + accel streams for a constant-rate rotation about a fixed axis.

    Single-axis rotation -> constant body-frame angular velocity; the accelerometer reads the
    (rotating) gravity direction exactly, ``C(t)^T @ (0,0,1)``, computed from the same
    Rodrigues form ``sim.arm3d`` uses for its rotation matrices.
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    omega = total_angle_rad / duration_s
    n = int(duration_s * fs)
    t = np.arange(n) / fs
    gyro_dps = np.rad2deg(np.tile(omega * axis, (n, 1)))

    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    accel = np.empty((n, 3))
    for i, theta in enumerate(omega * t):
        C = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        accel[i] = C.T @ np.array([0.0, 0.0, 1.0])
    return gyro_dps, accel


def test_90_degree_rotation_recovered_about_each_axis():
    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        gyro_dps, accel = _rotate_about_axis_trial(axis)
        f = MadgwickFilter(sample_rate_hz=FS, beta=0.1)
        qs = f.run(gyro_dps, accel)
        q_true = quat_from_axis_angle(axis, np.pi / 2)
        err = quat_angle_error_deg(qs[-1], q_true)
        assert err < 1.0, f"axis {axis}: {err:.3f} deg error recovering the 90 deg rotation"


# --------------------------------------------------------------------------- #
# Case 3 — noiseless synthetic input converges to under 0.5 deg error
# --------------------------------------------------------------------------- #
def test_noiseless_synthetic_converges_under_half_degree():
    """A 2-joint reach on the 7-DOF human arm (task 3.2), ideal (noiseless) IMU at S2.

    Ground truth is the analytic frame orientation for S2's segment (``sh_roll_upper``), which
    starts at identity (matching the filter's default ``q0``) since sh_roll_upper itself never
    moves in this trial — only its parent joints (sh_yaw, sh_pitch) do, so this exercises real
    tracking, not a static case. Error spikes during the fast active phase (specific force
    includes real dynamic acceleration there, not just gravity — a known accel-tilt-correction
    limitation, not a filter bug); "converges to" is checked after the motion returns to rest
    (the settle phase), which is the physically meaningful convergence criterion.
    """
    t = np.arange(0.0, 2.0, 1.0 / FS)
    arm = human_arm_7dof()
    n = t.size
    q = np.zeros((n, 7))
    qd = np.zeros((n, 7))
    qdd = np.zeros((n, 7))
    q[:, 0], qd[:, 0], qdd[:, 0] = min_jerk(t, 0.4, 1.2, 0.0, np.deg2rad(50))  # sh_yaw
    q[:, 1], qd[:, 1], qdd[:, 1] = min_jerk(t, 0.4, 1.2, 0.0, np.deg2rad(35))  # sh_pitch

    state = arm.state(q, qd, qdd)
    sensor = state["sensors"]["S2"]
    gyro_dps = np.rad2deg(sensor["gyro"])
    accel_g = sensor["accel"] / G
    q_true = _rotmat_to_quat_wxyz(state["frames"]["sh_roll_upper"])

    f = MadgwickFilter(sample_rate_hz=FS, beta=0.1, q0=(1.0, 0.0, 0.0, 0.0))
    qs = f.run(gyro_dps, accel_g)

    err = quat_angle_error_deg(qs, q_true)
    settle = t > 1.4  # motion (window ends 1.2 s) has fully settled back to rest
    assert err[settle].max() < 0.5, f"settle-phase error {err[settle].max():.3f} deg >= 0.5 deg"
