"""Tests for the multiplicative orientation EKF (task 4.4): the same three orientation cases
``test_madgwick.py`` covers — stationary gravity alignment, 90 deg/axis rotation recovery,
noiseless synthetic convergence — plus the case the EKF exists for, gyro-bias recovery.

The bias cases are split deliberately, because the physics splits:

  * **stationary** — only the two bias components *perpendicular* to gravity are observable;
    the vertical-axis one is not, for the same reason yaw is not (``fusion.ekf``'s
    "Observability" section). Asserting all three would be asserting something false.
  * **rotating** — the unobservable direction moves with the body, so a wide enough arc makes
    all three components observable. This is where the bias state earns its keep, and where
    it is checked head-to-head against Madgwick on the identical input stream.

Ground truth is analytic throughout (closed-form rotations, or ``sim.arm3d``'s own frame
orientation) — never a second filter run, which would only check a filter against itself.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from fusion.ekf import OrientationEKF, quat_mul, rotvec_to_quat, skew
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
    """(..., 3, 3) rotation matrices -> (..., 4) quaternions in (w,x,y,z) order."""
    xyzw = Rotation.from_matrix(C).as_quat()
    return np.concatenate([xyzw[..., 3:4], xyzw[..., :3]], axis=-1)


def _spin_about_axis(axis, rate_dps, duration_s, fs=FS):
    """Exact gyro + accel streams for a constant-rate rotation about a fixed body/world axis.

    Returns ``(gyro_dps (T,3), accel_g (T,3), q_true (T,4))``. Rotating about a single fixed
    axis keeps that axis common to both frames, so the accelerometer reads the exact rotated
    gravity direction ``C(t)^T (0,0,1)`` with no dynamic-acceleration component.
    """
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    omega = np.deg2rad(rate_dps)
    gyro_dps = np.tile(rate_dps * axis, (n, 1))
    k = skew(axis)
    q_true = np.empty((n, 4))
    accel = np.empty((n, 3))
    for i, th in enumerate(omega * t):
        C = np.eye(3) + np.sin(th) * k + (1 - np.cos(th)) * (k @ k)
        accel[i] = C.T @ np.array([0.0, 0.0, 1.0])
        q_true[i] = quat_from_axis_angle(axis, th)
    return gyro_dps, accel, q_true


# --------------------------------------------------------------------------- #
# Case 1 — stationary: converges to a gravity-aligned quaternion
# --------------------------------------------------------------------------- #
def test_stationary_converges_to_gravity_aligned_quaternion():
    """Held still (gyro = 0) with the estimate started 25 deg off about y; the accelerometer
    update alone must pull the predicted gravity direction onto the measured (0,0,1).

    ``p0_angle_deg=45`` because the initial error genuinely *is* large here — telling the
    filter it is 5 deg confident when it is 25 deg wrong is an inconsistent prior, and the
    correct fix is an honest prior, not a hand-tuned gain.
    """
    q_start = tuple(quat_from_axis_angle((0.0, 1.0, 0.0), np.deg2rad(25.0)))
    f = OrientationEKF(sample_rate_hz=FS, q0=q_start, p0_angle_deg=45.0)

    n = 2000  # 4 s at 500 Hz
    qs = f.run(np.zeros((n, 3)), np.tile([0.0, 0.0, 1.0], (n, 1)))

    pred = predicted_gravity_direction(qs[-1])
    assert np.allclose(pred, [0.0, 0.0, 1.0], atol=1e-3), pred


def test_stationary_already_aligned_stays_aligned():
    """Starting aligned with zero gyro input, the estimate does not drift."""
    f = OrientationEKF(sample_rate_hz=FS)
    n = 1000
    qs = f.run(np.zeros((n, 3)), np.tile([0.0, 0.0, 1.0], (n, 1)))
    assert quat_angle_error_deg(qs[-1], (1.0, 0.0, 0.0, 0.0)) < 0.1


# --------------------------------------------------------------------------- #
# Case 2 — a 90 deg rotation about each axis is recovered
# --------------------------------------------------------------------------- #
def test_90_degree_rotation_recovered_about_each_axis():
    for axis in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
        gyro_dps, accel, q_true = _spin_about_axis(axis, rate_dps=90.0, duration_s=1.0)
        f = OrientationEKF(sample_rate_hz=FS)
        qs = f.run(gyro_dps, accel)
        err = quat_angle_error_deg(qs[-1], q_true[-1])
        assert err < 1.0, f"axis {axis}: {err:.3f} deg error recovering the 90 deg rotation"


# --------------------------------------------------------------------------- #
# Case 3 — noiseless synthetic input converges to under 0.5 deg error
# --------------------------------------------------------------------------- #
def test_noiseless_synthetic_converges_under_half_degree():
    """The same 2-joint reach on the 7-DOF arm that ``test_madgwick.py`` case 3 uses, at S2.

    Graded on the settle phase for the same documented reason: during the fast active phase
    the specific force contains real linear acceleration, which violates *both* filters'
    "the accelerometer measures gravity" assumption. That is physics, not an implementation
    defect, and "converges to" is a statement about the settled value.
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
    q_true = _rotmat_to_quat_wxyz(state["frames"]["sh_roll_upper"])

    f = OrientationEKF(sample_rate_hz=FS, q0=(1.0, 0.0, 0.0, 0.0))
    qs = f.run(np.rad2deg(sensor["gyro"]), sensor["accel"] / G)

    err = quat_angle_error_deg(qs, q_true)
    settle = t > 1.4
    assert err[settle].max() < 0.5, f"settle-phase error {err[settle].max():.3f} deg >= 0.5 deg"


# --------------------------------------------------------------------------- #
# Case 4 — the reason this filter exists: a constant gyro bias is estimated and removed
# --------------------------------------------------------------------------- #
def test_constant_bias_estimated_under_rotation_all_three_axes():
    """A constant bias on all three axes, during a 240 deg sweep about x at 30 deg/s.

    The sweep is what makes the full 3-vector observable: the unobservable direction is the
    gravity direction *in the body frame*, which rotates with the body, so over a wide enough
    arc no component stays hidden. All three are required back to within 0.2 deg/s of truth.
    """
    bias_true = np.array([2.0, -1.5, 0.8])  # deg/s
    gyro_dps, accel, q_true = _spin_about_axis((1.0, 0.0, 0.0), rate_dps=30.0, duration_s=8.0)

    f = OrientationEKF(sample_rate_hz=FS, p0_bias_dps=5.0)
    qs, bias_hat = f.run(gyro_dps + bias_true, accel, return_bias=True)

    err_bias = np.abs(bias_hat[-1] - bias_true)
    assert (err_bias < 0.2).all(), f"bias estimate {bias_hat[-1]} vs true {bias_true}"

    err_deg = quat_angle_error_deg(qs[-1], q_true[-1])
    assert err_deg < 1.0, f"orientation error {err_deg:.3f} deg after bias convergence"


def test_bias_removal_beats_madgwick_on_the_same_biased_stream():
    """Head-to-head on one identical biased stream: the bias state is worth something.

    Madgwick has no bias state, so it pays the offset for the whole run — most visibly about
    the local vertical, where the accelerometer cannot correct it at all. The EKF learns the
    offset and stops paying it. Asserting a *factor* rather than an absolute number keeps this
    a test of the mechanism, not of a tuning constant.
    """
    bias_true = np.array([2.0, -1.5, 0.8])
    gyro_dps, accel, q_true = _spin_about_axis((1.0, 0.0, 0.0), rate_dps=30.0, duration_s=8.0)
    biased = gyro_dps + bias_true

    ekf_err = quat_angle_error_deg(
        OrientationEKF(sample_rate_hz=FS, p0_bias_dps=5.0).run(biased, accel)[-1], q_true[-1])
    mad_err = quat_angle_error_deg(
        MadgwickFilter(sample_rate_hz=FS, beta=0.1).run(biased, accel)[-1], q_true[-1])

    assert ekf_err < mad_err / 3.0, f"EKF {ekf_err:.3f} deg vs Madgwick {mad_err:.3f} deg"


def test_vertical_bias_is_not_observable_while_stationary():
    """The honest converse: held still, the bias component along gravity cannot be learned.

    Two horizontal components are recovered; the vertical one is not, and the filter's own
    covariance must *say* so — its variance may not shrink, because no information about it
    ever arrives. A filter that reported confidence here would be lying, and that is the
    failure mode this test exists to catch.
    """
    bias_true = np.array([1.5, -1.0, 2.0])  # z is along gravity for an aligned, level sensor
    n = 6000  # 12 s at 500 Hz
    f = OrientationEKF(sample_rate_hz=FS, p0_bias_dps=5.0)
    _, bias_hat = f.run(
        np.tile(bias_true, (n, 1)), np.tile([0.0, 0.0, 1.0], (n, 1)), return_bias=True)

    assert abs(bias_hat[-1, 0] - bias_true[0]) < 0.2, bias_hat[-1]
    assert abs(bias_hat[-1, 1] - bias_true[1]) < 0.2, bias_hat[-1]
    # the z variance must not have collapsed: no measurement ever constrained it
    var_z_dps2 = np.rad2deg(np.rad2deg(f.P[5, 5]))
    assert var_z_dps2 > 0.5 * f.p0_bias_dps**2, f"vertical bias variance collapsed to {var_z_dps2}"


# --------------------------------------------------------------------------- #
# Numerical hygiene — the failure mode the multiplicative formulation exists to prevent
# --------------------------------------------------------------------------- #
def test_quaternion_stays_unit_and_covariance_stays_valid():
    """Over a long noisy run: ``|q|`` never leaves 1, and ``P`` stays symmetric and PSD.

    ``|q| == 1`` here is a property of the *update rule* (every change to q is a product with
    a unit quaternion), not of the renormalization — which is why it is checked to 1e-12 and
    not to some loose tolerance.
    """
    rng = np.random.default_rng(0)
    n = 5000
    gyro_dps, accel, _ = _spin_about_axis((0.3, 1.0, -0.5), rate_dps=60.0, duration_s=n / FS)
    gyro_dps = gyro_dps + rng.normal(0.0, 1.0, gyro_dps.shape)
    accel = accel + rng.normal(0.0, 0.05, accel.shape)

    f = OrientationEKF(sample_rate_hz=FS)
    qs = f.run(gyro_dps, accel)

    assert np.abs(np.linalg.norm(qs, axis=-1) - 1.0).max() < 1e-12
    assert np.allclose(f.P, f.P.T, atol=1e-18)
    assert np.linalg.eigvalsh(f.P).min() > -1e-18


def test_quaternion_helpers_are_self_consistent():
    """``rotvec_to_quat`` and ``quat_mul`` agree with the composition they claim to implement."""
    a = rotvec_to_quat(np.deg2rad([30.0, 0.0, 0.0]))
    b = rotvec_to_quat(np.deg2rad([0.0, 0.0, 45.0]))
    ab = quat_mul(a, b)
    assert abs(np.linalg.norm(ab) - 1.0) < 1e-12
    # right-multiplication composes in the body frame: rotate 30 about x, then 45 about the
    # *new* z -- which scipy expresses as intrinsic 'xz' Euler angles.
    ref = Rotation.from_euler("XZ", [30.0, 45.0], degrees=True).as_quat()
    ref_wxyz = np.array([ref[3], *ref[:3]])
    assert quat_angle_error_deg(ab, ref_wxyz) < 1e-9
