"""Tests for the 3D serial-chain arm: planar reduction (2D special case) + invariants."""

import numpy as np

from sim.arm import G, PlanarArm, min_jerk
from sim.arm3d import Arm3D, Segment, human_arm_7dof, planar_2link


def _reach_cols(n_joints, moves, fs=500.0, dur=2.0):
    """Build (qs, qds, qdds) of shape (T, n_joints); `moves` maps joint index -> (q0, qf)."""
    t = np.arange(0.0, dur, 1.0 / fs)
    q = np.zeros((t.size, n_joints))
    qd = np.zeros((t.size, n_joints))
    qdd = np.zeros((t.size, n_joints))
    for j, (a0, a1) in moves.items():
        q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, 0.4, 1.2, a0, a1)
    return t, q, qd, qdd


# --- the headline: 2D is a special case of the 3D model -------------------- #
def test_planar_reduction_matches_planar_arm():
    """planar_2link (3D) reproduces sim.arm.PlanarArm gyro + accel exactly."""
    fs = 500.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, np.deg2rad(10), np.deg2rad(60))
    th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, np.deg2rad(20), np.deg2rad(90))

    planar = PlanarArm()
    ref = planar.ideal_imu(th1, th1d, th1dd, th2, th2d, th2dd)

    arm3d = planar_2link(l1=planar.l1, l2=planar.l2)
    qs = np.stack([th1, th2], axis=1)
    qds = np.stack([th1d, th2d], axis=1)
    qdds = np.stack([th1dd, th2dd], axis=1)
    got = arm3d.ideal_imu(qs, qds, qdds)

    for sid in ("S2", "S4"):
        # gyro: planar z-rate matches the 3D body-frame z component; x,y are zero
        assert np.allclose(got[sid]["gyro"][:, 2], ref[sid]["gyro_z"], atol=1e-12)
        assert np.allclose(got[sid]["gyro"][:, :2], 0.0, atol=1e-12)
        # accel: in-plane (x,y) matches; out-of-plane z is zero
        assert np.allclose(got[sid]["accel"][:, :2], ref[sid]["accel"], atol=1e-12)
        assert np.allclose(got[sid]["accel"][:, 2], 0.0, atol=1e-12)


def test_planar_reduction_matches_forward_kinematics():
    """planar_2link joint positions match PlanarArm forward kinematics (x,y), z=0."""
    planar = PlanarArm()
    t, q, qd, qdd = _reach_cols(2, {0: (0.1, 1.0), 1: (0.2, 1.3)})
    st = planar_2link(l1=planar.l1, l2=planar.l2).state(q, qd, qdd)
    fk = planar.forward_kinematics(q[:, 0], q[:, 1])
    assert np.allclose(st["joints"]["upper"][:, :2], fk["elbow"], atol=1e-12)
    assert np.allclose(st["joints"]["fore"][:, :2], fk["wrist"], atol=1e-12)
    assert np.allclose(st["joints"]["fore"][:, 2], 0.0, atol=1e-12)


# --- invariants for the genuinely 3D model --------------------------------- #
def test_rest_reads_gravity_zero_gyro_at_arbitrary_pose():
    """At any static pose (qd=qdd=0), every sensor reads |accel|=G and zero gyro."""
    arm = human_arm_7dof()
    T = 5
    q = np.tile(np.array([0.3, -0.4, 0.5, 0.9, 0.2, -0.3, 0.4]), (T, 1))  # arbitrary held pose
    z = np.zeros((T, 7))
    imu = arm.ideal_imu(q, z, z)
    for sid in ("S2", "S4", "S5"):
        assert np.allclose(np.linalg.norm(imu[sid]["accel"], axis=-1), G, atol=1e-9)
        assert np.allclose(imu[sid]["gyro"], 0.0, atol=1e-12)


def test_single_axis_shoulder_rotation_gyro():
    """Constant-rate rotation about the shoulder-yaw (+z) axis: S2 reads that rate on body z."""
    rate = 1.7  # rad/s
    T = 50
    t = np.linspace(0, 1, T)
    q = np.zeros((T, 7))
    q[:, 0] = rate * t
    qd = np.zeros((T, 7))
    qd[:, 0] = rate
    qdd = np.zeros((T, 7))
    imu = human_arm_7dof().ideal_imu(q, qd, qdd)
    # only joint 0 (about z) moves; S2's frame shares that z, so gyro = [0,0,rate]
    assert np.allclose(imu["S2"]["gyro"][:, 2], rate, atol=1e-12)
    assert np.allclose(imu["S2"]["gyro"][:, :2], 0.0, atol=1e-12)


def test_gravity_direction_recovered_at_rest():
    """A single segment tilted 90° about y points its body x-axis down; accel reads +G on x."""
    arm = Arm3D(segments=(Segment(axis=(0, 1, 0), link=(0.3, 0, 0), name="seg",
                                  has_sensor=True, sensor_id="S"),),
                g_world=(0.0, 0.0, -G))
    q = np.array([[np.pi / 2]])  # rotate +90° about y: local +x now points to world -z (down)
    imu = arm.ideal_imu(q, np.zeros((1, 1)), np.zeros((1, 1)))
    acc = imu["S"]["accel"][0]
    # specific force at rest points opposite gravity (up); body +x points down, so it reads -G on x
    assert np.isclose(np.linalg.norm(acc), G, atol=1e-9)
    assert np.isclose(acc[0], -G, atol=1e-9)
    assert np.allclose(acc[1:], 0.0, atol=1e-9)
