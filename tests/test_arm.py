"""Tests for the planar 2-DOF arm: forward kinematics + ideal IMU invariants."""

import numpy as np

from sim.arm import G, PlanarArm, min_jerk


def test_forward_kinematics_straight_arm():
    arm = PlanarArm(l1=0.30, l2=0.25)
    fk = arm.forward_kinematics(0.0, 0.0)  # fully extended along +x
    assert np.allclose(fk["elbow"], [0.30, 0.0])
    assert np.allclose(fk["wrist"], [0.55, 0.0])
    assert np.allclose(fk["S2"], [0.15, 0.0])
    assert np.allclose(fk["S4"], [0.30 + 0.125, 0.0])


def test_forward_kinematics_right_angle_elbow():
    arm = PlanarArm(l1=1.0, l2=1.0)
    fk = arm.forward_kinematics(0.0, np.pi / 2)  # upper arm +x, forearm +y
    assert np.allclose(fk["elbow"], [1.0, 0.0])
    assert np.allclose(fk["wrist"], [1.0, 1.0], atol=1e-9)


def test_imu_at_rest_reads_gravity_and_zero_gyro():
    arm = PlanarArm()
    z = np.zeros(10)
    th1 = np.full(10, 0.7)  # arbitrary fixed pose
    th2 = np.full(10, -0.4)
    imu = arm.ideal_imu(th1, z, z, th2, z, z)
    for s in ("S2", "S4"):
        assert np.allclose(imu[s]["gyro_z"], 0.0)
        assert np.allclose(np.linalg.norm(imu[s]["accel"], axis=-1), G)


def test_min_jerk_endpoints_have_zero_velocity_and_accel():
    t = np.array([0.4, 1.2])  # the two endpoints of the window
    q, qd, qdd = min_jerk(t, 0.4, 1.2, 0.0, 1.0)
    assert np.allclose(q, [0.0, 1.0])
    assert np.allclose(qd, 0.0)
    assert np.allclose(qdd, 0.0)


def test_min_jerk_endpoints_imply_rest_imu():
    # at a min-jerk endpoint, velocity and accel are zero -> IMU sees rest (|accel|=G)
    arm = PlanarArm()
    t = np.array([1.2])
    th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, 0.1, 1.0)
    th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, 0.2, 1.5)
    imu = arm.ideal_imu(th1, th1d, th1dd, th2, th2d, th2dd)
    for s in ("S2", "S4"):
        assert np.allclose(imu[s]["gyro_z"], 0.0)
        assert np.allclose(np.linalg.norm(imu[s]["accel"], axis=-1), G)


def test_gyro_matches_angular_velocity_during_motion():
    arm = PlanarArm()
    t = np.linspace(0, 2, 1000)
    th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, 0.0, np.deg2rad(60))
    th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, 0.0, np.deg2rad(90))
    imu = arm.ideal_imu(th1, th1d, th1dd, th2, th2d, th2dd)
    assert np.allclose(imu["S2"]["gyro_z"], th1d)          # upper arm
    assert np.allclose(imu["S4"]["gyro_z"], th1d + th2d)   # forearm (absolute)
