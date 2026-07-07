"""Tests for the source-agnostic ingest layer (task 3.11)."""

import numpy as np

from sim.arm import PlanarArm, min_jerk
from sim.arm3d import human_arm_7dof
from skilldata.encoder import build_record, validate
from skilldata.ingest import (
    Arm3DSimAdapter,
    GenericIMUAdapter,
    PlanarSimAdapter,
    assemble_base_layer,
)


def _planar_reach(fs=500.0):
    arm = PlanarArm()
    t = np.arange(0.0, 2.0, 1.0 / fs)
    j1 = min_jerk(t, 0.4, 1.2, np.deg2rad(10), np.deg2rad(60))
    j2 = min_jerk(t, 0.4, 1.2, np.deg2rad(20), np.deg2rad(90))
    return arm, t, j1, j2, fs


def _arm3d_reach(fs=500.0):
    arm = human_arm_7dof()
    t = np.arange(0.0, 2.0, 1.0 / fs)
    q = np.zeros((t.size, 7))
    qd = np.zeros((t.size, 7))
    qdd = np.zeros((t.size, 7))
    for j, (a0, a1) in {0: (0.0, 0.5), 1: (0.0, 0.9), 3: (0.2, 1.3), 4: (0.0, 0.8)}.items():
        q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, 0.4, 1.2, a0, a1)
    return arm, t, q, qd, qdd, fs


# --- contract --------------------------------------------------------------- #
def test_assemble_base_layer_infers_n_and_validates():
    n = 5
    streams = {"S2": {"timestamp_us": list(range(n)),
                      "angular_velocity_dps": [[0, 0, 0]] * n,
                      "linear_accel_g": [[0, 0, 1]] * n}}
    rec = assemble_base_layer(subject_id="S001", motion_class="reach", trial_index=0,
                              sample_rate_hz=500.0, imu_streams=streams)
    assert rec["session"]["n_samples"] == n
    assert "retarget" not in rec           # base layer only
    assert validate(rec) is True


# --- reference sim adapters ------------------------------------------------- #
def test_planar_adapter_matches_build_record():
    arm, t, j1, j2, fs = _planar_reach()
    ref = build_record(arm, t, *j1, *j2, subject_id="S001", motion_class="reach",
                       trial_index=0, sample_rate_hz=fs)
    got = PlanarSimAdapter(arm, t, *j1, *j2, sample_rate_hz=fs).to_base_layer(
        subject_id="S001", motion_class="reach", trial_index=0)
    assert got == ref                       # adapter is the reference planar source
    assert validate(got) is True


def test_arm3d_adapter_native_3axis_and_validates():
    arm, t, q, qd, qdd, fs = _arm3d_reach()
    rec = Arm3DSimAdapter(arm, t, q, qd, qdd, sample_rate_hz=fs).to_base_layer(
        subject_id="S001", motion_class="reach3d", trial_index=0)
    assert validate(rec) is True
    assert set(rec["imu_streams"]) == {"S2", "S4", "S5"}
    for s in rec["imu_streams"].values():
        assert np.shape(s["angular_velocity_dps"]) == (t.size, 3)   # genuine 3-axis
        assert np.shape(s["linear_accel_g"]) == (t.size, 3)
    # first sample is at rest -> |accel| == 1 g, gyro == 0
    a0 = np.array(rec["imu_streams"]["S4"]["linear_accel_g"][0])
    g0 = np.array(rec["imu_streams"]["S4"]["angular_velocity_dps"][0])
    assert np.isclose(np.linalg.norm(a0), 1.0, atol=1e-6)
    assert np.allclose(g0, 0.0, atol=1e-6)
    assert len(rec["segment_kinematics"]["joint_angles_rad"]) == 7   # 7 DOF


# --- generic (non-sim) device: the template for real hardware --------------- #
def test_generic_adapter_packs_external_device():
    n = 30
    rng = np.random.default_rng(0)
    ts = (np.arange(n) * 2000).astype(int)          # 500 Hz in microseconds
    gyro = rng.normal(0, 50, (n, 3))
    accel = rng.normal(0, 0.05, (n, 3)) + np.array([0, 0, 1.0])  # ~1 g downward
    quat = np.tile([1.0, 0, 0, 0], (n, 1))
    sat = np.zeros(n, bool)
    adapter = GenericIMUAdapter(
        500.0,
        {"S0": {"timestamp_us": ts, "angular_velocity_dps": gyro, "linear_accel_g": accel,
                "quaternion": quat, "saturation_flag": sat}},
        source_name="xsens_like", source="hardware",
    )
    rec = adapter.to_base_layer(subject_id="S001", motion_class="reach", trial_index=3)
    assert validate(rec) is True
    assert rec["session"]["source"] == "hardware"
    assert rec["session"]["n_samples"] == n
    assert len(rec["imu_streams"]["S0"]["quaternion"]) == n
    assert rec["imu_streams"]["S0"]["saturation_flag"][0] is False
