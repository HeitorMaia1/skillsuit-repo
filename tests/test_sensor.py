"""Tests for the sensor realism model (task 3.5): noise, bias drift, saturation, reproducibility."""

import numpy as np

from sim.arm import min_jerk
from sim.arm3d import human_arm_7dof
from sim.sensor import SensorModel, apply_to_streams
from skilldata.encoder import validate
from skilldata.ingest import Arm3DSimAdapter


def test_white_noise_statistics():
    m = SensorModel(gyro_noise_dps=5.0, accel_noise_g=0.02, gyro_range_dps=1e9, accel_range_g=1e9, seed=0)
    n = 5000
    gyro, accel, sat = m.corrupt_stream(np.zeros((n, 3)), np.zeros((n, 3)), dt=0.002)
    assert abs(gyro.mean()) < 0.3
    assert abs(gyro.std() - 5.0) < 0.4
    assert abs(accel.std() - 0.02) < 0.003
    assert not sat.any()               # huge range -> never saturates


def test_bias_random_walk_spreads_over_time():
    m = SensorModel(gyro_noise_dps=0.0, accel_noise_g=0.0,
                    gyro_bias_drift_dps_per_sqrt_s=40.0, gyro_range_dps=1e9, seed=1)
    n = 4000
    gyro, _, _ = m.corrupt_stream(np.zeros((n, 3)), np.zeros((n, 3)), dt=0.005)
    early = gyro[:500].var()
    late = gyro[-500:].var()
    assert late > 3 * early            # random walk variance grows with time


def test_saturation_clips_and_flags():
    m = SensorModel(gyro_noise_dps=0.0, accel_noise_g=0.0, gyro_range_dps=2000.0, accel_range_g=16.0)
    gyro_in = np.zeros((10, 3))
    gyro_in[:, 0] = 2500.0                                  # above the ±2000°/s D2 range
    accel_in = np.zeros((10, 3))
    accel_in[:, 2] = 20.0                                   # above ±16 g
    gyro, accel, sat = m.corrupt_stream(gyro_in, accel_in, dt=0.002)
    assert np.allclose(gyro[:, 0], 2000.0)                  # clipped
    assert sat.all()                                        # every sample flagged
    assert np.allclose(accel[:, 2], 16.0)
    # a within-range signal is untouched and unflagged
    _, _, sat2 = m.corrupt_stream(np.full((5, 3), 1500.0), np.zeros((5, 3)), dt=0.002)
    assert not sat2.any()


def test_reproducible_with_seed():
    a = SensorModel(gyro_noise_dps=3.0, seed=42).corrupt_stream(np.zeros((100, 3)), np.zeros((100, 3)), 0.002)
    b = SensorModel(gyro_noise_dps=3.0, seed=42).corrupt_stream(np.zeros((100, 3)), np.zeros((100, 3)), 0.002)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_apply_to_streams_adds_flag_and_perturbs():
    clean = {"S4": {"timestamp_us": [0, 1, 2],
                    "angular_velocity_dps": [[0, 0, 10.0]] * 3,
                    "linear_accel_g": [[0, 0, 1.0]] * 3}}
    noisy = apply_to_streams(clean, SensorModel(gyro_noise_dps=2.0, seed=0), dt=0.002)
    assert "saturation_flag" in noisy["S4"] and len(noisy["S4"]["saturation_flag"]) == 3
    assert noisy["S4"]["angular_velocity_dps"] != clean["S4"]["angular_velocity_dps"]  # perturbed


def test_adapter_with_sensor_model_validates():
    arm = human_arm_7dof()
    t = np.arange(0.0, 1.0, 1.0 / 200)
    q = np.zeros((t.size, 7))
    qd = np.zeros((t.size, 7))
    qdd = np.zeros((t.size, 7))
    q[:, 3], qd[:, 3], qdd[:, 3] = min_jerk(t, 0.2, 0.8, 0.2, 1.2)
    rec = Arm3DSimAdapter(arm, t, q, qd, qdd, sample_rate_hz=200.0,
                          sensor_model=SensorModel(seed=0)).to_base_layer(
        subject_id="S001", motion_class="reach3d", trial_index=0)
    assert validate(rec) is True
    for s in rec["imu_streams"].values():
        assert len(s["saturation_flag"]) == t.size
