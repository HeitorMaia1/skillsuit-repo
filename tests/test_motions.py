"""Tests for the motion-class library (task 3.6)."""

import numpy as np
import pytest

from sim.arm3d import human_arm_7dof
from sim.motions import MOTION_CLASSES, generate_trial
from sim.sensor import SensorModel
from skilldata.encoder import validate
from skilldata.ingest import Arm3DSimAdapter


def test_each_class_shapes_and_rest_endpoints():
    for mc in MOTION_CLASSES:
        t, q, qd, qdd = generate_trial(mc, fs=200.0, rng=np.random.default_rng(0))
        assert q.shape == (t.size, 7) and qd.shape == (t.size, 7) and qdd.shape == (t.size, 7)
        assert np.allclose(qd[0], 0.0) and np.allclose(qd[-1], 0.0)      # starts/ends at rest
        assert np.allclose(qdd[0], 0.0) and np.allclose(qdd[-1], 0.0)


def test_reproducible_with_seed():
    a = generate_trial("reach", rng=np.random.default_rng(7))
    b = generate_trial("reach", rng=np.random.default_rng(7))
    assert np.array_equal(a[1], b[1])


def test_classes_are_distinct():
    _, q_reach, qd_reach, _ = generate_trial("reach", rng=np.random.default_rng(0))
    _, q_wrist, _, _ = generate_trial("wrist_rotate", rng=np.random.default_rng(0))
    _, _, qd_throw, _ = generate_trial("throw", rng=np.random.default_rng(0))
    # wrist_rotate uses forearm pronation (joint 4); reach does not
    assert np.abs(q_wrist[:, 4]).max() > 0.5
    assert np.abs(q_reach[:, 4]).max() < 1e-9
    # throw is faster than reach
    assert np.abs(qd_throw).max() > np.abs(qd_reach).max()


def test_unknown_class_raises():
    with pytest.raises(ValueError):
        generate_trial("cartwheel")


def _record(mc, rng):
    arm = human_arm_7dof()
    t, q, qd, qdd = generate_trial(mc, fs=500.0, rng=rng)
    return Arm3DSimAdapter(arm, t, q, qd, qdd, sample_rate_hz=500.0,
                           sensor_model=SensorModel(seed=0)).to_base_layer(
        subject_id="S001", motion_class=mc, trial_index=0)


def test_throw_saturates_reach_does_not():
    throw = _record("throw", np.random.default_rng(0))
    reach = _record("reach", np.random.default_rng(0))
    # the fast throw pins the wrist gyro (S5) past ±2000°/s (D2 envelope); reach never does
    assert any(throw["imu_streams"]["S5"]["saturation_flag"])
    assert not any(reach["imu_streams"]["S5"]["saturation_flag"])


def test_adapter_sets_motion_class_and_validates():
    for mc in MOTION_CLASSES:
        rec = _record(mc, np.random.default_rng(1))
        assert validate(rec) is True
        assert rec["session"]["motion_class"] == mc
        assert set(rec["phase_labels"]) <= {"prep", "active", "settle"}
