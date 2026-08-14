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


# --------------------------------------------------------------------------- #
# The excitation class (task 3.15)
# --------------------------------------------------------------------------- #
def test_excitation_is_not_a_member_of_motion_classes():
    """Load-bearing separation, not cosmetic.

    `MOTION_CLASSES` drives `_class_counts()` in both dataset generators, so a fifth member would
    change `_class_counts(1000, ...)` from [250]*4 to [200]*5, change the RNG draw order, and
    silently break the dynamics slice's index alignment with the committed SkillData v1 dataset —
    invalidating the Phase 4 fusion numbers measured on it.
    """
    from sim.motions import ALL_CLASSES, EXCITATION_CLASS, MOTION_CLASSES

    assert MOTION_CLASSES == ("reach", "lift", "wrist_rotate", "throw")
    assert EXCITATION_CLASS not in MOTION_CLASSES
    assert ALL_CLASSES == MOTION_CLASSES + (EXCITATION_CLASS,)


def test_excitation_drives_every_joint():
    """The single property the naturalistic classes lack, and the whole point of the class."""
    from sim.motions import EXCITATION_CLASS

    _t, _q, qd, _qdd = generate_trial(EXCITATION_CLASS, fs=500.0, rng=np.random.default_rng(0))
    rms = np.sqrt((qd**2).mean(axis=0))
    assert (rms > 0.1).all(), f"every joint must move; got {rms.round(3)}"
    assert rms.max() / rms.min() < 5.0, (
        f"excitation must be roughly balanced across joints (the naturalistic classes are 43x "
        f"skewed); got {rms.max() / rms.min():.1f}x"
    )


def test_excitation_derivatives_are_analytic():
    """qd and qdd must be the exact derivatives, matching the exactness the rest of `sim` promises."""
    from sim.motions import EXCITATION_CLASS

    fs = 2000.0
    _t, q, qd, qdd = generate_trial(EXCITATION_CLASS, fs=fs, rng=np.random.default_rng(1))
    dt = 1.0 / fs
    assert np.abs(np.gradient(q, dt, axis=0)[5:-5] - qd[5:-5]).max() / np.abs(qd).max() < 1e-5
    assert np.abs(np.gradient(qd, dt, axis=0)[5:-5] - qdd[5:-5]).max() / np.abs(qdd).max() < 1e-5


def test_excitation_is_reproducible_and_varies_between_trials():
    from sim.motions import EXCITATION_CLASS

    a = generate_trial(EXCITATION_CLASS, rng=np.random.default_rng(3))[1]
    b = generate_trial(EXCITATION_CLASS, rng=np.random.default_rng(3))[1]
    assert np.array_equal(a, b), "same seed must give the same trial"
    rng = np.random.default_rng(4)
    c = generate_trial(EXCITATION_CLASS, rng=rng)[1]
    d = generate_trial(EXCITATION_CLASS, rng=rng)[1]
    assert not np.allclose(c, d), "consecutive draws must differ, or the trials do not span space"


def test_excitation_does_not_start_or_end_at_rest():
    """Documented difference from the naturalistic classes — anything assuming rest must exclude it."""
    from sim.motions import EXCITATION_CLASS

    _t, _q, qd, _qdd = generate_trial(EXCITATION_CLASS, rng=np.random.default_rng(5))
    assert np.abs(qd[0]).max() > 1e-3
    assert np.abs(qd[-1]).max() > 1e-3


def test_excitation_parameters_are_honoured():
    from sim.motions import EXCITATION_CLASS, excitation_trial

    t, _q, qd, _qdd = excitation_trial(fs=100.0, rng=np.random.default_rng(6), dur=2.0)
    assert t.size == 200
    small = excitation_trial(fs=200.0, rng=np.random.default_rng(7), amp=0.05)[2]
    large = excitation_trial(fs=200.0, rng=np.random.default_rng(7), amp=0.50)[2]
    assert np.abs(large).max() > 5 * np.abs(small).max()
    # dispatch through generate_trial carries the keywords
    t2 = generate_trial(EXCITATION_CLASS, fs=100.0, rng=np.random.default_rng(6), dur=2.0)[0]
    assert np.array_equal(t, t2)


def test_excitation_rejects_nonsense_parameters():
    from sim.motions import excitation_trial

    with pytest.raises(ValueError, match="n_harm"):
        excitation_trial(n_harm=0)
    for bad in ({"dur": 0.0}, {"f_base": -1.0}, {"amp": 0.0}):
        with pytest.raises(ValueError, match="positive"):
            excitation_trial(**bad)


def test_naturalistic_classes_reject_excitation_keywords():
    with pytest.raises(TypeError, match="takes no extra parameters"):
        generate_trial("reach", n_harm=3)


def test_unknown_class_names_the_full_set():
    with pytest.raises(ValueError, match="excite"):
        generate_trial("cartwheel")
