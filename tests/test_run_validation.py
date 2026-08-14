"""Tests for the fusion validation pipeline (task 4.5).

Two things need guarding here, and they are different in kind:

1. **The scoring machinery itself** — ground-truth reconstruction, the two error metrics, and
   the accelerometer-based initialiser. These are checked against closed-form answers, because
   a validation harness that silently mis-scores would corrupt every number in the paper and
   would do it *quietly*: the filters would still run, the figures would still render, and the
   RMS column would simply be wrong.
2. **The end-to-end run** — that `run()` produces every key the notebook (task 4.6) reads, on a
   tiny subsample. This is a contract test between the two deliverables.

The heavy full-dataset pass is not run here (it takes ~20 minutes); `--limit-per-class` exists
precisely so the same code path can be exercised cheaply.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from fusion.madgwick import quat_angle_error_deg, quat_from_axis_angle
from fusion.run_validation import (
    CLASSES,
    PHASES,
    SENSOR_SEGMENT,
    evaluate,
    gravity_dirs,
    ground_truth,
    q0_from_accel,
    run,
    tilt_error_deg,
)

DATA = "data/processed"


def _has_dataset():
    from pathlib import Path
    return (Path(DATA) / "manifest.json").exists()


needs_dataset = pytest.mark.skipif(
    not _has_dataset(), reason="synthetic dataset not generated (run `make synthetic`)")


# --------------------------------------------------------------------------- #
# The metrics
# --------------------------------------------------------------------------- #
def test_gravity_dirs_matches_the_scalar_implementation():
    """The batched form must agree with ``fusion.madgwick``'s scalar one, which carries the
    derivation. They are separate code, so this is a real cross-check, not a tautology."""
    from fusion.madgwick import predicted_gravity_direction

    rng = np.random.default_rng(0)
    qs = rng.normal(size=(20, 4))
    qs /= np.linalg.norm(qs, axis=-1, keepdims=True)
    expected = np.array([predicted_gravity_direction(q) for q in qs])
    assert np.allclose(gravity_dirs(qs), expected, atol=1e-12)


def test_tilt_error_is_zero_for_pure_heading_offset():
    """A yaw-only difference must register as *total* error but **zero tilt** error.

    This is the property the whole two-metric split rests on: heading is the channel a
    gyro+accel pair cannot observe, so a filter that is only wrong in heading must be scored
    as having perfect tilt. If this ever failed, the `accel`-init column would be meaningless.
    """
    q_ref = quat_from_axis_angle((0.0, 0.0, 1.0), 0.0)
    for deg in (5.0, 40.0, 120.0):
        q_yaw = quat_from_axis_angle((0.0, 0.0, 1.0), np.deg2rad(deg))
        assert tilt_error_deg(q_yaw[None], q_ref[None])[0] < 1e-9
        assert quat_angle_error_deg(q_yaw, q_ref) == pytest.approx(deg, abs=1e-9)


def test_tilt_error_equals_total_for_pure_tilt_offset():
    """Conversely, a rotation purely about a horizontal axis is fully observable, so the two
    metrics must agree — small angles only, since tilt saturates at 180° differently."""
    q_ref = quat_from_axis_angle((1.0, 0.0, 0.0), 0.0)
    for deg in (3.0, 25.0):
        q_tilt = quat_from_axis_angle((1.0, 0.0, 0.0), np.deg2rad(deg))
        assert tilt_error_deg(q_tilt[None], q_ref[None])[0] == pytest.approx(deg, abs=1e-9)


# --------------------------------------------------------------------------- #
# Initialisation
# --------------------------------------------------------------------------- #
def test_q0_from_accel_recovers_tilt_and_leaves_heading_at_zero():
    """Seeded from a tilted gravity reading, the initial estimate must have zero *tilt* error
    against the truth, and whatever heading error follows is unknowable, not a bug."""
    for axis, deg in (((1.0, 0.0, 0.0), 30.0), ((0.0, 1.0, 0.0), -20.0), ((1.0, 1.0, 0.0), 12.0)):
        q_true = quat_from_axis_angle(axis, np.deg2rad(deg))
        accel = gravity_dirs(q_true[None])[0]        # what a stationary sensor would read
        q0 = np.asarray(q0_from_accel(accel))
        assert tilt_error_deg(q0[None], q_true[None])[0] < 1e-8


def test_q0_from_accel_handles_degenerate_readings():
    """Free fall (zero reading) and an exactly-aligned reading must not divide by zero."""
    assert q0_from_accel([0.0, 0.0, 0.0]) == (1.0, 0.0, 0.0, 0.0)
    assert q0_from_accel([0.0, 0.0, 1.0]) == (1.0, 0.0, 0.0, 0.0)
    assert abs(np.linalg.norm(q0_from_accel([0.0, 0.0, -1.0])) - 1.0) < 1e-12


# --------------------------------------------------------------------------- #
# Ground truth
# --------------------------------------------------------------------------- #
@needs_dataset
def test_ground_truth_is_unit_quaternions_for_every_sensor():
    from pathlib import Path

    rec = json.loads((Path(DATA) / "S001_lift_clean_0000.json").read_text())
    truth = ground_truth(rec)
    assert set(truth) == set(SENSOR_SEGMENT)
    for sid, q in truth.items():
        assert q.shape == (rec["session"]["n_samples"], 4), sid
        assert np.abs(np.linalg.norm(q, axis=-1) - 1.0).max() < 1e-9


@needs_dataset
def test_ground_truth_reproduces_the_clean_accelerometer_at_rest():
    """The strongest available check on the truth reconstruction: at the very first sample
    every trial is at rest, so the clean accelerometer *is* the gravity direction, and the
    reconstructed orientation must predict it. This catches a transposed or mis-ordered
    rotation, which would otherwise produce a plausible-looking but wrong RMS column."""
    from pathlib import Path

    for mc in CLASSES:
        rec = json.loads((Path(DATA) / f"S001_{mc}_clean_0000.json").read_text())
        truth = ground_truth(rec)
        for sid, stream in rec["imu_streams"].items():
            a = np.asarray(stream["linear_accel_g"], float)[0]
            a = a / np.linalg.norm(a)
            pred = gravity_dirs(truth[sid][:1])[0]
            assert np.allclose(pred, a, atol=1e-3), f"{mc}/{sid}: {pred} vs {a}"


# --------------------------------------------------------------------------- #
# Scoring and the end-to-end contract with the notebook
# --------------------------------------------------------------------------- #
@needs_dataset
def test_evaluate_returns_every_aggregate_and_tilt_never_exceeds_total():
    from pathlib import Path

    files = sorted(Path(DATA).glob("S001_lift_noisy_000[01].json"))
    res = evaluate(files, "madgwick")
    assert res["by_class"]["lift"]["n"] == sum(
        json.loads(p.read_text())["session"]["n_samples"] * 3 for p in files)
    assert set(res["by_phase"]["lift"]) <= set(PHASES)
    assert set(res["by_sensor"]["lift"]) == set(SENSOR_SEGMENT)
    assert len(res["trial_rms"]["lift"]) == len(files)
    # tilt is a component of the total rotation error, so it can never be the larger of the two
    assert res["overall"]["tilt_rms"] <= res["overall"]["rms"] + 1e-9


@needs_dataset
def test_run_end_to_end_writes_every_key_the_notebook_reads(tmp_path):
    """Contract test against `notebooks/01_fusion_validation.ipynb` (task 4.6)."""
    res = run(DATA, tmp_path, limit_per_class=1, noise_trials=1, hold_s=2.0, figures=False)
    assert set(res["runs"]) == {"truth", "accel"}
    for runs in res["runs"].values():
        for key, r in runs.items():
            assert r["overall"]["rms"] >= 0, key
            for c in CLASSES:
                assert "tilt_rms" in r["by_class"][c]
    assert set(res["convergence"]) == set(CLASSES)
    assert res["long_horizon"]["filters"]["ekf"]["final"] >= 0
    assert len(res["noise_sensitivity"]["scales"]) >= 3

    saved = json.loads((tmp_path / "fusion_validation.json").read_text())
    assert saved["runs"]["truth"]["ekf"]["by_class"]["lift"]["rms"] == pytest.approx(
        res["runs"]["truth"]["ekf"]["by_class"]["lift"]["rms"])


@needs_dataset
def test_figures_are_written(tmp_path):
    run(DATA, tmp_path, limit_per_class=1, noise_trials=1, hold_s=2.0, figures=True)
    names = {p.name for p in tmp_path.glob("fusion_*.pdf")}
    assert names == {
        "fusion_rms_by_class.pdf", "fusion_rms_by_phase.pdf", "fusion_convergence.pdf",
        "fusion_ablation.pdf", "fusion_long_horizon.pdf", "fusion_noise_sensitivity.pdf",
        "fusion_tilt_vs_heading.pdf",
    }


# --------------------------------------------------------------------------- #
# The notebook's prose, guarded against the committed results (task 4.6)
# --------------------------------------------------------------------------- #
# `notebooks/01_fusion_validation.ipynb` states conclusions in words. Words do not recompute when
# the dataset or a filter changes, so every load-bearing sentence is asserted here against the
# committed `paper/figures/fusion_validation.json` that the notebook itself reads. Two of these
# started life as *false* claims written against a partial run (2026-08-14): "tilt is essentially
# unchanged between initialisations" (untrue for the gyro-only control, +8.0%) and "throw is the
# worst class for every filter" (untrue for `EKF, static R`, whose worst class is `reach`). Both
# are now stated correctly and pinned here.

RESULTS_JSON = Path("paper/figures/fusion_validation.json")
needs_results = pytest.mark.skipif(
    not RESULTS_JSON.exists(),
    reason="paper/figures/fusion_validation.json not generated (make fusion)",
)


@pytest.fixture(scope="module")
def results():
    return json.loads(RESULTS_JSON.read_text())


def _v(run, key, cls, metric="rms"):
    node = run[key]["overall"] if cls == "overall" else run[key]["by_class"][cls]
    return node[metric]


@needs_results
def test_claim_ekf_beats_madgwick_on_every_motion_class(results):
    truth = results["runs"]["truth"]
    for c in CLASSES:
        assert _v(truth, "ekf", c) < _v(truth, "madgwick", c), c


@needs_results
def test_claim_the_margin_is_deterministic_on_lift_and_marginal_on_reach(results):
    """Section 1b: the win is significant everywhere, consistent only where the accel is misled."""
    truth = results["runs"]["truth"]
    win = {c: float((np.array(truth["ekf"]["trial_rms"][c])
                     < np.array(truth["madgwick"]["trial_rms"][c])).mean())
           for c in CLASSES}
    assert win["lift"] == 1.0, "lift is claimed as 100% of trials"
    assert win["wrist_rotate"] > 0.98
    assert 0.55 < win["reach"] < 0.70, "reach is claimed as marginal, not decisive"
    assert win["reach"] < win["lift"] and win["reach"] < win["wrist_rotate"]


@needs_results
def test_claim_gyro_only_beats_madgwick_except_on_throw(results):
    """The control that makes the result interpretable: Madgwick's accel correction adds error."""
    truth = results["runs"]["truth"]
    for c in CLASSES:
        better = _v(truth, "gyro_only", c) < _v(truth, "madgwick", c)
        assert better == (c != "throw"), c


@needs_results
def test_claim_the_bias_state_is_not_what_produces_the_ekfs_advantage(results):
    truth = results["runs"]["truth"]
    assert abs(_v(truth, "ekf", "overall") - _v(truth, "ekf_no_bias", "overall")) < 0.1
    assert _v(truth, "ekf_static_r", "overall") / _v(truth, "madgwick", "overall") >= 8.0


@needs_results
def test_claim_initialisation_moves_only_the_unobservable_channel(results):
    """True for every filter that uses the accelerometer — and pointedly false for the one that does not."""
    truth, accel = results["runs"]["truth"], results["runs"]["accel"]
    for key in ("madgwick", "ekf", "ekf_no_bias", "ekf_static_r"):
        t = _v(truth, key, "overall", "tilt_rms")
        a = _v(accel, key, "overall", "tilt_rms")
        assert abs(a - t) / t < 0.03, key
    t = _v(truth, "gyro_only", "overall", "tilt_rms")
    a = _v(accel, "gyro_only", "overall", "tilt_rms")
    assert 0.05 < (a - t) / t < 0.11, "gyro-only is the documented exception, ~+8%"


@needs_results
def test_claim_static_r_fails_in_heading_not_tilt(results):
    truth = results["runs"]["truth"]
    assert _v(truth, "ekf_static_r", "overall", "tilt_rms") < 5.0
    assert _v(truth, "ekf_static_r", "overall") > 20.0


@needs_results
def test_claim_throw_is_worst_for_every_filter_except_static_r(results):
    """And that the exception is `reach`, which is what identifies the leak as correction-count driven."""
    truth = results["runs"]["truth"]
    for key in ("madgwick", "ekf", "gyro_only", "ekf_no_bias"):
        assert max(CLASSES, key=lambda c: _v(truth, key, c)) == "throw", key
    assert max(CLASSES, key=lambda c: _v(truth, "ekf_static_r", c)) == "reach"
    assert _v(truth, "ekf_static_r", "reach") / _v(truth, "ekf_static_r", "throw") > 2.5


@needs_results
def test_committed_results_cover_the_whole_dataset(results):
    """The committed JSON must be a full run, not a --limit-per-class sample."""
    assert results["dataset"]["n_noisy_records"] == 1000
    for init in ("truth", "accel"):
        for key in ("madgwick", "ekf", "gyro_only", "ekf_no_bias", "ekf_static_r"):
            assert len(results["runs"][init][key]["trial_rms"]["reach"]) == 250
