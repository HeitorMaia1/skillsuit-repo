"""Tests for the dynamics ground-truth slice (task 3.14)."""

import json
from pathlib import Path

import numpy as np
import pytest

from skilldata.generate_dynamics import (
    DEFAULT_DAMPING,
    FORMAT_VERSION,
    generate_dynamics_dataset,
    main,
)
from sim.motions import MOTION_CLASSES

V1_DIR = Path("data/processed")
JOINT_ORDER = ["sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
               "forearm_pron", "wrist_flex", "wrist_dev_hand"]


@pytest.fixture(scope="module")
def slice_(tmp_path_factory):
    out = tmp_path_factory.mktemp("dyn")
    manifest = generate_dynamics_dataset(out, per_class=2, decimate=20)
    return out, manifest


def test_writes_one_file_per_trial_plus_a_manifest(slice_):
    out, manifest = slice_
    assert manifest["format_version"] == FORMAT_VERSION
    assert manifest["n_trials"] == 2 * len(MOTION_CLASSES)
    assert len(list(out.glob("*.npz"))) == manifest["n_trials"]
    assert (out / "manifest.json").exists()
    for entry in manifest["trials"]:
        assert (out / entry["file"]).exists()


def test_every_documented_field_is_present_and_finite(slice_):
    out, manifest = slice_
    for entry in manifest["trials"]:
        d = np.load(out / entry["file"])
        for key in ("t", "q", "qd", "qdd", "p", "tau", "H", "T_kin", "U",
                    "power_dissipated", "lambda_min"):
            assert key in d, f"{key} missing from {entry['file']}"
            assert np.isfinite(d[key]).all(), f"{key} has non-finite values"
        T = d["t"].size
        assert d["q"].shape == (T, 7) and d["tau"].shape == (T, 7) and d["p"].shape == (T, 7)
        assert d["H"].shape == (T,) and d["lambda_min"].shape == (T,)
        assert T == entry["n_samples"]


def test_the_true_damping_is_recorded_in_the_manifest(slice_):
    """The whole point of task 3.14: `B` must be written down, or 5.9 has nothing to score against."""
    _out, manifest = slice_
    B = manifest["true_damping"]["B_diagonal_Nms_per_rad"]
    assert B == list(DEFAULT_DAMPING)
    assert manifest["true_damping"]["joint_order"] == JOINT_ORDER
    assert len(B) == 7 and all(b >= 0 for b in B)


def test_damping_override_reaches_the_manifest_and_the_data(tmp_path):
    B = [0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
    manifest = generate_dynamics_dataset(tmp_path, per_class=1, decimate=40, damping=tuple(B))
    assert manifest["true_damping"]["B_diagonal_Nms_per_rad"] == B
    d = np.load(tmp_path / manifest["trials"][0]["file"])
    expected = np.einsum("ta,a,ta->t", d["qd"], np.array(B, np.float32), d["qd"])
    assert np.allclose(d["power_dissipated"], expected, rtol=1e-4, atol=1e-7)


def test_power_dissipated_is_exactly_qd_B_qd(slice_):
    out, manifest = slice_
    B = np.array(manifest["true_damping"]["B_diagonal_Nms_per_rad"], np.float32)
    for entry in manifest["trials"]:
        d = np.load(out / entry["file"])
        expected = np.einsum("ta,a,ta->t", d["qd"], B, d["qd"])
        assert np.allclose(d["power_dissipated"], expected, rtol=1e-4, atol=1e-7)
        assert (d["power_dissipated"] >= 0).all(), "B >= 0 can only remove energy"


def test_trials_start_and_end_at_rest_so_dissipation_vanishes_there(slice_):
    """Min-jerk endpoints have zero velocity: momentum, kinetic energy and dissipation all vanish."""
    out, manifest = slice_
    for entry in manifest["trials"]:
        d = np.load(out / entry["file"])
        for k in (0, -1):
            assert np.abs(d["qd"][k]).max() < 1e-6
            assert np.abs(d["p"][k]).max() < 1e-6
            assert abs(d["T_kin"][k]) < 1e-9
            assert abs(d["power_dissipated"][k]) < 1e-9
            assert d["H"][k] == pytest.approx(d["U"][k], abs=1e-6)


def test_energy_decomposition_is_consistent(slice_):
    out, manifest = slice_
    for entry in manifest["trials"]:
        d = np.load(out / entry["file"])
        assert np.allclose(d["H"], d["T_kin"] + d["U"], rtol=1e-5, atol=1e-5)
        assert (d["T_kin"] >= -1e-9).all()
        assert (d["lambda_min"] > 0).all(), "M must stay positive definite along every trial"


def test_labels_match_a_fresh_computation_from_the_dynamics_module(slice_):
    """The stored arrays must be what `sim.dynamics` produces — not a stale or reshaped copy."""
    from sim.dynamics import human_arm_7dof_dynamics

    out, manifest = slice_
    dyn = human_arm_7dof_dynamics(damping=DEFAULT_DAMPING)
    d = np.load(out / manifest["trials"][0]["file"])
    k = d["t"].size // 2
    q, qd, qdd = d["q"][k].astype(float), d["qd"][k].astype(float), d["qdd"][k].astype(float)
    assert np.allclose(d["tau"][k], dyn.inverse_dynamics(q, qd, qdd), rtol=1e-4, atol=1e-4)
    assert np.allclose(d["p"][k], dyn.momentum(q, qd), rtol=1e-4, atol=1e-5)
    assert d["H"][k] == pytest.approx(dyn.energy(q, qd), rel=1e-4, abs=1e-4)


def test_throw_is_the_most_energetic_class(slice_):
    """Sanity that the labels track the physics: a throw dissipates more than a wrist rotation."""
    _out, manifest = slice_
    per = manifest["per_class"]
    assert per["throw"]["peak_power_dissipated"] > per["wrist_rotate"]["peak_power_dissipated"]
    assert per["throw"]["peak_abs_tau"] > per["reach"]["peak_abs_tau"]


def test_manifest_carries_the_conditioning_warning_and_the_schema_note(slice_):
    _out, manifest = slice_
    assert "gimbal" in manifest["conditioning_warning"]
    assert "lambda_min" in manifest["conditioning_warning"]
    assert "NOT modified" in manifest["schema_note"]
    assert "de Leva" in manifest["anthropometry"]["source"]
    assert "10.1016/0021-9290(95)00178-6" in manifest["anthropometry"]["source"]


def test_skilldata_v1_schema_is_untouched():
    """Task 3.14 must not mutate the locked contract."""
    from skilldata import SCHEMA_VERSION

    schema = json.loads((Path("skilldata") / "schema.json").read_text())
    assert SCHEMA_VERSION == "skilldata-v1"
    assert schema["properties"]["schema_version"]["const"] == "skilldata-v1"
    for banned in ("tau", "momentum", "hamiltonian", "damping"):
        assert banned not in schema["properties"], f"{banned} leaked into the v1 schema"


def test_per_class_larger_than_alignment_is_rejected():
    with pytest.raises(ValueError, match="exceeds"):
        generate_dynamics_dataset("/tmp/never_written", per_class=300, align_to_n=1000)


def test_cli_runs(tmp_path, capsys):
    manifest = main(["--out", str(tmp_path), "--per-class", "1", "--decimate", "40"])
    assert manifest["n_trials"] == 4
    assert "true B (diag)" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# the alignment claim, checked against the real v1 dataset when it is present
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not (V1_DIR / "manifest.json").exists(),
                    reason="SkillData v1 dataset not generated (make synthetic)")
def test_trial_indices_align_with_the_skilldata_v1_dataset(slice_):
    """A trial `(motion_class, trial_index)` here must be the SAME motion as v1's.

    This is the claim the manifest makes, and Phase 5 will join on it, so it is checked against
    the actual shipped records rather than asserted. The residual is v1's serialization rounding
    to 6 decimals (half-ulp 5e-7), not a difference in the motion.
    """
    out, manifest = slice_
    v1 = json.loads((V1_DIR / "manifest.json").read_text())
    if v1["n_base_trials"] != manifest["generator"]["align_to_n"] \
       or v1["generator"]["seed"] != manifest["generator"]["seed"]:
        pytest.skip("shipped v1 dataset was generated with different seed/n — alignment not claimed")

    dec = manifest["generator"]["decimate"]
    worst = 0.0
    for entry in manifest["trials"]:
        mc, idx = entry["motion_class"], entry["trial_index"]
        rec_path = V1_DIR / f"S001_{mc}_clean_{idx:04d}.json"
        if not rec_path.exists():
            pytest.skip(f"{rec_path.name} not present")
        ja = json.loads(rec_path.read_text())["segment_kinematics"]["joint_angles_rad"]
        v1_q = np.stack([np.asarray(ja[k], float) for k in JOINT_ORDER], axis=1)
        dyn_q = np.load(out / entry["file"])["q"]
        worst = max(worst, np.abs(v1_q[::dec][: dyn_q.shape[0]] - dyn_q).max())
    assert worst < 1e-6, f"trial indices do not align with SkillData v1 (max |dq| = {worst:.2e})"
