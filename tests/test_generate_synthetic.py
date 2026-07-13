"""Tests for the synthetic dataset generator (task 3.9)."""

import json

import numpy as np

from skilldata.encoder import validate
from skilldata.generate_synthetic import _class_counts, generate_dataset
from sim.motions import MOTION_CLASSES


def test_class_counts_balanced():
    assert _class_counts(1000, MOTION_CLASSES) == [250, 250, 250, 250]
    assert _class_counts(10, MOTION_CLASSES) == [3, 3, 2, 2]      # remainder to the first classes
    assert sum(_class_counts(1003, MOTION_CLASSES)) == 1003


def test_dataset_structure_and_manifest(tmp_path):
    # retarget off keeps the test fast; structure + variants + validation are the point here.
    m = generate_dataset(tmp_path, n=8, seed=0, retarget=False)
    assert m["n_base_trials"] == 8
    assert m["n_records"] == 16                      # clean + noisy per trial
    assert set(m["per_class"]) == set(MOTION_CLASSES)
    assert sum(c["n_trials"] for c in m["per_class"].values()) == 8

    files = list(tmp_path.glob("S001_*.json"))
    assert len(files) == 16
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["schema_version"] == "skilldata-v1"
    assert len(manifest["trials"]) == 16

    # every written record validates against the schema and carries its variant tag
    for f in files:
        rec = json.loads(f.read_text())
        assert validate(rec) is True
        assert rec["session"]["variant"] in ("clean", "noisy")
        if rec["session"]["variant"] == "noisy":
            assert "saturation_flag" in next(iter(rec["imu_streams"].values()))


def test_only_throw_saturates(tmp_path):
    m = generate_dataset(tmp_path, n=8, seed=0, retarget=False)
    pc = m["per_class"]
    assert pc["throw"]["saturated_records"] > 0            # throw pins S5 past ±2000°/s
    assert pc["throw"]["peak_gyro_dps"]["S5"] > 2000.0
    for mc in ("reach", "lift", "wrist_rotate"):
        assert pc[mc]["saturated_records"] == 0
        assert pc[mc]["peak_gyro_dps"]["S5"] <= 2000.0


def test_retarget_layer_attached(tmp_path):
    # one trial per class, retarget on: both variants must carry the robot-ready layer.
    m = generate_dataset(tmp_path, n=4, seed=3, retarget=True)
    for entry in m["trials"]:
        assert entry["robots"] == ["reference_humanoid_7dof"]
        assert entry["frac_reachable"] >= 0.0
    rec = json.loads((tmp_path / m["trials"][0]["file"]).read_text())
    block = rec["retarget"]["reference_humanoid_7dof"]
    assert np.array(block["joint_trajectory_rad"]).shape[1] == 7
    assert len(block["ik_residual_m"]) == rec["session"]["n_samples"]


def test_reproducible(tmp_path):
    a = generate_dataset(tmp_path / "a", n=4, seed=11, retarget=False)
    b = generate_dataset(tmp_path / "b", n=4, seed=11, retarget=False)
    ra = json.loads((tmp_path / "a" / a["trials"][0]["file"]).read_text())
    rb = json.loads((tmp_path / "b" / b["trials"][0]["file"]).read_text())
    assert ra["imu_streams"]["S5"]["angular_velocity_dps"] == rb["imu_streams"]["S5"]["angular_velocity_dps"]
