"""Tests for the SkillData v1 encoder: base-layer build, schema validation, IK retarget."""

import numpy as np

from sim.arm import PlanarArm, min_jerk
from skilldata import SCHEMA_VERSION
from skilldata.encoder import (
    TargetRobot,
    attach_retarget,
    build_record,
    export,
    planar_fk_2link,
    two_link_ik,
    validate,
)


def _reach_trial(fs=500.0, dur=2.0):
    """A canonical synthetic reach (matches sim.arm's self-test)."""
    arm = PlanarArm()
    t = np.arange(0.0, dur, 1.0 / fs)
    th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, np.deg2rad(10), np.deg2rad(60))
    th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, np.deg2rad(20), np.deg2rad(90))
    return arm, t, (th1, th1d, th1dd), (th2, th2d, th2dd)


def _build():
    arm, t, j1, j2 = _reach_trial()
    rec = build_record(
        arm, t, *j1, *j2,
        subject_id="S001", motion_class="reach", trial_index=0, sample_rate_hz=500.0,
    )
    return arm, t, j1, j2, rec


# --- base layer ------------------------------------------------------------- #
def test_build_record_structure():
    _, t, _, _, rec = _build()
    n = t.size
    assert rec["schema_version"] == SCHEMA_VERSION == "skilldata-v1"
    assert rec["session"]["n_samples"] == n
    for sid in ("S2", "S4"):
        s = rec["imu_streams"][sid]
        assert len(s["timestamp_us"]) == n
        assert np.shape(s["angular_velocity_dps"]) == (n, 3)
        assert np.shape(s["linear_accel_g"]) == (n, 3)
    assert len(rec["phase_labels"]) == n
    assert set(rec["phase_labels"]) <= {"prep", "active", "settle"}


def test_record_validates_against_schema():
    _, _, _, _, rec = _build()
    assert validate(rec) is True


def test_imu_rest_sample_reads_one_g():
    # first sample is at rest (min-jerk start) -> |accel| == 1 g, gyro == 0
    _, _, _, _, rec = _build()
    a0 = np.array(rec["imu_streams"]["S4"]["linear_accel_g"][0])
    g0 = np.array(rec["imu_streams"]["S4"]["angular_velocity_dps"][0])
    assert np.isclose(np.linalg.norm(a0), 1.0, atol=1e-6)
    assert np.allclose(g0, 0.0)


# --- inverse kinematics ----------------------------------------------------- #
def test_two_link_ik_roundtrip_reachable():
    rng = np.random.default_rng(0)
    a1, a2 = 0.35, 0.35
    # sample reachable targets in the annulus [|a1-a2|, a1+a2]
    ang = rng.uniform(-np.pi, np.pi, 500)
    rad = rng.uniform(0.05, a1 + a2 - 0.02, 500)
    xy = np.stack([rad * np.cos(ang), rad * np.sin(ang)], axis=-1)
    q, reachable, residual = two_link_ik(xy, a1, a2)
    assert reachable.all()
    assert np.allclose(planar_fk_2link(q, a1, a2), xy, atol=1e-9)
    assert residual.max() < 1e-9


def test_two_link_ik_flags_unreachable():
    a1, a2 = 0.35, 0.35
    xy = np.array([[0.90, 0.0]])  # beyond a1+a2 = 0.70
    q, reachable, residual = two_link_ik(xy, a1, a2)
    assert not reachable[0]
    assert residual[0] > 0.1  # clamp distance ~ 0.20 m
    assert np.isfinite(q).all()


# --- robot-ready layer ------------------------------------------------------ #
def test_retarget_reproduces_wrist_path_and_is_genuine():
    arm, _, (th1, *_), (th2, *_), rec = _build()
    ee = arm.forward_kinematics(th1, th2)["wrist"]  # (n, 2)
    target = TargetRobot(name="planar_2link_demo", a1=0.35, a2=0.35)
    attach_retarget(rec, ee, target)

    block = rec["retarget"]["planar_2link_demo"]
    q = np.array(block["joint_trajectory_rad"])
    assert all(block["reachable_flag"])
    # the robot end-effector reproduces the captured wrist path
    assert np.allclose(planar_fk_2link(q, 0.35, 0.35), ee, atol=1e-6)
    assert max(block["ik_residual_m"]) < 1e-6
    # genuine retarget: robot joints differ from the human joints (different morphology)
    human = np.stack([th1, th2], axis=-1)
    assert np.abs(q - human).max() > 0.1


def test_full_record_with_retarget_validates():
    arm, _, (th1, *_), (th2, *_), rec = _build()
    ee = arm.forward_kinematics(th1, th2)["wrist"]
    attach_retarget(rec, ee, TargetRobot(urdf_ref="robots/unitree_g1.urdf"))
    assert validate(rec) is True


# --- export ----------------------------------------------------------------- #
def test_export_writes_files_and_manifest(tmp_path):
    arm, _, (th1, *_), (th2, *_), rec = _build()
    ee = arm.forward_kinematics(th1, th2)["wrist"]
    attach_retarget(rec, ee, TargetRobot())
    paths = export([rec], tmp_path)
    assert len(paths) == 1 and paths[0].exists()
    assert (tmp_path / "manifest.json").exists()
