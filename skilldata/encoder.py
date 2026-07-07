"""skilldata.encoder — build, retarget, and export SkillData v1 records (tasks 3.7/3.8).

Three stages, mirroring the format's two layers (docs/skilldata_v1.md):

  build_record(...)   -> the BASE LAYER: captured human motion (IMU streams, joint
                         angles, phase labels) assembled into a SkillData v1 record.
  attach_retarget(..) -> the ROBOT-READY LAYER: the captured end-effector path mapped
                         onto a target robot's joints via inverse kinematics (IK).
  export(...)         -> write a batch of records to disk as a training-ready dataset.

The retargeter here is a closed-form planar 2-link IK (``method: analytic_2link``). It
proves the full pipeline end-to-end on the current planar arm; the humanoid URDF + general
IK backend swaps in after the 3D arm (task 3.2) with no change to the format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import SCHEMA_VERSION
from .ingest.base import assemble_base_layer, phase_labels_from_speed
from .ingest.sim import planar_imu_streams

_SCHEMA_PATH = Path(__file__).with_name("schema.json")


# --------------------------------------------------------------------------- #
# Planar 2-link kinematics for the demo retarget robot
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TargetRobot:
    """A planar 2-link target robot for retargeting. Lengths in metres.

    ``urdf_ref`` is carried into the record so a real robot (e.g. an open humanoid,
    or Tesla Optimus once a model exists) is identified by its URDF; the planar demo
    leaves it ``None``.
    """

    name: str = "planar_2link_demo"
    a1: float = 0.35
    a2: float = 0.35
    urdf_ref: str | None = None
    joint_names: tuple[str, ...] = ("q1", "q2")


def planar_fk_2link(q, a1, a2):
    """Forward kinematics of a planar 2-link arm. ``q`` is ``(..., 2)`` -> end-effector ``(..., 2)``."""
    q = np.asarray(q, float)
    phi1 = q[..., 0]
    phi2 = q[..., 0] + q[..., 1]
    x = a1 * np.cos(phi1) + a2 * np.cos(phi2)
    y = a1 * np.sin(phi1) + a2 * np.sin(phi2)
    return np.stack([x, y], axis=-1)


def two_link_ik(xy, a1, a2):
    """Closed-form planar 2-link inverse kinematics (elbow-down branch).

    Given desired end-effector positions ``xy`` of shape ``(..., 2)`` and link lengths
    ``a1, a2``, returns ``(q, reachable, residual)``:

      q         — joint angles ``(..., 2)`` = (shoulder absolute, elbow relative).
      reachable — bool ``(...)``; False where the target is outside the workspace annulus.
      residual  — metres ``(...)``; Euclidean error between the achieved end-effector and
                  the *desired* target (≈0 when reachable, = the clamp distance otherwise).

    Out-of-reach targets are radially clamped onto the nearest workspace boundary so a joint
    solution always exists; the residual and flag record that it was clamped.
    """
    xy = np.asarray(xy, float)
    x, y = xy[..., 0], xy[..., 1]
    r = np.hypot(x, y)

    r_min, r_max = abs(a1 - a2), a1 + a2
    tol = 1e-9
    reachable = (r >= r_min - tol) & (r <= r_max + tol)

    # radial clamp; guard r==0 (pick the +x direction)
    r_safe = np.where(r < tol, tol, r)
    r_clamped = np.clip(r, r_min, r_max)
    scale = r_clamped / r_safe
    xc = np.where(r < tol, r_clamped, x * scale)
    yc = np.where(r < tol, 0.0, y * scale)

    cos_q2 = np.clip((r_clamped**2 - a1**2 - a2**2) / (2 * a1 * a2), -1.0, 1.0)
    q2 = np.arccos(cos_q2)  # elbow-down: q2 >= 0
    q1 = np.arctan2(yc, xc) - np.arctan2(a2 * np.sin(q2), a1 + a2 * np.cos(q2))
    q = np.stack([q1, q2], axis=-1)

    achieved = planar_fk_2link(q, a1, a2)
    residual = np.linalg.norm(achieved - xy, axis=-1)
    return q, reachable, residual


# --------------------------------------------------------------------------- #
# Base layer — build a SkillData v1 record from a simulated trial
# --------------------------------------------------------------------------- #
def build_record(
    arm,
    t,
    th1,
    th1d,
    th1dd,
    th2,
    th2d,
    th2dd,
    *,
    subject_id,
    motion_class,
    trial_index,
    sample_rate_hz,
    source="synthetic",
):
    """Assemble the BASE LAYER of a SkillData v1 record from a planar-arm trial.

    Thin wrapper over the ingest layer: the planar arm is the reference capture source
    (``skilldata.ingest.PlanarSimAdapter``). Maps the planar IMU synthesis into the hardware-shaped
    3-axis fields and populates the instrumented sensors S2 (upper arm) and S4 (forearm).
    """
    streams = planar_imu_streams(arm, t, th1, th1d, th1dd, th2, th2d, th2dd)
    speed = np.abs(np.asarray(th1d, float)) + np.abs(np.asarray(th2d, float))
    return assemble_base_layer(
        subject_id=subject_id,
        motion_class=motion_class,
        trial_index=trial_index,
        sample_rate_hz=sample_rate_hz,
        imu_streams=streams,
        segment_kinematics={
            "joint_angles_rad": {
                "shoulder": np.asarray(th1, float).tolist(),
                "elbow": np.asarray(th2, float).tolist(),
            }
        },
        phase_labels=phase_labels_from_speed(speed),
        source=source,
    )


# --------------------------------------------------------------------------- #
# Robot-ready layer — retarget the captured motion onto a target robot
# --------------------------------------------------------------------------- #
def attach_retarget(record, ee_path, target: TargetRobot, robot_id=None):
    """Add a ROBOT-READY layer: retarget ``ee_path`` (the captured end-effector path,
    shape ``(n, 2)``) onto ``target`` via 2-link IK, and store it under ``record['retarget']``.

    Mutates and returns ``record``. Keyed by ``robot_id`` (defaults to the robot name), so
    several target robots can coexist in one record.
    """
    ee_path = np.asarray(ee_path, float)
    q, reachable, residual = two_link_ik(ee_path, target.a1, target.a2)
    block = {
        "target_robot": {
            "name": target.name,
            "dof": 2,
            "link_lengths_m": [target.a1, target.a2],
        },
        "method": "analytic_2link",
        "joint_names": list(target.joint_names),
        "joint_trajectory_rad": q.tolist(),
        "target_position_m": ee_path.tolist(),
        "ik_residual_m": residual.tolist(),
        "reachable_flag": [bool(b) for b in reachable.tolist()],
    }
    if target.urdf_ref is not None:
        block["target_robot"]["urdf_ref"] = target.urdf_ref
    record.setdefault("retarget", {})[robot_id or target.name] = block
    return record


# --------------------------------------------------------------------------- #
# Validation + export
# --------------------------------------------------------------------------- #
def validate(record):
    """Validate a record against schema.json. Raises jsonschema.ValidationError on failure."""
    import jsonschema  # lazy: only needed for QA, not for runtime encoding

    with _SCHEMA_PATH.open() as fh:
        schema = json.load(fh)
    jsonschema.validate(record, schema)
    return True


def save_record(record, path):
    """Write one record to ``path`` as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(record, fh)
    return path


def export(records, out_dir):
    """Write a batch of records to ``out_dir`` as a training-ready dataset.

    One JSON file per trial (``<subject>_<motion>_<trial:03d>.json``) plus a ``manifest.json``
    indexing them. Returns the list of written record paths.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    manifest = []
    for rec in records:
        s = rec["session"]
        name = f"{s['subject_id']}_{s['motion_class']}_{s['trial_index']:03d}.json"
        p = save_record(rec, out_dir / name)
        paths.append(p)
        manifest.append(
            {
                "file": name,
                "subject_id": s["subject_id"],
                "motion_class": s["motion_class"],
                "trial_index": s["trial_index"],
                "n_samples": s["n_samples"],
                "robots": sorted(rec.get("retarget", {}).keys()),
            }
        )
    with (out_dir / "manifest.json").open("w") as fh:
        json.dump({"schema_version": SCHEMA_VERSION, "n_trials": len(manifest), "trials": manifest}, fh, indent=2)
    return paths


if __name__ == "__main__":
    # End-to-end self-test: synthetic reach -> base record -> retarget -> validate -> export.
    from sim.arm import PlanarArm, min_jerk

    arm = PlanarArm()
    fs = 500.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, np.deg2rad(10), np.deg2rad(60))
    th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, np.deg2rad(20), np.deg2rad(90))

    rec = build_record(
        arm, t, th1, th1d, th1dd, th2, th2d, th2dd,
        subject_id="S001", motion_class="reach", trial_index=0, sample_rate_hz=fs,
    )
    ee = arm.forward_kinematics(th1, th2)["wrist"]
    attach_retarget(rec, ee, TargetRobot(name="planar_2link_demo", a1=0.35, a2=0.35))
    validate(rec)

    out = Path("data/raw/synthetic")
    paths = export([rec], out)
    block = rec["retarget"]["planar_2link_demo"]
    print(f"SkillData v1 record: {rec['session']['n_samples']} samples @ {fs:.0f} Hz, "
          f"motion={rec['session']['motion_class']}")
    print(f"  base layer: streams {sorted(rec['imu_streams'])}, "
          f"phases {sorted(set(rec['phase_labels']))}")
    print(f"  robot layer: {block['target_robot']['name']} ({block['method']}), "
          f"max IK residual = {max(block['ik_residual_m']):.2e} m, "
          f"all reachable = {all(block['reachable_flag'])}")
    print(f"  validated against schema.json; wrote {paths[0]}")
