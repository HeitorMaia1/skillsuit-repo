"""Ingest layer — the source-agnostic contract (task 3.11).

The SkillData v1 *base layer* (captured human motion) is device-independent. Any capture source —
the reference IMU sleeve/sim, an Xsens or Rokoko suit, markerless video, an exoskeleton — becomes an
``IngestAdapter`` whose ``to_base_layer`` returns a valid SkillData base-layer record. Downstream
(fusion, pHNN, retargeting, export) then works identically regardless of where the motion came from.

This module is a leaf (imports only numpy + the package version): ``assemble_base_layer`` packs the
common structure, ``phase_labels_from_speed`` derives prep/active/settle, and ``IngestAdapter`` is the
abstract contract every device adapter implements.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .. import SCHEMA_VERSION


def phase_labels_from_speed(speed, active_frac=0.01):
    """Per-sample prep/active/settle labels from a 1-D motion-speed array.

    'active' where speed exceeds ``active_frac`` of its peak; 'prep' before the first active sample,
    'settle' after the last. Returns a list of strings (SkillData ``phase_labels``).
    """
    speed = np.abs(np.asarray(speed, float))
    peak = speed.max() if speed.size else 0.0
    active = speed > active_frac * peak if peak > 0 else np.zeros_like(speed, bool)
    labels = np.where(active, "active", "prep").astype(object)
    if active.any():
        last = np.flatnonzero(active)[-1]
        labels[np.arange(speed.size) > last] = "settle"
    return labels.tolist()


def assemble_base_layer(
    *,
    subject_id,
    motion_class,
    trial_index,
    sample_rate_hz,
    imu_streams,
    segment_kinematics=None,
    phase_labels=None,
    calibration=None,
    source="synthetic",
):
    """Pack device-produced streams into a SkillData v1 **base-layer** record.

    ``imu_streams`` is the schema-shaped dict (sensor id -> {timestamp_us, angular_velocity_dps,
    linear_accel_g, ...}). ``n_samples`` is inferred from the first stream. The result is the base
    layer only (no ``retarget``); it validates against ``schema.json`` on its own.
    """
    if not imu_streams:
        raise ValueError("imu_streams must contain at least one sensor")
    n = len(next(iter(imu_streams.values()))["timestamp_us"])
    rec = {
        "schema_version": SCHEMA_VERSION,
        "session": {
            "subject_id": subject_id,
            "motion_class": motion_class,
            "trial_index": int(trial_index),
            "n_samples": int(n),
            "sample_rate_hz": float(sample_rate_hz),
            "source": source,
        },
        "calibration": calibration or {},
        "imu_streams": imu_streams,
    }
    if segment_kinematics is not None:
        rec["segment_kinematics"] = segment_kinematics
    if phase_labels is not None:
        rec["phase_labels"] = phase_labels
    return rec


class IngestAdapter(ABC):
    """Contract: a capture device/source -> a SkillData v1 base-layer record.

    Implement ``to_base_layer`` to map the source's native data into the base layer via
    ``assemble_base_layer``. The reference IMU sleeve/sim is one adapter; real devices (Xsens,
    Rokoko, video, exoskeletons) are additional adapters with the same output contract.
    """

    source_name: str = "generic"

    @abstractmethod
    def to_base_layer(self, *, subject_id, motion_class, trial_index) -> dict:
        """Return a validated SkillData v1 base-layer record for one trial."""
        raise NotImplementedError
