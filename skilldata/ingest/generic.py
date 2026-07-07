"""Generic device ingest adapter (task 3.11) — the template for real capture hardware.

``GenericIMUAdapter`` takes already-extracted per-sensor arrays (timestamps, gyro in deg/s, accel in
g, optional fused quaternion + saturation flags) and packs them into a SkillData v1 base layer. A
real Xsens / Rokoko / markerless-video / exoskeleton adapter is just this with a device-specific
parser in front that fills the same arrays — nothing downstream changes.
"""

from __future__ import annotations

import numpy as np

from .base import IngestAdapter, assemble_base_layer


class GenericIMUAdapter(IngestAdapter):
    """Any device that yields per-sensor timestamped gyro (deg/s) + accel (g) arrays.

    ``streams`` maps sensor id -> dict with ``timestamp_us`` (int, shape (N,)), ``angular_velocity_dps``
    and ``linear_accel_g`` (each (N, 3)), and optionally ``quaternion`` (N, 4) and ``saturation_flag``
    (N,). Arrays may be numpy or lists. ``segment_kinematics`` / ``phase_labels`` / ``calibration``
    are passed through if provided.
    """

    def __init__(self, sample_rate_hz, streams, *, source_name="generic_imu",
                 segment_kinematics=None, phase_labels=None, calibration=None, source="hardware"):
        self.sample_rate_hz = sample_rate_hz
        self.streams = streams
        self.source_name = source_name
        self.segment_kinematics = segment_kinematics
        self.phase_labels = phase_labels
        self.calibration = calibration
        self.source = source

    def to_base_layer(self, *, subject_id, motion_class, trial_index):
        packed = {}
        for sid, s in self.streams.items():
            out = {
                "timestamp_us": np.asarray(s["timestamp_us"]).astype(np.int64).tolist(),
                "angular_velocity_dps": np.asarray(s["angular_velocity_dps"], float).tolist(),
                "linear_accel_g": np.asarray(s["linear_accel_g"], float).tolist(),
            }
            if "quaternion" in s:
                out["quaternion"] = np.asarray(s["quaternion"], float).tolist()
            if "saturation_flag" in s:
                out["saturation_flag"] = [bool(b) for b in np.asarray(s["saturation_flag"]).tolist()]
            packed[sid] = out
        return assemble_base_layer(
            subject_id=subject_id, motion_class=motion_class, trial_index=trial_index,
            sample_rate_hz=self.sample_rate_hz, imu_streams=packed,
            segment_kinematics=self.segment_kinematics, phase_labels=self.phase_labels,
            calibration=self.calibration, source=self.source,
        )
