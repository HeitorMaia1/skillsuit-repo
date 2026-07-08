"""Simulation ingest adapters (task 3.11): the reference capture sources.

- ``planar_imu_streams`` — shared planar (2-DOF) IMU->streams mapping, used by both the encoder's
  ``build_record`` and ``PlanarSimAdapter``.
- ``PlanarSimAdapter`` — the legacy planar arm as an ingest source.
- ``Arm3DSimAdapter`` — the 3D serial-chain arm (``sim.arm3d``) as an ingest source, producing native
  3-axis IMU streams for every instrumented sensor. This is the reference 3D capture the humanoid
  retarget consumes.
"""

from __future__ import annotations

import numpy as np

from sim.arm import G
from sim.sensor import apply_to_streams

from .base import IngestAdapter, assemble_base_layer, phase_labels_from_speed


def planar_imu_streams(arm, t, th1, th1d, th1dd, th2, th2d, th2dd):
    """Planar 2-DOF IMU synthesis -> schema-shaped streams for S2, S4.

    Maps the planar readings into the hardware-shaped 3-axis fields: gyro ``[0, 0, deg/s]`` about the
    out-of-plane axis, accel ``[ax, ay, 0]`` in units of g.
    """
    t = np.asarray(t, float)
    n = t.size
    ts_us = np.round(t * 1e6).astype(np.int64)
    imu = arm.ideal_imu(th1, th1d, th1dd, th2, th2d, th2dd)
    streams = {}
    for sid in ("S2", "S4"):
        gyro_z_dps = np.rad2deg(np.asarray(imu[sid]["gyro_z"], float))
        accel = np.asarray(imu[sid]["accel"], float)  # (n, 2) m/s^2
        streams[sid] = {
            "timestamp_us": ts_us.tolist(),
            "angular_velocity_dps": np.stack([np.zeros(n), np.zeros(n), gyro_z_dps], axis=-1).tolist(),
            "linear_accel_g": np.stack([accel[:, 0] / G, accel[:, 1] / G, np.zeros(n)], axis=-1).tolist(),
        }
    return streams


class PlanarSimAdapter(IngestAdapter):
    """The planar 2-DOF arm (``sim.arm.PlanarArm``) as a capture source."""

    source_name = "sim_planar"

    def __init__(self, arm, t, th1, th1d, th1dd, th2, th2d, th2dd, sample_rate_hz, sensor_model=None):
        self.arm = arm
        self.t = np.asarray(t, float)
        self.j1 = (th1, th1d, th1dd)
        self.j2 = (th2, th2d, th2dd)
        self.sample_rate_hz = sample_rate_hz
        self.sensor_model = sensor_model

    def to_base_layer(self, *, subject_id, motion_class, trial_index):
        th1, th1d, _ = self.j1
        th2, th2d, _ = self.j2
        streams = planar_imu_streams(self.arm, self.t, *self.j1, *self.j2)
        if self.sensor_model is not None:
            streams = apply_to_streams(streams, self.sensor_model, 1.0 / self.sample_rate_hz)
        speed = np.abs(th1d) + np.abs(th2d)
        return assemble_base_layer(
            subject_id=subject_id, motion_class=motion_class, trial_index=trial_index,
            sample_rate_hz=self.sample_rate_hz, imu_streams=streams,
            segment_kinematics={"joint_angles_rad": {
                "shoulder": np.asarray(th1, float).tolist(), "elbow": np.asarray(th2, float).tolist()}},
            phase_labels=phase_labels_from_speed(speed), source="synthetic",
        )


class Arm3DSimAdapter(IngestAdapter):
    """The 3D serial-chain arm (``sim.arm3d.Arm3D``) as a capture source.

    Produces native 3-axis IMU streams (gyro deg/s, accel g) for every instrumented sensor, plus
    per-joint angles. ``joint_names`` defaults to the arm's segment names.
    """

    source_name = "sim_arm3d"

    def __init__(self, arm3d, t, qs, qds, qdds, sample_rate_hz, joint_names=None, sensor_model=None):
        self.arm3d = arm3d
        self.t = np.asarray(t, float)
        self.qs = np.asarray(qs, float)
        self.qds = np.asarray(qds, float)
        self.qdds = np.asarray(qdds, float)
        self.sample_rate_hz = sample_rate_hz
        self.joint_names = joint_names or [s.name for s in arm3d.segments]
        self.sensor_model = sensor_model

    def to_base_layer(self, *, subject_id, motion_class, trial_index):
        ts_us = np.round(self.t * 1e6).astype(np.int64).tolist()
        sensors = self.arm3d.ideal_imu(self.qs, self.qds, self.qdds)
        streams = {}
        for sid, s in sensors.items():
            streams[sid] = {
                "timestamp_us": ts_us,
                "angular_velocity_dps": np.rad2deg(np.asarray(s["gyro"], float)).tolist(),
                "linear_accel_g": (np.asarray(s["accel"], float) / G).tolist(),
            }
        if self.sensor_model is not None:
            streams = apply_to_streams(streams, self.sensor_model, 1.0 / self.sample_rate_hz)
        angles = {name: self.qs[:, i].tolist() for i, name in enumerate(self.joint_names)}
        speed = np.abs(self.qds).sum(axis=1)
        return assemble_base_layer(
            subject_id=subject_id, motion_class=motion_class, trial_index=trial_index,
            sample_rate_hz=self.sample_rate_hz, imu_streams=streams,
            segment_kinematics={"joint_angles_rad": angles},
            phase_labels=phase_labels_from_speed(speed), source="synthetic",
        )
