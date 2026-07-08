"""Sensor realism model (task 3.5): turn ideal IMU readings into believable ones.

Applies, in order, to each ideal sensor stream:
  1. **gyro bias** — a slow random walk (bias instability), starting from an initial offset;
  2. **Gaussian white noise** — independent per sample/axis, on gyro and accelerometer;
  3. **saturation / clipping** — hard-limit to the sensor's full-scale range (D2: ±2000°/s gyro,
     ±16 g accel) and flag every sample where the gyro pins at its range (reading is invalid there).

Defaults match the ICM-42688-P at the ranges chosen in decision D2. The model is seeded for
reproducibility; ``apply_to_streams`` corrupts a whole SkillData-shaped stream dict (adding the
``saturation_flag`` field).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SensorModel:
    """Configurable noise + bias-drift + saturation model for a 6-axis IMU.

    Ranges default to the D2 decision (ICM-42688-P: ±2000°/s, ±16 g). ``seed`` makes the corruption
    reproducible; each ``corrupt_stream`` call advances the internal RNG so distinct sensors get
    distinct noise realizations.
    """

    gyro_noise_dps: float = 0.1                     # gyro white-noise std (deg/s)
    accel_noise_g: float = 0.01                     # accel white-noise std (g)
    initial_gyro_bias_dps: float = 0.0              # constant gyro bias offset (deg/s)
    gyro_bias_drift_dps_per_sqrt_s: float = 0.0     # bias random-walk rate (deg/s / sqrt(s))
    gyro_range_dps: float = 2000.0                  # full-scale gyro range (D2)
    accel_range_g: float = 16.0                     # full-scale accel range
    seed: int | None = 0
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self):
        self._rng = np.random.default_rng(self.seed)

    def corrupt_stream(self, gyro_dps, accel_g, dt):
        """Corrupt ideal streams. ``gyro_dps``/``accel_g`` are ``(T, 3)``; ``dt`` seconds/sample.

        Returns ``(gyro_out (T,3), accel_out (T,3), saturation_flag (T,))``. ``saturation_flag[t]`` is
        True if any gyro axis reached the full-scale range at sample ``t`` (clipped -> invalid).
        """
        gyro = np.asarray(gyro_dps, float).copy()
        accel = np.asarray(accel_g, float).copy()
        n = gyro.shape[0]

        # gyro bias: initial offset + random walk (cumulative Gaussian steps)
        bias = np.full((n, 3), float(self.initial_gyro_bias_dps))
        if self.gyro_bias_drift_dps_per_sqrt_s > 0:
            steps = self._rng.normal(0.0, self.gyro_bias_drift_dps_per_sqrt_s * np.sqrt(dt), (n, 3))
            bias = bias + np.cumsum(steps, axis=0)
        gyro += bias

        # white noise
        if self.gyro_noise_dps > 0:
            gyro += self._rng.normal(0.0, self.gyro_noise_dps, (n, 3))
        if self.accel_noise_g > 0:
            accel += self._rng.normal(0.0, self.accel_noise_g, (n, 3))

        # saturation: flag where the gyro would pin, then hard-clip both sensors
        sat_flag = (np.abs(gyro) >= self.gyro_range_dps).any(axis=1)
        gyro = np.clip(gyro, -self.gyro_range_dps, self.gyro_range_dps)
        accel = np.clip(accel, -self.accel_range_g, self.accel_range_g)
        return gyro, accel, sat_flag


def apply_to_streams(streams, model: SensorModel, dt):
    """Return a copy of a SkillData ``imu_streams`` dict with ``model`` applied to every sensor.

    Replaces ``angular_velocity_dps`` / ``linear_accel_g`` with noisy values and adds
    ``saturation_flag``. Other fields (timestamps, quaternion) are carried through unchanged.
    """
    out = {}
    for sid, s in streams.items():
        gyro = np.asarray(s["angular_velocity_dps"], float)
        accel = np.asarray(s["linear_accel_g"], float)
        g2, a2, sat = model.corrupt_stream(gyro, accel, dt)
        new = dict(s)
        new["angular_velocity_dps"] = g2.tolist()
        new["linear_accel_g"] = a2.tolist()
        new["saturation_flag"] = [bool(b) for b in sat.tolist()]
        out[sid] = new
    return out
