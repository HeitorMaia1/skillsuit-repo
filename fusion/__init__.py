"""fusion — sensor-fusion orientation filters (Phase 4).

Turns raw gyro + accelerometer streams (as captured in a SkillData v1 record's
``imu_streams``) into a per-sample orientation estimate. Two filters, same input units, same
output convention (a Hamilton ``(w,x,y,z)`` body->earth quaternion), so they are drop-in
interchangeable:

  ``madgwick.py`` (task 4.1) — the gradient-descent AHRS filter, derived from scratch.
  ``ekf.py`` (task 4.3) — a multiplicative EKF carrying an explicit gyro-bias state.

``run_validation`` (task 4.5/4.6) scores both against the synthetic dataset's clean ground
truth, by motion class.
"""

__version__ = "0.1.0"
