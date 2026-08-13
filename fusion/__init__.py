"""fusion — sensor-fusion orientation filters (Phase 4).

Turns raw gyro + accelerometer streams (as captured in a SkillData v1 record's
``imu_streams``) into a per-sample orientation estimate. ``madgwick.py`` (task 4.1) is the
gradient-descent AHRS filter, derived from scratch; an EKF (task 4.3) is the planned
comparison point. ``run_validation`` (task 4.5/4.6) scores both against the synthetic
dataset's clean ground truth, by motion class.
"""

__version__ = "0.1.0"
