"""Generate a compact real SkillData sample for the visualizer (subsampled reach)."""
import json
import numpy as np
from sim.arm import PlanarArm, min_jerk
from skilldata.encoder import build_record, attach_retarget, TargetRobot

arm = PlanarArm()  # l1=0.30, l2=0.25
fs = 500.0
t = np.arange(0.0, 2.0, 1.0 / fs)
th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, np.deg2rad(10), np.deg2rad(60))
th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, np.deg2rad(20), np.deg2rad(90))

rec = build_record(arm, t, th1, th1d, th1dd, th2, th2d, th2dd,
                   subject_id="S001", motion_class="reach", trial_index=0, sample_rate_hz=fs)
ee = arm.forward_kinematics(th1, th2)["wrist"]
attach_retarget(rec, ee, TargetRobot(name="planar_2link_demo", a1=0.35, a2=0.35))

step = 5  # 1000 -> 200 frames, ~10 ms/frame
sl = slice(0, len(t), step)
S2, S4 = rec["imu_streams"]["S2"], rec["imu_streams"]["S4"]
q = np.array(rec["retarget"]["planar_2link_demo"]["joint_trajectory_rad"])


def gyro_z(stream):
    return [round(v[2], 2) for v in stream["angular_velocity_dps"][sl]]


def accel_mag(stream):
    return [round(float(np.hypot(v[0], v[1])), 4) for v in stream["linear_accel_g"][sl]]


data = {
    "meta": {"subject": "S001", "motion": "reach", "fs": fs, "n_full": len(t), "n_frames": len(t[sl]),
             "schema_version": rec["schema_version"]},
    "t": [round(x, 3) for x in t[sl].tolist()],
    "phase": rec["phase_labels"][sl],
    "human": {"l1": arm.l1, "l2": arm.l2,
              "th1": [round(x, 5) for x in th1[sl].tolist()],
              "th2": [round(x, 5) for x in th2[sl].tolist()]},
    "robot": {"a1": 0.35, "a2": 0.35,
              "q1": [round(x, 5) for x in q[sl, 0].tolist()],
              "q2": [round(x, 5) for x in q[sl, 1].tolist()]},
    "imu": {"S2_gyro": gyro_z(S2), "S4_gyro": gyro_z(S4),
            "S2_acc": accel_mag(S2), "S4_acc": accel_mag(S4)},
}
print(json.dumps(data))
