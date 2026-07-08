"""Generate a compact retargeted 3D capture (human + humanoid robot) for the viz."""
import json
import numpy as np
from sim.arm import min_jerk
from sim.arm3d import human_arm_7dof, reference_humanoid_7dof, G
from skilldata.retarget import dls_ik

EE = "wrist_dev_hand"
fs = 100.0
t = np.arange(0.0, 2.0, 1.0 / fs)
n = t.size

human = human_arm_7dof()
q = np.zeros((n, 7)); qd = np.zeros((n, 7)); qdd = np.zeros((n, 7))
for j, (a0, a1) in {0: (0.0, 0.5), 1: (0.0, 0.8), 3: (0.2, 1.3), 4: (0.0, 0.6)}.items():
    q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, 0.4, 1.2, a0, a1)
hs = human.state(q, qd, qdd)
hand = np.asarray(hs["joints"][EE], float)

robot = reference_humanoid_7dof()
q_robot, residual = dls_ik(robot, hand, EE)
rs = robot.state(q_robot, np.zeros_like(q_robot), np.zeros_like(q_robot))

step = 2
sl = slice(0, n, step)


def pts(a):
    return [[round(float(v), 4) for v in row] for row in np.asarray(a)[sl]]


def vec_dps(a):
    return [[round(float(x), 2) for x in row] for row in np.rad2deg(np.asarray(a))[sl]]


def vec_g(a):
    return [[round(float(x), 4) for x in row] for row in (np.asarray(a) / G)[sl]]


zeros3 = pts(np.zeros((n, 3)))
speed = np.abs(qd).sum(axis=1)
peak = speed.max()
active = speed > 0.01 * peak
phase = np.where(active, "active", "prep").astype(object)
if active.any():
    phase[np.arange(n) > np.flatnonzero(active)[-1]] = "settle"

data = {
    "meta": {"subject": "S001", "motion": "reach3d", "fs": fs, "n_full": n, "n_frames": len(t[sl]),
             "dof": 7, "robot": "reference_humanoid_7dof", "schema_version": "skilldata-v1"},
    "t": [round(x, 3) for x in t[sl].tolist()],
    "phase": phase[sl].tolist(),
    "human": {
        "chain": {"shoulder": zeros3, "elbow": pts(hs["joints"]["sh_roll_upper"]),
                  "wrist": pts(hs["joints"]["elbow_fore"]), "hand": pts(hs["joints"][EE])},
        "sensors": {sid: {"pos": pts(hs["sensors"][sid]["pos"]),
                          "gyro": vec_dps(hs["sensors"][sid]["gyro"]),
                          "accel": vec_g(hs["sensors"][sid]["accel"])} for sid in ("S2", "S4", "S5")},
    },
    "robot": {
        "chain": {"shoulder": zeros3, "elbow": pts(rs["joints"]["sh_roll_upper"]),
                  "wrist": pts(rs["joints"]["elbow_fore"]), "hand": pts(rs["joints"][EE])},
        "ik_residual_mm": [round(float(r) * 1000, 2) for r in residual[sl]],
    },
}
print(json.dumps(data))
