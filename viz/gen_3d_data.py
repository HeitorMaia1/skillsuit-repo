"""Generate a compact real 3D capture (human_arm_7dof) for the 3D visualizer."""
import json
import numpy as np
from sim.arm import min_jerk
from sim.arm3d import human_arm_7dof, G

fs = 500.0
t = np.arange(0.0, 2.0, 1.0 / fs)
arm = human_arm_7dof()  # l_upper=.30, l_fore=.25, l_hand=.08
n = t.size

# a natural 3D reach-and-rotate: yaw + lift + elbow bend + forearm twist + wrist
q = np.zeros((n, 7)); qd = np.zeros((n, 7)); qdd = np.zeros((n, 7))
moves = {0: (0.0, 0.5), 1: (0.0, 0.9), 3: (0.2, 1.3), 4: (0.0, 0.8), 6: (0.0, 0.4)}
for j, (a0, a1) in moves.items():
    q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, 0.4, 1.2, a0, a1)

st = arm.state(q, qd, qdd)
J = st["joints"]
S = st["sensors"]

step = 6  # 1000 -> 167 frames
sl = slice(0, n, step)


def pts(a):  # (T,3) -> subsampled rounded list
    return [[round(float(v), 4) for v in row] for row in np.asarray(a)[sl]]


def vec_dps(a):  # (T,3) rad/s -> deg/s rounded
    return [[round(float(x), 2) for x in row] for row in np.rad2deg(np.asarray(a))[sl]]


def vec_g(a):   # (T,3) m/s^2 -> g rounded
    return [[round(float(x), 4) for x in row] for row in (np.asarray(a) / G)[sl]]


# phase from total joint speed
speed = np.abs(qd).sum(axis=1)
peak = speed.max()
active = speed > 0.01 * peak
phase = np.where(active, "active", "prep").astype(object)
if active.any():
    last = np.flatnonzero(active)[-1]
    phase[np.arange(n) > last] = "settle"

data = {
    "meta": {"subject": "S001", "motion": "reach3d", "fs": fs, "n_full": n, "n_frames": len(t[sl]),
             "schema_version": "skilldata-v1", "dof": 7},
    "t": [round(x, 3) for x in t[sl].tolist()],
    "phase": phase[sl].tolist(),
    "chain": {  # drawable arm points (world, metres), z up
        "shoulder": pts(J["sh_roll_upper"] * 0),   # origin
        "elbow": pts(J["sh_roll_upper"]),
        "wrist": pts(J["elbow_fore"]),
        "hand": pts(J["wrist_dev_hand"]),
    },
    "sensors": {sid: {"pos": pts(S[sid]["pos"]), "gyro": vec_dps(S[sid]["gyro"]),
                      "accel": vec_g(S[sid]["accel"])} for sid in ("S2", "S4", "S5")},
}
print(json.dumps(data))
