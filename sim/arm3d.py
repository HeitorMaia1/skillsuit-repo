"""3D serial-chain arm: exact analytic kinematics + ideal IMU (task 3.2).

Generalizes the planar 2-DOF arm (``sim/arm.py``) to an arbitrary chain of revolute
joints in 3D. The planar model is recovered exactly as a special case (see
``planar_2link`` and ``tests/test_arm3d.py``).

Model
-----
The arm is a serial chain of ``n`` revolute joints. Segment ``i`` (0-indexed) has:

  axis          — the joint's rotation axis, a unit vector expressed in the *parent*
                  segment's frame (frame ``i-1``).
  link          — the segment vector (proximal joint -> distal joint), expressed in this
                  segment's *own* frame (frame ``i``).
  has_sensor    — whether an IMU rides this segment, at ``sensor_frac`` along ``link``.

Frame ``i``'s world orientation is ``C_i = C_{i-1} · R(axis_i, q_i)`` with ``C_{-1}=I``.

Kinematics (all exact, from the analytic joint trajectories q(t), q'(t), q''(t))
--------------------------------------------------------------------------------
World joint axis      k_i     = C_{i-1} · axis_i
Angular velocity      ω_i     = ω_{i-1} + k_i q'_i
Angular acceleration  α_i     = α_{i-1} + k_i q''_i + (ω_{i-1} × k_i) q'_i
Distal-joint accel    a_i     = a_{i-1} + α_i × d_i + ω_i × (ω_i × d_i),  d_i = C_i·link_i
A point fixed in frame i at world offset r from the proximal joint accelerates as
                      a(r)    = a_{i-1} + α_i × r + ω_i × (ω_i × r).

Ideal IMU on segment i (sensor aligned with frame i)
----------------------------------------------------
gyro  (body)  = C_i^T ω_i                       [rad/s]
accel (body)  = C_i^T (a_sensor − g_world)       specific force; at rest -> |accel| = G.
``g_world`` defaults to z-up gravity ``(0,0,−G)``. The planar special case uses in-plane
gravity ``(0,−G,0)`` so it matches ``sim/arm.py`` exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import G


def _rodrigues(axis, theta):
    """Rotation matrix about a unit ``axis`` by ``theta`` (Rodrigues' formula).

    Scalar ``theta`` -> ``(3,3)``; array ``theta`` of shape ``(T,)`` -> ``(T,3,3)``.
    """
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    KK = K @ K
    th = np.asarray(theta, float)
    s = np.sin(th)[..., None, None]
    c = np.cos(th)[..., None, None]
    return np.eye(3) + s * K + (1.0 - c) * KK


@dataclass(frozen=True)
class Segment:
    """One revolute joint + the rigid segment that follows it."""

    axis: tuple
    link: tuple
    name: str
    has_sensor: bool = False
    sensor_frac: float = 0.5
    sensor_id: str | None = None


@dataclass(frozen=True)
class Arm3D:
    """A serial chain of revolute joints in 3D. Lengths in metres, angles in radians."""

    segments: tuple
    g_world: tuple = (0.0, 0.0, -G)

    def state(self, qs, qds, qdds):
        """Full kinematic state from joint trajectories.

        ``qs, qds, qdds`` each have shape ``(T, n)`` (time × joints). Returns a dict with
        ``sensors`` (per sensor id: ``gyro``, ``accel``, ``pos``, ``orient``), ``frames``
        (segment name -> orientation ``C_i`` (T,3,3)), and ``joints`` (name -> distal
        joint world position (T,3), plus ``base``).
        """
        qs = np.asarray(qs, float)
        qds = np.asarray(qds, float)
        qdds = np.asarray(qdds, float)
        if qs.ndim != 2 or qs.shape[1] != len(self.segments):
            raise ValueError(f"qs must be (T, {len(self.segments)}); got {qs.shape}")
        T = qs.shape[0]
        g = np.asarray(self.g_world, float)

        C = np.broadcast_to(np.eye(3), (T, 3, 3)).copy()  # C_{-1} = I
        omega = np.zeros((T, 3))
        alpha = np.zeros((T, 3))
        p = np.zeros((T, 3))   # proximal joint position (base)
        a = np.zeros((T, 3))   # proximal joint acceleration

        sensors, frames, joints = {}, {}, {"base": p.copy()}
        for i, seg in enumerate(self.segments):
            axis = np.asarray(seg.axis, float)
            axis = axis / np.linalg.norm(axis)
            link = np.asarray(seg.link, float)
            qdi, qddi = qds[:, i, None], qdds[:, i, None]

            k = np.einsum("tij,j->ti", C, axis)          # world joint axis (uses C_{i-1})
            Ci = C @ _rodrigues(axis, qs[:, i])          # frame i orientation
            omega_i = omega + k * qdi
            alpha_i = alpha + k * qddi + np.cross(omega, k) * qdi

            d = np.einsum("tij,j->ti", Ci, link)         # link vector in world
            p_i = p + d
            a_i = a + np.cross(alpha_i, d) + np.cross(omega_i, np.cross(omega_i, d))

            if seg.has_sensor:
                r = np.einsum("tij,j->ti", Ci, seg.sensor_frac * link)
                a_s = a + np.cross(alpha_i, r) + np.cross(omega_i, np.cross(omega_i, r))
                sensors[seg.sensor_id or seg.name] = {
                    "gyro": np.einsum("tkj,tk->tj", Ci, omega_i),        # C_i^T ω_i
                    "accel": np.einsum("tkj,tk->tj", Ci, a_s - g),       # specific force
                    "pos": p + r,
                    "orient": Ci,
                }
            frames[seg.name] = Ci
            joints[seg.name] = p_i
            C, omega, alpha, p, a = Ci, omega_i, alpha_i, p_i, a_i

        return {"sensors": sensors, "frames": frames, "joints": joints}

    def ideal_imu(self, qs, qds, qdds):
        """Per-sensor ideal gyro (rad/s) + accelerometer specific force (m/s^2, body frame)."""
        return self.state(qs, qds, qdds)["sensors"]


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def planar_2link(l1: float = 0.30, l2: float = 0.25) -> Arm3D:
    """The planar 2-DOF arm as a 3D chain (both joints about +z, motion in x-y).

    With in-plane gravity ``(0,-G,0)`` this reproduces ``sim.arm.PlanarArm`` exactly —
    the proof that 2D is a special case of the 3D model.
    """
    return Arm3D(
        segments=(
            Segment(axis=(0, 0, 1), link=(l1, 0, 0), name="upper", has_sensor=True, sensor_id="S2"),
            Segment(axis=(0, 0, 1), link=(l2, 0, 0), name="fore", has_sensor=True, sensor_id="S4"),
        ),
        g_world=(0.0, -G, 0.0),
    )


def human_arm_7dof(l_upper: float = 0.30, l_fore: float = 0.25, l_hand: float = 0.08) -> Arm3D:
    """A 7-DOF right-arm model: shoulder (3) + elbow (1) + forearm pronation (1) + wrist (2).

    Sensor placement matches the hardware: S2 (upper-arm mid), S4 (forearm mid), S5 (hand).
    The scapula reference S0 is a static trunk sensor (added separately if needed); here the
    trunk is the fixed world base. Zero-length links stack the three shoulder axes at one point.
    """
    return Arm3D(
        segments=(
            Segment(axis=(0, 0, 1), link=(0, 0, 0), name="sh_yaw"),
            Segment(axis=(0, 1, 0), link=(0, 0, 0), name="sh_pitch"),
            Segment(axis=(1, 0, 0), link=(l_upper, 0, 0), name="sh_roll_upper",
                    has_sensor=True, sensor_id="S2"),
            Segment(axis=(0, 0, 1), link=(l_fore, 0, 0), name="elbow_fore",
                    has_sensor=True, sensor_id="S4"),
            Segment(axis=(1, 0, 0), link=(0, 0, 0), name="forearm_pron"),
            Segment(axis=(0, 0, 1), link=(0, 0, 0), name="wrist_flex"),
            Segment(axis=(0, 1, 0), link=(l_hand, 0, 0), name="wrist_dev_hand",
                    has_sensor=True, sensor_id="S5"),
        ),
        g_world=(0.0, 0.0, -G),
    )


def reference_humanoid_7dof(l_upper: float = 0.28, l_fore: float = 0.26, l_hand: float = 0.10) -> Arm3D:
    """A 7-DOF humanoid *robot* arm (retarget target). Same joint structure as the human arm but
    robot proportions (so retargeting is a genuine, non-identity mapping), and no sensors.

    Stand-in for an open humanoid (Unitree G1) until a real URDF is loaded into an Arm3D chain; the
    end-effector is the ``wrist_dev_hand`` joint. Kinematics are identical in form to a URDF chain, so
    swapping in a parsed URDF later is a drop-in.
    """
    return Arm3D(
        segments=(
            Segment(axis=(0, 0, 1), link=(0, 0, 0), name="sh_yaw"),
            Segment(axis=(0, 1, 0), link=(0, 0, 0), name="sh_pitch"),
            Segment(axis=(1, 0, 0), link=(l_upper, 0, 0), name="sh_roll_upper"),
            Segment(axis=(0, 0, 1), link=(l_fore, 0, 0), name="elbow_fore"),
            Segment(axis=(1, 0, 0), link=(0, 0, 0), name="forearm_pron"),
            Segment(axis=(0, 0, 1), link=(0, 0, 0), name="wrist_flex"),
            Segment(axis=(0, 1, 0), link=(l_hand, 0, 0), name="wrist_dev_hand"),
        ),
        g_world=(0.0, 0.0, -G),
    )


if __name__ == "__main__":
    from .arm import min_jerk

    fs = 500.0
    t = np.arange(0.0, 2.0, 1.0 / fs)
    arm = human_arm_7dof()
    # a 3D reach: shoulder pitch + yaw + elbow flex move together
    q = np.zeros((t.size, 7))
    qd = np.zeros((t.size, 7))
    qdd = np.zeros((t.size, 7))
    moves = {0: (0.0, 0.6), 1: (0.0, 0.8), 3: (0.2, 1.4)}  # joint -> (start, end) rad
    for j, (a0, a1) in moves.items():
        q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, 0.4, 1.2, a0, a1)
    imu = arm.ideal_imu(q, qd, qdd)
    print(f"human_arm_7dof 3D reach: {t.size} samples @ {fs:.0f} Hz")
    for sid in ("S2", "S4", "S5"):
        peak = np.rad2deg(np.linalg.norm(imu[sid]["gyro"], axis=-1)).max()
        rest = np.linalg.norm(imu[sid]["accel"][0])
        print(f"  {sid}: peak |gyro| = {peak:6.1f} deg/s | rest |accel| = {rest:.3f} m/s^2 (~{G})")
