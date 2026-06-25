"""Planar 2-DOF arm: forward kinematics (task 3.3) and ideal IMU synthesis (task 3.4).

Coordinates
-----------
The arm is a planar 2-link chain: an upper arm (length ``l1``) hinged at the shoulder
(world origin) and a forearm (length ``l2``) hinged at the elbow. Generalized
coordinates:

  theta1 — shoulder angle: absolute orientation of the upper arm, measured from world +x.
  theta2 — elbow angle: orientation of the forearm *relative* to the upper arm.

So the absolute segment orientations are ``phi1 = theta1`` and ``phi2 = theta1 + theta2``.
Sensors sit at segment midpoints: ``S2`` on the upper arm, ``S4`` on the forearm — the
same landmark IDs as the hardware. (In 2-DOF the scapula ``S0`` is the fixed trunk
reference and the wrist ``S5`` rides the forearm's distal end.)

IMU model
---------
For each instrumented segment the ideal IMU returns:

  gyro_z  — angular velocity about the out-of-plane z axis = the segment's ``phi_dot``.
  accel   — the **specific force** the accelerometer measures, in the sensor's own
            (body) frame: ``f_body = R(phi)^T (a_world - g_world)`` with gravity
            ``g_world = [0, -G]``. At rest (``a_world = 0``) this reads ``+G`` along the
            body axis opposing gravity, i.e. magnitude ``G`` — the usual "1 g at rest".

All kinematics are exact: velocities and accelerations come from the analytic
derivatives of the joint trajectory, not finite differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

G = 9.81  # standard gravity [m/s^2]


def _rot_T_apply(phi, vx, vy):
    """Apply R(phi)^T to a world vector (vx, vy) -> body frame. Vectorized over phi."""
    c, s = np.cos(phi), np.sin(phi)
    return c * vx + s * vy, -s * vx + c * vy


@dataclass(frozen=True)
class PlanarArm:
    """Planar 2-link arm model (task 3.1). Lengths in metres."""

    l1: float = 0.30  # upper-arm length
    l2: float = 0.25  # forearm length

    # ---- forward kinematics (task 3.3) -------------------------------------
    def forward_kinematics(self, theta1, theta2):
        """Joint and sensor positions for joint angles (theta1, theta2).

        Returns a dict of 2-D world positions: ``shoulder``, ``elbow``, ``wrist``,
        and the sensor midpoints ``S2`` (upper arm) and ``S4`` (forearm). Inputs may
        be scalars or arrays; arrays return shape ``(..., 2)``.
        """
        theta1, theta2 = np.asarray(theta1, float), np.asarray(theta2, float)
        phi1, phi2 = theta1, theta1 + theta2
        u1 = np.stack([np.cos(phi1), np.sin(phi1)], axis=-1)
        u2 = np.stack([np.cos(phi2), np.sin(phi2)], axis=-1)
        shoulder = np.zeros_like(u1)
        elbow = self.l1 * u1
        wrist = elbow + self.l2 * u2
        return {
            "shoulder": shoulder,
            "elbow": elbow,
            "wrist": wrist,
            "S2": 0.5 * self.l1 * u1,
            "S4": elbow + 0.5 * self.l2 * u2,
        }

    # ---- ideal IMU readings (task 3.4) -------------------------------------
    def ideal_imu(self, th1, th1d, th1dd, th2, th2d, th2dd):
        """Ideal gyro + accelerometer for sensors S2 and S4.

        Args are the joint angle, angular velocity, and angular acceleration time
        series for both joints (each broadcastable to a common shape). Returns
        ``{"S2": {"gyro_z", "accel"}, "S4": {...}}``; ``accel`` has shape ``(..., 2)``.
        """
        th1, th1d, th1dd = map(lambda a: np.asarray(a, float), (th1, th1d, th1dd))
        th2, th2d, th2dd = map(lambda a: np.asarray(a, float), (th2, th2d, th2dd))

        # absolute segment orientation and its time derivatives
        phi1, phi1d, phi1dd = th1, th1d, th1dd
        phi2, phi2d, phi2dd = th1 + th2, th1d + th2d, th1dd + th2dd

        def midpoint_accel(prev_x, prev_y, length, phi, phid, phidd):
            # world acceleration of a point at +length along a segment of orientation phi,
            # added to the accel of the segment's proximal joint (prev_*).
            ax = prev_x + length * (phidd * -np.sin(phi) + phid**2 * -np.cos(phi))
            ay = prev_y + length * (phidd * np.cos(phi) + phid**2 * -np.sin(phi))
            return ax, ay

        # S2: midpoint of upper arm (proximal joint = shoulder, at rest)
        a2x, a2y = midpoint_accel(0.0, 0.0, 0.5 * self.l1, phi1, phi1d, phi1dd)
        # elbow accel (full upper-arm length), then S4 = midpoint of forearm
        eax, eay = midpoint_accel(0.0, 0.0, self.l1, phi1, phi1d, phi1dd)
        a4x, a4y = midpoint_accel(eax, eay, 0.5 * self.l2, phi2, phi2d, phi2dd)

        # specific force in body frame: R(phi)^T (a_world - g_world), g_world = [0, -G]
        f2x, f2y = _rot_T_apply(phi1, a2x, a2y + G)
        f4x, f4y = _rot_T_apply(phi2, a4x, a4y + G)

        return {
            "S2": {"gyro_z": phi1d, "accel": np.stack([f2x, f2y], axis=-1)},
            "S4": {"gyro_z": phi2d, "accel": np.stack([f4x, f4y], axis=-1)},
        }


def min_jerk(t, t0, tf, q0, qf):
    """Minimum-jerk joint trajectory on [t0, tf] (task 3.6 building block).

    The 5th-order min-jerk profile has zero velocity and acceleration at both ends —
    a good model of a smooth, voluntary point-to-point arm motion (prep -> active ->
    settle). Returns (q, qdot, qddot), each the same shape as ``t``. Outside [t0, tf]
    the angle holds and the derivatives are zero.
    """
    t = np.asarray(t, float)
    dt = tf - t0
    tau = np.clip((t - t0) / dt, 0.0, 1.0)
    s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
    sd = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / dt
    sdd = (60 * tau - 180 * tau**2 + 120 * tau**3) / dt**2
    return q0 + (qf - q0) * s, (qf - q0) * sd, (qf - q0) * sdd


if __name__ == "__main__":
    arm = PlanarArm()
    fs = 500.0  # Hz
    t = np.arange(0.0, 2.0, 1.0 / fs)
    # a reach: shoulder 10->60 deg, elbow 20->90 deg, over [0.4, 1.2] s
    th1, th1d, th1dd = min_jerk(t, 0.4, 1.2, np.deg2rad(10), np.deg2rad(60))
    th2, th2d, th2dd = min_jerk(t, 0.4, 1.2, np.deg2rad(20), np.deg2rad(90))
    imu = arm.ideal_imu(th1, th1d, th1dd, th2, th2d, th2dd)
    peak = np.abs(imu["S4"]["gyro_z"]).max()
    rest_acc = np.linalg.norm(imu["S4"]["accel"][0])
    print(f"reach: {len(t)} samples @ {fs:.0f} Hz")
    print(f"  S4 peak gyro = {np.rad2deg(peak):.1f} deg/s")
    print(f"  S4 |accel| at rest = {rest_acc:.3f} m/s^2 (expect ~{G})")
    fk = arm.forward_kinematics(th1[-1], th2[-1])
    print(f"  final wrist position = {fk['wrist'].round(3)} m")
