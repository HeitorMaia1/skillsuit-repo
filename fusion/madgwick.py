"""fusion.madgwick — the Madgwick gradient-descent orientation filter (task 4.1), derived
from scratch (not a library port).

Estimates a quaternion ``q = (w, x, y, z)`` describing the orientation of the IMU (body)
frame relative to a fixed earth frame, such that a body-frame vector maps to the earth frame
by the Hamilton sandwich product ``v_E = q (x) v_B (x) q^-1``. This is the same body->world
convention ``sim.arm3d.Arm3D`` uses for its analytic frame orientation ``C_i`` (a rotation
matrix), so a filter estimate and the simulator's ground truth can be compared directly.

Two update terms are fused every sample:

  1. **Gyroscope integration** — fast, but drifts (integrates noise/bias without bound).
  2. **Accelerometer correction** — a gradient-descent step that rotates the estimate so its
     predicted gravity direction matches the measured one. Slow (no dynamics info) and noisy,
     but does not drift: gravity is a fixed, known direction in the earth frame.

Magnetometer fusion (the full 9-DOF "MARG" variant of Madgwick 2010) is **not implemented**.
The reference IMU (ICM-42688-P, decision D2) has no magnetometer, and no ``imu_streams``
field carries one (see ``skilldata/schema.json``), so there is nothing in this pipeline to
fuse a magnetometer term against. Consequence, stated once here rather than re-derived at
every call site: gyro+accel alone cannot observe rotation about the *local vertical* (the
gravity direction does not change under a yaw-only rotation), so **yaw is unobservable and
will drift freely** — everything else (roll/pitch, i.e. tilt relative to gravity) converges
and stays bounded. This is a physical limitation of the sensor pair, not an approximation in
the filter; a heading reference (compass or an external fix) is the only fix.

Derivation
==========

**1. Gyroscope term — quaternion kinemetics.**

For a body rotating with angular velocity :math:`\\omega = (\\omega_x,\\omega_y,\\omega_z)`
(measured in the body frame), the orientation quaternion evolves as

.. math::
    \\dot q = \\tfrac{1}{2}\\, q \\otimes \\omega_q, \\qquad
    \\omega_q = (0, \\omega_x, \\omega_y, \\omega_z)

(standard quaternion kinematics: differentiate the sandwich product that carries a
body-fixed vector along with the rotating body, and use :math:`\\dot q q^{-1}` being the pure
quaternion :math:`\\tfrac12\\omega_q` for a rotation with angular velocity :math:`\\omega`).
Writing out the Hamilton product :math:`q \\otimes \\omega_q` component-wise for
:math:`q=(q_1,q_2,q_3,q_4)=(w,x,y,z)`:

.. math::
    \\dot q_\\omega = \\tfrac12
    \\begin{pmatrix}
        -q_2\\omega_x - q_3\\omega_y - q_4\\omega_z \\\\
         q_1\\omega_x + q_3\\omega_z - q_4\\omega_y \\\\
         q_1\\omega_y - q_2\\omega_z + q_4\\omega_x \\\\
         q_1\\omega_z + q_2\\omega_y - q_3\\omega_x
    \\end{pmatrix}

**2. Accelerometer term — gradient descent on a gravity-alignment objective.**

Let :math:`\\hat a = (a_x,a_y,a_z)` be the *normalized* accelerometer reading (the measured
specific-force direction; at low dynamic acceleration this is the reaction to gravity, i.e.
the direction of "up" as seen in the body frame). The earth-frame "up" reference is the unit
quaternion :math:`g_E = (0,0,0,1)`. Rotating it into the body frame with the *current*
orientation estimate :math:`q` gives the *predicted* reading:

.. math::
    \\hat g_B(q) = q^{-1} \\otimes g_E \\otimes q

Expanding the sandwich product symbolically in the components of :math:`q` gives a closed
form (this is the same expansion that produces a quaternion's rotation matrix; only the
column that multiplies :math:`(0,0,1)` is needed):

.. math::
    \\hat g_B(q) = \\begin{pmatrix}
        2(q_2 q_4 - q_1 q_3) \\\\
        2(q_1 q_2 + q_3 q_4) \\\\
        q_1^2 - q_2^2 - q_3^2 + q_4^2
    \\end{pmatrix}

The orientation is correct exactly when the prediction matches the measurement. Define the
residual (objective) :math:`f(q) = \\hat g_B(q) - \\hat a` and the scalar cost
:math:`\\Phi(q) = \\tfrac12\\|f(q)\\|^2`; minimizing :math:`\\Phi` by gradient descent needs
:math:`\\nabla\\Phi = J(q)^T f(q)`, where :math:`J = \\partial f/\\partial q` is:

.. math::
    J(q) = \\begin{pmatrix}
        -2q_3 &  2q_4 & -2q_1 &  2q_2 \\\\
         2q_2 &  2q_1 &  2q_4 &  2q_3 \\\\
         2q_1 & -2q_2 & -2q_3 &  2q_4
    \\end{pmatrix}

(each entry is the partial derivative of the corresponding row of :math:`f` with respect to
each of :math:`q_1..q_4` — direct differentiation of the closed form above).

**3. Fusion.** Following Madgwick (2010), the descent direction is *normalized* before being
scaled, so the correction has bounded angular rate ``beta`` regardless of how large the
current residual is (this is what makes ``beta`` behave like "max correction rate in rad/s"
rather than a generic step-size):

.. math::
    \\dot q_{accel} = -\\beta\\,\\frac{J(q)^T f(q)}{\\lVert J(q)^T f(q) \\rVert}, \\qquad
    \\dot q = \\dot q_\\omega + \\dot q_{accel}

Then a first-order (Euler) integration step and renormalization keep :math:`q` a unit
quaternion:

.. math::
    q_{t+1} = \\operatorname{normalize}\\big(q_t + \\dot q\\, \\Delta t\\big), \\qquad
    \\Delta t = 1/f_s

``beta`` trades off the two terms: larger values trust the (drift-free, noisy) accelerometer
more and correct faster; smaller values trust the (smooth, drifting) gyro integration more.
Defaults here (``beta=0.1``, ``sample_rate_hz=500.0``) match the reference sleeve's capture
rate (task 3.9's dataset) and the mid-range value Madgwick's paper reports as effective
across the gyroscopes it was validated against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------- #
# Quaternion utilities (Hamilton convention, q = (w, x, y, z), body -> earth)
# --------------------------------------------------------------------------- #
def quat_normalize(q):
    """Normalize a quaternion (or a ``(..., 4)`` batch) to unit norm."""
    q = np.asarray(q, float)
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)  # guard the zero-quaternion edge case
    return q / n


def quat_angle_error_deg(q1, q2):
    """Angle (degrees) of the relative rotation between two orientation quaternions.

    ``q1``/``q2`` are ``(..., 4)`` arrays (broadcastable). Uses
    ``angle = 2*arccos(|dot(q1, q2)|)``; the absolute value handles the double cover
    (``q`` and ``-q`` represent the same rotation), so this is well-defined for any pair of
    unit quaternions regardless of sign convention.
    """
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


def quat_from_axis_angle(axis, angle_rad):
    """Unit quaternion for a rotation of ``angle_rad`` about the unit vector ``axis``."""
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    half = angle_rad / 2.0
    return np.array([np.cos(half), *(axis * np.sin(half))])


def predicted_gravity_direction(q):
    """The earth 'up' reference (0,0,1), rotated into the body frame by orientation ``q``.

    This is :math:`\\hat g_B(q)` from the module derivation — what the accelerometer *should*
    read (normalized) if ``q`` were the true orientation and the sensor were stationary.
    """
    w, x, y, z = np.asarray(q, float)
    return np.array([2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z])


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #
@dataclass
class MadgwickFilter:
    """The Madgwick gyro+accelerometer orientation filter (task 4.1).

    ``sample_rate_hz`` sets the integration step ``dt = 1/sample_rate_hz`` (defaults to the
    500 Hz reference capture rate, task 3.9). ``beta`` is the accelerometer-correction gain
    (rad/s scale — see the module derivation). ``q0`` seeds the initial orientation estimate;
    default is identity (body frame == earth frame at ``t=0``). Magnetometer fusion is not
    supported (see module docstring) — this is the accel+gyro ("IMU") variant.
    """

    sample_rate_hz: float = 500.0
    beta: float = 0.1
    q0: tuple = (1.0, 0.0, 0.0, 0.0)
    q: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.q = quat_normalize(np.asarray(self.q0, float))

    def reset(self, q0=None):
        """Reset the filter state to ``q0`` (or the constructor's ``q0`` if omitted)."""
        self.q = quat_normalize(np.asarray(q0 if q0 is not None else self.q0, float))

    def update(self, gyro_rad_s, accel):
        """Advance the filter by one sample; returns the updated quaternion ``(w,x,y,z)``.

        ``gyro_rad_s`` — body-frame angular velocity, rad/s, shape ``(3,)``.
        ``accel`` — body-frame specific force, any consistent unit (normalized internally);
        a zero (or near-zero) reading disables the accelerometer term for that sample and the
        filter falls back to pure gyro integration (there is no gravity direction to correct
        against — e.g. free fall, or a degenerate all-zero placeholder sample).
        """
        w, x, y, z = self.q
        gx, gy, gz = gyro_rad_s

        # gyroscope term: qDot_omega = 0.5 * q (x) (0, wx, wy, wz)
        qd_w = 0.5 * (-x * gx - y * gy - z * gz)
        qd_x = 0.5 * (w * gx + y * gz - z * gy)
        qd_y = 0.5 * (w * gy - x * gz + z * gx)
        qd_z = 0.5 * (w * gz + x * gy - y * gx)

        a = np.asarray(accel, float)
        a_norm = np.linalg.norm(a)
        if a_norm > 1e-8:
            ax, ay, az = a / a_norm

            # objective f(q) = predicted gravity direction - measured direction
            f0 = 2.0 * (x * z - w * y) - ax
            f1 = 2.0 * (w * x + y * z) - ay
            f2 = (w * w - x * x - y * y + z * z) - az

            # gradient = J(q)^T f(q), J from the module derivation
            g_w = -2.0 * y * f0 + 2.0 * x * f1 + 2.0 * w * f2
            g_x = 2.0 * z * f0 + 2.0 * w * f1 - 2.0 * x * f2
            g_y = -2.0 * w * f0 + 2.0 * z * f1 - 2.0 * y * f2
            g_z = 2.0 * x * f0 + 2.0 * y * f1 + 2.0 * z * f2

            g_norm = np.sqrt(g_w * g_w + g_x * g_x + g_y * g_y + g_z * g_z)
            if g_norm > 1e-12:
                qd_w -= self.beta * g_w / g_norm
                qd_x -= self.beta * g_x / g_norm
                qd_y -= self.beta * g_y / g_norm
                qd_z -= self.beta * g_z / g_norm

        dt = 1.0 / self.sample_rate_hz
        self.q = quat_normalize(self.q + np.array([qd_w, qd_x, qd_y, qd_z]) * dt)
        return self.q

    def run(self, gyro_dps, accel, *, q0=None):
        """Run the filter over a whole stream. Returns the orientation quaternion per sample.

        ``gyro_dps`` — ``(T, 3)`` angular velocity in deg/s (the ``imu_streams`` unit; converted
        to rad/s internally). ``accel`` — ``(T, 3)`` specific force in any consistent unit (e.g.
        the ``imu_streams`` unit of g). Pass ``q0`` to seed the run without mutating the
        constructor default. Returns ``(T, 4)`` quaternions, one per input sample.
        """
        if q0 is not None:
            self.reset(q0)
        gyro = np.deg2rad(np.asarray(gyro_dps, float))
        accel = np.asarray(accel, float)
        n = gyro.shape[0]
        if accel.shape[0] != n:
            raise ValueError(f"gyro has {n} samples but accel has {accel.shape[0]}")
        out = np.empty((n, 4))
        for i in range(n):
            out[i] = self.update(gyro[i], accel[i])
        return out
