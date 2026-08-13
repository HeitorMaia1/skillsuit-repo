"""fusion.ekf — a multiplicative Extended Kalman Filter for orientation **with an explicit
gyro-bias state** (task 4.3), derived from scratch (not a library port).

This is the comparison point for ``fusion.madgwick``. It consumes exactly the same inputs
(body-frame gyroscope + accelerometer, same units, same call signature) and returns exactly
the same output representation — a unit quaternion ``q = (w, x, y, z)`` in the Hamilton,
body->earth convention ``v_E = q (x) v_B (x) q^-1``, matching ``sim.arm3d.Arm3D``'s analytic
frame orientation ``C_i`` — so the two filters are drop-in interchangeable in validation code.

**What it adds over Madgwick.** Madgwick has no mechanism to represent a *gyroscope bias*: a
slowly-varying offset added to every angular-rate reading (``sim.sensor.SensorModel`` injects
one — a 0.5 deg/s turn-on offset plus a random walk). Madgwick can only fight a bias
indirectly, by letting the accelerometer term pull the estimate back; it never learns the
offset, so it pays for it forever, and it cannot fight it at all in the one direction the
accelerometer cannot see (see "Observability" below). This filter carries the bias as three
extra state variables, estimates them online, and subtracts them from the gyro before
integrating.

State
=====

The *nominal* state carried between samples is 7 numbers::

    q  (4)   orientation quaternion, unit norm, body -> earth
    b  (3)   gyroscope bias, rad/s, in the body frame

The *estimated* state — the thing the covariance describes — is a 6-vector of small errors::

    dx = [ dtheta (3) , db (3) ]

``dtheta`` is a rotation vector (axis * angle, radians) expressed in the **body** frame,
defined multiplicatively::

    q_true = q_hat (x) dq(dtheta),      dq(dtheta) ~= (1, dtheta/2)

and ``db = b_true - b_hat``. This is the standard Multiplicative EKF (MEKF) formulation.

Why the covariance is 6x6 and not 7x7 — the unit-norm constraint
================================================================

**This is the single place naive quaternion EKFs go wrong silently, so it is spelled out.**

A unit quaternion has 4 numbers but only 3 degrees of freedom: ``|q| = 1`` removes one. An
"additive" EKF that puts the quaternion itself in the state vector and carries a 4x4
covariance block is therefore describing an error that *cannot happen* — an error along the
``q`` direction itself, which would change the norm. The consequences are not loud, which is
exactly the danger:

  * the true error covariance is singular in that direction, so the maintained 4x4 ``P``
    is inconsistent with reality (it claims uncertainty that the constraint forbids);
  * the Kalman update ``q <- q + K y`` moves the estimate *off* the unit sphere, and the
    usual fix — renormalizing afterwards — is a projection that ``P`` knows nothing about,
    so ``P`` no longer describes the error of the state actually being carried;
  * over a long run this shows up as a filter that is quietly over-confident, or that goes
    numerically indefinite, rather than as an obvious blow-up.

The multiplicative formulation removes the problem at the root rather than patching it:

  1. ``P`` is 6x6 and describes ``dtheta`` (3 parameters, a minimal local chart on the
     rotation group at ``q_hat``) and ``db``. There is no redundant direction, so ``P`` is
     full rank and consistent by construction.
  2. The quaternion is never added to. Every change to it is a **quaternion product** with a
     unit quaternion — the propagation step multiplies by the exact exponential map of the
     rotation increment, and the correction step multiplies by the exact exponential map of
     the estimated ``dtheta``. Both operands are unit by construction, so the product is unit
     by construction. ``q`` stays on the unit sphere *exactly*, in exact arithmetic.
  3. ``quat_normalize`` is still called after each product, but only to sweep up float
     round-off (parts in 1e-16), never to enforce a constraint the update violated. Removing
     it would change results in the 15th decimal place, not the 1st.
  4. After the correction is folded into ``q_hat``, the error state is **reset to zero** (its
     content now lives in the nominal quaternion) and ``P`` is mapped through the **reset
     Jacobian**::

         G = I - 0.5 [dtheta_hat]_x        (applied to the dtheta block only)

     which follows from ``exp(dtheta+/2) = exp(-dtheta_hat/2) (x) exp(dtheta/2)`` and the
     Baker-Campbell-Hausdorff expansion. It is common to drop this — it is ``O(correction
     angle)`` from the identity — and it *is* negligible once the filter has converged and
     corrections are microradians. It is **not** negligible during a transient, and dropping
     it has a specific failure mode worth naming, because it was measured here rather than
     assumed: ``P``'s eigenframe stops tracking ``g_hat``, ``P H^T`` therefore stops being
     perpendicular to ``g_hat``, and the correction acquires a small component about the very
     axis the measurement cannot see. Yaw has nothing to pull it back, so that component
     accumulates. Keeping ``G`` costs one 3x3 product per sample and roughly halves the
     settled error on the synthetic reach; it is kept.

Process model
=============

Gyroscope measurement model, with ``w_m`` the reading and ``n_g`` white noise::

    w_true = w_m - b - n_g,        b_dot = n_b        (bias random walk)

**Nominal propagation** (no linearization — this part is exact)::

    w_hat      = w_m - b_hat
    q_hat(k+1) = q_hat(k) (x) exp_q(w_hat * dt)
    b_hat(k+1) = b_hat(k)

``exp_q(phi) = (cos(|phi|/2), sin(|phi|/2) * phi/|phi|)`` is the exact unit quaternion of the
rotation vector ``phi``; the body-frame rate multiplies on the **right** because ``q`` is a
body->earth rotation. Using the exponential map rather than the first-order Euler step
``q + 0.5 q (x) w dt`` matters here: Euler is what forces a renormalization that changes the
answer, and it also accumulates a systematic error under sustained high rate — precisely the
regime (``throw``) where this filter is being asked to do better.

**Error propagation** (this is where the linearization lives). Differentiating the
multiplicative error definition gives the standard MEKF error dynamics::

    dtheta_dot = -[w_hat]_x dtheta - db - n_g
    db_dot     = n_b

so the discrete transition matrix over one step ``dt`` is::

            | Theta   -I*dt |                                  | sigma_g^2 dt I    0            |
    Phi =   |               |,   Theta = exp(-[w_hat]_x dt),  Q=|                                |
            |  0        I   |                                  |   0   sigma_b^2 dt I           |

``Theta`` is computed exactly (Rodrigues on the rotation vector ``-w_hat*dt``). The
off-diagonal block is exactly ``-integral_0^dt exp(-[w_hat]_x s) ds``, approximated by
``-I*dt``; likewise ``Q`` drops the ``O(dt^2)`` cross-term and the ``sigma_b^2 dt^3/3``
contribution to the ``dtheta`` block. At the 500 Hz reference rate ``dt = 2 ms`` and the
neglected terms are ~1e-3 of the retained ones. Documented, not hidden.

    P(k+1) = Phi P(k) Phi^T + Q

Measurement model
=================

The accelerometer, **normalized to a unit direction**, is treated as a measurement of the
earth "up" axis seen in the body frame::

    z      = a / |a|                                (measured)
    h(q)   = R(q)^T e_z = ( 2(xz - wy), 2(wx + yz), w^2 - x^2 - y^2 + z^2 )   (predicted)

which is the same ``predicted_gravity_direction`` the Madgwick module derives — the two
filters are corrected by the identical physical fact, so any performance difference comes
from *how* the correction is applied, not from a different measurement.

**Measurement Jacobian.** Substituting ``q_true = q_hat (x) dq`` and
``R(dq) ~= I + [dtheta]_x``::

    h(q_true) = R(dq)^T R(q_hat)^T e_z ~= (I - [dtheta]_x) g_hat = g_hat + [g_hat]_x dtheta

so, with ``g_hat = h(q_hat)``::

    H = [ [g_hat]_x   0_{3x3} ]        (3 x 6)

The bias block is zero: the accelerometer says nothing *instantaneously* about gyro bias. The
bias is observed only indirectly, through the propagation coupling ``-I*dt`` that lets a bias
error grow into an attitude error the accelerometer *can* see. That is why bias estimation
needs time, and why it needs motion (below).

Standard EKF update, in Joseph form::

    y = z - g_hat,   S = H P H^T + R_meas,   K = P H^T S^-1,   dx = K y
    P <- (I - K H) P (I - K H)^T + K R_meas K^T

Joseph form rather than the shorter ``(I - K H) P`` because ``H`` here is **rank 2**, not 3
(see below); with a rank-deficient measurement the short form loses symmetry and positive
definiteness to round-off over millions of samples, and Joseph does not.

Measurement noise — why a static ``R`` is not merely suboptimal but *unstable*
=============================================================================

``h(q) = R(q)^T e_z`` says the accelerometer measures gravity. During real motion it does
not: it measures specific force, gravity **plus the sensor's own linear acceleration**. On
this project's synthetic reach that is a direction error of up to 7.6 degrees — 13x the
0.6 degrees implied by the D2 accelerometer's 0.01 g white noise.

Feeding that mismatch into a filter with a small static ``R`` does not just produce a few
degrees of tilt error. It produces an unbounded **heading** error, and the mechanism is worth
stating because it is not obvious and it was measured here, not assumed. Each correction is
confined to the plane perpendicular to ``g_hat`` (that is what ``H``'s rank-2 structure
means), so no single correction touches yaw. But the body is rotating, so that plane is
rotating too, and a long sequence of large corrections about a moving axis does not commute:
the composition leaks into the one direction none of the individual corrections touched.
Yaw has no measurement to pull it back, so the leak is permanent and accumulates. Measured on
the noiseless synthetic reach with ``accel_noise=0.01``: tilt error stays under 4 degrees
throughout while total orientation error reaches **132 degrees**, essentially all of it
heading. Madgwick escapes this only because ``beta`` caps its correction rate at 5.7 deg/s —
it is too weak to leak. An EKF has no such cap; it must be told the truth about ``R`` instead.

So ``R`` carries two terms::

    R = ( accel_noise^2 + (accel_dynamic_gain * | ||a|| - 1 g |)^2 ) * I

The second is a *measurable* proxy for the unmodeled linear acceleration: whenever the
specific-force magnitude departs from 1 g, the difference is linear acceleration, and the
filter down-weights that sample in proportion. It is a lower bound (acceleration exactly
perpendicular to gravity changes the direction without changing the magnitude much), which
is why ``accel_noise`` itself also defaults well above the sensor's white-noise figure — it
absorbs the part of the dynamic error the proxy cannot see. Set ``accel_dynamic_gain=0`` for
the pure static model; the validation table in ``WORK/work11.md`` reports both.

Observability — the honest limit, identical to Madgwick's
=========================================================

``[g_hat]_x`` has ``g_hat`` in its null space. Rotation *about the local vertical* (yaw)
therefore produces no accelerometer residual and is **unobservable**, exactly as in
``fusion.madgwick`` — this is a property of the gyro+accel sensor pair (the reference
ICM-42688-P, decision D2, has no magnetometer), not of the estimator. Adding a bias state
does not repeal it. Two consequences that matter when reading validation numbers:

  * **Yaw drifts** in both filters. What the bias state can do is make it drift *slower*, by
    removing the deterministic part of the rate error that drives it.
  * **The bias component along ``g_hat`` is unobservable at any single instant.** All three
    bias components become observable only if the body *rotates*, because ``g_hat`` moves in
    the body frame and the unobservable direction moves with it. A sensor held perfectly
    still can never learn its vertical-axis bias; a sensor that tilts through a wide arc
    learns all three. ``tests/test_ekf.py`` tests both regimes explicitly.

Defaults
========

``sample_rate_hz=500`` (task 3.9's capture rate). The *gyro* noise parameters are taken
straight from the D2 sensor model that generated the dataset
(``skilldata.generate_synthetic._NOISE``): white noise 0.1 deg/s, bias random walk
0.02 deg/s/sqrt(s). ``accel_noise`` is expressed in the *normalized* measurement's units
(dimensionless direction error, so 0.01 g of noise on a 1 g vector is ~0.01) and deliberately
does **not** match the sensor's 0.01 g figure — it defaults to 0.15, fifteen times larger,
because the dominant error in this measurement is unmodeled linear acceleration, not sensor
noise. See "Measurement noise" above. ``accel_noise=0.15``, ``accel_dynamic_gain=20`` and
``p0_bias_dps=0.5`` were chosen by parameter sweep against the synthetic dataset; the sweep
and the reasoning for taking the mildest point of a flat optimum are in ``WORK/work11.md``.

What the bias state was actually worth — read this before believing the headline
===============================================================================

Honesty about a negative result, because the temptation to let the module's own framing stand
is exactly how a wrong belief survives. This filter **does** beat Madgwick on the synthetic
dataset, including on the ``lift`` class task 4.3 singled out. **The gyro-bias state is not
why.** Disabling the bias state entirely changes the dataset RMS by hundredths of a degree
(the ablation is in ``WORK/work11.md``); the improvement comes almost entirely from the
dynamic-acceleration term in ``R`` above. The reason is trial length: the dataset's motions
run 1.2-2.5 s, and a 0.5 deg/s turn-on bias only integrates to ~1 deg over 2.5 s — too small
to matter and, more to the point, too small to *identify* against an accelerometer carrying
several degrees of dynamic-acceleration error. The bias state needs a longer horizon before
it pays: on a 10 s hold appended to a real trial it cuts the error from 3.5 deg to 0.9 deg.
Both facts are real; only the second one is the case for carrying three extra states.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .madgwick import quat_normalize

_I3 = np.eye(3)
_I6 = np.eye(6)


# --------------------------------------------------------------------------- #
# Small helpers (Hamilton convention, q = (w, x, y, z), body -> earth)
# --------------------------------------------------------------------------- #
def skew(v):
    """The skew-symmetric matrix ``[v]_x`` with ``[v]_x u = v x u`` (cross product)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def rotvec_to_quat(phi):
    """Exact unit quaternion of the rotation vector ``phi`` (axis * angle, radians).

    ``exp_q(phi) = (cos(|phi|/2), sin(|phi|/2) * phi/|phi|)``, with the small-angle limit
    handled by the series so the zero-rotation case is exact rather than 0/0.
    """
    phi = np.asarray(phi, float)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-12:
        # sin(a/2)/a -> 1/2 as a -> 0; second-order term is O(a^2) and below float noise here
        return np.array([1.0, *(0.5 * phi)])
    half = 0.5 * angle
    return np.array([np.cos(half), *(phi * (np.sin(half) / angle))])


def rotvec_to_matrix(phi):
    """Exact rotation matrix of the rotation vector ``phi`` (Rodrigues' formula)."""
    phi = np.asarray(phi, float)
    angle = float(np.linalg.norm(phi))
    if angle < 1e-12:
        return _I3 + skew(phi)  # first order is exact to float precision at this scale
    k = skew(phi / angle)
    return _I3 + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def quat_mul(a, b):
    """Hamilton product ``a (x) b`` of two quaternions in ``(w, x, y, z)`` order."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


# --------------------------------------------------------------------------- #
# The filter
# --------------------------------------------------------------------------- #
@dataclass
class OrientationEKF:
    """Multiplicative EKF for orientation with an explicit gyro-bias state (task 4.3).

    Interface parity with ``fusion.madgwick.MadgwickFilter`` is deliberate: ``update()`` takes
    one sample of body-frame gyro (rad/s) + accelerometer (any consistent unit), ``run()``
    takes a whole stream with gyro in deg/s, and both return the orientation quaternion
    ``(w, x, y, z)``. The two filters can therefore be swapped in validation code without any
    other change.

    Parameters (all noise figures match the D2 sensor model unless overridden):

    ``sample_rate_hz``   integration/prediction rate; ``dt = 1/sample_rate_hz``.
    ``gyro_noise_dps``   gyro white-noise std, deg/s -> ``sigma_g``.
    ``bias_rw_dps_sqrt_s``  bias random-walk rate, deg/s per sqrt(s) -> ``sigma_b``.
    ``accel_noise``      std of the *normalized* accelerometer direction (dimensionless).
    ``q0``               initial orientation (default identity).
    ``b0``               initial bias estimate, deg/s (default zero — nothing is assumed known).
    ``p0_angle_deg``     initial 1-sigma attitude uncertainty, degrees.
    ``p0_bias_dps``      initial 1-sigma bias uncertainty, deg/s.
    ``accel_dynamic_gain``  dynamic-acceleration term (see "Measurement noise" above); set to
                         0 for the pure static model. Requires the accelerometer in **g**.
    """

    sample_rate_hz: float = 500.0
    gyro_noise_dps: float = 0.1
    bias_rw_dps_sqrt_s: float = 0.02
    accel_noise: float = 0.15
    q0: tuple = (1.0, 0.0, 0.0, 0.0)
    b0: tuple = (0.0, 0.0, 0.0)
    p0_angle_deg: float = 5.0
    p0_bias_dps: float = 0.5
    accel_dynamic_gain: float = 20.0

    q: np.ndarray = field(init=False, repr=False)
    b: np.ndarray = field(init=False, repr=False)
    P: np.ndarray = field(init=False, repr=False)

    def __post_init__(self):
        self.reset()

    # -- state management ---------------------------------------------------- #
    def reset(self, q0=None, b0=None):
        """Reset orientation, bias and covariance to their initial values.

        ``q0``/``b0`` override the constructor defaults for this run without mutating them
        (``b0`` in deg/s, as in the constructor).
        """
        self.q = quat_normalize(np.asarray(self.q0 if q0 is None else q0, float))
        self.b = np.deg2rad(np.asarray(self.b0 if b0 is None else b0, float)).copy()
        self.P = np.zeros((6, 6))
        self.P[:3, :3] = _I3 * np.deg2rad(self.p0_angle_deg) ** 2
        self.P[3:, 3:] = _I3 * np.deg2rad(self.p0_bias_dps) ** 2

    @property
    def bias_dps(self):
        """The current gyro-bias estimate in deg/s (the unit the streams are stored in)."""
        return np.rad2deg(self.b)

    def predicted_gravity(self):
        """``h(q_hat)`` — the earth up-axis rotated into the body frame by the estimate."""
        w, x, y, z = self.q
        return np.array([2 * (x * z - w * y), 2 * (w * x + y * z), w * w - x * x - y * y + z * z])

    # -- one sample ---------------------------------------------------------- #
    def update(self, gyro_rad_s, accel):
        """Advance one sample (predict + accelerometer correct); returns ``q = (w,x,y,z)``.

        ``gyro_rad_s`` — body-frame angular velocity, rad/s, shape ``(3,)``.
        ``accel`` — body-frame specific force, any consistent unit (normalized internally). A
        zero/near-zero reading skips the correction entirely (free fall, or a degenerate
        placeholder sample): there is no gravity direction to measure, so the filter coasts on
        gyro integration, and ``P`` correctly grows to say so.
        """
        dt = 1.0 / self.sample_rate_hz
        w_hat = np.asarray(gyro_rad_s, float) - self.b

        # ---- predict: nominal (exact) -------------------------------------- #
        self.q = quat_normalize(quat_mul(self.q, rotvec_to_quat(w_hat * dt)))

        # ---- predict: covariance ------------------------------------------- #
        theta_blk = rotvec_to_matrix(-w_hat * dt)          # exp(-[w_hat]_x dt)
        phi = np.zeros((6, 6))
        phi[:3, :3] = theta_blk
        phi[:3, 3:] = -_I3 * dt
        phi[3:, 3:] = _I3
        sig_g = np.deg2rad(self.gyro_noise_dps)
        sig_b = np.deg2rad(self.bias_rw_dps_sqrt_s)
        q_proc = np.zeros((6, 6))
        q_proc[:3, :3] = _I3 * (sig_g**2 * dt)
        q_proc[3:, 3:] = _I3 * (sig_b**2 * dt)
        self.P = phi @ self.P @ phi.T + q_proc

        # ---- correct: accelerometer ---------------------------------------- #
        a = np.asarray(accel, float)
        a_norm = float(np.linalg.norm(a))
        if a_norm > 1e-8:
            z_meas = a / a_norm
            g_hat = self.predicted_gravity()

            h_mat = np.zeros((3, 6))
            h_mat[:, :3] = skew(g_hat)

            # measurement noise = static direction noise + a dynamic-acceleration term,
            # inferred from how far the specific-force magnitude sits from 1 g (see the
            # module docstring's "Measurement noise" section for why this is not optional)
            var = self.accel_noise**2
            if self.accel_dynamic_gain:
                var += (self.accel_dynamic_gain * (a_norm - 1.0)) ** 2
            r_meas = _I3 * var

            s_mat = h_mat @ self.P @ h_mat.T + r_meas
            k_gain = np.linalg.solve(s_mat.T, (self.P @ h_mat.T).T).T   # P H^T S^-1
            dx = k_gain @ (z_meas - g_hat)

            # Joseph form: symmetry + positive definiteness survive a rank-2 H
            ikh = _I6 - k_gain @ h_mat
            self.P = ikh @ self.P @ ikh.T + k_gain @ r_meas @ k_gain.T

            # ---- reset: fold the error state into the nominal state --------- #
            # Multiplicative, via the exact exponential map -> |q| is preserved by
            # construction, never restored by projection (see the module docstring).
            self.q = quat_normalize(quat_mul(self.q, rotvec_to_quat(dx[:3])))
            self.b = self.b + dx[3:]

            # ...and carry P through the reset Jacobian, so P's eigenframe keeps tracking
            # g_hat. Dropping this leaks correction into the unobservable yaw axis.
            g_reset = _I6.copy()
            g_reset[:3, :3] = _I3 - 0.5 * skew(dx[:3])
            self.P = g_reset @ self.P @ g_reset.T

        self.P = 0.5 * (self.P + self.P.T)   # kill accumulated round-off asymmetry
        return self.q

    # -- a whole stream ------------------------------------------------------ #
    def run(self, gyro_dps, accel, *, q0=None, b0=None, return_bias=False):
        """Run over a whole stream. Returns ``(T, 4)`` quaternions, one per input sample.

        ``gyro_dps`` — ``(T, 3)`` angular velocity in deg/s (the ``imu_streams`` unit).
        ``accel`` — ``(T, 3)`` specific force; supply it in **g** if ``accel_gate_g`` is set.
        ``q0``/``b0`` seed the run without mutating the constructor defaults (and reset ``P``).
        ``return_bias=True`` additionally returns the ``(T, 3)`` bias estimate in deg/s — the
        state Madgwick has no equivalent of, and the thing task 4.4's bias test grades.
        """
        self.reset(q0=q0, b0=b0)
        gyro = np.deg2rad(np.asarray(gyro_dps, float))
        accel = np.asarray(accel, float)
        n = gyro.shape[0]
        if accel.shape[0] != n:
            raise ValueError(f"gyro has {n} samples but accel has {accel.shape[0]}")
        out = np.empty((n, 4))
        bias = np.empty((n, 3)) if return_bias else None
        for i in range(n):
            out[i] = self.update(gyro[i], accel[i])
            if return_bias:
                bias[i] = self.b
        if return_bias:
            return out, np.rad2deg(bias)
        return out
