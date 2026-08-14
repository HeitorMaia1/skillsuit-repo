"""Rigid-body dynamics for the 3D serial-chain arm (task 3.12).

Why this module exists
----------------------
`sim/arm.py` and `sim/arm3d.py` are **pure kinematics**: a min-jerk joint trajectory is
prescribed, and exact analytic differentiation gives omega/alpha and an ideal IMU. Nothing in
them knows about mass, inertia, torque or energy. That is fine for the fusion work (Phase 4,
which only needs orientation) but it is fatal for Phase 5.

Task 5.9 claims that a port-Hamiltonian network's *learned dissipation* concentrates in the
prep and settle phases and is near zero during the active phase. Fitted to purely kinematic
data, that claim cannot be tested: the generating process has **no dissipation at all**, so the
true damping is identically zero everywhere. A pHNN would still draw a plausible curve, because
min-jerk deceleration *looks* like damping. The figure would be an artefact of the trajectory
generator rather than a result. This module supplies the missing physics — momentum, energy,
torque, and a **known** viscous damping matrix `B` — so that 5.8/5.9 can be scored as "the
network recovered `B` to within X%" instead of "the curve matches the story".

Model
-----
An `Arm3D` chain of `n` revolute joints. Segment `i` carries a rigid `Body` with mass `m_i`,
a centre of mass offset `c_i` from the segment's **proximal joint**, and an inertia tensor
`I_i` taken **about the centre of mass** — both expressed in segment `i`'s own frame. Segments
that exist only to stack a rotation axis at a point (zero-length links, e.g. the three shoulder
axes) carry zero mass and zero inertia; they still enter the dynamics, because they move the
bodies downstream of them.

Kinematics reused from `arm3d`: frame `i`'s orientation is `C_i = C_{i-1} R(axis_i, q_i)`, joint
`i` rotates about the world axis `k_i = C_{i-1} axis_i` through the world point `o_i` (the
segment's proximal joint), and the body's centre of mass sits at `r_i = o_i + C_i c_i`.

Derivation
----------
**Geometric Jacobians.** For a chain of revolute joints, the world velocity of body `i`'s centre
of mass and its angular velocity are linear in `qdot`:

    v_i = J_v^i qdot,      J_v^i[:, j] = k_j x (r_i - o_j)   for j <= i, else 0
    w_i = J_w^i qdot,      J_w^i[:, j] = k_j                  for j <= i, else 0

The `j <= i` condition is the serial-chain structure: joint `j` cannot move body `i` if it sits
further out along the chain.

**Inertia matrix.** Kinetic energy is `T = 1/2 sum_i (m_i |v_i|^2 + w_i^T I_i^world w_i)` with
`I_i^world = C_i I_i C_i^T` (the inertia tensor rotated into world coordinates). Substituting the
Jacobians and factoring out `qdot`:

    M(q) = sum_i [ m_i J_v^i^T J_v^i + J_w^i^T C_i I_i C_i^T J_w^i ],     T = 1/2 qdot^T M qdot

`M` is symmetric by construction (each term is a congruence `A^T S A` with `S` symmetric) and
positive definite whenever every body's inertia is positive definite and every joint has some
mass downstream of it. Test (a) checks this numerically rather than trusting the argument.

**Gravity.** Potential energy is `U(q) = -sum_i m_i g_world . r_i`; with the default z-up gravity
`g_world = (0,0,-G)` this is the familiar `sum_i m_i G z_i`. The generalized gravity torque is

    g(q)_k = dU/dq_k = -sum_i m_i ( g_world . J_v^i[:, k] )

which reuses the Jacobians already computed for `M` — no separate derivation, and no finite
differencing.

**Coriolis / centrifugal.** `C(q, qdot)` is built from the Christoffel symbols of the first kind:

    C_kj = sum_i c_ijk qdot_i,    c_ijk = 1/2 ( dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k )

This particular construction matters, and it is why the passivity test is the one that
discriminates a correct `C` from a merely plausible one. Writing `Mdot_kj = sum_i (dM_kj/dq_i)
qdot_i` and subtracting:

    (Mdot - 2C)_kj = sum_i [ dM_ij/dq_k - dM_ki/dq_j ] qdot_i

Adding the transpose and using the symmetry of `M` (`dM_ij/dq_k = dM_ji/dq_k`) makes every term
cancel, so `Mdot - 2C` is **skew-symmetric identically** — not approximately, and not only at
equilibrium. That is the passivity identity, and it is what makes the energy balance below exact.
Many wrong-but-plausible `C` matrices reproduce the correct `C qdot` product while failing this.

`dM/dq` is evaluated by **complex-step differentiation**: `df/dx = Im(f(x + ih))/h` with
`h = 1e-20`. Unlike a finite difference this has no subtractive cancellation, so it is accurate to
machine precision (~1e-16 relative) rather than to `sqrt(eps)` (~1e-8). It works here because
every operation in the mass-matrix path — Rodrigues rotations, matrix products, cross products —
is analytic and dtype-generic; the only real-valued step, normalising the joint axes, is done once
at construction time on constants. Tests (b) and (e) both rely on this being essentially exact.

**Equations of motion.** With actuation `tau` and viscous joint damping `tau_d = -B qdot` for a
configurable diagonal `B >= 0`:

    M(q) qddot + C(q, qdot) qdot + g(q) + B qdot = tau

    forward  dynamics:  qddot = M^-1 ( tau - C qdot - g - B qdot )
    inverse  dynamics:  tau   = M qddot + C qdot + g + B qdot

The inverse path is what turns the existing min-jerk trials into dynamically labelled data
(task 3.14) instead of throwing their torques away.

**Energy balance — the quantity task 5.8's loss will use.** With `H = 1/2 qdot^T M qdot + U(q)`,

    Hdot = qdot^T M qddot + 1/2 qdot^T Mdot qdot + qdot^T g
         = qdot^T ( tau - C qdot - g - B qdot ) + 1/2 qdot^T Mdot qdot + qdot^T g
         = qdot^T tau - qdot^T B qdot            [using qdot^T (Mdot - 2C) qdot = 0]

so with no actuation `Hdot = -qdot^T B qdot` exactly: energy leaves the system only through the
damping, at a rate that is quadratic in velocity and known in closed form. Setting `B = 0` and
`tau = 0` gives `Hdot = 0`, conservation — test (c). Setting `B != 0` gives the balance residual
`|Hdot + qdot^T B qdot|` — test (d), and the residual a pHNN must learn to reproduce.

**Momentum.** The port-Hamiltonian state is `(q, p)` with `p = M(q) qdot`, and
`H(q, p) = 1/2 p^T M(q)^-1 p + U(q)`. Both are provided so Phase 5 never has to reconstruct them.

Anthropometry
-------------
Segment masses, centre-of-mass positions and radii of gyration are the **adjusted
Zatsiorsky-Seluyanov parameters of de Leva (1996)**, male column:

    de Leva, P. (1996). "Adjustments to Zatsiorsky-Seluyanov's segment inertia parameters."
    Journal of Biomechanics 29(9), 1223-1230. doi:10.1016/0021-9290(95)00178-6

de Leva re-referenced the original Zatsiorsky-Seluyanov gamma-ray-scan parameters from bony
landmarks to **joint centres**, which is exactly the convention this chain uses, so the numbers
drop in without re-referencing. Values were taken from the C-Motion/Visual3D documentation table
of those adjusted parameters (`has-motion.com`, retrieved 2026-08-14), which reproduces de Leva's
Tables and cites the DOI above; the Elsevier full text is paywalled, so the primary table was not
read directly. The three numbers used per segment are relative mass (fraction of total body mass),
relative CoM position (fraction of segment length from the proximal joint), and the three relative
radii of gyration (fractions of segment length), from which `I = m (l r)^2` about each principal
axis through the CoM.

**Axis mapping, stated because it is a real modelling choice.** de Leva reports radii about the
sagittal, transverse and longitudinal axes. This chain lays every link along its own frame's `+x`,
so the *longitudinal* radius maps to local `x` and the two transverse radii map to local `y` and
`z`. Which of sagittal/transverse goes to `y` versus `z` is not determined by this model, since
`Arm3D` does not fix an anatomical plane; the assignment below is sagittal -> `y`, transverse ->
`z`. For the upper arm the two differ by about 6% (0.285 vs 0.269), so the choice moves the
inertia tensor by a few percent at most, well inside the between-subject spread de Leva reports.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .arm import G
from .arm3d import Arm3D, human_arm_7dof

# --------------------------------------------------------------------------- #
# de Leva (1996), male column — see the module docstring for the citation.
# (relative mass, relative CoM from proximal, (r_sagittal, r_transverse, r_longitudinal))
# --------------------------------------------------------------------------- #
DE_LEVA_MALE = {
    "upper_arm": (0.0271, 0.5772, (0.285, 0.269, 0.158)),
    "forearm": (0.0162, 0.4574, (0.276, 0.265, 0.121)),
    "hand": (0.0061, 0.7900, (0.628, 0.513, 0.401)),
}

_COMPLEX_STEP = 1e-20  # h in Im(f(x+ih))/h; no cancellation, so it can be this small


@dataclass(frozen=True)
class Body:
    """Rigid body riding one segment. Zero mass marks a pure axis-stacking segment."""

    mass: float = 0.0
    com: tuple = (0.0, 0.0, 0.0)  # CoM offset from the proximal joint, segment frame [m]
    inertia: tuple = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # about the CoM

    @property
    def I(self) -> np.ndarray:  # noqa: E743 - matches the symbol used in the derivation
        return np.asarray(self.inertia, float)


def body_from_de_leva(segment: str, length: float, body_mass: float) -> Body:
    """Build a `Body` for `segment` ('upper_arm' | 'forearm' | 'hand') of the given length.

    `length` is the segment length [m], `body_mass` the subject's total mass [kg]. Applies
    `m = m_rel * body_mass`, `c = c_rel * length` along the segment's own `+x`, and
    `I_principal = m (length * r)^2` with the axis mapping documented in the module docstring
    (longitudinal -> x, sagittal -> y, transverse -> z).
    """
    if segment not in DE_LEVA_MALE:
        raise ValueError(f"unknown segment {segment!r}; choose from {tuple(DE_LEVA_MALE)}")
    m_rel, c_rel, (r_sag, r_trans, r_long) = DE_LEVA_MALE[segment]
    m = m_rel * body_mass
    principal = m * (length * np.array([r_long, r_sag, r_trans])) ** 2
    return Body(mass=m, com=(c_rel * length, 0.0, 0.0), inertia=tuple(map(tuple, np.diag(principal))))


def _rodrigues(axis_unit: np.ndarray, theta):
    """Rotation about a **pre-normalised** real axis by `theta`. dtype-generic in `theta`."""
    a = axis_unit
    K = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])
    return np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)


@dataclass(frozen=True)
class ArmDynamics:
    """Rigid-body dynamics on an `Arm3D` chain: M(q), C(q,qdot), g(q), energy, momentum, rollout.

    `bodies` is one `Body` per segment, in the chain's order. `damping` is the diagonal of the
    viscous joint-damping matrix `B` [N m s / rad]; a scalar broadcasts to every joint. `B = 0`
    makes the system conservative, which is what test (c) exercises.
    """

    arm: Arm3D
    bodies: tuple
    damping: tuple = ()

    def __post_init__(self):
        if len(self.bodies) != len(self.arm.segments):
            raise ValueError(
                f"need one Body per segment: {len(self.bodies)} bodies, "
                f"{len(self.arm.segments)} segments"
            )
        # cache pre-normalised real axes so the complex-step path never touches np.linalg.norm
        axes = np.array([np.asarray(s.axis, float) for s in self.arm.segments])
        axes = axes / np.linalg.norm(axes, axis=1, keepdims=True)
        object.__setattr__(self, "_axes", axes)
        object.__setattr__(self, "_links", np.array([np.asarray(s.link, float) for s in self.arm.segments]))
        object.__setattr__(self, "_masses", np.array([b.mass for b in self.bodies], float))
        object.__setattr__(self, "_inertias", np.array([b.I for b in self.bodies], float))
        object.__setattr__(self, "_coms", np.array([np.asarray(b.com, float) for b in self.bodies]))
        b = np.zeros(len(self.bodies)) if len(self.damping) == 0 else np.asarray(self.damping, float)
        if b.ndim == 0:
            b = np.full(len(self.bodies), float(b))
        if b.shape != (len(self.bodies),):
            raise ValueError(f"damping must be scalar or length {len(self.bodies)}; got {b.shape}")
        if np.any(b < 0):
            raise ValueError("damping must be non-negative (B >= 0 or the system gains energy)")
        object.__setattr__(self, "B", np.diag(b))

    @property
    def n(self) -> int:
        return len(self.bodies)

    # ---- kinematics needed by the dynamics (dtype-generic in q) ------------ #
    def _chain(self, q):
        """Joint world axes/origins, segment orientations, and body CoM positions at `q`.

        Returns `(axes_w, origins_w, rots, coms_w)` with shapes `(n,3) (n,3) (n,3,3) (n,3)`.
        `origins_w[i]` is the world position of joint `i` (the segment's proximal end).
        """
        q = np.asarray(q)
        dt = np.result_type(q.dtype, float)
        n = self.n
        C = np.eye(3, dtype=dt)
        p = np.zeros(3, dtype=dt)
        axes_w = np.zeros((n, 3), dtype=dt)
        origins_w = np.zeros((n, 3), dtype=dt)
        rots = np.zeros((n, 3, 3), dtype=dt)
        coms_w = np.zeros((n, 3), dtype=dt)

        for i in range(n):
            axes_w[i] = C @ self._axes[i]  # k_i uses C_{i-1}
            origins_w[i] = p  # joint i sits at the previous distal point
            Ci = C @ _rodrigues(self._axes[i], q[i])
            rots[i] = Ci
            coms_w[i] = p + Ci @ np.asarray(self.bodies[i].com, float)
            p = p + Ci @ self._links[i]
            C = Ci
        return axes_w, origins_w, rots, coms_w

    def jacobians(self, q, chain=None):
        """Linear and angular Jacobians of every body's CoM: `(J_v, J_w)`, each `(n, 3, n)`.

        Fully vectorized: `J_v[i,:,j] = k_j x (r_i - o_j)` and `J_w[i,:,j] = k_j`, masked by the
        lower-triangular `j <= i` serial-chain condition. Pass a precomputed `chain` to avoid
        walking the chain twice.
        """
        axes_w, origins_w, _rots, coms_w = self._chain(q) if chain is None else chain
        n = self.n
        mask = np.tril(np.ones((n, n)))                      # (i, j), 1 where j <= i
        diff = coms_w[:, None, :] - origins_w[None, :, :]     # (i, j, 3) = r_i - o_j
        Jv = np.cross(np.broadcast_to(axes_w[None, :, :], diff.shape), diff) * mask[:, :, None]
        Jw = np.broadcast_to(axes_w[None, :, :], diff.shape) * mask[:, :, None]
        return Jv.transpose(0, 2, 1), Jw.transpose(0, 2, 1)   # -> (i, 3, j)

    # ---- M, g, C ----------------------------------------------------------- #
    def mass_matrix(self, q, chain=None):
        """Inertia matrix `M(q)`, shape `(n, n)`. Symmetric by construction."""
        chain = self._chain(q) if chain is None else chain
        rots = chain[2]
        J_v, J_w = self.jacobians(q, chain)
        Iw = np.einsum("iab,ibc,idc->iad", rots, self._inertias, rots)  # C_i I_i C_i^T
        M = np.einsum("i,iaj,iak->jk", self._masses, J_v, J_v)
        M = M + np.einsum("iaj,iab,ibk->jk", J_w, Iw, J_w)
        return 0.5 * (M + M.T)  # kill the ~1e-18 asymmetry from floating-point summation order

    def potential_energy(self, q, chain=None) -> float:
        """`U(q) = -sum_i m_i g_world . r_i` [J]."""
        coms_w = (self._chain(q) if chain is None else chain)[3]
        g_w = np.asarray(self.arm.g_world, float)
        return -float(self._masses @ (coms_w @ g_w))

    def gravity(self, q, chain=None):
        """Generalized gravity torque `g(q)_k = dU/dq_k`, shape `(n,)`."""
        chain = self._chain(q) if chain is None else chain
        J_v = self.jacobians(q, chain)[0]
        g_w = np.asarray(self.arm.g_world, float)
        return -np.einsum("i,a,iaj->j", self._masses, g_w, J_v)

    def velocities(self, q, qd, chain=None):
        """Body angular velocities and CoM linear velocities: `(w, v)`, each `(n, 3)`.

        Standard O(n) forward velocity recursion — no Jacobian assembly, so it is the cheap path
        `_two_T` (and hence `coriolis_times_qd`) uses:

            w_i       = w_{i-1} + k_i qdot_i
            v_com_i   = v_o_i + w_i x (r_i - o_i)
            v_o_{i+1} = v_o_i + w_i x d_i,        d_i = C_i link_i
        """
        axes_w, origins_w, rots, coms_w = self._chain(q) if chain is None else chain
        qd = np.asarray(qd)
        dt = np.result_type(axes_w.dtype, qd.dtype, float)
        n = self.n
        w = np.zeros((n, 3), dtype=dt)
        v = np.zeros((n, 3), dtype=dt)
        w_i = np.zeros(3, dtype=dt)
        v_o = np.zeros(3, dtype=dt)  # velocity of joint i's origin point; the base is fixed
        for i in range(n):
            w_i = w_i + axes_w[i] * qd[i]
            w[i] = w_i
            v[i] = v_o + np.cross(w_i, coms_w[i] - origins_w[i])
            v_o = v_o + np.cross(w_i, rots[i] @ self._links[i])
        return w, v

    def _two_T(self, q, qd, chain=None):
        """`qdot^T M(q) qdot` (= twice the kinetic energy) via the O(n) recursion, dtype-generic."""
        chain = self._chain(q) if chain is None else chain
        w, v = self.velocities(q, qd, chain)
        rots = chain[2]
        Iw = np.einsum("iab,ibc,idc->iad", rots, self._inertias, rots)
        return np.einsum("i,ia,ia->", self._masses, v, v) + np.einsum("ia,iab,ib->", w, Iw, w)

    def dM_dq(self, q):
        """`dM/dq` by complex-step differentiation, shape `(n, n, n)` indexed `[k, j, i]`.

        Element `[k, j, i]` is `dM_kj/dq_i`. Accurate to machine precision — see the module
        docstring for why the complex step beats a finite difference here.
        """
        q = np.asarray(q, float)
        out = np.zeros((self.n, self.n, self.n))
        for i in range(self.n):
            qc = q.astype(complex)
            qc[i] += 1j * _COMPLEX_STEP
            out[:, :, i] = self.mass_matrix(qc).imag / _COMPLEX_STEP
        return out

    def coriolis(self, q, qd):
        """Coriolis/centrifugal matrix `C(q, qdot)` via Christoffel symbols, shape `(n, n)`.

        Built so that `Mdot - 2C` is skew-symmetric identically (the passivity identity), which
        is what makes the energy balance `Hdot = qdot^T tau - qdot^T B qdot` exact.
        """
        qd = np.asarray(qd, float)
        dM = self.dM_dq(q)
        # c_ijk = 1/2 (dM_kj/dq_i + dM_ki/dq_j - dM_ij/dq_k); C_kj = sum_i c_ijk qdot_i
        term = dM.transpose(0, 1, 2) + dM.transpose(0, 2, 1) - dM.transpose(2, 1, 0)
        return 0.5 * np.einsum("kji,i->kj", term, qd)

    def mass_matrix_dot(self, q, qd):
        """`Mdot = sum_i (dM/dq_i) qdot_i`, shape `(n, n)`.

        One complex step along `qdot` — a directional derivative, so this costs a single
        mass-matrix evaluation rather than `n` of them.
        """
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        return self.mass_matrix(q + 1j * _COMPLEX_STEP * qd).imag / _COMPLEX_STEP

    def coriolis_times_qd(self, q, qd):
        """`C(q, qdot) qdot`, shape `(n,)` — the fast path, without forming `C`.

        From the Christoffel definition, the two terms symmetric in `i <-> j` collapse onto
        `Mdot qdot` and the third is a plain gradient:

            C qdot = Mdot qdot - 1/2 grad_q ( qdot^T M(q) qdot )

        `Mdot qdot` is one complex step along `qdot`; the gradient is `n` complex steps on the
        **scalar** `qdot^T M qdot`, which the O(n) velocity recursion computes without ever
        assembling `M`. That is `n + 1` cheap evaluations instead of the `n` full mass matrices
        `coriolis()` needs, and it is what makes rollouts and task 3.14 tractable.
        `tests/test_dynamics.py` asserts this equals `coriolis(q, qd) @ qd`.
        """
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        Mdot_qd = self.mass_matrix_dot(q, qd) @ qd
        grad = np.empty(self.n)
        for k in range(self.n):
            qc = q.astype(complex)
            qc[k] += 1j * _COMPLEX_STEP
            grad[k] = self._two_T(qc, qd).imag / _COMPLEX_STEP
        return Mdot_qd - 0.5 * grad

    # ---- energy and momentum ----------------------------------------------- #
    def momentum(self, q, qd):
        """Generalized momentum `p = M(q) qdot`, shape `(n,)`."""
        return self.mass_matrix(q) @ np.asarray(qd, float)

    def kinetic_energy(self, q, qd) -> float:
        qd = np.asarray(qd, float)
        return 0.5 * float(qd @ self.mass_matrix(q) @ qd)

    def energy(self, q, qd) -> float:
        """Total mechanical energy `H = T + U` from `(q, qdot)` [J]."""
        return self.kinetic_energy(q, qd) + self.potential_energy(q)

    def hamiltonian(self, q, p) -> float:
        """`H(q, p) = 1/2 p^T M(q)^-1 p + U(q)` — the port-Hamiltonian form [J]."""
        p = np.asarray(p, float)
        return 0.5 * float(p @ np.linalg.solve(self.mass_matrix(q), p)) + self.potential_energy(q)

    # ---- equations of motion ------------------------------------------------ #
    def forward_dynamics(self, q, qd, tau=None):
        """`qddot = M^-1 (tau - C qdot - g - B qdot)`, shape `(n,)`."""
        qd = np.asarray(qd, float)
        tau = np.zeros(self.n) if tau is None else np.asarray(tau, float)
        chain = self._chain(q)
        rhs = tau - self.coriolis_times_qd(q, qd) - self.gravity(q, chain) - self.B @ qd
        return np.linalg.solve(self.mass_matrix(q, chain), rhs)

    def inverse_dynamics(self, q, qd, qdd):
        """`tau = M qddot + C qdot + g + B qdot`, shape `(n,)`.

        This is the path that gives the prescribed min-jerk trials their torques (task 3.14)
        rather than discarding them.
        """
        qd, qdd = np.asarray(qd, float), np.asarray(qdd, float)
        chain = self._chain(q)
        return (self.mass_matrix(q, chain) @ qdd + self.coriolis_times_qd(q, qd)
                + self.gravity(q, chain) + self.B @ qd)

    def inverse_dynamics_traj(self, qs, qds, qdds):
        """Vectorized `inverse_dynamics` over a trajectory. `(T,n)` in -> `(T,n)` torques out."""
        qs, qds, qdds = (np.asarray(a, float) for a in (qs, qds, qdds))
        return np.stack([self.inverse_dynamics(qs[k], qds[k], qdds[k]) for k in range(qs.shape[0])])

    def trajectory_energetics(self, qs, qds):
        """Per-sample `(p, H, U)` along a trajectory: `(T,n)`, `(T,)`, `(T,)`."""
        lab = self.label_trajectory(qs, qds)
        return lab["p"], lab["H"], lab["U"]

    def label_trajectory(self, qs, qds, qdds=None):
        """Everything task 3.14 needs, in one pass over the trajectory.

        Returns a dict of arrays: `p` (momentum), `H` (total energy), `U` (potential), `T_kin`,
        `power_dissipated` (`qdot^T B qdot`, the true dissipation rate), `lambda_min` (smallest
        eigenvalue of `M`, so Phase 5 can see where the chain is near-singular), and — when `qdds`
        is given — `tau` from inverse dynamics.

        Single-pass because the chain is walked once per sample and shared between `M`, `g` and
        the energy terms; the Coriolis term dominates the remaining cost.
        """
        qs, qds = np.asarray(qs, float), np.asarray(qds, float)
        qdds = None if qdds is None else np.asarray(qdds, float)
        T = qs.shape[0]
        out = {
            "p": np.zeros((T, self.n)),
            "H": np.zeros(T),
            "U": np.zeros(T),
            "T_kin": np.zeros(T),
            "power_dissipated": np.zeros(T),
            "lambda_min": np.zeros(T),
        }
        if qdds is not None:
            out["tau"] = np.zeros((T, self.n))

        for k in range(T):
            q, qd = qs[k], qds[k]
            chain = self._chain(q)
            M = self.mass_matrix(q, chain)
            out["p"][k] = M @ qd
            out["T_kin"][k] = 0.5 * qd @ M @ qd
            out["U"][k] = self.potential_energy(q, chain)
            out["H"][k] = out["T_kin"][k] + out["U"][k]
            out["power_dissipated"][k] = qd @ self.B @ qd
            out["lambda_min"][k] = np.linalg.eigvalsh(M)[0]
            if qdds is not None:
                out["tau"][k] = (M @ qdds[k] + self.coriolis_times_qd(q, qd)
                                 + self.gravity(q, chain) + self.B @ qd)
        return out

    def simulate(self, q0, qd0, t, tau=None):
        """Fixed-step RK4 rollout of the forward dynamics. Returns `(qs, qds)`, each `(T, n)`.

        `t` must be uniformly spaced. `tau` is `None`, a constant `(n,)` vector, or a callable
        `tau(time, q, qdot) -> (n,)`.
        """
        t = np.asarray(t, float)
        dt = float(t[1] - t[0])
        if not np.allclose(np.diff(t), dt):
            raise ValueError("simulate() needs a uniformly spaced time vector")

        if tau is None:
            tau_fn = lambda _t, _q, _qd: None  # noqa: E731
        elif callable(tau):
            tau_fn = tau
        else:
            tau_const = np.asarray(tau, float)
            tau_fn = lambda _t, _q, _qd: tau_const  # noqa: E731

        def deriv(time, y):
            q, qd = y[: self.n], y[self.n :]
            return np.concatenate([qd, self.forward_dynamics(q, qd, tau_fn(time, q, qd))])

        y = np.concatenate([np.asarray(q0, float), np.asarray(qd0, float)])
        out = np.zeros((t.size, 2 * self.n))
        out[0] = y
        for k in range(t.size - 1):
            tk = t[k]
            k1 = deriv(tk, y)
            k2 = deriv(tk + 0.5 * dt, y + 0.5 * dt * k1)
            k3 = deriv(tk + 0.5 * dt, y + 0.5 * dt * k2)
            k4 = deriv(tk + dt, y + dt * k3)
            y = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            out[k + 1] = y
        return out[:, : self.n], out[:, self.n :]


# --------------------------------------------------------------------------- #
# Factories
# --------------------------------------------------------------------------- #
def human_arm_7dof_dynamics(
    body_mass: float = 75.0,
    l_upper: float = 0.30,
    l_fore: float = 0.25,
    l_hand: float = 0.08,
    damping=0.05,
) -> ArmDynamics:
    """The 7-DOF right arm of `arm3d.human_arm_7dof` with de Leva (1996) male inertial parameters.

    Mass sits on the three real segments — upper arm (joint 2), forearm (joint 3), hand (joint 6).
    The other four joints have zero-length links: they stack rotation axes at a point and carry no
    body of their own, but they still move everything downstream, so they appear in `M(q)`.

    `damping` is the diagonal of `B` [N m s / rad], scalar or per-joint. The default 0.05 is a
    deliberately round, uncalibrated placeholder: this module's job is to make the dissipation
    **known**, not to claim it is the physiological value. Task 5.9 scores a pHNN against whatever
    `B` generated the data, so the number only has to be recorded, not correct.
    """
    zero = Body()
    bodies = (
        zero,  # sh_yaw
        zero,  # sh_pitch
        body_from_de_leva("upper_arm", l_upper, body_mass),  # sh_roll_upper
        body_from_de_leva("forearm", l_fore, body_mass),  # elbow_fore
        zero,  # forearm_pron
        zero,  # wrist_flex
        body_from_de_leva("hand", l_hand, body_mass),  # wrist_dev_hand
    )
    arm = human_arm_7dof(l_upper=l_upper, l_fore=l_fore, l_hand=l_hand)
    damping = (damping,) * 7 if np.isscalar(damping) else tuple(damping)
    return ArmDynamics(arm=arm, bodies=bodies, damping=damping)


def planar_2link_dynamics(
    m1: float = 2.0,
    m2: float = 1.2,
    l1: float = 0.30,
    l2: float = 0.25,
    c1: float = 0.5,
    c2: float = 0.5,
    damping=0.0,
) -> ArmDynamics:
    """The planar 2-link arm as an `ArmDynamics`, matching `arm3d.planar_2link`'s geometry.

    Uniform-rod inertia about the CoM (`m l^2 / 12` about the two axes perpendicular to the rod,
    zero about the rod's own axis) so the closed-form reference in `planar_2link_closed_form`
    describes exactly the same body. `c1`, `c2` are CoM positions as fractions of link length.
    Gravity is in-plane `(0,-G,0)`, the same convention `planar_2link` uses.
    """
    from .arm3d import Segment

    def rod(m, ell, c):
        Iyy = Izz = m * ell**2 / 12.0
        return Body(mass=m, com=(c * ell, 0.0, 0.0),
                    inertia=((0.0, 0.0, 0.0), (0.0, Iyy, 0.0), (0.0, 0.0, Izz)))

    arm = Arm3D(
        segments=(
            Segment(axis=(0, 0, 1), link=(l1, 0, 0), name="upper", has_sensor=True, sensor_id="S2"),
            Segment(axis=(0, 0, 1), link=(l2, 0, 0), name="fore", has_sensor=True, sensor_id="S4"),
        ),
        g_world=(0.0, -G, 0.0),
    )
    damping = (damping,) * 2 if np.isscalar(damping) else tuple(damping)
    return ArmDynamics(arm=arm, bodies=(rod(m1, l1, c1), rod(m2, l2, c2)), damping=damping)


def planar_2link_closed_form(q, qd, m1=2.0, m2=1.2, l1=0.30, l2=0.25, c1f=0.5, c2f=0.5):
    """Textbook closed-form `M`, `C`, `g` for the planar 2-link arm — an **independent** reference.

    Derived by hand rather than by the general Jacobian machinery, so agreement between the two is
    a genuine cross-check of `ArmDynamics` (test (e)) rather than a restatement of it. Uses the
    same convention as `sim.arm.PlanarArm`: `q[0]` is the shoulder angle measured from world `+x`,
    `q[1]` the elbow angle **relative** to the upper arm; gravity acts along `-y`.

    With `c_i` the CoM distance from joint `i` and `I_i` the rod inertia about its own CoM:

        M11 = I1 + m1 c1^2 + I2 + m2 (l1^2 + c2^2 + 2 l1 c2 cos q2)
        M12 = M21 = I2 + m2 (c2^2 + l1 c2 cos q2)
        M22 = I2 + m2 c2^2
        h   = -m2 l1 c2 sin q2
        C   = [[h qdot2, h (qdot1 + qdot2)], [-h qdot1, 0]]
        g1  = G ( m1 c1 cos q1 + m2 ( l1 cos q1 + c2 cos(q1 + q2) ) )
        g2  = G m2 c2 cos(q1 + q2)
    """
    q, qd = np.asarray(q, float), np.asarray(qd, float)
    c1, c2 = c1f * l1, c2f * l2
    I1, I2 = m1 * l1**2 / 12.0, m2 * l2**2 / 12.0
    q1, q2 = q
    qd1, qd2 = qd
    cos2, sin2 = np.cos(q2), np.sin(q2)

    M = np.array([
        [I1 + m1 * c1**2 + I2 + m2 * (l1**2 + c2**2 + 2 * l1 * c2 * cos2),
         I2 + m2 * (c2**2 + l1 * c2 * cos2)],
        [I2 + m2 * (c2**2 + l1 * c2 * cos2), I2 + m2 * c2**2],
    ])
    h = -m2 * l1 * c2 * sin2
    C = np.array([[h * qd2, h * (qd1 + qd2)], [-h * qd1, 0.0]])
    g = np.array([
        G * (m1 * c1 * np.cos(q1) + m2 * (l1 * np.cos(q1) + c2 * np.cos(q1 + q2))),
        G * m2 * c2 * np.cos(q1 + q2),
    ])
    return M, C, g


if __name__ == "__main__":
    from .motions import generate_trial

    dyn = human_arm_7dof_dynamics()
    print(f"7-DOF arm: {dyn.n} joints, "
          f"total limb mass {sum(b.mass for b in dyn.bodies):.3f} kg "
          f"(de Leva 1996 male, 75 kg subject)")

    q = np.zeros(7)
    M = dyn.mass_matrix(q)
    print(f"  M(0) diagonal [kg m^2] = {np.diag(M).round(5)}")
    print(f"  eigenvalues            = {np.linalg.eigvalsh(M).round(6)}  (all > 0 => PD)")

    # conservative 5 s rollout: how well is energy held?
    free = human_arm_7dof_dynamics(damping=0.0)
    t = np.arange(0.0, 5.0, 1.0 / 500.0)
    qs, qds = free.simulate(np.full(7, 0.3), np.zeros(7), t)
    H = np.array([free.energy(qs[k], qds[k]) for k in range(0, t.size, 25)])
    print(f"  5 s conservative rollout: energy drift = {abs(H - H[0]).max():.3e} J "
          f"(|H| ~ {abs(H[0]):.3f} J)")

    # torques for one real min-jerk trial
    t2, q2, qd2, qdd2 = generate_trial("reach", fs=500.0, rng=np.random.default_rng(0))
    tau = dyn.inverse_dynamics_traj(q2[::10], qd2[::10], qdd2[::10])
    print(f"  reach trial inverse dynamics: peak |tau| = {np.abs(tau).max():.3f} N m")
