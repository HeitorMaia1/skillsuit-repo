"""Tests for the rigid-body dynamics layer (task 3.13).

These tests are the deliverable, not decoration. The claim task 5.9 wants to make — that a
pHNN's learned dissipation concentrates in prep/settle — is only meaningful if the data it is
fitted to has a *known, correct* dissipation in it. That requires `M`, `C`, `g` to be right, and
"right" here means five specific things:

  (a) `M` is symmetric positive definite across the configuration space.
  (b) `Mdot - 2C` is skew-symmetric — the passivity identity. This is the test that actually
      discriminates a correct `C` from a plausible one: many wrong `C` matrices reproduce the
      correct `C qdot` product and fail this.
  (c) With `B = 0` and `tau = 0`, energy is conserved over a 5 s rollout.
  (d) With `B != 0`, the energy balance `Hdot = -qdot^T B qdot` holds. This residual is exactly
      the quantity task 5.8's loss will use, so it is measured here first.
  (e) The planar reduction agrees with an independently derived closed form, in the spirit of
      3.2's 1e-12 planar check.

Measured numbers are printed by `test_report_the_numbers` at the end and repeated in
`WORK/work12.md`. A tolerance chosen is fine; a tolerance not reported is not.
"""

import numpy as np
import pytest

from sim.arm import PlanarArm
from sim.dynamics import (
    DE_LEVA_MALE,
    ArmDynamics,
    Body,
    body_from_de_leva,
    human_arm_7dof_dynamics,
    planar_2link_closed_form,
    planar_2link_dynamics,
)

# Tolerances, all chosen here and all reported by the final test.
TOL_SYMMETRY = 1e-15        # |M - M^T|, machine precision
TOL_CLOSED_FORM = 1e-12     # general machinery vs hand-derived planar closed form
TOL_SKEW = 1e-9             # |(Mdot - 2C) + (Mdot - 2C)^T|, absolute
TOL_FASTPATH = 1e-10        # coriolis_times_qd vs coriolis @ qd
DRIFT_PLANAR = 5e-6         # |H(t) - H(0)| over 5 s at 1 kHz, B = 0, tau = 0  [J]
DRIFT_7DOF = 5e-3           # ditto, 7-DOF at 500 Hz (see the note in test (c))  [J]
TOL_BALANCE_REL = 2e-3      # |Hdot + qdot^T B qdot| relative to max |qdot^T B qdot|

_MEASURED: dict[str, str] = {}


def _rand_states(n, count, seed, qd_scale=3.0):
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield rng.uniform(-np.pi, np.pi, n), rng.uniform(-qd_scale, qd_scale, n)


def _pitch_elbow_3d(damping=0.0):
    """A 2-DOF *3D* chain with no singularity anywhere: shoulder pitch (about y) + elbow (about z).

    The elbow axis is `R(y, q0) z = (sin q0, 0, cos q0)`, which is orthogonal to the pitch axis
    `y` for every `q0`, so the two axes can never align and `M` never loses rank. That makes this
    the right vehicle for long rollouts in 3D — unlike the 7-DOF arm, whose resting pose is itself
    a gimbal lock (see `test_the_arms_resting_pose_is_itself_a_shoulder_gimbal_lock`). Real
    de Leva segment parameters, z-up gravity.
    """
    from sim.arm3d import Arm3D, Segment

    l_upper, l_fore = 0.30, 0.25
    arm = Arm3D(
        segments=(
            Segment(axis=(0, 1, 0), link=(l_upper, 0, 0), name="upper"),
            Segment(axis=(0, 0, 1), link=(l_fore, 0, 0), name="fore"),
        ),
        g_world=(0.0, 0.0, -9.81),
    )
    bodies = (body_from_de_leva("upper_arm", l_upper, 75.0),
              body_from_de_leva("forearm", l_fore, 75.0))
    damping = (damping, damping) if np.isscalar(damping) else tuple(damping)
    return ArmDynamics(arm=arm, bodies=bodies, damping=damping)


def test_pitch_elbow_chain_is_never_singular():
    """The premise of `_pitch_elbow_3d`: its two joint axes stay orthogonal for every `q`."""
    dyn = _pitch_elbow_3d()
    worst_lambda, worst_align = np.inf, 1.0
    for q, _ in _rand_states(2, 200, seed=30):
        axes_w = dyn._chain(q)[0]
        worst_align = min(worst_align, np.linalg.norm(np.cross(axes_w[0], axes_w[1])))
        worst_lambda = min(worst_lambda, np.linalg.eigvalsh(dyn.mass_matrix(q))[0])
    assert worst_align > 0.999, "axes stay orthogonal, so there is no gimbal lock"
    assert worst_lambda > 1e-4
    _MEASURED["3D pitch+elbow min eigenvalue"] = f"{worst_lambda:.3e} kg m^2"


# --------------------------------------------------------------------------- #
# (a) M symmetric positive definite
# --------------------------------------------------------------------------- #
def test_mass_matrix_symmetric_positive_definite_7dof():
    dyn = human_arm_7dof_dynamics()
    worst_asym, worst_lambda = 0.0, np.inf
    for q, _ in _rand_states(7, 300, seed=1):
        M = dyn.mass_matrix(q)
        worst_asym = max(worst_asym, np.abs(M - M.T).max())
        worst_lambda = min(worst_lambda, np.linalg.eigvalsh(M)[0])
    assert worst_asym < TOL_SYMMETRY
    assert worst_lambda > 0.0, "M must be positive definite everywhere"
    _MEASURED["7dof M asymmetry"] = f"{worst_asym:.2e}"
    _MEASURED["7dof min eigenvalue over 300 random q"] = f"{worst_lambda:.2e} kg m^2"


def test_mass_matrix_symmetric_positive_definite_planar():
    dyn = planar_2link_dynamics()
    worst = np.inf
    for q, _ in _rand_states(2, 200, seed=2):
        M = dyn.mass_matrix(q)
        assert np.abs(M - M.T).max() < TOL_SYMMETRY
        worst = min(worst, np.linalg.eigvalsh(M)[0])
    assert worst > 1e-6
    _MEASURED["planar min eigenvalue"] = f"{worst:.3e} kg m^2"


def test_wrist_gimbal_lock_is_real_and_documented():
    """The 3-axis wrist gimbal-locks and `M` loses rank there. Not a bug — a property.

    At `q[5] = pi/2` the pronation axis (joint 4) and the deviation axis (joint 6) become
    parallel, so the wrist's three rotational DOF span only two directions and `M`'s smallest
    eigenvalue collapses by ~6 orders of magnitude. Locked in as a test so it cannot be
    rediscovered later as a mystery.
    """
    dyn = human_arm_7dof_dynamics()
    lam, align = [], []
    for q5 in (0.0, 0.5, 1.0, np.pi / 2 - 1e-3):
        q = np.zeros(7)
        q[5] = q5
        axes_w = dyn._chain(q)[0]
        lam.append(np.linalg.eigvalsh(dyn.mass_matrix(q))[0])
        align.append(np.linalg.norm(np.cross(axes_w[4], axes_w[6])))

    assert lam[0] > 1e-4, "away from the singularity M is well conditioned"
    assert lam[-1] < 1e-8, "at gimbal lock M is numerically singular"
    assert align[-1] < 1e-2, "the two wrist axes really are parallel there"
    assert all(a > b for a, b in zip(lam, lam[1:])), "conditioning degrades monotonically"
    _MEASURED["wrist gimbal lock: lambda_min(M)"] = f"{lam[0]:.2e} -> {lam[-1]:.2e} kg m^2"


def test_the_arms_resting_pose_is_itself_a_shoulder_gimbal_lock():
    """**The finding Phase 5 most needs to know.** The arm's gravitational rest pose is singular.

    The three shoulder axes are yaw about `z` (joint 0), pitch about `y` (joint 1) and roll about
    `x` (joint 2). Rotating pitch by `pi/2` carries the roll axis `x` onto `-z`, i.e. onto the yaw
    axis: `k_0` and `k_2` become parallel and the shoulder loses a degree of freedom. But
    `q[1] = pi/2` is exactly the configuration where the arm hangs straight down — the **stable
    equilibrium under gravity**.

    So the configuration this arm settles into when you let go is a kinematic singularity of its
    own parameterisation. Consequences, none of them optional to know about:

      * any damped free rollout converges *into* the singularity, so `M^-1` blows up as it
        settles and no explicit integrator can finish the transient (this is what makes a long
        damped 7-DOF rollout the wrong instrument for the energy-balance test);
      * `H(q, p) = 1/2 p^T M(q)^-1 p` is ill-conditioned near rest, which is precisely where a
        motion trial begins and ends;
      * synthetic trials that start or finish near the hanging pose will carry huge condition
        numbers into Phase 5's training data.

    This is a property of the joint parameterisation, not of the dynamics code — the physical arm
    is perfectly well behaved hanging down; it is the yaw/roll coordinate pair that degenerates.
    """
    dyn = human_arm_7dof_dynamics()
    lam, align = [], []
    for q1 in (0.0, 0.6, 1.2, np.pi / 2 - 1e-3):
        q = np.zeros(7)
        q[1] = q1
        axes_w = dyn._chain(q)[0]
        lam.append(np.linalg.eigvalsh(dyn.mass_matrix(q))[0])
        align.append(np.linalg.norm(np.cross(axes_w[0], axes_w[2])))

    assert align[0] > 0.99, "upright: yaw and roll axes are perpendicular"
    assert align[-1] < 1e-2, "at pitch = pi/2 they are parallel — gimbal lock"
    assert lam[-1] < lam[0] / 100.0, "and M's conditioning collapses with them"
    _MEASURED["shoulder lock at rest pose: lambda_min(M)"] = (
        f"{lam[0]:.2e} -> {lam[-1]:.2e} kg m^2 (|k0 x k2| {align[0]:.3f} -> {align[-1]:.1e})"
    )


# --------------------------------------------------------------------------- #
# (b) the passivity identity — the test that discriminates C
# --------------------------------------------------------------------------- #
def test_passivity_mdot_minus_2c_is_skew_symmetric():
    """`Mdot - 2C` skew-symmetric, on both systems, at random states."""
    worst = 0.0
    for dyn, n, seed in ((planar_2link_dynamics(), 2, 3), (human_arm_7dof_dynamics(), 7, 4)):
        for q, qd in _rand_states(n, 60, seed=seed):
            S = dyn.mass_matrix_dot(q, qd) - 2.0 * dyn.coriolis(q, qd)
            worst = max(worst, np.abs(S + S.T).max())
    assert worst < TOL_SKEW
    _MEASURED["passivity |(Mdot-2C)+(Mdot-2C)^T|"] = f"{worst:.2e}"


def test_passivity_quadratic_form_vanishes():
    """The scalar consequence `qdot^T (Mdot - 2C) qdot = 0`, which the energy balance rests on."""
    dyn = human_arm_7dof_dynamics()
    worst = 0.0
    for q, qd in _rand_states(7, 60, seed=5):
        S = dyn.mass_matrix_dot(q, qd) - 2.0 * dyn.coriolis(q, qd)
        worst = max(worst, abs(qd @ S @ qd))
    assert worst < TOL_SKEW
    _MEASURED["passivity qdot^T(Mdot-2C)qdot"] = f"{worst:.2e}"


def test_mdot_equals_c_plus_c_transpose():
    """Equivalent statement of the same identity, checked independently: `Mdot = C + C^T`."""
    dyn = planar_2link_dynamics()
    for q, qd in _rand_states(2, 50, seed=6):
        C = dyn.coriolis(q, qd)
        assert np.abs(dyn.mass_matrix_dot(q, qd) - (C + C.T)).max() < TOL_SKEW


def test_coriolis_fast_path_matches_full_matrix():
    """`coriolis_times_qd` (used in every rollout) must equal `coriolis(q,qd) @ qd`."""
    dyn = human_arm_7dof_dynamics()
    worst = 0.0
    for q, qd in _rand_states(7, 80, seed=7):
        worst = max(worst, np.abs(dyn.coriolis_times_qd(q, qd) - dyn.coriolis(q, qd) @ qd).max())
    assert worst < TOL_FASTPATH
    _MEASURED["fast C qdot vs C @ qdot"] = f"{worst:.2e}"


# --------------------------------------------------------------------------- #
# (c) energy conservation, B = 0, tau = 0, over 5 s
# --------------------------------------------------------------------------- #
def test_energy_conserved_planar_5s():
    dyn = planar_2link_dynamics(damping=0.0)
    t = np.arange(0.0, 5.0, 1.0 / 1000.0)
    qs, qds = dyn.simulate([0.4, 0.9], [0.0, 0.0], t)
    assert np.isfinite(qs).all()
    H = np.array([dyn.energy(qs[k], qds[k]) for k in range(0, t.size, 10)])
    drift = np.abs(H - H[0]).max()
    assert drift < DRIFT_PLANAR
    _MEASURED["(c) planar 5 s drift @1 kHz"] = f"{drift:.3e} J on |H0|={abs(H[0]):.4f} J " \
                                               f"(rel {drift / abs(H[0]):.2e})"


def test_energy_conserved_3d_5s():
    """5 s conservative rollout of the non-singular 3D chain — the clean 3D conservation number."""
    dyn = _pitch_elbow_3d(damping=0.0)
    t = np.arange(0.0, 5.0, 1.0 / 1000.0)
    qs, qds = dyn.simulate([0.4, 0.9], [0.0, 0.0], t)
    assert np.isfinite(qs).all()
    H = np.array([dyn.energy(qs[k], qds[k]) for k in range(0, t.size, 10)])
    drift = np.abs(H - H[0]).max()
    assert drift < DRIFT_PLANAR
    _MEASURED["(c) 3D pitch+elbow 5 s drift @1 kHz"] = (
        f"{drift:.3e} J on |H0|={abs(H[0]):.4f} J (rel {drift / abs(H[0]):.2e})"
    )


def test_energy_conserved_7dof_5s():
    """5 s conservative rollout of the full 7-DOF arm, started near the hanging equilibrium.

    The initial condition matters and is not arbitrary: from a horizontal start the arm free-falls
    through the wrist gimbal lock documented above, where `M` goes numerically singular and any
    explicit integrator diverges. Starting near hanging keeps `lambda_min(M)` off the floor for
    the whole 5 s. Drift is RK4 truncation error and converges with step size — measured
    1.39e-3 J at 500 Hz, 3.44e-4 J at 1 kHz, 8.85e-6 J at 2 kHz on |H| = 11.0 J. The test runs at
    500 Hz to keep the suite fast and asserts the corresponding tolerance.
    """
    dyn = human_arm_7dof_dynamics(damping=0.0)
    t = np.arange(0.0, 5.0, 1.0 / 500.0)
    q0 = np.array([0.0, np.pi / 2 + 0.15, 0.0, 0.1, 0.0, 0.0, 0.0])
    qs, qds = dyn.simulate(q0, np.zeros(7), t)
    assert np.isfinite(qs).all(), "rollout diverged — check the wrist singularity"
    idx = range(0, t.size, 5)
    H = np.array([dyn.energy(qs[k], qds[k]) for k in idx])
    lam = min(np.linalg.eigvalsh(dyn.mass_matrix(qs[k]))[0] for k in idx)
    drift = np.abs(H - H[0]).max()
    assert drift < DRIFT_7DOF
    assert lam > 1e-8, "stayed clear of gimbal lock"
    _MEASURED["(c) 7-DOF 5 s drift @500 Hz"] = f"{drift:.3e} J on |H0|={abs(H[0]):.4f} J " \
                                               f"(rel {drift / abs(H[0]):.2e})"


# --------------------------------------------------------------------------- #
# (d) the energy balance with damping — the quantity 5.8's loss uses
# --------------------------------------------------------------------------- #
def test_energy_balance_with_damping():
    """`Hdot = -qdot^T B qdot` along a damped rollout, with `Hdot` from a central difference.

    `Hdot` is measured numerically from the `H(t)` series and compared against the closed-form
    dissipation rate. The residual is the number task 5.8's loss will be asked to drive to zero,
    so it is reported rather than merely asserted.
    """
    dyn = planar_2link_dynamics(damping=[0.06, 0.04])
    t = np.arange(0.0, 3.0, 1.0 / 1000.0)
    qs, qds = dyn.simulate([0.5, 0.8], [0.0, 0.0], t)
    H = np.array([dyn.energy(qs[k], qds[k]) for k in range(t.size)])
    dt = t[1] - t[0]
    Hdot = (H[2:] - H[:-2]) / (2 * dt)
    diss = -np.einsum("ta,ab,tb->t", qds[1:-1], dyn.B, qds[1:-1])
    scale = np.abs(diss).max()
    resid = np.abs(Hdot - diss).max()
    assert resid / scale < TOL_BALANCE_REL
    assert H[-1] < H[0], "damping must remove energy"
    _MEASURED["(d) planar |Hdot + qdot^T B qdot|"] = (
        f"{resid:.3e} W (max dissipation {scale:.3f} W, rel {resid / scale:.2e}); "
        f"H fell {H[0]:.4f} -> {H[-1]:.4f} J"
    )

def test_energy_balance_7dof_pointwise():
    """The same balance on the 7-DOF arm, checked **pointwise** rather than along a rollout.

    A long damped rollout of this arm is not the right instrument. Viscous damping on the wrist
    joints combined with the near-singular `M` documented in
    `test_wrist_singularity_is_real_and_documented` makes the system genuinely stiff: the
    dissipation time constant is `lambda_min(M) / b`, which collapses from ~20 ms in a good
    configuration to nanoseconds at gimbal lock, and no explicit integrator survives it. That is a
    property of the arm, not of the balance.

    So the identity is verified where it actually lives — in the equations, at random states
    across the whole configuration space, with `Hdot` assembled from its closed form

        Hdot = qdot^T M qddot + 1/2 qdot^T Mdot qdot + qdot^T g,   qddot from forward_dynamics

    and compared against `qdot^T tau - qdot^T B qdot`. This is stronger coverage than one
    trajectory, not weaker: it samples configurations a stable rollout could never reach.
    """
    B = np.array([0.08, 0.08, 0.05, 0.05, 0.02, 0.02, 0.02])
    dyn = human_arm_7dof_dynamics(damping=B)
    rng = np.random.default_rng(21)
    worst_rel, worst_abs = 0.0, 0.0
    for q, qd in _rand_states(7, 60, seed=20):
        tau = rng.uniform(-2.0, 2.0, 7)
        qdd = dyn.forward_dynamics(q, qd, tau)
        Hdot = (qd @ (dyn.mass_matrix(q) @ qdd)
                + 0.5 * qd @ dyn.mass_matrix_dot(q, qd) @ qd
                + qd @ dyn.gravity(q))
        predicted = qd @ tau - qd @ dyn.B @ qd
        err = abs(Hdot - predicted)
        worst_abs = max(worst_abs, err)
        worst_rel = max(worst_rel, err / max(abs(predicted), 1e-12))
    assert worst_rel < 1e-9
    _MEASURED["(d) 7-DOF pointwise |Hdot - (qdot^T tau - qdot^T B qdot)|"] = (
        f"{worst_abs:.2e} W (rel {worst_rel:.2e}, 60 random states)"
    )


def test_damping_removes_energy_monotonically_3d():
    """A damped rollout on a genuinely 3D chain: `H` decreases and never increases.

    Uses `_pitch_elbow_3d`, whose two axes can never align, so unlike the 7-DOF arm it has no
    singularity to fall into and a damped rollout can actually be integrated to rest. This shows
    the *sign and monotonicity* of the energy flow, which the pointwise test above cannot.
    """
    dyn = _pitch_elbow_3d(damping=[0.06, 0.04])
    t = np.arange(0.0, 4.0, 1.0 / 1000.0)
    qs, qds = dyn.simulate([0.5, 0.8], [0.0, 0.0], t)
    assert np.isfinite(qs).all()
    H = np.array([dyn.energy(qs[k], qds[k]) for k in range(0, t.size, 5)])
    assert H[-1] < H[0], "damping must remove energy"
    assert np.diff(H).max() < 1e-6, "H must never increase (beyond integrator noise)"
    diss = np.einsum("ta,ab,tb->t", qds, dyn.B, qds)
    assert (diss >= 0).all()
    _MEASURED["(d) 3D damped 4 s"] = (
        f"H fell {H[0]:.4f} -> {H[-1]:.4f} J monotonically, peak dissipation {diss.max():.3f} W"
    )


def test_zero_damping_is_exactly_conservative_in_the_balance():
    """With `B = 0` the balance says `Hdot = 0` — checked against the closed form, not a rollout."""
    dyn = human_arm_7dof_dynamics(damping=0.0)
    assert np.abs(dyn.B).max() == 0.0
    for q, qd in _rand_states(7, 20, seed=8):
        S = dyn.mass_matrix_dot(q, qd) - 2.0 * dyn.coriolis(q, qd)
        assert abs(qd @ S @ qd) < TOL_SKEW


# --------------------------------------------------------------------------- #
# (e) planar reduction vs an independent closed form
# --------------------------------------------------------------------------- #
def test_planar_matches_independent_closed_form():
    """`M`, `C`, `g` from the general 3D machinery vs the hand-derived planar expressions.

    The reference in `planar_2link_closed_form` was derived separately, so this is a genuine
    cross-check of the Jacobian assembly, the Christoffel construction and the complex-step
    derivative all at once — not a restatement of them.
    """
    dyn = planar_2link_dynamics()
    eM = eC = eg = 0.0
    for q, qd in _rand_states(2, 300, seed=9):
        Mr, Cr, gr = planar_2link_closed_form(q, qd)
        eM = max(eM, np.abs(dyn.mass_matrix(q) - Mr).max())
        eC = max(eC, np.abs(dyn.coriolis(q, qd) - Cr).max())
        eg = max(eg, np.abs(dyn.gravity(q) - gr).max())
    assert eM < TOL_CLOSED_FORM and eC < TOL_CLOSED_FORM and eg < TOL_CLOSED_FORM
    _MEASURED["(e) planar vs closed form"] = f"M {eM:.2e} · C {eC:.2e} · g {eg:.2e}"


def test_planar_com_positions_match_planar_arm():
    """The dynamics chain's CoM positions reproduce `PlanarArm`'s sensor midpoints to 1e-12.

    `PlanarArm` carries no dynamics to compare against, so the link to it is kinematic: with the
    CoM at each link's midpoint, the dynamics model's body centres must land exactly on
    `PlanarArm.forward_kinematics`' `S2` and `S4`.
    """
    planar, dyn = PlanarArm(), planar_2link_dynamics(c1=0.5, c2=0.5)
    for q, _ in _rand_states(2, 100, seed=10):
        fk = planar.forward_kinematics(q[0], q[1])
        coms = dyn._chain(q)[3]
        assert np.abs(coms[0][:2] - fk["S2"]).max() < 1e-12
        assert np.abs(coms[1][:2] - fk["S4"]).max() < 1e-12
        assert np.abs(coms[:, 2]).max() < 1e-15, "planar motion stays in the z = 0 plane"


# --------------------------------------------------------------------------- #
# consistency of the remaining API surface
# --------------------------------------------------------------------------- #
def test_forward_and_inverse_dynamics_are_inverses():
    dyn = human_arm_7dof_dynamics(damping=0.03)
    worst = 0.0
    for q, qd in _rand_states(7, 50, seed=11):
        qdd = np.random.default_rng(int(q[0] * 1e6) % 2**31).uniform(-5, 5, 7)
        tau = dyn.inverse_dynamics(q, qd, qdd)
        worst = max(worst, np.abs(dyn.forward_dynamics(q, qd, tau) - qdd).max())
    assert worst < 1e-8
    _MEASURED["inverse/forward round-trip"] = f"{worst:.2e} rad/s^2"


def test_hamiltonian_matches_energy_through_momentum():
    """`H(q, M qdot) == T + U`: the (q,p) and (q,qdot) forms agree, as Phase 5 assumes."""
    dyn = human_arm_7dof_dynamics()
    worst = 0.0
    for q, qd in _rand_states(7, 50, seed=12):
        worst = max(worst, abs(dyn.hamiltonian(q, dyn.momentum(q, qd)) - dyn.energy(q, qd)))
    assert worst < 1e-9
    _MEASURED["H(q,p) vs T+U"] = f"{worst:.2e} J"


def test_gravity_is_the_gradient_of_potential_energy():
    """`g(q)` is computed from the Jacobians; check it against a numerical gradient of `U`."""
    dyn = human_arm_7dof_dynamics()
    h = 1e-6
    for q, _ in _rand_states(7, 20, seed=13):
        num = np.array([
            (dyn.potential_energy(q + h * np.eye(7)[k]) - dyn.potential_energy(q - h * np.eye(7)[k]))
            / (2 * h) for k in range(7)
        ])
        assert np.abs(dyn.gravity(q) - num).max() < 1e-6


def test_velocities_recursion_matches_jacobians():
    """The O(n) velocity recursion must agree with `J qdot` from the Jacobians."""
    dyn = human_arm_7dof_dynamics()
    for q, qd in _rand_states(7, 30, seed=14):
        w, v = dyn.velocities(q, qd)
        J_v, J_w = dyn.jacobians(q)
        assert np.abs(v - np.einsum("iaj,j->ia", J_v, qd)).max() < 1e-12
        assert np.abs(w - np.einsum("iaj,j->ia", J_w, qd)).max() < 1e-12


def test_kinetic_energy_paths_agree():
    dyn = human_arm_7dof_dynamics()
    for q, qd in _rand_states(7, 30, seed=15):
        assert abs(0.5 * dyn._two_T(q, qd) - dyn.kinetic_energy(q, qd)) < 1e-12


def test_zero_mass_segments_do_not_break_positive_definiteness():
    """Four of the seven segments carry no body; `M` must still be PD because of what is downstream."""
    dyn = human_arm_7dof_dynamics()
    zero_mass = [i for i, b in enumerate(dyn.bodies) if b.mass == 0.0]
    assert zero_mass == [0, 1, 4, 5]
    assert np.linalg.eigvalsh(dyn.mass_matrix(np.zeros(7)))[0] > 0.0


# --------------------------------------------------------------------------- #
# anthropometry
# --------------------------------------------------------------------------- #
def test_de_leva_parameters_land_where_the_source_says():
    """Spot-check `body_from_de_leva` against the published male relative values."""
    body_mass, ell = 75.0, 0.30
    b = body_from_de_leva("upper_arm", ell, body_mass)
    m_rel, c_rel, (r_sag, r_trans, r_long) = DE_LEVA_MALE["upper_arm"]
    assert b.mass == pytest.approx(m_rel * body_mass)
    assert b.com[0] == pytest.approx(c_rel * ell)
    inertia = b.I
    assert inertia[0, 0] == pytest.approx(b.mass * (ell * r_long) ** 2)   # longitudinal -> x
    assert inertia[1, 1] == pytest.approx(b.mass * (ell * r_sag) ** 2)    # sagittal     -> y
    assert inertia[2, 2] == pytest.approx(b.mass * (ell * r_trans) ** 2)  # transverse   -> z
    assert np.allclose(inertia, np.diag(np.diag(inertia))), "principal axes: the tensor is diagonal"
    assert inertia[0, 0] < inertia[2, 2] < inertia[1, 1], \
        "smallest inertia is about the segment's own long axis"


def test_total_limb_mass_is_a_plausible_fraction_of_body_mass():
    dyn = human_arm_7dof_dynamics(body_mass=75.0)
    total = sum(b.mass for b in dyn.bodies)
    frac = total / 75.0
    assert frac == pytest.approx(0.0271 + 0.0162 + 0.0061, abs=1e-9)
    assert 0.045 < frac < 0.055, "one arm is about 5% of body mass"
    _MEASURED["limb mass (75 kg male)"] = f"{total:.3f} kg = {100 * frac:.2f}% of body mass"


def test_unknown_segment_name_rejected():
    with pytest.raises(ValueError, match="unknown segment"):
        body_from_de_leva("tentacle", 0.3, 75.0)


# --------------------------------------------------------------------------- #
# construction-time validation
# --------------------------------------------------------------------------- #
def test_body_count_must_match_segment_count():
    from sim.arm3d import planar_2link

    with pytest.raises(ValueError, match="one Body per segment"):
        ArmDynamics(arm=planar_2link(), bodies=(Body(mass=1.0),))


def test_negative_damping_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        planar_2link_dynamics(damping=[-0.1, 0.0])


def test_damping_wrong_length_rejected():
    with pytest.raises(ValueError, match="damping must be"):
        human_arm_7dof_dynamics(damping=[0.1, 0.2])


def test_scalar_damping_broadcasts():
    dyn = human_arm_7dof_dynamics(damping=0.07)
    assert np.allclose(np.diag(dyn.B), 0.07)


def test_simulate_rejects_non_uniform_time():
    dyn = planar_2link_dynamics()
    with pytest.raises(ValueError, match="uniformly spaced"):
        dyn.simulate([0.1, 0.2], [0.0, 0.0], np.array([0.0, 0.01, 0.05]))


def test_simulate_accepts_constant_and_callable_torque():
    dyn = planar_2link_dynamics(damping=0.0)
    t = np.arange(0.0, 0.2, 1e-3)
    q_const = dyn.simulate([0.1, 0.2], [0.0, 0.0], t, tau=np.array([0.5, 0.0]))[0]
    q_call = dyn.simulate([0.1, 0.2], [0.0, 0.0], t, tau=lambda _t, _q, _qd: np.array([0.5, 0.0]))[0]
    assert np.allclose(q_const, q_call)
    assert not np.allclose(q_const, dyn.simulate([0.1, 0.2], [0.0, 0.0], t)[0])


# --------------------------------------------------------------------------- #
# the min-jerk trials get their torques computed rather than thrown away
# --------------------------------------------------------------------------- #
def test_inverse_dynamics_on_a_min_jerk_trial():
    """A prescribed min-jerk reach now carries torque, momentum and energy."""
    from sim.motions import generate_trial

    dyn = human_arm_7dof_dynamics(damping=0.05)
    t, q, qd, qdd = generate_trial("reach", fs=500.0, rng=np.random.default_rng(0))
    sl = slice(None, None, 10)
    tau = dyn.inverse_dynamics_traj(q[sl], qd[sl], qdd[sl])
    p, H, U = dyn.trajectory_energetics(q[sl], qd[sl])

    assert tau.shape == q[sl].shape and np.isfinite(tau).all()
    assert np.isfinite(p).all() and np.isfinite(H).all()
    # at rest at both ends the motion contributes nothing: torque is pure gravity, momentum zero
    assert np.abs(p[0]).max() < 1e-12 and np.abs(p[-1]).max() < 1e-12
    assert np.allclose(tau[0], dyn.gravity(q[0]), atol=1e-12)
    assert H[0] == pytest.approx(U[0]) and H[-1] == pytest.approx(U[-1])
    # and something actually happens in between
    assert np.abs(p).max() > 1e-3
    _MEASURED["reach trial peak |tau|"] = f"{np.abs(tau).max():.3f} N m"


def test_dissipated_power_is_known_in_closed_form():
    """The point of the whole module: the true dissipation rate is a number we can write down."""
    from sim.motions import generate_trial

    B_diag = 0.05
    dyn = human_arm_7dof_dynamics(damping=B_diag)
    _t, q, qd, _qdd = generate_trial("throw", fs=500.0, rng=np.random.default_rng(1))
    power = np.einsum("ta,ab,tb->t", qd, dyn.B, qd)
    assert (power >= 0).all(), "a positive-definite B can only remove energy"
    assert power[0] == 0.0 and power[-1] == 0.0, "trials start and end at rest"
    assert power.max() > 0.0
    _MEASURED["throw trial peak dissipation"] = f"{power.max():.3f} W (B = {B_diag} N m s/rad)"


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def test_report_the_numbers(capsys):
    """Print every measured number. A tolerance not reported is not a tolerance.

    Defined last in the file so pytest's default (definition-order) collection runs it last. Under
    a shuffling plugin it may run early and simply report less; it never fails for that reason.
    """
    if not _MEASURED:
        pytest.skip("run the whole module in definition order to collect the measurements")
    with capsys.disabled():
        print("\n  --- task 3.13 measured values " + "-" * 44)
        for k, v in _MEASURED.items():
            print(f"  {k:44s} {v}")
        print("  " + "-" * 74)
