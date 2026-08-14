"""Tests for the damping-identifiability analysis (task 3.15).

`analysis/identifiability.py` started life as a throwaway script written during the 14 Aug depth
pass. Its conclusion — that two of the seven joints carry no recoverable damping in the
naturalistic dataset — is now load-bearing: it is why the excitation class exists, why Phase 5
carries a do-not-start note, and what the dataset manifest reports about itself. A conclusion that
important should not rest on a script nobody tests, so the machinery is pinned here.

The regressor is the part that must be right. If `Y` is wrong, every number downstream is wrong in
a way that looks perfectly plausible, so the tests below check it against properties that hold by
construction rather than against remembered outputs.
"""

import numpy as np
import pytest

from analysis.identifiability import (
    _theta0,
    _with_theta,
    build_Y,
    damping_error_bars,
    regressor_block,
    verify_linearity,
)
from sim.dynamics import human_arm_7dof_dynamics
from sim.motions import excitation_trial, generate_trial


@pytest.fixture(scope="module")
def dyn():
    return human_arm_7dof_dynamics()


def _states(n, seed=0, scale=2.0):
    rng = np.random.default_rng(seed)
    return [(rng.uniform(-np.pi, np.pi, 7), rng.uniform(-scale, scale, 7),
             rng.uniform(-scale, scale, 7)) for _ in range(n)]


# --------------------------------------------------------------------------- #
# The regressor is exact, not approximate
# --------------------------------------------------------------------------- #
def test_regressor_is_step_independent(dyn):
    """`tau` is exactly linear in the parameters, so the central difference must be exact.

    If this drifts, the 'finite difference' is measuring curvature that should not exist and every
    conditioning number in DEPTH.md is suspect.
    """
    err = verify_linearity(dyn, _states(1, seed=11)[0])
    assert err < 1e-6


def test_regressor_is_the_jacobian_of_inverse_dynamics(dyn):
    """`Y` predicts the torque change from a parameter perturbation, to first order.

    This is the property that matters and the one that is actually true. `Y theta != tau` under
    this parameterisation, because the parallel-axis term makes the model quadratic in `com_x` —
    see `test_model_is_linear_in_mass_and_inertia_but_not_in_com`. Local identifiability is a
    statement about the Jacobian, so the Jacobian is what gets tested.
    """
    theta0 = _theta0(dyn)
    for n, (q, qd, qdd) in enumerate(_states(6, seed=12)):
        Y = regressor_block(dyn, q, qd, qdd)[:, :15]
        base = _with_theta(dyn, theta0).inverse_dynamics(q, qd, qdd)
        direction = np.random.default_rng(120 + n).normal(size=15) * np.maximum(np.abs(theta0), 1e-3)

        def residual(scale, Y=Y, base=base, direction=direction, q=q, qd=qd, qdd=qdd):
            step = scale * direction
            actual = _with_theta(dyn, theta0 + step).inverse_dynamics(q, qd, qdd) - base
            return np.abs(Y @ step - actual).max(), np.abs(actual).max()

        # (i) at a small step the Jacobian predicts the change to within rounding
        r_small, a_small = residual(1e-6)
        assert r_small / a_small < 1e-3, f"state {n}: relative residual {r_small / a_small:.2e}"

        # (ii) and the residual falls quadratically with step size, which is what makes it a
        #      first-order derivative rather than a coincidence at one scale
        r_big, _ = residual(1e-2)
        r_mid, _ = residual(1e-4)
        assert r_big / r_mid > 100, f"state {n}: {r_big:.2e} -> {r_mid:.2e} is not second order"


def test_model_is_linear_in_mass_and_inertia_but_not_in_com(dyn):
    """Pins the nonlinearity that makes `Y` a Jacobian rather than a global linear model.

    Found by the Loop Method on 2026-08-14: the module originally claimed `tau` was linear in every
    inertial parameter. It is linear in mass and in the inertia tensor, but the parallel-axis term
    contributes `m |c|^2`, so it is quadratic in the CoM offset. The finding the module reports is
    unaffected — error bars and condition numbers are Jacobian properties — but the claim was wrong
    and is now pinned so it cannot be restated.
    """
    q, qd, qdd = _states(1, seed=140)[0]
    theta0 = _theta0(dyn)
    Y = regressor_block(dyn, q, qd, qdd)[:, :15]
    base = _with_theta(dyn, theta0).inverse_dynamics(q, qd, qdd)

    def curvature(idx):
        h = max(abs(theta0[idx]), 1e-3) * 1e-3
        bumped = theta0.copy()
        bumped[idx] += h
        actual = _with_theta(dyn, bumped).inverse_dynamics(q, qd, qdd) - base
        return np.abs(Y[:, idx] * h - actual).max()

    assert curvature(0) < 1e-10, "mass: exactly linear"
    assert curvature(2) < 1e-10, "Ixx: exactly linear"
    assert curvature(1) > 1e-9, "com_x: quadratic via the parallel-axis term"


def test_damping_columns_are_exactly_qdot_on_the_diagonal(dyn):
    """`d tau_j / d b_j = qdot_j`, and damping never couples across joints."""
    for q, qd, qdd in _states(10, seed=13):
        damp = regressor_block(dyn, q, qd, qdd)[:, 15:]
        assert np.allclose(damp, np.diag(qd), atol=0)


def test_a_motionless_joint_has_a_zero_damping_column(dyn):
    """The mechanism behind the whole finding: no motion, no information about damping."""
    q, _qd, qdd = _states(1, seed=14)[0]
    qd = np.zeros(7)
    qd[3] = 1.5                       # elbow only
    damp = regressor_block(dyn, q, qd, qdd)[:, 15:]
    assert np.abs(damp[:, 3]).max() > 0
    for j in (0, 1, 2, 4, 5, 6):
        assert np.abs(damp[:, j]).max() == 0.0


def test_with_theta_round_trips(dyn):
    theta = _theta0(dyn)
    rebuilt = _with_theta(dyn, theta)
    assert np.allclose(_theta0(rebuilt), theta)
    assert np.allclose(np.diag(rebuilt.B), np.diag(dyn.B))


# --------------------------------------------------------------------------- #
# The finding itself: naturalistic motion cannot identify B, excitation can
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def naturalistic_samples():
    """Samples from the four naturalistic classes, drawn the same way the depth pass drew them."""
    rng = np.random.default_rng(0)
    out = []
    for mc in ("reach", "lift", "wrist_rotate", "throw"):
        for _ in range(10):
            _t, q, qd, qdd = generate_trial(mc, fs=500.0, rng=rng)
            lo, hi = int(0.15 * q.shape[0]), int(0.9 * q.shape[0])
            for k in np.linspace(lo, hi - 1, 8, dtype=int):
                out.append((q[k], qd[k], qdd[k]))
    return out


@pytest.fixture(scope="module")
def excitation_samples():
    rng = np.random.default_rng(10_000)
    out = []
    for _ in range(10):
        _t, q, qd, qdd = excitation_trial(fs=500.0, rng=rng)
        for k in np.linspace(0, q.shape[0] - 1, 32, dtype=int):
            out.append((q[k], qd[k], qdd[k]))
    return out


def test_naturalistic_classes_cannot_identify_two_joints(dyn, naturalistic_samples):
    """The depth-pass finding, pinned. `sh_roll` and `wrist_dev` come back as noise."""
    r = damping_error_bars(dyn, naturalistic_samples)
    bars = r["relative_error_bar"]
    assert bars["sh_roll_upper"] > 1.0, "error bar should exceed the value itself"
    assert bars["wrist_dev_hand"] > 0.4
    for good in ("elbow_fore", "forearm_pron", "wrist_flex"):
        assert bars[good] < 0.2, good
    assert r["condition_number"] > 1e10


def test_excitation_makes_every_joint_identifiable(dyn, excitation_samples):
    """The fix, pinned: no joint's damping estimate is dominated by its own error bar."""
    r = damping_error_bars(dyn, excitation_samples)
    assert r["regressor_rank"] == r["regressor_cols"] == 22
    assert r["worst_relative_error_bar"] < 0.35, r["relative_error_bar"]
    assert all(v < 0.35 for v in r["relative_error_bar"].values())


def test_excitation_beats_naturalistic_on_both_criteria(dyn, naturalistic_samples,
                                                        excitation_samples):
    """The comparison the excitation class exists to win — conditioning and worst-joint error."""
    nat = damping_error_bars(dyn, naturalistic_samples)
    exc = damping_error_bars(dyn, excitation_samples)
    assert exc["condition_number"] < nat["condition_number"] / 5
    assert exc["worst_relative_error_bar"] < nat["worst_relative_error_bar"] / 4


def test_error_bars_scale_linearly_with_torque_noise(dyn, excitation_samples):
    """Sanity on the statistics: doubling sigma doubles every error bar."""
    a = damping_error_bars(dyn, excitation_samples, sigma_frac=0.01)["relative_error_bar"]
    b = damping_error_bars(dyn, excitation_samples, sigma_frac=0.02)["relative_error_bar"]
    for k in a:
        # `damping_error_bars` rounds to 6 significant figures for the manifest, so the tolerance
        # is set by that rounding rather than by the arithmetic.
        assert b[k] == pytest.approx(2 * a[k], rel=1e-5)


def test_damping_error_bars_output_is_json_serialisable(dyn, excitation_samples):
    """It is written into the dataset manifest, so it must survive a round trip."""
    import json

    r = damping_error_bars(dyn, excitation_samples[:40])
    assert json.loads(json.dumps(r)) == r
    assert set(r["relative_error_bar"]) == {
        "sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
        "forearm_pron", "wrist_flex", "wrist_dev_hand",
    }


def test_build_Y_shape_matches_the_sample_count(dyn):
    samples = _states(9, seed=15)
    assert build_Y(dyn, samples).shape == (9 * 7, 22)
