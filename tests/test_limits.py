"""Task 3.16 — the dataset must describe an arm a person actually has.

The bug this file exists to keep dead: `excitation_trial` centred every joint at zero, and a
straight elbow is already the *end* of the elbow's travel, so a zero-mean oscillation hyperextended
it to -66.2 degrees against a -2 degree limit — in 10 of 10 trials. Nothing caught it because every
test in the repo asked whether the simulator was self-consistent and none asked whether it was a
person.

So the tests below come in three groups: the limits table has to be honest about where its numbers
came from, the generator has to produce motions inside those limits, and the repair has to not
disturb anything it was not meant to touch.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from sim.dynamics import human_arm_7dof_dynamics
from sim.limits import (
    JOINT_LIMITS,
    JOINT_ORDER,
    SINGULAR_ANGLES,
    UNSOURCED,
    feasibility,
    limits_manifest,
    usable_interval,
)
from sim.motions import MOTION_CLASSES, excitation_trial, generate_trial

DATA = Path(__file__).resolve().parents[1] / "data" / "processed" / "dynamics"


# --------------------------------------------------------------------------- #
# 1. The limits table has to be honest
# --------------------------------------------------------------------------- #
def test_every_limit_is_ordered_and_sourced():
    for name, lim in JOINT_LIMITS.items():
        if lim is None:
            continue
        assert lim.lo < lim.hi, f"{name}: empty interval"
        assert lim.tier in ("A", "B"), f"{name}: tier must say how good the source is"
        assert len(lim.source) > 40, f"{name}: a source has to be citable, not a word"
        assert "doi:" in lim.source or "AAOS" in lim.source, f"{name}: no traceable source"


def test_joint_order_matches_the_arm_it_constrains():
    from sim.arm3d import human_arm_7dof
    assert tuple(s.name for s in human_arm_7dof().segments) == JOINT_ORDER


def test_unsourced_joints_are_declared_not_silently_absent():
    """A joint with no limit must be visible as such — that is what makes the flag a lower bound."""
    assert UNSOURCED == ("sh_yaw",)
    assert JOINT_LIMITS["sh_yaw"] is None
    m = limits_manifest()
    assert m["unsourced_joints"] == ["sh_yaw"]
    assert m["n_joints_scored"] == 6
    assert "LOWER BOUND" in m["reading"]
    json.dumps(m)  # must survive the manifest


def test_elbow_limit_is_the_primary_source_and_agrees_with_the_secondary_one():
    """Zwerus 2017 measured 146/-2; AAOS says 150/0. The 4-degree agreement is why tier B is usable."""
    lim = JOINT_LIMITS["elbow_fore"]
    assert lim.tier == "A"
    assert math.degrees(lim.hi) == pytest.approx(146.0)
    assert math.degrees(lim.lo) == pytest.approx(-2.0)
    assert abs(math.degrees(lim.hi) - 150.0) <= 5.0
    assert abs(math.degrees(lim.lo) - 0.0) <= 5.0


# --------------------------------------------------------------------------- #
# 2. usable_interval — anatomy AND the gimbal locks, which are not the same thing
# --------------------------------------------------------------------------- #
def test_usable_interval_avoids_the_shoulder_gimbal_lock():
    """The naive ROM midpoint is +30 deg, which walks the oscillation into the singularity at +90."""
    lo, hi = usable_interval("sh_pitch")
    assert not (lo < math.pi / 2 < hi), "the interval must not straddle the singularity"
    assert math.degrees(lo) == pytest.approx(-90.0)
    assert math.degrees(hi) == pytest.approx(70.0)
    assert math.degrees(0.5 * (lo + hi)) == pytest.approx(-10.0)  # not +30


def test_usable_interval_picks_the_wider_side_not_the_first():
    """Excluding the lock leaves [-90,70] (width 160) and [110,150] (width 40). Wider must win."""
    lo, hi = usable_interval("sh_pitch")
    assert hi - lo == pytest.approx(math.radians(160.0))


def test_usable_interval_is_the_full_range_when_no_singularity_is_inside_it():
    """The wrist lock sits at +90, outside the wrist's own ROM of [-70, 80] — so nothing is cut."""
    assert math.pi / 2 in SINGULAR_ANGLES["wrist_flex"]
    lim = JOINT_LIMITS["wrist_flex"]
    assert lim.hi < math.pi / 2
    assert usable_interval("wrist_flex") == (lim.lo, lim.hi)


def test_usable_interval_is_none_for_an_unsourced_joint():
    assert usable_interval("sh_yaw") is None


def test_singularities_do_not_affect_the_feasibility_flag():
    """A person can hang their arm at the shoulder singularity; it is a coordinate defect, not anatomy."""
    q = np.zeros((1, 7))
    q[0, 1] = math.pi / 2                       # exactly the shoulder gimbal lock
    assert feasibility(q)["human_feasible"] is True


# --------------------------------------------------------------------------- #
# 3. feasibility scores what it says it scores
# --------------------------------------------------------------------------- #
def test_feasibility_catches_the_original_bug():
    q = np.zeros((1, 7))
    q[0, 3] = math.radians(-66.2)               # the elbow the shipped excitation produced
    r = feasibility(q)
    assert r["human_feasible"] is False
    assert r["worst_joint"] == "elbow_fore"
    assert math.degrees(r["excess_rad"][3]) == pytest.approx(64.2, abs=0.1)


def test_feasibility_ignores_the_unsourced_joint():
    """sh_yaw is not scored, so a wild yaw must not flip the flag — that is the lower-bound property."""
    q = np.zeros((1, 7))
    q[0, 0] = 100.0                             # radians: absurd, and deliberately unscored
    r = feasibility(q)
    assert r["human_feasible"] is True
    assert r["excess_rad"][0] == 0.0


def test_feasibility_excess_is_the_distance_past_the_limit():
    q = np.zeros((1, 7))
    q[0, 6] = JOINT_LIMITS["wrist_dev_hand"].hi + math.radians(7.0)
    assert math.degrees(feasibility(q)["excess_rad"][6]) == pytest.approx(7.0)


def test_feasibility_accepts_a_single_pose_and_a_trajectory():
    assert feasibility(np.zeros(7))["human_feasible"] is True
    assert feasibility(np.zeros((10, 7)))["human_feasible"] is True
    with pytest.raises(ValueError):
        feasibility(np.zeros((10, 5)))


# --------------------------------------------------------------------------- #
# 4. The repaired excitation
# --------------------------------------------------------------------------- #
def test_constrained_excitation_is_feasible_and_unconstrained_is_not():
    """Both halves matter: the fix works, and the bug it fixed was real and total."""
    bad = good = 0
    for seed in range(20):
        _, q_off, _, _ = excitation_trial(rng=np.random.default_rng(seed), respect_limits=False)
        _, q_on, _, _ = excitation_trial(rng=np.random.default_rng(seed), respect_limits=True)
        bad += not feasibility(q_off)["human_feasible"]
        good += feasibility(q_on)["human_feasible"]
    assert bad == 20, "the original trajectory was infeasible in every trial — keep that recorded"
    assert good == 20


def test_the_datasets_own_excitation_stream_is_feasible_end_to_end():
    """Replays the exact stream the generator uses, because 20 hand-picked seeds were not enough.

    The first regenerated dataset came back 95/100 on `excite`: five trials overran
    `wrist_dev_hand` by an excess that rounded to 0.000 degrees. Scaling filled the range exactly,
    which puts the extremum *on* the stop and lets float64 rounding decide a strict comparison.
    `FIT_MARGIN` fixes it. This test walks the generator's own RNG rather than seeds 0..19, so it
    would have caught it.
    """
    from skilldata.generate_dynamics import EXCITE_SEED_OFFSET

    rng = np.random.default_rng(0 + EXCITE_SEED_OFFSET)
    infeasible = []
    for i in range(100):
        _, q, _, _ = excitation_trial(fs=500.0, rng=rng)
        if not feasibility(q)["human_feasible"]:
            infeasible.append(i)
    assert infeasible == [], f"trials {infeasible} overran a joint limit"


def test_fit_margin_leaves_a_real_standoff_from_the_joint_stop():
    """The margin has to be visible in the data, not just enough to win a floating-point tie."""
    from sim.motions import FIT_MARGIN

    _, q, _, _ = excitation_trial(rng=np.random.default_rng(0), respect_limits=True)
    dev = JOINT_ORDER.index("wrist_dev_hand")
    lim = JOINT_LIMITS["wrist_dev_hand"]
    assert q[:, dev].max() < lim.hi
    assert q[:, dev].min() > lim.lo
    slack = min(lim.hi - q[:, dev].max(), q[:, dev].min() - lim.lo)
    assert slack > math.radians(0.1), "a standoff smaller than 0.1 deg is a rounding fix, not a design"
    assert 0.9 < FIT_MARGIN < 1.0


def test_constrained_excitation_still_drives_every_joint():
    """The whole point of the class is 7 moving joints; shrinking must not silence one."""
    _, _, qd, _ = excitation_trial(rng=np.random.default_rng(3))
    assert (np.abs(qd).max(axis=0) > 0.05).all()


def test_respect_limits_does_not_disturb_the_rng_stream():
    """Scaling happens after every draw, so a trial's coefficients cannot depend on the flag."""
    for flag in (True, False):
        rng = np.random.default_rng(11)
        excitation_trial(rng=rng, respect_limits=flag)
        state = rng.bit_generator.state["state"]
        if flag:
            first = state
    assert first == state, "turning the flag on must consume exactly the same random numbers"


def test_scaling_is_affine_and_consistent_across_derivatives():
    """q must be scaled about the new centre by exactly the factor qd and qdd are scaled by."""
    from sim.motions import FIT_MARGIN

    _, q_on, qd_on, qdd_on = excitation_trial(rng=np.random.default_rng(5), respect_limits=True)
    _, q_off, qd_off, qdd_off = excitation_trial(rng=np.random.default_rng(5), respect_limits=False)
    for j, name in enumerate(JOINT_ORDER):
        iv = usable_interval(name)
        if iv is None:
            np.testing.assert_allclose(q_on[:, j], q_off[:, j], atol=1e-15)
            continue
        mid = 0.5 * (iv[0] + iv[1])
        peak = np.abs(q_off[:, j]).max()
        s = min(1.0, 0.5 * (iv[1] - iv[0]) * FIT_MARGIN / peak)
        np.testing.assert_allclose(q_on[:, j], mid + q_off[:, j] * s, atol=1e-12)
        np.testing.assert_allclose(qd_on[:, j], qd_off[:, j] * s, atol=1e-12)
        np.testing.assert_allclose(qdd_on[:, j], qdd_off[:, j] * s, atol=1e-12)


def test_derivatives_stay_analytic_after_scaling():
    """The class's selling point is exact derivatives; an affine map must not spoil them."""
    t, q, qd, qdd = excitation_trial(fs=2000.0, rng=np.random.default_rng(2))
    dt = t[1] - t[0]
    np.testing.assert_allclose(np.gradient(q, dt, axis=0)[5:-5], qd[5:-5], rtol=0, atol=2e-5)
    np.testing.assert_allclose(np.gradient(qd, dt, axis=0)[5:-5], qdd[5:-5], rtol=0, atol=2e-3)


def test_naturalistic_classes_are_untouched_by_this_task():
    """3.16 must not perturb _SPECS — that would break SkillData v1 alignment (routed to Heitor)."""
    for mc in MOTION_CLASSES:
        a = generate_trial(mc, rng=np.random.default_rng(4))
        b = generate_trial(mc, rng=np.random.default_rng(4))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        with pytest.raises(TypeError):
            generate_trial(mc, rng=np.random.default_rng(4), respect_limits=True)


def test_feasibility_cost_falls_on_the_tightest_joint():
    """Anatomy is not free, and the bill goes to the joint with the least room. Direction, pinned.

    wrist_dev_hand has a 50-degree range against the elbow's 148, so it is the only joint whose
    amplitude must shrink to fit. Measured 11.9% -> 16.4% while the other six move <0.1pp.

    **The noise level must be shared between the two runs**, and getting that wrong is not a
    hypothetical: the first version of this test set sigma to 1% of each set's *own* RMS torque, and
    since constraining the trajectory lowers RMS torque (5.80 -> 4.89 N m) it handed the constrained
    run a smaller sigma and made every joint look 16% better. That is the same confound task 3.15
    already had to fix once in `damping_error_bars`. One absolute sigma, both runs.
    """
    from analysis.identifiability import _theta0, build_Y
    dyn = human_arm_7dof_dynamics()
    b_true = np.diag(dyn.B)
    SHARED_SIGMA_NM = 0.0849804          # the dataset's own shared torque noise

    def err_bars(constrain):
        rng = np.random.default_rng(7)
        S = []
        for _ in range(4):
            t, q, qd, qdd = excitation_trial(rng=rng, respect_limits=constrain)
            for k in np.linspace(int(0.05 * len(t)), int(0.95 * len(t)) - 1, 24, dtype=int):
                S.append((q[k], qd[k], qdd[k]))
        Y = build_Y(dyn, S)
        _ = Y @ np.concatenate([_theta0(dyn), b_true])   # asserts the shapes line up
        cov = np.linalg.pinv(Y.T @ Y, rcond=1e-12)
        return SHARED_SIGMA_NM * np.sqrt(np.maximum(np.diag(cov)[15:], 0.0)) / b_true

    off, on = err_bars(False), err_bars(True)
    dev = JOINT_ORDER.index("wrist_dev_hand")
    assert on[dev] > off[dev] * 1.15, "the tight joint must visibly pay"
    others = [j for j in range(7) if j != dev]
    assert np.allclose(on[others], off[others], rtol=0.05), "no other joint should move much"


# --------------------------------------------------------------------------- #
# 5. The shipped dataset carries the flag, and the flag is true
# --------------------------------------------------------------------------- #
requires_dataset = pytest.mark.skipif(
    not (DATA / "manifest.json").exists(), reason="dynamics dataset not generated")


@requires_dataset
def test_manifest_carries_feasibility_with_its_provenance():
    m = json.loads((DATA / "manifest.json").read_text())
    hf = m["human_feasibility"]
    assert hf["n_trials"] == m["n_trials"]
    assert hf["unsourced_joints"] == ["sh_yaw"]
    assert "gate" in hf and "Phase-5" in hf["gate"]
    for name, blk in hf["joints"].items():
        assert (blk is None) == (name in hf["unsourced_joints"])


@requires_dataset
def test_stored_flag_matches_a_recomputation_from_the_stored_angles():
    m = json.loads((DATA / "manifest.json").read_text())
    checked = 0
    for entry in m["trials"]:
        if entry["trial_index"] >= 3:
            continue
        z = np.load(DATA / entry["file"])
        recomputed = feasibility(z["q"].astype(float))
        assert bool(z["human_feasible"]) == entry["human_feasible"]
        assert recomputed["human_feasible"] == entry["human_feasible"], entry["file"]
        np.testing.assert_allclose(np.rad2deg(recomputed["excess_rad"]), z["rom_excess_deg"],
                                   atol=1e-3)
        checked += 1
    assert checked >= 15


def test_feasibility_can_be_rescored_in_place_without_regenerating(tmp_path, monkeypatch):
    """The limits table is expected to change (tier B is a substitute), so re-scoring must be cheap.

    Also pins the property that makes it safe: re-scoring rewrites only the feasibility fields and
    leaves the identifiability block, the true damping and the trial trajectories alone.
    """
    from skilldata import generate_dynamics as gd

    gd.generate_dynamics_dataset(tmp_path, per_class=2, excite_trials=2, identifiability=False)
    before = json.loads((tmp_path / "manifest.json").read_text())
    q_before = np.load(tmp_path / before["trials"][0]["file"])["q"].copy()
    assert before["human_feasibility"]["per_class"]["throw"]["n_human_feasible"] == 0

    # Widen every limit far past anything the arm does; everything must now pass. This stands in for
    # the real reason this path exists — swapping tier B for a better source changes the yardstick.
    from sim.limits import JointLimit
    wide = {n: (None if v is None else JointLimit(-99.0, 99.0, v.tier, v.movement, v.source))
            for n, v in JOINT_LIMITS.items()}
    monkeypatch.setattr("sim.limits.JOINT_LIMITS", wide)

    after = gd.recompute_feasibility(tmp_path)
    assert all(tr["human_feasible"] for tr in after["trials"])
    assert after["human_feasibility"]["n_trials_feasible"] == after["n_trials"]
    assert after["human_feasibility"]["per_class"]["throw"]["n_human_feasible"] == 2

    # untouched
    assert after["true_damping"] == before["true_damping"]
    assert after["n_trials"] == before["n_trials"]
    np.testing.assert_array_equal(np.load(tmp_path / after["trials"][0]["file"])["q"], q_before)
    assert bool(np.load(tmp_path / after["trials"][0]["file"])["human_feasible"]) is True


def test_feasibility_only_cli_round_trips(tmp_path, capsys):
    from skilldata.generate_dynamics import generate_dynamics_dataset, main

    generate_dynamics_dataset(tmp_path, per_class=2, excite_trials=2, identifiability=False)
    m = main(["--out", str(tmp_path), "--feasibility-only"])
    assert m["human_feasibility"]["n_trials"] == m["n_trials"]
    assert "human feasibility" in capsys.readouterr().out


@requires_dataset
def test_the_excitation_class_is_now_feasible_and_throw_is_still_not():
    """The half 3.16 fixed, and the half that is Heitor's to decide. Both stated, neither hidden."""
    m = json.loads((DATA / "manifest.json").read_text())
    per = m["human_feasibility"]["per_class"]
    assert per["excite"]["n_human_feasible"] == per["excite"]["n_trials"]
    assert per["reach"]["n_human_feasible"] == per["reach"]["n_trials"]
    assert per["lift"]["n_human_feasible"] == per["lift"]["n_trials"]
    assert per["throw"]["n_human_feasible"] == 0
    assert per["wrist_rotate"]["n_human_feasible"] < per["wrist_rotate"]["n_trials"]
    assert "forearm_pron" in per["throw"]["rom_violating_joints"]


# --------------------------------------------------------------------------- #
# 6. The two identifiability code paths must agree about the same dataset
# --------------------------------------------------------------------------- #
def test_inline_and_reloaded_identifiability_agree(tmp_path):
    """A number that depends on which function computed it is a number nobody can reproduce.

    Before task 3.16 they disagreed: generation built the regressor from float64 memory while
    `--identifiability-only` reloaded float32 from disk, giving cond(Y) 2.7128e11 vs 2.5329e11 on
    the identical 320 naturalistic samples — a 6.6% gap, because the smallest singular value sits
    where float32 storage noise lives. The shipped 3.15 manifest carried the reloaded number and a
    fresh run produced the other one, which is how it surfaced.
    """
    from skilldata.generate_dynamics import generate_dynamics_dataset, recompute_identifiability

    inline = generate_dynamics_dataset(tmp_path, per_class=4, excite_trials=4)
    reloaded = recompute_identifiability(tmp_path)

    for key in ("naturalistic_classes", "excite", "combined"):
        a, b = inline["identifiability"][key], reloaded["identifiability"][key]
        assert a["regressor_rank"] == b["regressor_rank"]
        assert a["condition_number"] == pytest.approx(b["condition_number"], rel=1e-6), key
        for joint, v in a["relative_error_bar"].items():
            assert v == pytest.approx(b["relative_error_bar"][joint], rel=1e-6), f"{key}/{joint}"


def test_float32_storage_moves_cond_but_not_the_error_bars():
    """Pins *why* the paths used to disagree, so the next person does not re-derive it.

    This is the measurement behind `_as_stored`: rounding the trajectories to the precision the
    dataset is stored at shifts the condition number by a few percent and leaves everything that
    matters alone.
    """
    from analysis.identifiability import build_Y
    from skilldata.generate_dynamics import _as_stored

    dyn = human_arm_7dof_dynamics()
    b_true = np.diag(dyn.B)
    rng = np.random.default_rng(0)
    raw = []
    for mc in MOTION_CLASSES:
        for _ in range(3):
            t, q, qd, qdd = generate_trial(mc, fs=500.0, rng=rng)
            for k in np.linspace(int(0.15 * t.size), int(0.9 * t.size) - 1, 8, dtype=int):
                raw.append((q[k], qd[k], qdd[k]))

    def analyse(samples):
        Y = build_Y(dyn, samples)
        s = np.linalg.svd(Y, compute_uv=False)
        tol = max(Y.shape) * s[0] * np.finfo(float).eps
        rank = int((s > tol).sum())
        cov = np.linalg.pinv(Y.T @ Y, rcond=1e-12)
        return s[0] / s[rank - 1], np.sqrt(np.maximum(np.diag(cov)[15:], 0.0)) / b_true

    c64, e64 = analyse(raw)
    c32, e32 = analyse([_as_stored(*s) for s in raw])

    assert c64 > 1e9, "this only bites on an ill-conditioned regressor — check the premise holds"
    assert abs(c32 / c64 - 1.0) > 0.01, "float32 must visibly move cond, else the note is wrong"
    np.testing.assert_allclose(e32, e64, rtol=1e-5)   # and must not move what we actually quote


def test_manifest_reports_identifiability_on_the_subset_phase5_may_use(tmp_path):
    """The gate restricts Phase 5 to feasible trials, so the manifest must score that subset.

    Otherwise Phase 5 plans against `combined` — which is dominated by `throw` and `wrist_rotate`,
    the two classes the gate excludes — and meets the real conditioning only after training.
    """
    from skilldata.generate_dynamics import generate_dynamics_dataset

    m = generate_dynamics_dataset(tmp_path, per_class=4, excite_trials=4)
    idb = m["identifiability"]
    fo = idb["feasible_only"]
    assert fo is not None
    assert set(fo["classes_contributing"]) <= {"reach", "lift", "wrist_rotate", "excite"}
    assert "throw" not in fo["classes_contributing"], "throw is infeasible in every trial"
    assert fo["regressor_rank"] == fo["regressor_cols"], "the allowed subset must still be full rank"
    assert set(fo["relative_error_bar"]) == set(JOINT_ORDER)
    # every sample it used must come from a trial the flag passed
    n_feas = sum(m["identifiability"]["sampled_per_class_feasible"].values())
    assert 0 < n_feas <= sum(m["identifiability"]["sampled_per_class"].values())
