"""Human joint limits for `human_arm_7dof`, with the source of every number attached (task 3.16).

Why this module exists
----------------------
The depth pass of 2026-08-14 measured the task-3.15 dataset against published human joint limits
and found that **three of its five motion classes describe movements no person can perform** — the
`excite` class hyperextends the elbow by 64 degrees in every trial, and `throw` and `wrist_rotate`
rotate the forearm to roughly twice its travel. Those amplitudes were written by hand in June and
never checked against a human.

SkillSuit's whole claim is about recovering the dynamics of *human* motion from a *wearable*. A
result obtained on motions nobody can make is a statement about the simulator's conditioning, not
about human movement, and no sleeve is needed to establish it. So the limits belong in the code,
beside the arm they constrain, and every one of them has to carry the source it came from.

The two-tier rule, and why it is visible in the data rather than buried here
---------------------------------------------------------------------------
Not every joint has an equally good source, and pretending otherwise is the exact failure this
module exists to prevent. So each limit carries a `tier`:

* **Tier A — primary.** A number read verbatim from a peer-reviewed paper. Two joints qualify:
  elbow flexion/extension and forearm pronation/supination, from **Zwerus, Willigenburg, Scholtes,
  Somford, Eygendaal & van den Bekerom (2017), *Normative values and affecting factors for the
  elbow range of motion*, Shoulder & Elbow 11(3):215-224, doi:10.1177/1758573217728711** — n = 352
  healthy adults, universal goniometer, *active* range of the dominant arm, reported in the
  abstract as "mean flexion was 146 deg, extension -2 deg, pronation 80 deg and supination 87 deg".

* **Tier B — secondary.** The AAOS normative reference standard (*Joint Motion: Method of Measuring
  and Recording*, AAOS, Chicago; also tabulated in Norkin & White, *Measurement of Joint Motion: A
  Guide to Goniometry*, 5th ed., F.A. Davis). Used for the shoulder and wrist. **This project has
  not read either book**; the values were taken from a reference chart that reproduces them
  (https://goniometer.io/range-of-motion, retrieved 2026-08-14, which cites both). That is a
  tertiary source and it is recorded as such rather than dressed up.

  The right primary source is **Aizawa, Masuda, Hyodo, Jinno, Yagishita, Nakamaru, Koyama & Morita
  (2013), *Ranges of active joint motion for the shoulder, elbow, and wrist in healthy adults*,
  Disabil Rehabil 35(16):1342-1349, doi:10.3109/09638288.2012.731133** (n = 20, FASTRAK
  electromagnetic tracking, exactly these joints in 3D). It is paywalled; Unpaywall returns no OA
  copy and Springer/JBJS full text was unreachable. Replacing Tier B with it is an open task.

  One reassurance about Tier B, and it is a real check rather than a hope: AAOS gives elbow flexion
  150 deg / extension 0 deg, against Zwerus's measured 146 / -2. The two disagree by 4 and 2
  degrees on the only joint where they can be compared, which is a point in favour of the AAOS
  numbers being usable for the joints Zwerus does not cover.

* **Unsourced.** `sh_yaw` has **no limit at all**, deliberately. It is the arm's azimuth in the
  transverse plane, and mapping AAOS's frontal-plane abduction and across-body adduction onto it
  requires asserting which way the model's +y axis points for a right arm. The previous depth pass
  ruled on exactly this: inventing a joint mapping is worse than admitting the coarseness. So the
  feasibility flag is computed over six joints and is **a lower bound on violations, not a count**,
  and the dataset says so.

The sign conventions
--------------------
`human_arm_7dof` puts the arm horizontal and forward at q = 0, gravity along -z. Reading the axes
straight off `sim.arm3d.human_arm_7dof`:

| j | name | axis | anatomical movement | zero pose means |
|---|------|------|---------------------|-----------------|
| 0 | `sh_yaw` | z | horizontal ab/adduction | arm pointing forward |
| 1 | `sh_pitch` | y | shoulder elevation | arm horizontal; **+pi/2 is hanging at the side** |
| 2 | `sh_roll_upper` | x | humeral axial rotation | neutral rotation |
| 3 | `elbow_fore` | z | elbow flexion | fully extended |
| 4 | `forearm_pron` | x | pronation/supination | neutral |
| 5 | `wrist_flex` | z | wrist flexion/extension | neutral |
| 6 | `wrist_dev_hand` | y | radial/ulnar deviation | neutral |

`sh_pitch` needs its mapping stated, because the model's zero is not the anatomical one. Anatomical
neutral is the arm hanging, which is `sh_pitch = +pi/2`. AAOS shoulder flexion of 180 deg (hanging
-> forward -> overhead) therefore runs to `sh_pitch = -pi/2`, and AAOS extension of 60 deg (hanging
-> backward) runs to `sh_pitch = +150 deg`. Hence [-90, +150].

For the three joints whose two directions have different limits and whose positive sense this model
does not fix (`sh_roll_upper`, `wrist_flex`, `wrist_dev_hand`, and `forearm_pron`), the larger
excursion is assigned to the positive side. That is a convention, it is flagged per joint as
`sign_convention_assumed`, and it matters only for trajectories that are asymmetric about zero.

Singularities are a separate constraint, and not anatomy
--------------------------------------------------------
Task 3.13 found two configurations where the inertia matrix `M` loses rank: the shoulder at
`sh_pitch = pi/2` — which is the arm's own hanging rest pose — and the wrist at `wrist_flex = pi/2`.
Those are defects of the *joint parameterisation*, not of the person: a real arm hangs quite happily.

They are kept out of `human_feasible` for that reason, and exposed separately through
`usable_interval`, which returns the widest stretch of a joint's anatomical range that stays clear
of them. That is what a trajectory *generator* should centre itself in; whether a recorded motion
was humanly possible is a different question with a different answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Joint order of `sim.arm3d.human_arm_7dof`.
JOINT_ORDER = ("sh_yaw", "sh_pitch", "sh_roll_upper", "elbow_fore",
               "forearm_pron", "wrist_flex", "wrist_dev_hand")

ZWERUS = ("Zwerus et al. (2017), Shoulder & Elbow 11(3):215-224, "
          "doi:10.1177/1758573217728711, n=352, active ROM, dominant arm")
AAOS = ("AAOS, Joint Motion: Method of Measuring and Recording (via the reference chart at "
        "goniometer.io/range-of-motion, retrieved 2026-08-14; also in Norkin & White, "
        "Measurement of Joint Motion, 5th ed.). SECONDARY -- see sim/limits.py")


@dataclass(frozen=True)
class JointLimit:
    """One joint's anatomical travel, in radians, with the source it came from."""

    lo: float
    hi: float
    tier: str            # "A" primary, "B" secondary
    movement: str        # the anatomical movement this axis corresponds to
    source: str
    sign_convention_assumed: bool = False

    @property
    def mid(self) -> float:
        return 0.5 * (self.lo + self.hi)

    @property
    def half(self) -> float:
        return 0.5 * (self.hi - self.lo)


def _deg(lo, hi, **kw):
    return JointLimit(math.radians(lo), math.radians(hi), **kw)


#: Per-joint anatomical limits. ``None`` means no source was available -- see the module docstring.
JOINT_LIMITS: dict[str, JointLimit | None] = {
    # No defensible mapping from AAOS's frontal-plane abduction onto this axis. Not guessed.
    "sh_yaw": None,
    "sh_pitch": _deg(-90.0, 150.0, tier="B", movement="shoulder elevation (AAOS flexion 180 deg / "
                     "extension 60 deg, re-zeroed: model 0 is horizontal, +pi/2 is hanging)",
                     source=AAOS),
    "sh_roll_upper": _deg(-70.0, 90.0, tier="B", movement="humeral axial rotation "
                          "(AAOS external 90 deg / internal 70 deg)",
                          source=AAOS, sign_convention_assumed=True),
    "elbow_fore": _deg(-2.0, 146.0, tier="A", movement="elbow flexion/extension",
                       source=ZWERUS),
    "forearm_pron": _deg(-87.0, 80.0, tier="A", movement="forearm pronation/supination",
                         source=ZWERUS, sign_convention_assumed=True),
    "wrist_flex": _deg(-70.0, 80.0, tier="B", movement="wrist flexion/extension "
                       "(AAOS flexion 80 deg / extension 70 deg)",
                       source=AAOS, sign_convention_assumed=True),
    "wrist_dev_hand": _deg(-20.0, 30.0, tier="B", movement="radial/ulnar deviation "
                           "(AAOS radial 20 deg / ulnar 30 deg)",
                           source=AAOS, sign_convention_assumed=True),
}

UNSOURCED = tuple(n for n, v in JOINT_LIMITS.items() if v is None)

#: Configurations where `M(q)` loses rank (task 3.13). NOT anatomy -- see the module docstring.
SINGULAR_ANGLES: dict[str, tuple] = {
    "sh_pitch": (math.pi / 2,),
    "wrist_flex": (math.pi / 2,),
}

#: How far a generated trajectory should stay from a singular angle, in radians.
SINGULARITY_MARGIN = math.radians(20.0)


def usable_interval(joint: str, margin: float = SINGULARITY_MARGIN) -> tuple | None:
    """Widest stretch of `joint`'s anatomical range that stays `margin` clear of a singularity.

    Returns ``(lo, hi)`` in radians, or ``None`` if the joint has no sourced limit.

    This is the interval a *generator* should centre a trajectory in. It is deliberately not what
    `feasibility` scores against: a pose near the shoulder singularity is perfectly possible for a
    person, it is only badly conditioned for this parameterisation.

    The shoulder is the case that makes this necessary. Its anatomical range is [-90, 150] degrees,
    so the naive midpoint is +30 degrees -- which walks a large-amplitude oscillation straight into
    the gimbal lock at +90. Excluding a band around that leaves [-90, 70] and [110, 150]; the wider
    one wins, and the centre moves to -10 degrees.
    """
    lim = JOINT_LIMITS.get(joint)
    if lim is None:
        return None
    cuts = [a for a in SINGULAR_ANGLES.get(joint, ()) if lim.lo < a < lim.hi]
    if not cuts:
        return (lim.lo, lim.hi)
    edges = [lim.lo]
    for a in sorted(cuts):
        edges += [a - margin, a + margin]
    edges.append(lim.hi)
    best, best_w = None, -1.0
    for lo, hi in zip(edges[::2], edges[1::2]):
        if hi - lo > best_w:
            best, best_w = (lo, hi), hi - lo
    return best


def feasibility(q) -> dict:
    """Is this trajectory anatomically possible? `q` is (T, 7) or (7,) in radians.

    Returns ``{"human_feasible": bool, "excess_rad": (7,), "n_joints_scored": int,
    "worst_joint": str|None}``. ``excess_rad[j]`` is how far joint *j* travels beyond its limit,
    0.0 when it is inside and for any joint with no sourced limit.

    **`human_feasible` is a lower bound.** Joints in `UNSOURCED` are not scored at all, so a trial
    can be flagged feasible and still be impossible in a way this project cannot yet measure.
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    if q.shape[1] != len(JOINT_ORDER):
        raise ValueError(f"q must have {len(JOINT_ORDER)} joints; got {q.shape[1]}")
    excess = np.zeros(len(JOINT_ORDER))
    for j, name in enumerate(JOINT_ORDER):
        lim = JOINT_LIMITS[name]
        if lim is None:
            continue
        excess[j] = max(0.0, float(q[:, j].max() - lim.hi), float(lim.lo - q[:, j].min()))
    worst = int(excess.argmax())
    return {
        "human_feasible": bool(excess.max() <= 0.0),
        "excess_rad": excess,
        "n_joints_scored": len(JOINT_ORDER) - len(UNSOURCED),
        "worst_joint": JOINT_ORDER[worst] if excess[worst] > 0 else None,
    }


def limits_manifest() -> dict:
    """The limits table as plain JSON, so a dataset can carry its own provenance."""
    return {
        "what_this_is": (
            "The human joint limits every trial in this dataset was scored against, with the "
            "source of each one. Added by task 3.16 after the depth pass found three of five "
            "motion classes describing movements no person can perform. A dataset that claims to "
            "be about human motion without saying which of its trials a human could produce is "
            "the same trap the identifiability block exists to close."
        ),
        "reading": (
            "human_feasible is a LOWER BOUND: joints listed in unsourced_joints are not scored, "
            "so a trial can pass and still be impossible. Tier A is a number read verbatim from a "
            "peer-reviewed paper; tier B is the AAOS reference standard reached through a "
            "tertiary chart rather than the book itself."
        ),
        "unsourced_joints": list(UNSOURCED),
        "n_joints_scored": len(JOINT_ORDER) - len(UNSOURCED),
        "singularity_note": (
            "SINGULAR_ANGLES are configurations where M(q) loses rank (task 3.13). They are a "
            "defect of the joint parameterisation, not of the person, so they do NOT affect "
            "human_feasible. They constrain trajectory generation only, via usable_interval()."
        ),
        "joints": {
            name: (None if lim is None else {
                "lo_deg": round(math.degrees(lim.lo), 3),
                "hi_deg": round(math.degrees(lim.hi), 3),
                "tier": lim.tier,
                "movement": lim.movement,
                "source": lim.source,
                "sign_convention_assumed": lim.sign_convention_assumed,
            })
            for name, lim in JOINT_LIMITS.items()
        },
    }
