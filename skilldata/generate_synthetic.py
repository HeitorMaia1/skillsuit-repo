"""Task 3.9 — generate the synthetic SkillData v1 dataset (>=1000 trials).

Draws trials across the four motion classes (reach / lift / wrist_rotate / throw) on the 7-DOF human
arm (``sim.motions`` on ``sim.arm3d.human_arm_7dof``). Each trial is exported in **two variants**:

  clean  — ideal IMU streams (ground-truth reference for fusion validation).
  noisy  — the same motion pushed through the D2 sensor model (``sim.sensor.SensorModel``): gyro/accel
           white noise + gyro bias drift + hard saturation to +-2000 deg/s / +-16 g, with a per-sample
           ``saturation_flag``. This is what the real 4-IMU sleeve would produce.

The captured hand path is retargeted onto the reference humanoid (``reference_humanoid_7dof``) via
DLS-IK (``skilldata.retarget``) once per trial and attached to both variants (the retarget depends on
the kinematics, not the sensor noise). Every record is validated against ``schema.json`` and written to
``<out>`` with a rich ``manifest.json`` carrying per-class and saturation statistics.

Run:  ``uv run python -m skilldata.generate_synthetic --out data/processed``  (see ``--help``).
"""

from __future__ import annotations

import argparse
import copy
import datetime as _dt
import json
from pathlib import Path

import numpy as np

from sim.arm3d import human_arm_7dof, reference_humanoid_7dof
from sim.motions import MOTION_CLASSES, generate_trial
from sim.sensor import SensorModel

from . import SCHEMA_VERSION
from .ingest.sim import Arm3DSimAdapter
from .retarget import attach_retarget_3d
from .encoder import save_record, validate

SUBJECT_ID = "S001"
HUMAN_EE_JOINT = "wrist_dev_hand"          # end-effector (hand tip) joint on human_arm_7dof
ROBOT_NAME = "reference_humanoid_7dof"     # retarget target (Unitree-G1 stand-in)

# D2 sensor-model parameters shared by every noisy record (ranges default to +-2000 dps / +-16 g).
_NOISE = dict(
    gyro_noise_dps=0.1,                    # MEMS gyro white-noise std
    accel_noise_g=0.01,                    # accel white-noise std
    initial_gyro_bias_dps=0.5,             # small turn-on bias
    gyro_bias_drift_dps_per_sqrt_s=0.02,   # bias-instability random walk
)

# Record array fields rounded before serialization (keeps files lean; 1e-6 precision is ample:
# 1e-6 rad ~ 6e-5 deg, 1e-6 m = 1 micron, 1e-6 deg/s / g well below sensor resolution).
_ROUND_DP = 6


def _sensor_model(seed: int) -> SensorModel:
    """A D2 ``SensorModel`` seeded for this trial (reproducible, per-trial-distinct noise)."""
    return SensorModel(seed=seed, **_NOISE)


def _round_arr(x, dp):
    return np.round(np.asarray(x, float), dp).tolist()


def _round_record(rec, dp):
    """Round the float arrays of a record in place (cosmetic size reduction; flags/ints untouched)."""
    for s in rec["imu_streams"].values():
        s["angular_velocity_dps"] = _round_arr(s["angular_velocity_dps"], dp)
        s["linear_accel_g"] = _round_arr(s["linear_accel_g"], dp)
    ja = rec.get("segment_kinematics", {}).get("joint_angles_rad", {})
    for k in ja:
        ja[k] = _round_arr(ja[k], dp)
    for block in rec.get("retarget", {}).values():
        for key in ("joint_trajectory_rad", "target_position_m", "ik_residual_m"):
            if key in block:
                block[key] = _round_arr(block[key], dp)
    return rec


def _class_counts(n, classes):
    """Split ``n`` trials as evenly as possible across ``classes`` (remainder to the first ones)."""
    base, rem = divmod(n, len(classes))
    return [base + (1 if i < rem else 0) for i in range(len(classes))]


def _saturation_stats(rec):
    """(saturated_samples, n_samples, per_sensor_counts) for a noisy record.

    A sample counts as saturated if *any* instrumented sensor pinned its gyro at that sample.
    """
    n = rec["session"]["n_samples"]
    union = np.zeros(n, bool)
    per_sensor = {}
    for sid, s in rec["imu_streams"].items():
        flag = np.asarray(s.get("saturation_flag", []), bool)
        if flag.size == n:
            union |= flag
            per_sensor[sid] = int(flag.sum())
    return int(union.sum()), n, per_sensor


def _peak_gyro_dps(rec):
    """Per-sensor peak |gyro| (deg/s) over a record's streams."""
    out = {}
    for sid, s in rec["imu_streams"].items():
        g = np.asarray(s["angular_velocity_dps"], float)
        out[sid] = float(np.linalg.norm(g, axis=-1).max()) if g.size else 0.0
    return out


def generate_dataset(out, *, n=1000, seed=0, fs=500.0, retarget=True, round_dp=_ROUND_DP, verbose=False):
    """Generate the synthetic dataset into ``out``. Returns the manifest dict.

    ``n`` base motions are split across the four classes; each is written as a clean and a noisy
    record (2*n files). Set ``retarget=False`` to skip the (slower) IK layer.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    arm = human_arm_7dof()
    robot = reference_humanoid_7dof()
    rng = np.random.default_rng(seed)
    counts = _class_counts(n, MOTION_CLASSES)

    trials = []
    per_class = {}
    global_trial = 0
    for mc, count in zip(MOTION_CLASSES, counts):
        cls = {
            "n_trials": count,
            "n_samples": None,          # samples per record (constant within a class)
            "peak_gyro_dps": {},        # max over trials of the clean per-sensor peak
            "saturated_records": 0,     # noisy records with >=1 saturated sample
            "saturated_samples": 0,     # total saturated samples across noisy records
            "total_samples": 0,         # total samples across noisy records
            "per_sensor_saturated": {}, # per-sensor saturated-sample totals (noisy)
        }
        for local_idx in range(count):
            t, qs, qds, qdds = generate_trial(mc, fs=fs, rng=rng)
            ee_path = np.asarray(arm.state(qs, qds, qdds)["joints"][HUMAN_EE_JOINT], float)

            clean = Arm3DSimAdapter(arm, t, qs, qds, qdds, fs).to_base_layer(
                subject_id=SUBJECT_ID, motion_class=mc, trial_index=local_idx)
            clean["session"]["variant"] = "clean"

            noisy = Arm3DSimAdapter(arm, t, qs, qds, qdds, fs,
                                    sensor_model=_sensor_model(seed * 100003 + global_trial)).to_base_layer(
                subject_id=SUBJECT_ID, motion_class=mc, trial_index=local_idx)
            noisy["session"]["variant"] = "noisy"

            retarget_summary = None
            if retarget:
                attach_retarget_3d(clean, ee_path, robot, robot_name=ROBOT_NAME,
                                   urdf_ref="stand-in:unitree_g1")
                noisy["retarget"] = copy.deepcopy(clean["retarget"])
                block = clean["retarget"][ROBOT_NAME]
                retarget_summary = {
                    "robot": ROBOT_NAME,
                    "max_ik_residual_m": float(max(block["ik_residual_m"])),
                    "frac_reachable": float(np.mean(block["reachable_flag"])),
                }

            # clean peak gyro (ground-truth) feeds the per-class envelope stat
            peaks = _peak_gyro_dps(clean)
            for sid, p in peaks.items():
                cls["peak_gyro_dps"][sid] = max(cls["peak_gyro_dps"].get(sid, 0.0), p)

            sat_samples, n_samp, per_sensor = _saturation_stats(noisy)
            cls["n_samples"] = n_samp
            cls["total_samples"] += n_samp
            cls["saturated_samples"] += sat_samples
            cls["saturated_records"] += int(sat_samples > 0)
            for sid, c in per_sensor.items():
                cls["per_sensor_saturated"][sid] = cls["per_sensor_saturated"].get(sid, 0) + c

            for variant, rec in (("clean", clean), ("noisy", noisy)):
                _round_record(rec, round_dp)
                validate(rec)
                fname = f"{SUBJECT_ID}_{mc}_{variant}_{local_idx:04d}.json"
                save_record(rec, out / fname)
                entry = {
                    "file": fname,
                    "subject_id": SUBJECT_ID,
                    "motion_class": mc,
                    "variant": variant,
                    "trial_index": local_idx,
                    "n_samples": rec["session"]["n_samples"],
                    "robots": sorted(rec.get("retarget", {}).keys()),
                }
                if variant == "noisy":
                    entry["saturated_samples"] = sat_samples
                    entry["saturation_frac"] = round(sat_samples / n_samp, 6) if n_samp else 0.0
                if retarget_summary is not None:
                    entry.update(retarget_summary)
                trials.append(entry)
            global_trial += 1
            if verbose and global_trial % 50 == 0:
                print(f"  ... {global_trial}/{n} trials", flush=True)
        cls["saturation_frac"] = round(cls["saturated_samples"] / cls["total_samples"], 6) if cls["total_samples"] else 0.0
        per_class[mc] = cls

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generator": {
            "module": "skilldata.generate_synthetic",
            "seed": seed,
            "sample_rate_hz": fs,
            "n_base_trials": n,
            "variants": ["clean", "noisy"],
            "arm": "human_arm_7dof",
            "retarget_robot": ROBOT_NAME if retarget else None,
            "sensor_model": {**_NOISE, "gyro_range_dps": 2000.0, "accel_range_g": 16.0},
            "round_decimals": round_dp,
            "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        },
        "n_base_trials": n,
        "n_records": len(trials),
        "motion_classes": list(MOTION_CLASSES),
        "per_class": per_class,
        "saturation_note": (
            "Only 'throw' peaks past the D2 +-2000 deg/s gyro range (distal wrist sensor S5 ~3000 deg/s); "
            "reach/lift/wrist_rotate stay inside the envelope. This is the documented cheap-gyro limit."
        ),
        "trials": trials,
    }
    with (out / "manifest.json").open("w") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest


def _print_summary(manifest):
    print(f"SkillData v1 synthetic dataset -> {manifest['n_records']} records "
          f"({manifest['n_base_trials']} trials x clean+noisy), seed {manifest['generator']['seed']}")
    print(f"{'class':13s} {'trials':>6s} {'n_samp':>7s} "
          f"{'peakGyro S5':>12s} {'sat.recs':>8s} {'sat.frac':>8s}")
    for mc in manifest["motion_classes"]:
        c = manifest["per_class"][mc]
        s5 = c["peak_gyro_dps"].get("S5", 0.0)
        nsamp = c["n_samples"] or 0  # samples per record
        print(f"{mc:13s} {c['n_trials']:6d} {nsamp:7d} {s5:11.0f}  "
              f"{c['saturated_records']:8d} {c['saturation_frac']:8.4f}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate the synthetic SkillData v1 dataset (task 3.9).")
    ap.add_argument("--out", default="data/processed", help="output directory")
    ap.add_argument("--n", type=int, default=1000, help="number of base trials (split across 4 classes)")
    ap.add_argument("--seed", type=int, default=0, help="master RNG seed")
    ap.add_argument("--fs", type=float, default=500.0, help="sample rate (Hz)")
    ap.add_argument("--no-retarget", action="store_true", help="skip the DLS-IK robot-ready layer")
    ap.add_argument("--round-dp", type=int, default=_ROUND_DP, help="decimals to round float arrays to")
    args = ap.parse_args(argv)
    manifest = generate_dataset(
        args.out, n=args.n, seed=args.seed, fs=args.fs,
        retarget=not args.no_retarget, round_dp=args.round_dp, verbose=True)
    _print_summary(manifest)


if __name__ == "__main__":
    main()
