# SkillData v1 — the AI-training-data contract

SkillData v1 is the stable format that turns one trial of captured arm motion into an
AI-training-ready record. It is the boundary between the hardware/capture side and every
downstream consumer (sensor fusion, the HNN/pHNN modeling, and motor-policy training).
The machine-readable schema is [`skilldata/schema.json`](../skilldata/schema.json) (JSON
Schema draft-07); this document is the authoritative prose. Locked in task 3.7.

One file = **one trial** = one subject performing one motion class once.

## Two layers

SkillData v1 records carry two layers. The split is deliberate and load-bearing.

1. **Base layer — captured human motion (always present, robot-independent).**
   What the sleeve actually measured plus what fusion derives from it: per-IMU streams,
   the recovered joint angles, calibration, and phase labels. This is the irreplaceable
   data; nothing about a robot leaks into it.

2. **Robot-ready layer — `retarget` (optional, derived).**
   The same motion *retargeted* (adjusted to fit a robot's body) onto one or more target
   robot morphologies, computed from the base layer by **inverse kinematics (IK)** — solving
   for the joint angles that put the robot's end-effector on the captured trajectory.
   Because the base layer is robot-independent, adding a new target robot is a *re-run*, not
   a re-capture. Multiple robots can coexist in one record, keyed by robot id.

This protects the project: if retargeting or the pHNN ever fails on a robot, the base
dataset stands alone (the pivot condition), and the generalized framing survives.

## Base layer fields

| Field | Type | Units | Notes |
|---|---|---|---|
| `schema_version` | string | — | Always `"skilldata-v1"`. |
| `session.subject_id` | string | — | e.g. `S001`. |
| `session.motion_class` | string | — | `reach`, `lift`, `wrist_rotate`, `throw`, … (first-class label). |
| `session.trial_index` | int | — | 0-based trial counter. |
| `session.n_samples` | int | — | Length of every per-sample array. |
| `session.sample_rate_hz` | number | Hz | 200–500 (D8). |
| `session.source` | string | — | `synthetic` or `hardware`. |
| `calibration.*` | object | — | Per-sensor gyro bias, accel scale, T-pose alignment quaternions. May be empty for synthetic. |
| `imu_streams.<Sx>.timestamp_us` | int[] | µs | Per-sample timestamp. |
| `imu_streams.<Sx>.angular_velocity_dps` | number[][3] | deg/s | Gyroscope `[x,y,z]`. |
| `imu_streams.<Sx>.linear_accel_g` | number[][3] | g | Accelerometer specific force `[x,y,z]`. |
| `imu_streams.<Sx>.quaternion` | number[][4] | — | Fused orientation `[w,x,y,z]` (added by fusion). |
| `imu_streams.<Sx>.saturation_flag` | bool[] | — | True where the gyro pinned at full scale (D2). |
| `segment_kinematics.joint_angles_rad.<joint>` | number[] | rad | `shoulder` (S0–S2), `elbow` (S2–S4), `wrist` (S4–S5). |
| `phase_labels` | string[] | — | Per-timestep `prep` / `active` / `settle`. |

Sensor IDs match the hardware: `S0` scapula, `S2` upper arm, `S4` forearm, `S5` wrist.
The 2-DOF synthetic source populates `S2` and `S4`; `S0`/`S5` arrive with the 3D arm
(task 3.2) and the hardware capture (Phase 9).

## Robot-ready layer (`retarget.<robot_id>`)

| Field | Type | Units | Notes |
|---|---|---|---|
| `target_robot.name` | string | — | e.g. `unitree_g1`, `planar_2link_demo`, `tesla_optimus`. |
| `target_robot.dof` | int | — | Joints in the targeted chain. |
| `target_robot.urdf_ref` | string | — | Path/URL/identifier of the URDF defining the morphology. |
| `target_robot.link_lengths_m` | number[] | m | Link lengths used for IK, if applicable. |
| `method` | string | — | `analytic_2link`, `ikpy_urdf`, … |
| `joint_names` | string[] | — | Names for each column of the trajectory. |
| `joint_trajectory_rad` | number[][dof] | rad | Per-sample robot joint angles `[n_samples × dof]`. |
| `target_position_m` | number[][] | m | The base-layer end-effector path the IK tracked. |
| `ik_residual_m` | number[] | m | Per-sample Euclidean IK position error. |
| `reachable_flag` | bool[] | — | False where the target left the workspace and was clamped. |

**URDF** (Unified Robot Description Format) = the standard file describing a robot's links
and joints; robotics tooling reads it. The retarget step is parameterized by a URDF so any
robot is a swappable target.

## Reference target & current scope

The buildable reference robot is an **open humanoid** (Unitree G1/H1 — real, research-standard
URDFs, ~7-DOF arms). **Tesla Optimus** is the named aspirational target; the format accepts it
unchanged once a model exists.

Full humanoid (URDF + general IK) retargeting depends on the **3D arm model (task 3.2)** —
you cannot meaningfully map a *planar* human arm onto a 3D humanoid. Until 3.2 lands, the
encoder ships a working **closed-form planar 2-link IK** retargeter (`method: analytic_2link`)
that proves the full mechanism end-to-end — human kinematics → end-effector path → IK → robot
joints → stored in the robot-ready layer — onto a 2-link demo robot. Swapping in the humanoid
is then a URDF + IK-backend change, not a format change.
