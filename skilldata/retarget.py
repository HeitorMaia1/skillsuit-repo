"""3D retargeting — map a captured end-effector path onto a humanoid arm via IK (post-3.2).

The robot is an ``Arm3D`` chain (e.g. ``sim.arm3d.reference_humanoid_7dof`` — a stand-in for an open
humanoid like Unitree G1 until a URDF is parsed into an Arm3D). We solve **damped-least-squares (DLS)
inverse kinematics** — reusing the exact Arm3D forward kinematics and a finite-difference Jacobian, so
no external IK dependency — to place the robot's end-effector on the captured hand path. The result
populates the SkillData v1 robot-ready ``retarget`` layer for a 3D target.

This is the 3D counterpart to ``encoder.attach_retarget`` (planar 2-link). It consumes any base-layer
capture's end-effector path, so it works with the source-agnostic ingest layer.
"""

from __future__ import annotations

import numpy as np


def _fk_batch(robot, Q, ee_name):
    """End-effector positions for a batch of configs ``Q`` (M, n) -> (M, 3).

    Uses ``Arm3D.state`` with zero velocities (the "time" axis carries the M configs).
    """
    Q = np.atleast_2d(np.asarray(Q, float))
    z = np.zeros_like(Q)
    return np.asarray(robot.state(Q, z, z)["joints"][ee_name], float)


def dls_ik(robot, targets, ee_name, *, q0=None, lam=0.05, iters=120, eps=1e-6, tol=1e-4):
    """Damped-least-squares IK for a 3D position path.

    Solves for joint angles that place ``robot``'s ``ee_name`` on each target in ``targets`` (T, 3).
    Warm-starts each frame from the previous solution for temporal continuity. Returns
    ``(q_traj (T, n), residual (T,))`` where residual is the final Euclidean position error (m).

    DLS update: ``dq = Jᵀ (J Jᵀ + λ²I)⁻¹ e`` with ``e = target − fk(q)``; damping ``λ`` keeps the
    step stable near singular/near-unreachable configs.
    """
    targets = np.atleast_2d(np.asarray(targets, float))
    n = len(robot.segments)
    q = np.zeros(n) if q0 is None else np.asarray(q0, float).copy()
    I3 = np.eye(3)
    q_traj = np.zeros((targets.shape[0], n))
    residual = np.zeros(targets.shape[0])

    for t, target in enumerate(targets):
        for _ in range(iters):
            # base config + n forward-perturbed configs, evaluated in one FK batch
            Q = np.vstack([q, q + eps * np.eye(n)])
            P = _fk_batch(robot, Q, ee_name)
            p0 = P[0]
            J = (P[1:] - p0).T / eps          # (3, n)
            e = target - p0
            if np.linalg.norm(e) < tol:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * I3, e)
            q = q + dq
        q_traj[t] = q
        residual[t] = np.linalg.norm(target - _fk_batch(robot, q, ee_name)[0])
    return q_traj, residual


def attach_retarget_3d(
    record,
    ee_path,
    robot,
    *,
    robot_name,
    ee_name="wrist_dev_hand",
    joint_names=None,
    urdf_ref=None,
    reach_tol=5e-3,
    **ik_kwargs,
):
    """Retarget ``ee_path`` (T, 3) onto a 3D ``robot`` (Arm3D) via DLS-IK; store the robot-ready layer.

    Mutates and returns ``record``, adding ``record['retarget'][robot_name]`` in SkillData v1 shape.
    ``reachable_flag`` is True where the IK residual is under ``reach_tol``.
    """
    ee_path = np.asarray(ee_path, float)
    joint_names = joint_names or [s.name for s in robot.segments]
    q_traj, residual = dls_ik(robot, ee_path, ee_name, **ik_kwargs)

    target_robot = {"name": robot_name, "dof": len(robot.segments)}
    if urdf_ref is not None:
        target_robot["urdf_ref"] = urdf_ref
    record.setdefault("retarget", {})[robot_name] = {
        "target_robot": target_robot,
        "method": "dls_ik_arm3d",
        "joint_names": list(joint_names),
        "joint_trajectory_rad": q_traj.tolist(),
        "target_position_m": ee_path.tolist(),
        "ik_residual_m": residual.tolist(),
        "reachable_flag": [bool(r < reach_tol) for r in residual],
    }
    return record
