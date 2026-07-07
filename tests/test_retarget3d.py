"""Tests for the 3D humanoid retarget (DLS-IK on an Arm3D robot chain)."""

import numpy as np

from sim.arm import min_jerk
from sim.arm3d import human_arm_7dof, reference_humanoid_7dof
from skilldata.encoder import validate
from skilldata.ingest import Arm3DSimAdapter
from skilldata.retarget import _fk_batch, attach_retarget_3d, dls_ik

EE = "wrist_dev_hand"


def _human_reach(fs=100.0, dur=1.5):
    arm = human_arm_7dof()
    t = np.arange(0.0, dur, 1.0 / fs)
    q = np.zeros((t.size, 7))
    qd = np.zeros((t.size, 7))
    qdd = np.zeros((t.size, 7))
    for j, (a0, a1) in {0: (0.0, 0.4), 1: (0.0, 0.7), 3: (0.2, 1.1)}.items():
        q[:, j], qd[:, j], qdd[:, j] = min_jerk(t, 0.3, 1.0, a0, a1)
    hand = np.asarray(arm.state(q, qd, qdd)["joints"][EE], float)
    return arm, t, q, qd, qdd, hand, fs


def test_fk_batch_zero_config():
    robot = reference_humanoid_7dof(l_upper=0.28, l_fore=0.26, l_hand=0.10)
    ee = _fk_batch(robot, np.zeros((1, 7)), EE)[0]
    assert np.allclose(ee, [0.28 + 0.26 + 0.10, 0.0, 0.0], atol=1e-9)


def test_dls_ik_recovers_reachable_targets():
    robot = reference_humanoid_7dof()
    rng = np.random.default_rng(1)
    q_true = rng.uniform(-0.5, 0.5, (8, 7))
    targets = _fk_batch(robot, q_true, EE)          # guaranteed reachable
    q_sol, residual = dls_ik(robot, targets, EE, iters=200)
    assert residual.max() < 1e-3                     # ee reproduced, despite 7-DOF redundancy
    assert np.allclose(_fk_batch(robot, q_sol, EE), targets, atol=1e-3)


def test_attach_retarget_3d_on_human_reach():
    human, t, q, qd, qdd, hand, fs = _human_reach()
    rec = Arm3DSimAdapter(human, t, q, qd, qdd, sample_rate_hz=fs).to_base_layer(
        subject_id="S001", motion_class="reach3d", trial_index=0)
    robot = reference_humanoid_7dof()
    attach_retarget_3d(rec, hand, robot, robot_name="reference_humanoid_7dof",
                       urdf_ref="pending:unitree_g1.urdf")

    assert validate(rec) is True                     # full record: base + 3D robot layer
    block = rec["retarget"]["reference_humanoid_7dof"]
    assert block["method"] == "dls_ik_arm3d"
    assert block["target_robot"]["dof"] == 7
    q_traj = np.array(block["joint_trajectory_rad"])
    assert q_traj.shape == (t.size, 7)
    # the human hand path is within the robot's reach -> essentially all frames solved
    assert np.mean(block["reachable_flag"]) > 0.95
    assert max(block["ik_residual_m"]) < 1e-2
    # the robot end-effector actually reproduces the captured hand path
    assert np.allclose(_fk_batch(robot, q_traj, EE), hand, atol=5e-3)
