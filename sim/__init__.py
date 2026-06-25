"""sim — synthetic motion + ideal IMU generation via analytic rigid-body dynamics.

Synthetic data is produced from analytic kinematics (no physics engine): exact
ground-truth poses, velocities, accelerations, and energies. For a 2-DOF arm and a
pendulum this is both lighter and more correct than a contact-physics simulator —
it gives clean ground truth, which is exactly what HNN/pHNN validation needs.
"""

__version__ = "0.1.0"
