"""skilldata.ingest — source-agnostic capture -> SkillData v1 base layer (task 3.11).

Every capture device is an ``IngestAdapter`` producing the same base-layer record, so the fusion /
pHNN / retarget / export pipeline is device-independent. Reference adapters ship for the simulation
(``PlanarSimAdapter``, ``Arm3DSimAdapter``); ``GenericIMUAdapter`` is the template for real hardware.
"""

from .base import IngestAdapter, assemble_base_layer, phase_labels_from_speed
from .generic import GenericIMUAdapter
from .sim import Arm3DSimAdapter, PlanarSimAdapter, planar_imu_streams

__all__ = [
    "IngestAdapter",
    "assemble_base_layer",
    "phase_labels_from_speed",
    "planar_imu_streams",
    "PlanarSimAdapter",
    "Arm3DSimAdapter",
    "GenericIMUAdapter",
]
