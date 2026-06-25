"""skilldata — the SkillData v1 format and capture/export pipeline.

This package is the core deliverable's home: the stable contract between the
wearable hardware and every downstream consumer (sensor fusion, modeling, and
AI-training pipelines). It turns raw multi-IMU arm-motion capture into a clean,
labeled, AI-training-ready motor dataset.

The format is specified in ``schema.json`` (placeholder at the Phase 1 skeleton
stage) and finalized as SkillData v1 in Phase 3.7. The Python serializer
(``encoder.py``) and the training-ready export path land in Phase 3.8.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "skilldata-v1-draft"
