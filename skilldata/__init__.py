"""skilldata — the SkillData v1 format and capture/export pipeline.

This package is the core deliverable's home: the stable contract between the
wearable hardware and every downstream consumer (sensor fusion, modeling, and
AI-training pipelines). It turns raw multi-IMU arm-motion capture into a clean,
labeled, AI-training-ready motor dataset.

The format is specified in ``schema.json`` and ``docs/skilldata_v1.md`` (locked as
SkillData v1 in task 3.7). The Python serializer + training-ready export path live in
``encoder.py`` (task 3.8): ``build_record`` (base layer), ``attach_retarget`` (robot-ready
layer), ``export``, and ``validate``.
"""

__version__ = "0.1.0"
SCHEMA_VERSION = "skilldata-v1"
