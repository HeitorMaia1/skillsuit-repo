# SkillSuit

A generalized wearable system that captures human motor data across arbitrary arm motions and
exports it as **AI-training-ready data**. A 4-IMU arm sleeve streams synchronized inertial data over
USB; the pipeline fuses it into clean kinematics, labels it by motion class, and serializes it to a
stable format — **SkillData v1** — that a model can train on. Port-Hamiltonian Neural Networks
(pHNN) form the scientific core: they learn the energy and dissipation structure of the captured
motion, making the exported data physically grounded rather than raw trajectories alone.

This is a **cheap feasibility proof-of-concept**: prove the capture is possible and that the data is
usable for training. Hardware is a tethered v0 (USB serial stream, USB-powered, no SD/battery).

## Quick Start

```bash
uv sync      # install the pinned dependency set (Python 3.12)
make all     # run the full synthetic pipeline end-to-end
```

`make all` runs the synthetic stages in order (generate → fuse → model → figures). The hardware
capture path is documented in [docs/](docs/) once Track B lands. Run `make help` to list targets.

> Requires [uv](https://docs.astral.sh/uv/) and Python 3.12. The synthetic stages (`synthetic`,
> `fusion`, `models`) are implemented across Phases 3–5; until then those targets stop with a clear
> error.

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `notebooks/00_synthetic_dataset.ipynb` | Load and plot synthetic trajectories per motion class; visualize IMU noise; dataset stats. |
| `notebooks/01_fusion_validation.ipynb` | Madgwick vs EKF convergence curves and RMS orientation error by motion class / phase. |
| `notebooks/02_hnn_synthetic.ipynb` | HNN on conservative-phase data; true vs predicted phase-space; learned Hamiltonian level sets. |
| `notebooks/03_porthnn_synthetic.ipynb` | pHNN on full motions; learned dissipation magnitude vs time across motion classes (core result). |

Notebooks are exploratory; `make all` reproduces every figure non-interactively from scripts.

## Layout

| Path | What |
|------|------|
| `src/skilldata/` | The SkillData v1 format + capture/export pipeline — the core deliverable. |
| `src/fusion/` | Sensor fusion (Madgwick, EKF). *(Phase 4)* |
| `src/models/` | HNN and pHNN modeling core. *(Phase 5)* |
| `src/capture/` | Host-side serial receiver/decoder for real captures. *(Phase 9)* |
| `firmware/` | ESP32-S3 capture firmware + host-side encoder tests. |
| `notebooks/` | Exploratory analysis and figure generation. |
| `paper/` | LaTeX manuscript (secondary deliverable) + figures. |
| `data/raw/`, `data/processed/` | Raw captures and exported training-ready datasets. |
| `docs/` | SkillData v1 spec, calibration protocol, dataset datasheet. |

## Scope

This repository is the open technical and scientific artifact of SkillSuit: the capture hardware,
the SkillData v1 format, the sensor-fusion and pHNN pipeline, and the released datasets. It
intentionally contains no business, financial, or go-to-market material.

## License

MIT — see [LICENSE](LICENSE).
