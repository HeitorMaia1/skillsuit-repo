# SkillSuit — reproducible synthetic pipeline (Phase 1.5).
#
# `make all` runs the full synthetic pipeline end-to-end. The stage targets
# reference pipeline modules that land in Phases 3-5; until each is implemented,
# that stage stops with a clear error. Phase 10.5 verifies `make all` reproduces
# every figure from scratch in a clean env.

PY := uv run python

.PHONY: all sync synthetic fusion models figures paper test clean help

help:  ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN{FS=":.*?## "}{printf "  %-12s %s\n", $$1, $$2}'

all: synthetic fusion models figures  ## full synthetic pipeline end-to-end

sync:  ## install the locked dependency set into the uv-managed venv
	uv sync

synthetic:  ## Phase 3: generate + export the synthetic SkillData v1 dataset (task 3.9)
	$(PY) -m skilldata.generate_synthetic --out data/processed --n 1000

fusion:  ## Phase 4: run Madgwick + EKF, compute RMS orientation error
	$(PY) -m fusion.run_validation --data data/processed --figures paper/figures

models:  ## Phase 5: train HNN (conservative phase) + pHNN (full motions)
	$(PY) -m models.train_hnn  --data data/processed --figures paper/figures
	$(PY) -m models.train_phnn --data data/processed --figures paper/figures

figures: fusion models  ## (alias) regenerate all paper figures

paper:  ## Phase 7/10.7 (secondary): compile the LaTeX PDF
	cd paper && latexmk -pdf main.tex

test:  ## run the test suite (Python + host-side firmware encoder tests)
	uv run pytest

clean:  ## remove generated data, figures, and LaTeX build artifacts
	rm -rf data/raw/synthetic/* data/processed/* paper/figures/*.pdf
	cd paper && latexmk -C 2>/dev/null || true
