SHELL := /bin/bash

ARTIFACT_ROOT ?= artifacts
FULL_OUTPUT ?= $(ARTIFACT_ROOT)/full-run
GPU_INDEX ?= 0
STAGES ?= nominal,sensitivity,horizon,evaluate,landscapes,aggregate,figures,verify
REPRO_FLAGS ?=

.PHONY: setup setup-cuda test figures verify reproduce-plan reproduce smoke

setup:
	uv sync --frozen --group dev

setup-cuda:
	uv sync --frozen --extra cuda --group dev

test:
	uv run python -m pytest -q

figures:
	uv run python reproduce.py --source frozen --output "$(ARTIFACT_ROOT)"

verify: test
	@set -eu; temporary=$$(mktemp -d); \
	trap 'rm -rf "$$temporary"' EXIT; \
	uv run python reproduce.py --source frozen --output "$$temporary"; \
	test "$$(find "$$temporary/figures" -maxdepth 1 -name '*.pdf' | wc -l)" -eq 9

reproduce-plan:
	uv run python reproduce.py --source simulation --gpu-index "$(GPU_INDEX)" \
		--stages "$(STAGES)" --output "$(FULL_OUTPUT)"

reproduce:
	uv run python reproduce.py --source simulation --execute \
		--gpu-index "$(GPU_INDEX)" --stages "$(STAGES)" \
		--output "$(FULL_OUTPUT)" $(REPRO_FLAGS)

smoke:
	CUDA_VISIBLE_DEVICES="$(GPU_INDEX)" uv run python -m cbo_mppi.reproduction.smoke \
		--output-root "$(ARTIFACT_ROOT)/smoke" $(REPRO_FLAGS)
