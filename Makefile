# --------
# Makefile (full)
# --------

# Python executables
PY        ?= python
PIP       ?= $(PY) -m pip

# Paths
REPO_ROOT := $(CURDIR)

# ----------------------
# Default / Help
# ----------------------
.PHONY: help
help:
	@echo ""
	@echo "Targets:"
	@echo "  install        - Upgrade pip and install package (editable) + dev deps"
	@echo "  test           - Run full pytest"
	@echo "  test-fast      - Quick tests (cli / ingest / runner_dry)"
	@echo "  run-smoke      - CPU smoke run on gpu_smoke.csv (safe params)"
	@echo "  run-gpu        - GPU run on gpu_smoke.csv (safe params; set your GPU id)"
	@echo "  diagnose       - Show simple diagnose for nf_auto_runs"
	@echo "  clean-plan     - Show workspace-clean plan (dry-run)"
	@echo "  clean-all      - Clean workspace (no confirm)"
	@echo ""
	@echo "  tidy            - レイアウト整理を実行"
	@echo "  tidy-dry        - ドライラン（変更なし）"
	@echo "  tidy-aggressive - 退避/egg-info も削除"
	@echo "  tidy-move-data  - サンプルCSVを tests/data へ移動"
	@echo "  tidy-no-symlink - lightning_logs の symlink を作らない"

# ----------------------
# Setup
# ----------------------
.PHONY: install
install:
	$(PIP) install -U pip
	$(PIP) install -e .
	$(PIP) install black ruff pytest

# ----------------------
# Tests
# ----------------------
.PHONY: test
test:
	pytest -q

.PHONY: test-fast
test-fast:
	pytest -q -k "cli or ingest or runner_dry"

# ----------------------
# Runs
# ----------------------
# NOTE:
#  - gpu_smoke.csv は極端に短いデータなので、空ウィンドウを避けるために
#    --gpu-smoke と --val-size h を明示。
#  - CPU 実行を強制するため CUDA_VISIBLE_DEVICES="" を前置。
.PHONY: run-smoke
run-smoke:
	CUDA_VISIBLE_DEVICES="" $(PY) -m tsl.cli.tsl run-auto \
		--data-csv "$(REPO_ROOT)/gpu_smoke.csv" \
		--num-samples 1 \
		--gpu-smoke \
		--val-size h \
		--save-model

# GPU で走らせたいとき。使用する GPU は環境に合わせて変更（例: 0）
.PHONY: run-gpu
run-gpu:
	CUDA_VISIBLE_DEVICES=0 $(PY) -m tsl.cli.tsl run-auto \
		--data-csv "$(REPO_ROOT)/gpu_smoke.csv" \
		--num-samples 1 \
		--gpu-smoke \
		--val-size h \
		--save-model

# ----------------------
# CLI helpers
# ----------------------
.PHONY: diagnose
diagnose:
	$(PY) -m tsl.cli.tsl diagnose --base "$(REPO_ROOT)/nf_auto_runs"

.PHONY: clean-plan
clean-plan:
	$(PY) -m tsl.cli.tsl workspace-clean --all --dry-run

.PHONY: clean-all
clean-all:
	$(PY) -m tsl.cli.tsl workspace-clean --all -y

.PHONY: tidy tidy-dry tidy-aggressive tidy-move-data tidy-no-symlink

TIDY := scripts/tidy_layout.sh

tidy:
	- $(TIDY)

tidy-dry:
	- $(TIDY) --dry-run

tidy-aggressive:
	- $(TIDY) --aggressive

tidy-move-data:
	- $(TIDY) --move-sample-data

tidy-no-symlink:
	- $(TIDY) --no-symlink

