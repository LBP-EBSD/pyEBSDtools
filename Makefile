CONFIG ?= config.yaml
# Use .venv if it exists (created by `make venv`), otherwise fall back to system python3
PYTHON ?= $(shell [ -f .venv/bin/python ] && echo .venv/bin/python || [ -f venv/bin/python ] && echo venv/bin/python || (command -v python3 || command -v python))

# Remote sync target — override on command line if needed:
#   make sync REMOTE_HOST=user@otherhost REMOTE_DIR=~/some/path
REMOTE_HOST ?= cosign@kratos.sdslabs.org
REMOTE_DIR  ?= ~/lbp/pyEBSDtools/
SSH_KEY     ?= ~/.ssh/id_ed25519
SSH_PORT    ?= 22

.PHONY: help generate generate-bg sample simulate convert validate \
        setup-xtal preview venv sync ebsd resample \
        docker-pull docker-build docker-check clean clean-raw clean-processed

# ─────────────────────────────────────────────────────────────────────────────
# Default target
# ─────────────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  EMsoft EBSD Data Generation Pipeline"
	@echo "  ─────────────────────────────────────────────────────────────"
	@echo "  FIRST TIME SETUP (run once):"
	@echo "  make venv                Create .venv and install Python deps"
	@echo "  make docker-pull         Pull pre-built EMsoft image"
	@echo ""
	@echo "  REMOTE SYNC:"
	@echo "  make sync                Push code to $(REMOTE_HOST):$(REMOTE_DIR)"
	@echo "  make sync REMOTE_HOST=user@host REMOTE_DIR=~/path  (override)"
	@echo ""
	@echo "  GENERATE DATA:"
	@echo "  make generate            Full pipeline (all 4 stages)"
	@echo "  make generate CONFIG=x   Use a different config file"
	@echo ""
	@echo "  ── Partial runs (skip stages) ──────────────────────────────"
	@echo "  make sample              Stage 1 only: generate angle + label files"
	@echo "  make simulate            Stage 2 only: run EMsoft in Docker"
	@echo "  make convert             Stage 3 only: HDF5 + labels → .npy"
	@echo "  make validate            Stage 4 only: sanity-check .npy output"
	@echo "  make skip-simulate       Stages 1, 3, 4  (reuse existing .h5)"
	@echo "  make skip-sample         Stages 2, 3, 4  (reuse existing angles)"
	@echo ""
	@echo "  INSPECT OUTPUT:"
	@echo "  make preview             Visualise generated patterns (opens window)"
	@echo ""
	@echo "  ── Docker ──────────────────────────────────────────────────"
	@echo "  make docker-pull         Pull pre-built EMsoft image from Hub"
	@echo "  make docker-build        Build EMsoft image locally from source"
	@echo "  make docker-check        Verify Docker + GPU are available"
	@echo ""
	@echo "  ── Cleanup ─────────────────────────────────────────────────"
	@echo "  make clean               Remove all generated data"
	@echo "  make clean-raw           Remove raw .h5 files only"
	@echo "  make clean-processed     Remove processed .npy files only"
	@echo ""
	@echo "  CONFIG = $(CONFIG)"
	@echo ""

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────

venv:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install numpy h5py pyyaml
	@echo ""
	@echo "  Virtual env ready. You can now run: make generate"
	@echo "  (The Makefile uses .venv automatically — no need to activate it.)"

sync:
	@echo "Syncing to $(REMOTE_HOST):$(REMOTE_DIR) ..."
	rsync -avz --exclude='.git' --exclude='.venv' --exclude='data/' \
		-e "ssh -i $(SSH_KEY) -p $(SSH_PORT)" \
		./ $(REMOTE_HOST):$(REMOTE_DIR)
	@echo "Done. Run 'make generate' on $(REMOTE_HOST)."

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline targets
# ─────────────────────────────────────────────────────────────────────────────

generate:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG)

generate-bg:
	@LOG=generate_$$(date +%Y%m%d_%H%M%S).log; \
	nohup make generate CONFIG=$(CONFIG) > $$LOG 2>&1 & \
	echo "Pipeline running in background (PID $$!)"; \
	echo "Log: $$LOG"; \
	echo "Watch: tail -f $$LOG"

sample:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG) \
		--skip-simulate --skip-convert

simulate:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG) \
		--skip-sample --skip-convert

convert:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG) \
		--skip-sample --skip-simulate

validate:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG) --validate-only

skip-simulate:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG) --skip-simulate

skip-sample:
	$(PYTHON) datagen/pipeline.py --config $(CONFIG) --skip-sample

# Re-generate angles.txt then run only EMEBSD (reuses existing MC+master output)
resample:
	$(PYTHON) -c "\
import yaml, sys; sys.path.insert(0, '.'); \
from datagen import sampler; \
cfg = yaml.safe_load(open('$(CONFIG)')); \
sampler.run_from_config(cfg)"

ebsd:
	@DATA_DIR=$$($(PYTHON) -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); print(os.path.expanduser(cfg['paths']['data_dir']))"); \
	EXP=$$($(PYTHON) -c "import yaml; cfg=yaml.safe_load(open('$(CONFIG)')); print(cfg['paths']['experiment_name'])"); \
	IMAGE=$$($(PYTHON) -c "import yaml; cfg=yaml.safe_load(open('$(CONFIG)')); print(cfg['docker']['image'])"); \
	GPU_FLAGS="--gpus all"; \
	for dev in /dev/nvidia0 /dev/nvidiactl /dev/nvidia-uvm; do \
		[ -e "$$dev" ] && GPU_FLAGS="$$GPU_FLAGS --device $$dev"; \
	done; \
	OPENCL_MOUNT=""; \
	[ -d /etc/OpenCL/vendors ] && OPENCL_MOUNT="-v /etc/OpenCL/vendors:/etc/OpenCL/vendors:ro"; \
	docker run --rm $$GPU_FLAGS $$OPENCL_MOUNT \
		-v "$$DATA_DIR:/home/EMuser/EMPlay" \
		"$$IMAGE" \
		bash -c "set -e && \
			XTAL_SRC=\$$(find /home/EMs/EMsoftData -name 'Ni.xtal' 2>/dev/null | head -1) && \
			mkdir -p /tmp/XtalFolder && cp \"\$$XTAL_SRC\" /tmp/XtalFolder/Fe_FCC.xtal && \
			python3 -c \"import json; cfg='/home/EMuser/.config/EMsoft/EMsoftConfig.json'; c=json.load(open(cfg)); c['EMXtalFolderpathname']='/tmp/XtalFolder'; json.dump(c,open(cfg,'w'),indent=4)\" && \
			cd /home/EMuser/EMPlay && EMEBSD $$EXP/EMEBSD.nml"

# ─────────────────────────────────────────────────────────────────────────────
# Setup targets
# ─────────────────────────────────────────────────────────────────────────────

setup-xtal:
	@echo "Setting up Fe_FCC.xtal crystal file..."
	@XTAL_DIR=$$($(PYTHON) -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); \
		print(os.path.expanduser(cfg['paths']['xtal_dir']))"); \
	XTAL_NAME=$$($(PYTHON) -c "import yaml; cfg=yaml.safe_load(open('$(CONFIG)')); \
		print(cfg['emsoft']['xtalname'])"); \
	DATA_DIR=$$($(PYTHON) -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); \
		print(os.path.expanduser(cfg['paths']['data_dir']))"); \
	mkdir -p $$XTAL_DIR; \
	TARGET="$$XTAL_DIR/$$XTAL_NAME"; \
	echo "xtal_dir : $$XTAL_DIR"; \
	echo "target   : $$TARGET"; \
	if [ -f "$$TARGET" ]; then \
		echo "[setup-xtal] $$XTAL_NAME already exists — skipping."; \
	elif [ -f "$$DATA_DIR/$$XTAL_NAME" ]; then \
		cp "$$DATA_DIR/$$XTAL_NAME" "$$TARGET"; \
		echo "[setup-xtal] Copied $$DATA_DIR/$$XTAL_NAME → $$TARGET"; \
	else \
		IMAGE=$$($(PYTHON) -c "import yaml; cfg=yaml.safe_load(open('$(CONFIG)')); \
			print(cfg['docker']['image'])"); \
		echo "[setup-xtal] Not found on host — fetching from Docker image $$IMAGE..."; \
		docker run --rm \
			-v "$$XTAL_DIR:/xtal_out" \
			"$$IMAGE" \
			bash -c "set -e; \
				SRC=\$$(find /home/EMs /root -name 'Ni.xtal' 2>/dev/null | head -1); \
				if [ -z \"\$$SRC\" ]; then \k
					cd /tmp && git clone --depth 1 https://github.com/EMsoft-org/EMsoftData.git; \
					SRC=/tmp/EMsoftData/Ni.xtal; \
				fi; \
				cp \"\$$SRC\" /xtal_out/$$XTAL_NAME; \
				echo '[setup-xtal] Done: '\$$SRC' → /xtal_out/$$XTAL_NAME'"; \
	fi; \
	echo "[setup-xtal] Complete."

preview:
	@H5=$$($(PYTHON) -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); \
		data=os.path.expanduser(cfg['paths']['data_dir']); \
		exp=cfg['paths']['experiment_name']; \
		print(os.path.join(data, exp, 'Fe_EBSD_patterns.h5'))"); \
	echo "Previewing: $$H5"; \
	$(PYTHON) scripts/visualize.py $$H5 2>/dev/null || \
	$(PYTHON) visualize.py $$H5

# ─────────────────────────────────────────────────────────────────────────────
# Docker targets
# ─────────────────────────────────────────────────────────────────────────────

docker-pull:
	docker pull marcdegraef/emsoft:buildx-latest

docker-build:
	@echo "Building EMsoft Docker image from local source..."
	@echo "This requires the EMsoft repo to be at ../../ relative to EBSDtools"
	docker build -t emsoft:local -f Dockerfile.emsoft ../../

docker-check:
	@echo "── Docker ──────────────────────────────────────────────────"
	@docker --version
	@docker info --format '{{.ServerVersion}}' | xargs -I{} echo "  Docker server: {}"
	@echo ""
	@echo "── GPU ─────────────────────────────────────────────────────"
	@nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null \
		|| echo "  nvidia-smi not found (GPU may still work via Docker)"
	@docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi 2>/dev/null \
		&& echo "  GPU passthrough: OK" \
		|| echo "  GPU passthrough: not available (CPU mode will be used)"

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup
# ─────────────────────────────────────────────────────────────────────────────

clean: clean-raw clean-processed
	@echo "All generated data removed."

clean-raw:
	rm -rf data/raw
	@echo "Removed data/raw/"

clean-processed:
	rm -rf data/processed
	@echo "Removed data/processed/"
