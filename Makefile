CONFIG ?= config.yaml

.PHONY: help generate sample simulate convert validate \
        setup-xtal preview \
        docker-pull docker-build docker-check clean clean-raw clean-processed

# ─────────────────────────────────────────────────────────────────────────────
# Default target
# ─────────────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "  EMsoft EBSD Data Generation Pipeline"
	@echo "  ─────────────────────────────────────────────────────────────"
	@echo "  FIRST TIME SETUP (run once):"
	@echo "  make docker-pull         Pull pre-built EMsoft image"
	@echo "  make setup-xtal          Copy Ni.xtal → Fe_FCC.xtal in xtal_dir"
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
# Pipeline targets
# ─────────────────────────────────────────────────────────────────────────────

generate:
	python datagen/pipeline.py --config $(CONFIG)

sample:
	python datagen/pipeline.py --config $(CONFIG) \
		--skip-simulate --skip-convert

simulate:
	python datagen/pipeline.py --config $(CONFIG) \
		--skip-sample --skip-convert

convert:
	python datagen/pipeline.py --config $(CONFIG) \
		--skip-sample --skip-simulate

validate:
	python datagen/pipeline.py --config $(CONFIG) --validate-only

skip-simulate:
	python datagen/pipeline.py --config $(CONFIG) --skip-simulate

skip-sample:
	python datagen/pipeline.py --config $(CONFIG) --skip-sample

# ─────────────────────────────────────────────────────────────────────────────
# Setup targets
# ─────────────────────────────────────────────────────────────────────────────

setup-xtal:
	@echo "Setting up Fe_FCC.xtal crystal file..."
	@XTAL_DIR=$$(python -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); \
		print(os.path.expanduser(cfg['paths']['xtal_dir']))"); \
	XTAL_NAME=$$(python -c "import yaml; cfg=yaml.safe_load(open('$(CONFIG)')); \
		print(cfg['emsoft']['xtalname'])"); \
	DATA_DIR=$$(python -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); \
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
		IMAGE=$$(python -c "import yaml; cfg=yaml.safe_load(open('$(CONFIG)')); \
			print(cfg['docker']['image'])"); \
		echo "[setup-xtal] Not found on host — fetching from Docker image $$IMAGE..."; \
		docker run --rm \
			-v "$$XTAL_DIR:/xtal_out" \
			"$$IMAGE" \
			bash -c "set -e; \
				SRC=\$$(find /home/EMs /root -name 'Ni.xtal' 2>/dev/null | head -1); \
				if [ -z \"\$$SRC\" ]; then \
					cd /tmp && git clone --depth 1 https://github.com/EMsoft-org/EMsoftData.git; \
					SRC=/tmp/EMsoftData/Ni.xtal; \
				fi; \
				cp \"\$$SRC\" /xtal_out/$$XTAL_NAME; \
				echo '[setup-xtal] Done: '\$$SRC' → /xtal_out/$$XTAL_NAME'"; \
	fi; \
	echo "[setup-xtal] Complete."

preview:
	@H5=$$(python -c "import yaml,os; cfg=yaml.safe_load(open('$(CONFIG)')); \
		raw=os.path.expanduser(cfg['paths']['raw_dir']); \
		exp=cfg['paths']['experiment_name']; \
		print(os.path.join(raw, exp, 'Fe_EBSD_patterns.h5'))"); \
	echo "Previewing: $$H5"; \
	python scripts/visualize.py $$H5 2>/dev/null || \
	python visualize.py $$H5

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
