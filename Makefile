# GSNet fine-tuning helpers.
#
# Typical usage:
#   make check-env
#   make preprocess-simplify
#   make preprocess-graspness
#   make train
#   make tensorboard
#   make eval-epoch EPOCH=10
#   make vis-epoch EPOCH=10
#
# Override variables as needed, for example:
#   make eval-epoch EPOCH=6 CAMERA=kinect
#   make vis-official SCENE=scene_0100 FRAME=0000

SHELL := /bin/bash

CONDA_ENV ?= gsnet-repro-sm120
REPO_ROOT ?= $(CURDIR)
DATASET_ROOT ?= $(REPO_ROOT)/data/graspnet
OFFICIAL_CKPT ?= $(REPO_ROOT)/weights/kinect/minkuresunet_kinect.tar

CAMERA ?= kinect
SPLIT ?= scenes
SCENE ?= scene_0100
FRAME ?= 0000
EPOCH ?= 10
RUN_NAME ?= weekend_finetune_probe_v4_clean
MAX_EPOCH ?= 10
BATCH_SIZE ?= 4
LR ?= 1e-4
NUM_POINT ?= 15000
VOXEL_SIZE ?= 0.005
VAL_SCENES ?= 100-102
VAL_FRAMES ?= 0-31
VAL_TOP_K ?= 50
TB_PORT ?= 6007
TOPK ?= 50
REPORT_TOPKS ?= 1,5,50

OUTPUT_ROOT ?= $(REPO_ROOT)/gsnet_reproduction/outputs
RUN_ROOT ?= $(OUTPUT_ROOT)/finetune_runs/$(RUN_NAME)
TRAIN_LOG_DIR ?= $(RUN_ROOT)/train_logs
EPOCH_PADDED := $(shell printf "%02d" $(EPOCH))
EPOCH_CKPT ?= $(TRAIN_LOG_DIR)/$(RUN_NAME)_epoch$(EPOCH_PADDED).tar
EVAL_DUMP_DIR ?= $(OUTPUT_ROOT)/test_seen_epoch$(EPOCH_PADDED)_full
VIS_OUTPUT_ROOT ?= $(OUTPUT_ROOT)/vis_epoch$(EPOCH_PADDED)
VIS_OFFICIAL_OUTPUT_ROOT ?= $(OUTPUT_ROOT)/vis_official

RUNTIME_ENV = \
	export OMP_NUM_THREADS=12; \
	export LD_LIBRARY_PATH=$$CONDA_PREFIX/lib:$$CONDA_PREFIX/lib64:$$LD_LIBRARY_PATH; \
	export LD_PRELOAD=$$CONDA_PREFIX/lib/libstdc++.so.6:$$CONDA_PREFIX/lib/libgcc_s.so.1$${LD_PRELOAD:+:$$LD_PRELOAD}; \
	export PYTHONPATH=$(REPO_ROOT)/external/graspness_unofficial/pointnet2:$(REPO_ROOT)/external/graspness_unofficial/knn:$$PYTHONPATH;

CONDA_RUN = conda run -n $(CONDA_ENV) bash -lc

.PHONY: help check-env check-dataset preprocess-simplify preprocess-graspness train tensorboard \
	tail-train eval-official eval-epoch eval-epoch-infer eval-epoch-ap vis-official vis-epoch \
	show-result list-results clean-pyc

help:
	@echo "GSNet reproduction commands"
	@echo ""
	@echo "Setup and checks:"
	@echo "  make check-env                 Check Python/CUDA/MinkowskiEngine/custom extensions"
	@echo "  make check-dataset             Check expected dataset paths"
	@echo ""
	@echo "Preprocessing:"
	@echo "  make preprocess-simplify       Run GraspNet dataset simplification for $(CAMERA)"
	@echo "  make preprocess-graspness      Generate graspness labels for $(CAMERA)"
	@echo ""
	@echo "Training:"
	@echo "  make train                     Fine-tune from official checkpoint with periodic probe"
	@echo "  make tail-train                Follow training log"
	@echo "  make tensorboard               Start TensorBoard on port $(TB_PORT)"
	@echo ""
	@echo "Evaluation:"
	@echo "  make eval-official             Full test_seen evaluation for official checkpoint"
	@echo "  make eval-epoch EPOCH=10       Full test_seen evaluation for a trained checkpoint"
	@echo "  make show-result EPOCH=10      Print saved AP summary for an epoch"
	@echo "  make list-results              List available AP summaries"
	@echo ""
	@echo "Visualization:"
	@echo "  make vis-official              RGB top-k overlays for official checkpoint"
	@echo "  make vis-epoch EPOCH=10        RGB top-k overlays for a trained checkpoint"
	@echo ""
	@echo "Key variables:"
	@echo "  REPO_ROOT=$(REPO_ROOT)"
	@echo "  DATASET_ROOT=$(DATASET_ROOT)"
	@echo "  RUN_NAME=$(RUN_NAME)"
	@echo "  CAMERA=$(CAMERA) SCENE=$(SCENE) FRAME=$(FRAME) EPOCH=$(EPOCH)"

check-env:
	cd $(REPO_ROOT) && conda run -n $(CONDA_ENV) python gsnet_reproduction/scripts/check_env.py

check-dataset:
	cd $(REPO_ROOT) && conda run -n $(CONDA_ENV) python gsnet_reproduction/scripts/check_dataset.py --dataset-root $(DATASET_ROOT)

preprocess-simplify:
	cd $(REPO_ROOT)/external/graspness_unofficial/dataset && \
	conda run -n $(CONDA_ENV) python simplify_dataset.py \
	  --dataset_root $(DATASET_ROOT) \
	  --camera_type $(CAMERA)

preprocess-graspness:
	cd $(REPO_ROOT)/external/graspness_unofficial/dataset && \
	$(CONDA_RUN) '$(RUNTIME_ENV) python generate_graspness.py \
	  --dataset_root $(DATASET_ROOT) \
	  --camera_type $(CAMERA)'

train:
	cd $(REPO_ROOT) && \
	$(CONDA_RUN) '$(RUNTIME_ENV) python gsnet_reproduction/scripts/run_finetune_with_periodic_probe.py \
	  --dataset-root $(DATASET_ROOT) \
	  --camera $(CAMERA) \
	  --checkpoint-path $(OFFICIAL_CKPT) \
	  --run-name $(RUN_NAME) \
	  --max-epoch $(MAX_EPOCH) \
	  --batch-size $(BATCH_SIZE) \
	  --learning-rate $(LR) \
	  --num-point $(NUM_POINT) \
	  --voxel-size $(VOXEL_SIZE) \
	  --val-every 1 \
	  --val-scenes $(VAL_SCENES) \
	  --val-frames $(VAL_FRAMES) \
	  --val-top-k $(VAL_TOP_K)'

tail-train:
	tail -f $(TRAIN_LOG_DIR)/log_train.txt

tensorboard:
	cd $(REPO_ROOT) && conda run -n $(CONDA_ENV) tensorboard \
	  --logdir $(RUN_ROOT)/tensorboard \
	  --port $(TB_PORT) \
	  --bind_all

eval-official:
	$(MAKE) eval-epoch EPOCH=official EPOCH_CKPT=$(OFFICIAL_CKPT) EVAL_DUMP_DIR=$(OUTPUT_ROOT)/test_seen_official_full

eval-epoch: eval-epoch-infer eval-epoch-ap

eval-epoch-infer:
	cd $(REPO_ROOT) && \
	$(CONDA_RUN) '$(RUNTIME_ENV) cd $(REPO_ROOT)/external/graspness_unofficial && \
	python test.py \
	  --camera $(CAMERA) \
	  --split test_seen \
	  --dataset_root $(DATASET_ROOT) \
	  --checkpoint_path $(EPOCH_CKPT) \
	  --dump_dir $(EVAL_DUMP_DIR) \
	  --batch_size 1 \
	  --infer'

eval-epoch-ap:
	cd $(REPO_ROOT) && \
	$(CONDA_RUN) '$(RUNTIME_ENV) python gsnet_reproduction/scripts/eval_dumps_by_scene.py \
	  --dataset_root $(DATASET_ROOT) \
	  --dump_dir $(EVAL_DUMP_DIR) \
	  --camera $(CAMERA) \
	  --split test_seen \
	  --cache_dir $(EVAL_DUMP_DIR)/scene_eval_test_seen_$(CAMERA)'

show-result:
	@cat $(EVAL_DUMP_DIR)/ap_$(CAMERA)_summary.json

list-results:
	@find $(OUTPUT_ROOT) -path "*ap_$(CAMERA)_summary.json" | sort

vis-official:
	$(MAKE) vis-epoch EPOCH=official EPOCH_CKPT=$(OFFICIAL_CKPT) VIS_OUTPUT_ROOT=$(VIS_OFFICIAL_OUTPUT_ROOT)

vis-epoch:
	cd $(REPO_ROOT) && \
	$(CONDA_RUN) '$(RUNTIME_ENV) python gsnet_reproduction/scripts/run_single_frame_inference.py \
	  --dataset-root $(DATASET_ROOT) \
	  --split $(SPLIT) \
	  --scene $(SCENE) \
	  --camera $(CAMERA) \
	  --frame $(FRAME) \
	  --checkpoint-path $(EPOCH_CKPT) \
	  --output-root $(VIS_OUTPUT_ROOT) \
	  --topk $(TOPK) \
	  --report-rgb-only \
	  --report-topks $(REPORT_TOPKS) \
	  --overlay-edge-thickness 1 \
	  --overlay-fill-alpha 0.035 \
	  --overlay-outline-extra 1 \
	  --skip-render'

clean-pyc:
	find $(REPO_ROOT) -type d -name "__pycache__" -prune -exec rm -rf {} +
	find $(REPO_ROOT) -type d -name ".pytest_cache" -prune -exec rm -rf {} +
