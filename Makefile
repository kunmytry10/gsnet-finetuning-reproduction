# AnyGraspNet dataset visualization helpers.
# Override variables at runtime, for example:
#   make vis-rgb SCENE=scene_0001 FRAME=0010
DATA_ROOT ?= /home/zky-miakho/datas
SPLIT ?= train_1
SCENE ?= scene_0000
CAMERA ?= kinect
FRAME ?= 0000
RECT_SELECTION ?= balanced
RECT_PER_OBJECT_LIMIT ?= 12
GRASP_OBJECT_ID ?= 14
GRASP_POINT_INDEX ?=
GRASP_DIRECTION_INDEX ?=
GRASP_DIRECTION_LENGTH ?=
GRASP_TOPK_PER_OBJECT ?= 3
ANTIPODAL_MAX_MODEL_POINTS ?= 9000

.PHONY: help vis-rgb vis-depth vis-rgb-depth vis-label vis-pointcloud vis-annotations vis-pose-axes vis-rect-labels vis-grasp-label vis-topk-3d-grasps vis-collision-label vis-antipodal test-vis

help:
	@echo "AnyGraspNet visualization commands:"
	@echo "  make vis-rgb        Export RGB image for the selected scene/frame"
	@echo "  make vis-depth      Export colorized depth image for the selected scene/frame"
	@echo "  make vis-rgb-depth  Export side-by-side RGB/depth figure for notes"
	@echo "  make vis-label      Export label overlay for the selected scene/frame"
	@echo "  make vis-pointcloud Export RGB point cloud and a preview image"
	@echo "  make vis-annotations Export object pose table and top-down plot"
	@echo "  make vis-pose-axes  Project annotation pose axes onto the RGB image"
	@echo "  make vis-rect-labels Export 2D grasp rectangle overlay"
	@echo "  make vis-grasp-label Export object-level 3D grasp label overview and 300 directions"
	@echo "  make vis-topk-3d-grasps Export top-k 3D grippers on RGB"
	@echo "  make vis-collision-label Export collision label summaries and filtered grasps"
	@echo "  make vis-antipodal  Export antipodal-analysis figures on the real object model"
	@echo "  make test-vis       Run visualization tests"
	@echo ""
	@echo "Variables: DATA_ROOT=$(DATA_ROOT) SPLIT=$(SPLIT) SCENE=$(SCENE) CAMERA=$(CAMERA) FRAME=$(FRAME)"

vis-rgb:
	python3 -m visualization.scripts.visualize_rgb --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-depth:
	python3 -m visualization.scripts.visualize_depth --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-rgb-depth:
	python3 -m visualization.scripts.visualize_rgb_depth_pair --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-label:
	python3 -m visualization.scripts.visualize_label --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-pointcloud:
	python3 -m visualization.scripts.visualize_pointcloud --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-annotations:
	python3 -m visualization.scripts.visualize_annotations --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-pose-axes:
	python3 -m visualization.scripts.visualize_pose_axes --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME)

vis-rect-labels:
	python3 -m visualization.scripts.visualize_rect_labels --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME) --selection $(RECT_SELECTION) --per-object-limit $(RECT_PER_OBJECT_LIMIT)

vis-grasp-label:
	python3 -m visualization.scripts.visualize_grasp_label --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME) --object-id $(GRASP_OBJECT_ID) $(if $(GRASP_POINT_INDEX),--point-index $(GRASP_POINT_INDEX),) $(if $(GRASP_DIRECTION_INDEX),--direction-index $(GRASP_DIRECTION_INDEX),) $(if $(GRASP_DIRECTION_LENGTH),--direction-length $(GRASP_DIRECTION_LENGTH),)

vis-topk-3d-grasps:
	python3 -m visualization.scripts.visualize_topk_grasps --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME) --topk-per-object $(GRASP_TOPK_PER_OBJECT)

vis-collision-label:
	python3 -m visualization.scripts.visualize_collision_label --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME) --topk-per-object $(GRASP_TOPK_PER_OBJECT)

vis-antipodal:
	python3 -m visualization.scripts.visualize_antipodal_analysis --data-root $(DATA_ROOT) --split $(SPLIT) --scene $(SCENE) --camera $(CAMERA) --frame $(FRAME) --object-id $(GRASP_OBJECT_ID) --max-model-points $(ANTIPODAL_MAX_MODEL_POINTS)

test-vis:
	python3 -m pytest visualization/tests -q
