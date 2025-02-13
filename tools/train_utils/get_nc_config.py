from mmcv import Config
import sys

if len(sys.argv) != 3:
    print("Usage: python check_config.py <config_file> <num_classes>")
    sys.exit(1)

cfg = Config.fromfile(sys.argv[1])
num_classes = int(sys.argv[2])
heads = []

# Check query_head
if 'query_head' in cfg.model:
    heads.append(f"model.query_head.num_classes={num_classes}")

# Check roi_head
if 'roi_head' in cfg.model:
    if isinstance(cfg.model.roi_head, list):
        if hasattr(cfg.model.roi_head[0], 'bbox_head'):
            heads.append(f"model.roi_head.0.bbox_head.num_classes={num_classes}")
        if hasattr(cfg.model.roi_head[0], 'mask_head'):
            heads.append(f"model.roi_head.0.mask_head.num_classes={num_classes}")
    else:
        if hasattr(cfg.model.roi_head, 'bbox_head'):
            heads.append(f"model.roi_head.bbox_head.num_classes={num_classes}")
        if hasattr(cfg.model.roi_head, 'mask_head'):
            heads.append(f"model.roi_head.mask_head.num_classes={num_classes}")

# Check bbox_head
if 'bbox_head' in cfg.model:
    if isinstance(cfg.model.bbox_head, list):
        if hasattr(cfg.model.bbox_head[0], 'num_classes'):
            heads.append(f"model.bbox_head.0.num_classes={num_classes}")
    else:
        if hasattr(cfg.model.bbox_head, 'num_classes'):
            heads.append(f"model.bbox_head.num_classes={num_classes}")

# Check mask_iou_head
if 'mask_iou_head' in cfg.model:
    heads.append(f"model.mask_iou_head.num_classes={num_classes}")

# Check mask_head for stage_num_classes
if 'mask_head' in cfg.model and hasattr(cfg.model.mask_head, 'stage_num_classes'):
    num_classes_list = cfg.model.mask_head.stage_num_classes
    if isinstance(num_classes_list, list) and len(num_classes_list) > 1:
        stage_num_classes = [num_classes] * (len(num_classes_list) - 1)
        stage_num_classes.append(num_classes_list[-1])
        stage_num_classes_str = '[' + ','.join(map(str, stage_num_classes)) + ']'
        heads.append(f"model.mask_head.stage_num_classes={stage_num_classes_str}")

# Print list of parameters that need to be overridden
print(" ".join(heads))