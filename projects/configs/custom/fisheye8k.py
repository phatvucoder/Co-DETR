# Kế thừa từ config gốc
_base_ = '../co_deformable_detr/co_deformable_detr_r50_1x_coco.py'
load_from = 'models/co_deformable_detr_r50_1x_coco.pth'

# 1. Dataset settings
dataset_type = 'CocoDataset'
classes = ('Bus', 'Bike', 'Car', 'Pedestrian', 'Truck')  # Fisheye8k classes
data_root = '/kaggle/input/fisheye8k/'

num_classes = len(classes)

img_norm_cfg = dict(
	mean=[97.08351897397036, 98.15350265093993, 95.42513796452337],
	std=[62.79758906384603, 62.18993362009985, 62.71828160370258],
	to_rgb=True
) # Fisheye8k train dataset

# img_norm_cfg = dict(
# 	mean=[93.61435634923271, 95.52524449855612, 92.45636147586735],
# 	std=[62.51498717031526, 62.28997892560787, 62.335156725331025],
# 	to_rgb=True
# ) # Fisheye8k train+test dataset

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train/train.json',
        img_prefix=data_root + 'train/images/',
        pipeline=train_pipeline,
        classes=classes),  
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'test/test.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline,
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'test/test.json',
        img_prefix=data_root + 'test/images/',
        pipeline=test_pipeline),
        classes=classes
)

evaluation = dict(interval=1, metric='bbox')
