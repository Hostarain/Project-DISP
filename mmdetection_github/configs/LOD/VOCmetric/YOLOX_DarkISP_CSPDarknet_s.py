_base_ = [
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
]

# ==================================================================
# Dark-ISP + YOLOX (CSPDarknet-s)  —  Plan C
# ISP processes RAW, CSPDarknet extracts features, YOLOX head detects.
# ==================================================================
model = dict(
    type='RAWYOLOV3',
    data_preprocessor=dict(
        type='DetRAWDataPreprocessor',
        mean=None,
        std=None,
        bgr_to_rgb=False,
        pad_size_divisor=32,
    ),
    backbone=dict(
        type='DarkISP_CSPDarknet',
        ISP_version='v2',
        # CSPDarknet-s pre-trained on COCO (extracted from YOLOX-s checkpoint)
        cspdarknet_init_cfg=dict(
            type='Pretrained',
            checkpoint='checkpoints/cspdarknet_s_coco.pth'),
        # CSPDarknet-s: widen=0.5, deepen=0.33  (~5M backbone params)
        arch='P5',
        deepen_factor=0.33,
        widen_factor=0.5,
        out_indices=(2, 3, 4),           # P3/P4/P5 → channels [128, 256, 512]
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    neck=dict(
        type='YOLOXPAFPN',
        in_channels=[128, 256, 512],     # match CSPDarknet-s channels
        out_channels=128,                  # smaller head for smaller backbone
        num_csp_blocks=3,
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
    ),
    bbox_head=dict(
        type='YOLOXHead',
        num_classes=8,
        in_channels=128,
        feat_channels=128,
        stacked_convs=2,
        strides=[8, 16, 32],
        use_depthwise=False,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='Swish'),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True,
            reduction='sum', loss_weight=1.0),
        loss_bbox=dict(
            type='IoULoss', mode='square', eps=1e-16,
            reduction='sum', loss_weight=5.0),
        loss_obj=dict(
            type='CrossEntropyLoss', use_sigmoid=True,
            reduction='sum', loss_weight=1.0),
        loss_l1=dict(type='L1Loss', reduction='sum', loss_weight=1.0),
    ),
    train_cfg=dict(
        assigner=dict(
            type='SimOTAAssigner',
            center_radius=2.5,
            candidate_topk=10,
            iou_weight=3.0,
            cls_weight=1.0,
        ),
    ),
    test_cfg=dict(
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.65),
        max_per_img=100,
    ),
    cri_pix=dict(type='L1Loss', loss_weight=1.0)
)

# ==================================================================
# dataset — LOD low-light RAW
# ==================================================================
dataset_type = 'LOD_RAWDAtaset'
data_root = 'E:/!yan/SoftWare/code/Dark-ISP/Dark-ISP/dataset/RAWTest'
backend_args = None

train_pipeline = [
    dict(type='LoadRAWImageFromFile', backend_args=backend_args, postfix='npz'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(672, 416), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'flip', 'flip_direction', 'gt',
                    'iso', 'wb', 'white_level', 'black_level', 'ccm',
                    'raw_pattern', 'exp_time', 'ratio'))
]

test_pipeline = [
    dict(type='LoadRAWImageFromFile', backend_args=backend_args, postfix='npz'),
    dict(type='Resize', scale=(672, 416), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'iso', 'wb', 'white_level', 'ratio',
                    'black_level', 'ccm', 'raw_pattern', 'exp_time'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=0,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type='ConcatDataset',
            ignore_keys=['dataset_type'],
            datasets=[
                dict(
                    type=dataset_type,
                    data_root=data_root,
                    ann_file='trainval/train.txt',
                    img_subdir='RAW_dark_npz',
                    ann_subdir='RGB-dark-Annotations',
                    data_prefix=dict(sub_data_root='E:/!yan/SoftWare/code/Dark-ISP/Dark-ISP/dataset/RAWTest'),
                    filter_cfg=dict(
                        filter_empty_gt=True, min_size=32, bbox_min_size=32),
                    pipeline=train_pipeline,
                    backend_args=backend_args,
                    clean_subdir=None)
            ])))

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='trainval/val.txt',
        img_subdir='RAW_dark_npz',
        ann_subdir='RGB-dark-Annotations',
        data_prefix=dict(sub_data_root='E:/!yan/SoftWare/code/Dark-ISP/Dark-ISP/dataset/RAWTest'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args)
)
test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001),
    clip_grad=dict(max_norm=35, norm_type=2))

max_epochs = 15
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=500),
]
train_cfg = dict(max_epochs=max_epochs)

auto_scale_lr = dict(base_batch_size=16)
find_unused_parameters = True

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

# ==================================================================
# NOTE: Input (672, 416) divisible by 32. Mosaic disabled for RAW 4ch.
# YOLOXModeSwitchHook enables L1 loss for final 3 epochs.
# ==================================================================
custom_hooks = [
    dict(
        type='YOLOXModeSwitchHook',
        num_last_epochs=3,
        skip_type_keys=('Mosaic', 'RandomAffine', 'MixUp'),
    ),
]
