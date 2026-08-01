_base_ = [
    '../../_base_/schedules/schedule_1x.py',
    '../../_base_/default_runtime.py',
]

# ============================================================
# Dark-ISP + RTMDet-x (最大参数量, 服务器 2xH100)
# ============================================================
model = dict(
    type='SingleStageRAWDetector',
    data_preprocessor=dict(
        type='DetRAWDataPreprocessor',
        mean=None, std=None, bgr_to_rgb=False, pad_size_divisor=32),
    backbone=dict(
        type='DarkISP_CSPNeXt',
        ISP_version='v2',       # 原始 Dark-ISP, 无 GGNC
        use_grad_guide=False,  # 显式关闭 GGNC 梯度引导
        arch='P5',
        deepen_factor=1.33,
        widen_factor=1.25,
        out_indices=(2, 3, 4),
        channel_attention=True,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True),
        cspnext_init_cfg=dict(
            type='Kaiming', layer='Conv2d', a=2.236,
            distribution='uniform', mode='fan_in',
            nonlinearity='leaky_relu'),
    ),
    neck=dict(
        type='CSPNeXtPAFPN',
        in_channels=[320, 640, 1280],
        out_channels=320,
        num_csp_blocks=4,
        expand_ratio=0.5,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    bbox_head=dict(
        type='RTMDetSepBNHead',
        num_classes=8,
        in_channels=320,
        stacked_convs=2,
        feat_channels=320,
        anchor_generator=dict(
            type='MlvlPointGenerator', offset=0, strides=[8, 16, 32]),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        loss_cls=dict(
            type='QualityFocalLoss', use_sigmoid=True, beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0),
        with_objectness=False,
        exp_on_reg=True,
        share_conv=True,
        pred_kernel_size=1,
        norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
        act_cfg=dict(type='SiLU', inplace=True)),
    train_cfg=dict(
        assigner=dict(type='DynamicSoftLabelAssigner', topk=13),
        allowed_border=-1, pos_weight=-1, debug=False),
    test_cfg=dict(
        nms_pre=30000, min_bbox_size=0, score_thr=0.001,
        nms=dict(type='nms', iou_threshold=0.65), max_per_img=300),
    cri_pix=dict(type='L1Loss', loss_weight=1.0),
)

dataset_type = 'LOD_RAWDAtaset'
data_root = '/path/to/LOD/dataset'
backend_args = None

train_pipeline = [
    dict(type='LoadRAWImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(672, 416), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','flip','flip_direction','gt','iso','wb','white_level','black_level','ccm','raw_pattern','exp_time','ratio')),
]
test_pipeline = [
    dict(type='LoadRAWImageFromFile', backend_args=backend_args),
    dict(type='Resize', scale=(672, 416), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id','img_path','ori_shape','img_shape','scale_factor','iso','wb','white_level','ratio','black_level','ccm','raw_pattern','exp_time')),
]

train_dataloader = dict(
    batch_size=16, num_workers=4, persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file='trainval/train.txt', img_subdir='RAW_dark_npz', ann_subdir='RGB-dark-Annotations',
        data_prefix=dict(sub_data_root=data_root),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=train_pipeline, backend_args=backend_args, clean_subdir=None),
)
val_dataloader = dict(
    batch_size=1, num_workers=2, persistent_workers=False, drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type, data_root=data_root,
        ann_file='trainval/val.txt', img_subdir='RAW_dark_npz', ann_subdir='RGB-dark-Annotations',
        data_prefix=dict(sub_data_root=data_root),
        test_mode=True, pipeline=test_pipeline, backend_args=backend_args),
)
test_dataloader = val_dataloader
val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='11points')
test_evaluator = val_evaluator

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005),
    clip_grad=dict(max_norm=35, norm_type=2),
)
max_epochs = 100
param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=2000),
    dict(type='CosineAnnealingLR', by_epoch=True, begin=5, end=100),
]
train_cfg = dict(max_epochs=max_epochs, val_interval=5)
auto_scale_lr = dict(base_batch_size=64, enable=True)
find_unused_parameters = True
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='spawn', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
