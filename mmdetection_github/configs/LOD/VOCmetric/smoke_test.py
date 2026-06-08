_base_ = [
    './R_Net_denoise_50.py',
]

# Smoke test: 1 epoch on GPU, verify pipeline then exit
train_cfg = dict(_delete_=True, type='EpochBasedTrainLoop', max_epochs=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

train_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
    dataset=dict(
        times=1,
        dataset=dict(
            datasets=[
                dict(
                    type='LOD_RAWDAtaset',
                    data_root = 'E:/!yan/SoftWare/code/Dark-ISP/Dark-ISP/dataset/RAWTest',
                    ann_file='trainval/train.txt',
                    img_subdir='RAW_dark_npz',
                    ann_subdir='RGB-dark-Annotations',
                    data_prefix=dict(sub_data_root = 'E:/!yan/SoftWare/code/Dark-ISP/Dark-ISP/dataset/RAWTest'),
                    pipeline=[
                        dict(type='LoadRAWImageFromFile', postfix='npz'),
                        dict(type='LoadAnnotations', with_bbox=True),
                        dict(type='Resize', scale=(667, 400), keep_ratio=True),
                        dict(type='RandomFlip', prob=0.5),
                        dict(type='PackDetInputs',
                             meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                                        'scale_factor', 'flip', 'flip_direction', 'gt',
                                        'iso', 'wb', 'white_level', 'black_level', 'ccm',
                                        'raw_pattern', 'exp_time', 'ratio')),
                    ],
                    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
                )
            ]
        )
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=0,
    persistent_workers=False,
)

test_dataloader = val_dataloader

auto_scale_lr = None
