"""Quick test: verify dataset loads correctly before training."""
import sys
sys.path.insert(0, r'E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\mmdetection_github')

import mmcv
import mmengine
import mmdet

# Must import transforms to register PackDetInputs, LoadRAWImageFromFile etc.
# in their respective registries BEFORE Compose tries to look them up.
import mmdet.datasets.transforms  # registers PackDetInputs
import mmcv.transforms            # registers LoadRAWImageFromFile
from mmdet.datasets.lod_bmvc21 import LOD_RAWDAtaset

data_root = r'E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\dataset\RAWTest'

pipeline = [
    dict(type='LoadRAWImageFromFile', postfix='npz'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(667, 400), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                     'scale_factor', 'flip', 'flip_direction', 'gt',
                     'iso', 'wb', 'white_level', 'black_level', 'ccm',
                     'raw_pattern', 'exp_time', 'ratio'))
]

dataset = LOD_RAWDAtaset(
    data_root=data_root,
    ann_file='trainval/train.txt',
    img_subdir='RAW_dark_npz',
    ann_subdir='RGB-dark-Annotations',
    data_prefix=dict(sub_data_root=''),
    filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
    pipeline=pipeline,
)

print(f'Dataset size: {len(dataset)} images')

# Load first sample
data = dataset[0]
print(f'Sample 0 keys: {list(data.keys())}')
print(f'  img_id: {data.get("img_id")}')
if 'inputs' in data:
    print(f'  input shape: {data["inputs"].shape}')
if 'data_samples' in data:
    ds = data['data_samples']
    print(f'  gt_bboxes: {ds.gt_bboxes.shape if hasattr(ds, "gt_bboxes") else "N/A"}')

# Load a few more to check no errors
for i in [100, 500, 1000]:
    d = dataset[i]
    print(f'  Sample {i}: img_id={d.get("img_id")}, ok')

print('Dataset test PASSED')
