"""
ISP + Detection inference visualization script.

Usage:
    conda activate dark-isp
    python infer_vis.py        # run from project root

Outputs (saved to vis_output/):
    *_01_raw.png       - RAW Bayer input (simple demosaic to RGB)
    *_02_linear.png    - Linear stage output (CCM+WB, gamma-corrected)
    *_03_enhanced.png  - Nonlinear enhanced image (what detector sees)
    *_04_detection.png - Detection results (pred+GT bboxes overlaid)
    *_05_npz_raw.png   - RAW directly from npz file (demosaiced)

Modify NUM_IMAGES to control how many images to process.
"""

import os
import sys
import numpy as np
import cv2
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, 'mmdetection_github'))

from mmengine.config import Config
from mmengine.runner import Runner

# ============================================================
# Configuration
# ============================================================
MMDET_DIR = os.path.join(BASE_DIR, 'mmdetection_github')
CONFIG_PATH = os.path.join(MMDET_DIR, 'configs', 'LOD', 'VOCmetric', 'R_Net_DarkISP_18.py')
CHECKPOINT_PATH = os.path.join(MMDET_DIR, 'checkpoints', '2026_06_04_DarkISP_ResNet18',
                               'best_pascal_voc_mAP_epoch_13.pth')
DATA_ROOT = 'E:/!yan/SoftWare/code/Dark-ISP/Dark-ISP/dataset/RAWTest'
NPZ_DIR = os.path.join(DATA_ROOT, 'RAW_dark_npz')
VAL_TXT = os.path.join(DATA_ROOT, 'trainval', 'val.txt')
OUTPUT_DIR = os.path.join(BASE_DIR, 'vis_output')
NUM_IMAGES = 5  # How many images to process

# VOC class names (8 classes)
CLASSES = ('bicycle', 'car', 'motorbike', 'chair',
           'diningtable', 'bottle', 'tvmonitor', 'bus')
# BGR palette for OpenCV drawing
PALETTE_BGR = [(228, 0, 106), (32, 11, 119), (42, 42, 165), (192, 0, 0),
               (255, 226, 197), (100, 60, 0), (142, 0, 0), (255, 77, 255)]

# Globals to capture ISP intermediates via hooks
_isp_data = {}


def _nonlinear_pre_hook(module, input):
    """Capture x_g (linear output) before nonlinear processing."""
    _isp_data['x_g'] = input[0].detach().clone()


def _nonlinear_hook(module, input, output):
    """Capture x_l (nonlinear/enhanced output)."""
    _isp_data['x_l'] = output.detach().clone()


def _backbone_pre_hook(module, input):
    """Capture the raw input x to backbone, compute x_i = clip(x, 0, 1)."""
    x = input[0]
    _isp_data['x_i'] = x.clamp(0, 1).detach().clone()
    if len(input) > 1:
        _isp_data['metainfo'] = input[1]


# ============================================================
# RAW Bayer -> RGB (simple demosaic)
# ============================================================
def simple_demosaic(bayer_4ch):
    """Packed RGGB Bayer (4, H, W) -> BGR image (H*2, W*2, 3)."""
    if torch.is_tensor(bayer_4ch):
        bayer_4ch = bayer_4ch.cpu().numpy()
    if bayer_4ch.ndim == 4:
        bayer_4ch = bayer_4ch[0]

    R, G1, B, G2 = bayer_4ch.astype(np.float32)
    H, W = R.shape

    R_ch = R
    G_ch = (G1 + G2) / 2.0
    B_ch = B

    # Resize to full resolution
    R_full = cv2.resize(R_ch, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
    G_full = cv2.resize(G_ch, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
    B_full = cv2.resize(B_ch, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)

    # Gaussian smooth
    R_full = cv2.GaussianBlur(R_full, (5, 5), 0)
    G_full = cv2.GaussianBlur(G_full, (5, 5), 0)
    B_full = cv2.GaussianBlur(B_full, (5, 5), 0)

    bgr = np.stack([B_full, G_full, R_full], axis=-1)
    return np.clip(bgr, 0, 1)


def load_npz_raw(npz_path):
    """Load npz, return (4ch_bayer_CHW, raw_pattern)."""
    data = np.load(npz_path)
    img_key = None
    for key in data.keys():
        if isinstance(data[key], np.ndarray) and data[key].ndim >= 2:
            img_key = key
            break
    if img_key is None:
        raise KeyError(f'No image array in {npz_path}, keys: {list(data.keys())}')

    img = data[img_key].astype(np.float32)
    raw_pattern = data.get('raw_pattern', np.array([[0, 1], [3, 2]]))

    # Handle HWC format -> CHW
    if img.ndim == 3 and img.shape[-1] in (3, 4):
        img = img.transpose(2, 0, 1)  # HWC -> CHW

    # Normalize to [0, 1] using white_level
    if img.max() > 1.0:
        white_level = data.get('white_level', img.max())
        if hasattr(white_level, 'ndim') and white_level.ndim > 0:
            # Per-channel white_level, reshape to (C, 1, 1)
            wl = white_level.reshape(-1)[0].item()
        else:
            wl = float(white_level.item() if hasattr(white_level, 'item') else white_level)
        img = img / wl
    return np.clip(img, 0, 1), raw_pattern


def tensor_to_bgr(tensor, apply_gamma=False, auto_brightness=True):
    """(1,3,H,W) or (3,H,W) tensor -> BGR uint8 numpy."""
    if tensor.ndim == 4:
        tensor = tensor[0]
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = np.clip(img, 0, 1)
    if apply_gamma:
        img = np.power(img, 1.0 / 2.2)
    if auto_brightness and img.max() > 0:
        # Stretch 2nd to 98th percentile for dark scene visibility
        lo, hi = np.percentile(img, (2, 98))
        if hi > lo:
            img = np.clip((img - lo) / (hi - lo), 0, 1)
    if img.shape[-1] == 3:
        img = img[:, :, ::-1]
    return (img * 255).astype(np.uint8)


def draw_bboxes(img_bgr, bboxes, labels, scores=None):
    """Draw detection boxes on BGR image."""
    H, W = img_bgr.shape[:2]
    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = [int(max(0, v)) for v in bbox[:4]]
        x2, y2 = min(x2, W), min(y2, H)
        label = int(labels[i])
        color = PALETTE_BGR[label % len(PALETTE_BGR)]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
        text = CLASSES[label] if label < len(CLASSES) else f'cls{label}'
        if scores is not None:
            text += f' {scores[i]:.2f}'
        cv2.putText(img_bgr, text, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return img_bgr


def draw_gt_bboxes(img_bgr, data_sample, scale_factor):
    """Draw GT boxes in green, scaled by scale_factor to match image coordinates."""
    H, W = img_bgr.shape[:2]
    sf_h, sf_w = scale_factor  # (H_scale, W_scale)
    if hasattr(data_sample, 'gt_instances') and data_sample.gt_instances is not None:
        gt = data_sample.gt_instances
        if hasattr(gt, 'bboxes') and len(gt.bboxes) > 0:
            bboxes = gt.bboxes.cpu().numpy()
            labels = gt.labels.cpu().numpy()
            for i, bbox in enumerate(bboxes):
                x1 = int(max(0, bbox[0] * sf_w))
                y1 = int(max(0, bbox[1] * sf_h))
                x2 = int(min(W, bbox[2] * sf_w))
                y2 = int(min(H, bbox[3] * sf_h))
                label = int(labels[i])
                cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                text = CLASSES[label] + ' GT'
                cv2.putText(img_bgr, text, (x1, max(y1 - 5, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
    return img_bgr


# ============================================================
# Main
# ============================================================
def main():
    # Ensure CWD is mmdetection_github (some mmdet internals rely on it)
    os.chdir(MMDET_DIR)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'Output: {OUTPUT_DIR}')
    print(f'Config: {CONFIG_PATH}')
    print(f'Checkpoint: {CHECKPOINT_PATH}')
    print()

    with open(VAL_TXT, 'r') as f:
        img_ids = [line.strip() for line in f if line.strip()]
    print(f'Total val images: {len(img_ids)}')
    img_ids = img_ids[:NUM_IMAGES]
    print(f'Processing {len(img_ids)}: {img_ids}')
    print()

    # Build model
    print('Loading model...')
    cfg = Config.fromfile(CONFIG_PATH)
    cfg.load_from = CHECKPOINT_PATH
    cfg.work_dir = 'work_dirs/_vis_infer'
    cfg.launcher = 'none'

    runner = Runner.from_cfg(cfg)
    runner.call_hook('before_run')
    # Explicitly load checkpoint (Runner.from_cfg doesn't auto-load)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=False)
    runner.model.load_state_dict(checkpoint['state_dict'], strict=True)
    model = runner.model
    model.eval()
    device = next(model.parameters()).device
    print(f'Device: {device}')
    print()

    # Register hooks on backbone ISP pipeline
    backbone = model.backbone
    hook_handles = [
        backbone.register_forward_pre_hook(_backbone_pre_hook),
        backbone.nonlinear.register_forward_pre_hook(_nonlinear_pre_hook),
        backbone.nonlinear.register_forward_hook(_nonlinear_hook),
    ]
    print('ISP hooks registered.\n')

    dataloader = runner.val_dataloader

    try:
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(dataloader):
                if batch_idx >= len(img_ids):
                    break
                img_id = img_ids[batch_idx]
                print(f'===== [{batch_idx+1}/{len(img_ids)}] {img_id} =====')

                # val_step: data_preprocessor(padding) + predict(rescale=True)
                # This ensures predictions match the official test.py results
                results = model.val_step(data_batch)

                # Get scale factor to map original coords -> resized image coords
                ds = data_batch['data_samples'][0]
                sf = ds.metainfo.get('scale_factor', (1.0, 1.0))

                # ---- 1. RAW Bayer input ----
                if 'x_i' in _isp_data:
                    bgr = simple_demosaic(_isp_data['x_i'])
                    sp = os.path.join(OUTPUT_DIR, f'{img_id}_01_raw.png')
                    cv2.imwrite(sp, (bgr * 255).astype(np.uint8))
                    print(f'  [1/5] RAW Bayer -> {os.path.basename(sp)}')
                else:
                    print('  [1/5] x_i not captured!')

                # ---- 2. Linear output (x_g) with gamma correction ----
                if 'x_g' in _isp_data:
                    sp = os.path.join(OUTPUT_DIR, f'{img_id}_02_linear.png')
                    cv2.imwrite(sp, tensor_to_bgr(_isp_data['x_g'], apply_gamma=True))
                    print(f'  [2/5] Linear+Gamma -> {os.path.basename(sp)}')
                else:
                    print('  [2/5] x_g not captured!')

                # ---- 3. Enhanced output (x_l) ----
                if 'x_l' in _isp_data:
                    sp = os.path.join(OUTPUT_DIR, f'{img_id}_03_enhanced.png')
                    cv2.imwrite(sp, tensor_to_bgr(_isp_data['x_l'], apply_gamma=False))
                    print(f'  [3/5] Enhanced -> {os.path.basename(sp)}')
                else:
                    print('  [3/5] x_l not captured!')

                # ---- 4. Detection results ----
                if 'x_l' in _isp_data:
                    vis = tensor_to_bgr(_isp_data['x_l'], apply_gamma=False)
                    # Scale GT boxes from original coords to padded image coords
                    vis = draw_gt_bboxes(vis, ds, scale_factor=sf)
                    result = results[0]
                    pred = result.pred_instances
                    if hasattr(pred, 'bboxes') and len(pred.bboxes) > 0:
                        # Pred bboxes are in original coords (val_step uses rescale=True)
                        # Scale them to match the padded image
                        bboxes_orig = pred.bboxes.cpu().numpy()
                        labels = pred.labels.cpu().numpy()
                        scores = pred.scores.cpu().numpy()
                        bboxes_scaled = bboxes_orig.copy()
                        bboxes_scaled[:, 0] *= sf[1]  # x1, x2 * W_scale
                        bboxes_scaled[:, 2] *= sf[1]
                        bboxes_scaled[:, 1] *= sf[0]  # y1, y2 * H_scale
                        bboxes_scaled[:, 3] *= sf[0]
                        vis = draw_bboxes(vis, bboxes_scaled, labels, scores)
                        for i in range(len(bboxes_orig)):
                            cls_name = CLASSES[int(labels[i])]
                            print(f'         Pred: {cls_name} score={scores[i]:.3f}')
                    else:
                        print('         No predictions')
                    sp = os.path.join(OUTPUT_DIR, f'{img_id}_04_detection.png')
                    cv2.imwrite(sp, vis)
                    print(f'  [4/5] Detection -> {os.path.basename(sp)}')
                else:
                    print('  [4/5] Skipped (no enhanced image)')

                # ---- 5. NPZ RAW visualization ----
                npz_path = os.path.join(NPZ_DIR, f'{img_id}.npz')
                sp = os.path.join(OUTPUT_DIR, f'{img_id}_05_npz_raw.png')
                if os.path.exists(npz_path):
                    bayer, _ = load_npz_raw(npz_path)
                    bgr = simple_demosaic(bayer)
                    cv2.imwrite(sp, (bgr * 255).astype(np.uint8))
                    print(f'  [5/5] NPZ raw -> {os.path.basename(sp)}')
                else:
                    print(f'  [5/5] NPZ not found: {npz_path}')

                print()

    finally:
        for h in hook_handles:
            h.remove()

    print(f'===== DONE =====')
    print(f'Outputs in: {OUTPUT_DIR}')
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f'  {f}')


if __name__ == '__main__':
    main()