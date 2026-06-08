# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dark-ISP is a research project implementing the paper "Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection." It builds on OpenMMLab's mmdetection framework (mmcv 2.1.0, mmengine 0.10.5) to perform object detection directly on RAW Bayer images under low-light conditions.

## Environment & Installation

- **Conda env**: Python 3.8, CUDA 10.2
- Three libraries require **editable installs in this exact order** (do not let pip auto-update them):
  1. `mmcv-2.1.0/` — `pip install -v -e .`
  2. `mmengine-0.10.5/` — `pip install -v -e .`
  3. `mmdetection_github/` — `pip install -v -e .` (after installing its `requirements.txt`)
- LibRaw and rawpy must be built from source (`downloads/`). On systems without sudo, set `LD_LIBRARY_PATH` and `PKG_CONFIG_PATH` after installing to a custom prefix.

## Commands

```bash
# Preprocess RAW (.CR2) files to .npz training data
python mmdetection_github/LOD_RAW_preprocess.py

# Train
python mmdetection_github/tools/train.py <config_file>

# Distributed training
bash mmdetection_github/tools/dist_train.sh <config_file> <num_gpus>

# Test/evaluation
python mmdetection_github/tools/test.py <config_file> <checkpoint_path>
```

Config files and checkpoints must have their paths updated to match the local environment before use.

## Architecture

### Custom Detectors (`mmdet/models/detectors/`)

Three custom detector classes handle RAW image inputs by extracting metadata (ISO, white balance, black/white level, CCM, exposure time, brightness ratio) from each sample:

- **`SingleStageRAWDetector`** — Extends single-stage detector. Normalizes RAW input via `(x - black_level) / (white_level - black_level)`, optionally applies virtual noise, then passes metadata to the backbone. Used with RetinaNet (`RAWRetinaNet`), YOLOv3 (`RAWYOLOV3`).
- **`SingleStageRAWPartDetector`** — Variant that receives a ground-truth RGB image (`gts`) alongside RAW for auxiliary loss.
- **`TwoStageRAWDetector`** — Two-stage equivalent, used with Faster R-CNN (`RAWFasterRCNN`) and Sparse R-CNN (`RAWSparseRCNN`).

### Backbone: `Dark_ResNet` (`mmdet/models/backbones/dark_resnet.py`)

The core backbone. The `model` parameter selects the ISP strategy; `ISP_version` controls the preprocessing pipeline within the backbone:

- **`ISP_version='v0'`**: Default ISP (white balance → CCM → gamma). Output detached (no gradient to RAW).
- **`ISP_version='v1'`**: Demosaic only — packs RGBG channels to 3-ch RGB via channel averaging. No color correction.
- **`ISP_version='v2'`** (Dark-ISP): Learnable color transform. A `Matrix_Predictor` predicts per-pixel adjustments (`dmat`) to the CCM matrix, then `color_transform()` applies WB+CCM. A `Color_Level_Process` handles nonlinear tone mapping. A matrix loss (`mat_loss2`) regularizes the learned CCM against a least-squares solution.

**`model` variants** (applied after ISP):
- `"our"`: No additional module (Dark-ISP alone).
- `"Raw-adapter"`: Input-level + model-level adapters with merge blocks at ResNet stages (`RAW_Adapter/`, `LOD_Adapter/`).
- `"Enhancer"`: Feature enhancer module (`FeatEnhancer/`).
- `"LIS"`: Adaptive downsampling convolutions (`LIS/CustomConv.py`) replacing max pooling at each stage.
- `"RAOD"`: Adaptive preprocessing (`RAOD_Adapter/`).
- `"IA_ISP"`: Image-adaptive ISP pipeline (`IA_ISP/`).

### Dark Modules (`mmdet/models/backbones/Dark_modules/`)

- `lut.py` — 3D LUT and 1D LUT generators for learnable color transforms.
- `noise_predictor.py` — `Matrix_Predictor`, `Color_Level_Process`, `Zero_DCE`, `NILUT` — learnable ISP components.
- `utils.py` — RAW normalization, white balance, CCM application, gamma expansion, Taylor expansion operators.

### Data Pipeline

- **Dataset**: `LOD_RAWDAtaset` (`mmdet/datasets/lod_bmvc21.py`) — loads `.npz` files containing packed RAW Bayer (4-channel RGBG), plus metadata arrays: `wb`, `ccm`, `iso`, `exp_time`, `ratio`, `white_level`, `black_level`, `raw_pattern`.
- **Preprocessing**: `LOD_RAW_preprocess.py` — reads `.CR2` files via rawpy, extracts metadata, packs Bayer pattern, resizes, saves as `.npz`.
- **Data preprocessor**: `DetRAWDataPreprocessor` (registered in `data_preprocessor.py`) — handles RAW-specific batching without standard ImageNet normalization (mean=None, std=None).
- **Pipeline**: `LoadRAWImageFromFile` → `LoadAnnotations` → `Resize` → `RandomFlip` → `PackDetInputs` (with RAW-specific meta_keys).

### Config Structure

Configs are in `configs/LOD/`:
- `VOCmetric/` — Uses VOC-style evaluation (11-point mAP). Most experiment configs (denoise, SID, LIS, enhancer, etc.).
- `COCOmetric/` — Uses COCO-style evaluation.

Base configs reference `configs/_base_/schedules/schedule_1x.py` and `configs/_base_/default_runtime.py`.
