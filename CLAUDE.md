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

---

## Paper Core Concepts: Dark-ISP

**Source**: "Dark-ISP: Enhancing RAW Image Processing for Low-Light Object Detection", Fudan University, arXiv:2509.09183v1 (Sep 2025)

### Problem & Motivation

- Low-light object detection: conventional RGB pipeline loses information during ISP (bit-depth reduction, noise amplification, dynamic range compression).
- RAW (Bayer RGGB) images preserve richer photon-level information, camera metadata (ISO, WB, CCM, exposure time), and superior dynamic range.
- Existing RAW-based methods either: (a) quantize to 8-bit RAW-RGB (lossy), (b) use complex ISP parameter search (expensive), or (c) require auxiliary normal-light RGB (impractical).

### Dark-ISP Pipeline (3 Components)

#### 1. Linear Component — Dynamic Linear Mapping (`ISP_version='v2'`, in `Matrix_Predictor`)

Deconstructs traditional ISP linear operations into a single learnable matrix `P ∈ ℝ³ˣ⁴`:

```
I' = P · I    where P = C · B · W
```

- **W** (White Balance): diagonal matrix ∈ ℝ⁴ˣ⁴
- **B** (Binning): sparse matrix ∈ ℝ³ˣ⁴, merges 4 Bayer channels → 3 RGB channels (green channels averaged)
- **C** (Color Correction Matrix/CCM): ∈ ℝ³ˣ³

Enhanced with **Local-Global Attention**:
- Local branch: pixel-level features `F_l` → `P_l ∈ ℝ³ˣ⁴ˣᴴˣᵂ` (per-pixel adjustments)
- Global branch: image-level features `F_g` (after 16× pooling) → `P_g ∈ ℝ³ˣ⁴` (global adjustments)
- Final: `P' = P_l + P_g + P` (broadcast + skip connection)

Implemented in `noise_predictor.py` → `Matrix_Predictor.forward()`:
- Local stream: 5 conv blocks (lc_block1-5) with InstanceNorm + GELU
- Global stream: 5 GaussianConv2d blocks (gl_block1-5) with AvgPool2d(16)
- Cross-attention (`FA`): local features attend to camera parameter matrix
- Global attention: camera params as Q, pooled features as K/V → 12-dim global adjustment

#### 2. Nonlinear Component — Polynomial Tone Mapping (`Color_Level_Process`)

Instead of a black-box CNN, uses **8th-order non-convex polynomial bases** with strong physical interpretability:

```
F(x) = Σₖ₌₀⁸ Cₖ(i,j) · fₖ(x)
```

where `f₀(x)=0`, and each `fₖ(x)` (k≥1) is a polynomial of order k passing through (0,0) and (1,1).

Key property: these bases are **non-convex** — derivative is large near x→0 (stretch dark regions), small near x→1 (compress bright regions). This physically models the tone-mapping needed in low-light.

Implemented in `utils.py` → `Tayler_res()`:
- 8 basis functions applied per RGB channel independently
- Network (`Color_Level_Process`) predicts 24 coefficient maps (3 channels × 8 bases) via softmax
- Skip connection preserves gradient flow: `output = input + Tayler_res(input, coefficients)`
- Parameters: only 0.136 MB

#### 3. Self-Boost Regularization (`mat_loss2`)

Promotes synergy between linear and nonlinear components:

```
L_sb = Σ ||1 - cos(p'_i, p̃_i)||
```

where:
- `p'_i` = row vectors of learned linear matrix P'
- `p̃_i` = row vectors of approximate linear matrix P̃ = U · Iᵀ · (I · Iᵀ)⁻¹
- U = output of nonlinear component (used as pseudo-target)

The loss aligns the **direction** (not magnitude) of corresponding row vectors, avoiding premature convergence. Multiplied by λ=0.01 (in code: 0.05).

### Experimental Results

| Dataset | Backbone | Dark-ISP mAP | Best Competitor |
|---------|----------|-------------|-----------------|
| LOD (Canon 5D IV) | ResNet-18 | 64.9 | 63.6 (default ISP) |
| LOD | ResNet-50 | **70.4** | 67.9 (LIS) |
| NOD Sony | R-50 | **31.5** | 30.3 (FeatEnHancer) |
| NOD Nikon | R-50 | **29.9** | 28.8 (FeatEnHancer) |
| SynCOCO | R-50 | **23.1** | 22.4 (FeatEnHancer) |

Ablation on LOD R-50: Linear only 66.6 → +Nonlinear 68.7 → +Self-Boost **70.4**

### Key Insight: Structural Information Preservation

The nonlinear polynomial bases are designed to **stretch dark regions while compressing bright regions**. This means:
- Weak gradients and edges in dark areas (e.g., 0.01→0.02) get amplified (e.g., 0.02→0.06 after nonlinear mapping)
- Structural information that would normally be lost in ISP denoising survives
- The output RGB retains richer edge/contour information than conventional ISP

---

## Research Direction: Structural Response Operators for Dark-ISP Output

> *Goal: exploit the structural information preserved by Dark-ISP by designing structural response operators between Dark-ISP output and ResNet stem.*

### Rationale
- Dark-ISP's non-convex polynomial bases stretch dark regions, amplifying weak edges BUT also amplifying noise
- Standard ResNet stem (7×7 conv, stride=2, ImageNet-pretrained) expects natural-light gradient distributions
- A structural response operator should learn to distinguish "real edges" from "amplified noise" in Dark-ISP output

### Baseline (user's local config)
- Config: `R_Net_DarkISP_18.py` (model="our", ISP_version='v2', depth=18)
- Result: **61.9% mAP** on LOD (VOC 11-point)
- Training: batch_size=2, RepeatDataset×3, 15 epochs, SGD lr=0.001
- GPU: RTX 2060 6GB, Windows, conda env `dark-isp`

### Experiment v1: `StructureAwareBlock` (model="Structure")

**Implemented in**: `Dark_modules/structure_aware.py` | Config: `R_Net_Structure_18.py`

**Design**: Fixed Sobel x/y kernels (groups=3) → 6ch gradient → small attention head (6→16→3) → spatial weight α → `output = x + x * α * 0.1`

**Rationale**: Give ResNet an explicit gradient-based hint about edge locations.

**Result**: Best **61.23%** mAP (epoch 13), essentially identical to baseline (61.9%). **No measurable improvement.**

**Root cause analysis**:
1. Sobel assumes clean step edges — Dark-ISP amplifies noise alongside edges, Sobel cannot distinguish them
2. Fixed 2-kernel response (|Gx|, |Gy|) loses directional structure information
3. scale=0.1 may be too conservative — 5% enhancement signal drowned by BN+ReLU
4. Attention head input (6ch) is too impoverished to learn fine spatial decisions

### Experiment v2: `LearnedStructureAttention` (model="LSA")

**Implemented in**: `Dark_modules/learned_structure_attention.py` | Config: `R_Net_LSA_18.py`

**Design**: 12 learnable 3×3 differential kernels (groups=3, 4 kernels per RGB channel) → 12ch structural response → same attention head (12→16→3) → spatial weight α → `output = x + x * α * gamma`

**Key differences from v1**:
| | v1 (Structure) | v2 (LSA) |
|------|:---:|:---:|
| Structural response | 2 fixed Sobel kernels | 12 learnable differential kernels |
| Kernel init | Sobel x/y (frozen) | 4 types per channel: Sobel x/y, center diff x/y |
| Enhancement strength | scale=0.1 (fixed) | gamma (learnable, init 0.1, unconstrained) |
| Input to attention head | 6ch | 12ch |
| Parameters | 1,334 | 2,307 |

**Design philosophy**:
- NOT "replacing Sobel" — exploring whether learnable differential kernels can better model Dark-ISP's specific gradient distribution
- Kernels initialised with differential priors but free to evolve during training
- gamma unconstrained (may become negative → general residual modulator, not pure enhancement)
- Attention head kept identical to v1 to isolate the effect of the structural response operator
- No identity kernels, zero-mean regularization, or complex attention in v1 — minimal viable version

**Status**: NOT YET TRAINED. Awaiting training run.

### Implementation Pattern (both v1 and v2)

All modules follow the same non-breaking pattern:
1. New `.py` file in `Dark_modules/`
2. Import added to `dark_resnet.py`
3. `elif self.model == "XXX"` branch in both `__init__` and `forward`
4. New config copies `R_Net_DarkISP_18.py`, only changes `model` field
5. **All existing configs and code paths untouched**

### Further Candidate Directions (not yet implemented)
- **Octave Convolution** in early ResNet blocks (explicit high/low frequency separation)
- **Deformable Convolution** in layer1/2 (adapt sampling grid to fragmented low-light edges)
- **Feature-level gradient attention** via Bottleneck plugins (only applicable to R-50, not R-18)
- **Zero-mean regularization** on learned kernels if they degenerate toward non-differential patterns

### Git Workflow
- Remote: `git@github.com:Hostarain/Project-DISP.git` (SSH via port 443, `ssh.github.com`)
- Branch: `main`
- SSH config at `~/.ssh/config`
- Conda env: `dark-isp` (CUDA_LAUNCH_BLOCKING=1 auto-set via `etc/conda/activate.d/env_vars.bat`)
