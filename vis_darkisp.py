"""Render RAW .npz through Dark-ISP + GGNC pipeline."""
import numpy as np, os, sys, torch
from PIL import Image
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(SCRIPT_DIR, "mmdetection_github"))
from mmdet.models.backbones.Dark_modules.utils import color_transform
from mmdet.models.backbones.Dark_modules.noise_predictor import Matrix_Predictor, Color_Level_Process
from mmdet.models.backbones.dark_resnet import Dark_ResNet

SAMPLES = {"bicycle":100,"car":10,"motorbike":10,"chair":1000,"diningtable":1000,"bottle":1000,"tvmonitor":1134,"bus":140}
NPZ_DIR = os.path.join(SCRIPT_DIR, "dataset", "RAWTest", "RAW_dark_npz")
OUT_DIR = os.path.join(SCRIPT_DIR, "vis_darkisp_output")
CKPT = os.path.join(SCRIPT_DIR, "mmdetection_github", "checkpoints", "R_Net_GGNC_18", "best_pascal_voc_mAP_epoch_8.pth")
os.makedirs(OUT_DIR, exist_ok=True)

# Build Dark_ResNet with GGNC config and load weights
model = Dark_ResNet(depth=18, model="our", ISP_version="v2", use_grad_guide=True, norm_eval=True)
ckpt = torch.load(CKPT, map_location="cpu")
state = ckpt.get("state_dict", ckpt)
state = {k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")}
model.load_state_dict(state, strict=False)
model.eval()

for cat, img_id in SAMPLES.items():
    data = np.load(os.path.join(NPZ_DIR, f"{img_id}.npz"), allow_pickle=True)
    raw = torch.from_numpy(data["im"].astype(np.float32))
    bl  = torch.from_numpy(data["black_level"].astype(np.float32)).reshape(-1)
    wl  = torch.from_numpy(data["white_level"].astype(np.float32)).reshape(-1)
    
    # Normalize and prepare input [1, 4, H, W]
    x = ((raw - bl) / (wl - bl)).permute(2,0,1).unsqueeze(0).clamp(0,1)
    
    # Build metainfo dict
    metainfo = {
        "wb": torch.from_numpy(data["wb"]).float().unsqueeze(0),
        "ccm": torch.from_numpy(data["ccm"]).float().unsqueeze(0),
    }
    
    with torch.no_grad():
        # Run Dark-ISP linear + nonlinear (GGNC) only
        mat = model.joint_mat(metainfo)
        dmat, _, _ = model.params_predictor(x, mat)
        x_g, _ = color_transform(x, mat, dmat=dmat)
        x_g = x_g.clip(0, 1)
        x_l = model.nonlinear(x_g)  # GGNC output, [1, 3, H, W]
    
    rgb_np = x_l.squeeze(0).permute(1,2,0).clamp(0,1).numpy()
    Image.fromarray((rgb_np*255).astype(np.uint8)).save(os.path.join(OUT_DIR, f"{cat}_{img_id}.png"))
    print(f"[OK] {cat:12s} (id={img_id})")
print(f"Done. Saved to {OUT_DIR}")
