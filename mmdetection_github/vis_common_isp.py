"""Render RAW .npz through traditional ISP pipeline."""
import numpy as np, os, sys, torch
from PIL import Image
SCRIPT_DIR = r'E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP'
sys.path.insert(0, os.path.join(SCRIPT_DIR, 'mmdetection_github'))
from mmdet.models.backbones.Dark_modules.utils import default_ISP

SAMPLES = {'bicycle':100,'car':10,'motorbike':10,'chair':1000,'diningtable':1000,'bottle':1000,'tvmonitor':1134,'bus':140}
NPZ_DIR = os.path.join(SCRIPT_DIR, 'dataset', 'RAWTest', 'RAW_dark_npz')
OUT_DIR = r'D:\!bishe\开题相关\CommonISP'
os.makedirs(OUT_DIR, exist_ok=True)

for cat, img_id in SAMPLES.items():
    data = np.load(os.path.join(NPZ_DIR, f"{img_id}.npz"), allow_pickle=True)
    raw = torch.from_numpy(data["im"].astype(np.float32))
    bl  = torch.from_numpy(data["black_level"].astype(np.float32)).reshape(-1)
    wl  = torch.from_numpy(data["white_level"].astype(np.float32)).reshape(-1)
    raw = ((raw - bl) / (wl - bl)).permute(2,0,1).unsqueeze(0).clamp(0,1)
    meta = {"wb": torch.from_numpy(data["wb"]).float().unsqueeze(0),
            "ccm": torch.from_numpy(data["ccm"]).float().unsqueeze(0)}
    with torch.no_grad():
        rgb, _ = default_ISP(raw, meta)
    rgb_np = rgb.squeeze(0).permute(1,2,0).clamp(0,1).numpy()
    Image.fromarray((rgb_np*255).astype(np.uint8)).save(os.path.join(OUT_DIR, f"{cat}_{img_id}.png"))
    print(f"[OK] {cat:12s} (id={img_id})")
print(f"Done. Saved to {OUT_DIR}")
