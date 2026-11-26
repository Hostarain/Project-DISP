"""测量 L_grad 正则项在三个模型输出上的数值。

目的：判断"哪个注入方式给方向一致性留下了更多可优化的空间"。
L_grad 惩罚 x_g 与 x_l 的梯度方向偏差；数值越小，说明该模型的方向本来就保得很好，
约束加进去几乎无事可做；数值越大，约束才有发挥余地。

测量口径与 `Dark_ResNet.grad_consistency_loss` 完全一致：
  tau  = mean(|∇x_g|) + 2 * std(|∇x_g|)
  mask = |∇x_g| > tau          （只统计强梯度像素）
  loss = mean( (1 - cos(∇x_g, ∇x_l)) over mask )

注意：必须非受限运行（受限进程导入 mmdet 会卡在 yapf）。
"""
import os
import sys

MM = r'E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\mmdetection_github'
sys.path.insert(0, MM)
os.chdir(MM)

import numpy as np
import torch
import torch.nn.functional as F

import mmdet  # noqa: F401
from mmdet.models.backbones.dark_resnet import Dark_ResNet
from mmdet.models.backbones.Dark_modules.utils import color_transform, normlize

NPZ_DIR = os.path.join(os.path.dirname(MM), 'dataset', 'RAWTest', 'RAW_dark_npz')

MODELS = [
    ('Dark-ISP 基线', 'checkpoints/2026_06_04_DarkISP_ResNet18/best_pascal_voc_mAP_epoch_13.pth', dict(use_grad_guide=False)),
    ('GGNC v1', 'checkpoints/R_Net_GGNC_18/best_pascal_voc_mAP_epoch_8.pth', dict(use_grad_guide=True, grad_guide_mode='v1')),
    ('GGNC v2', 'checkpoints/R_Net_GGNC_v2_18/best_pascal_voc_mAP_epoch_15.pth', dict(use_grad_guide=True, grad_guide_mode='v2')),
]

SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.


def build(flags):
    bb = Dark_ResNet(depth=18, model='our', ISP_version='v2', norm_eval=True, **flags)
    return bb


def load(bb, path):
    sd = torch.load(path, map_location='cpu')
    sd = sd.get('state_dict', sd)
    st = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    missing, unexpected = bb.load_state_dict(st, strict=False)
    bb.eval()
    bb = bb.cuda()
    return bb, len(missing), len(unexpected)


def inputs(path):
    d = np.load(path, allow_pickle=True)
    im = torch.from_numpy(d['im'].astype(np.float32))
    bl = torch.from_numpy(d['black_level'].astype(np.float32)).reshape(-1)
    wl = torch.from_numpy(d['white_level'].astype(np.float32)).reshape(-1)
    x = normlize(im.permute(2, 0, 1).unsqueeze(0),
                 torch.tensor(wl).view(1, 4, 1, 1), torch.tensor(bl).view(1, 4, 1, 1))
    ratio = float(np.asarray(d['ratio']).reshape(-1)[0])
    x = (x * ratio).clamp(0, 1).cuda()
    meta = {'wb': torch.from_numpy(d['wb'].astype(np.float32)).unsqueeze(0).cuda(),
            'ccm': torch.from_numpy(d['ccm'].astype(np.float32)).unsqueeze(0).cuda()}
    return x, meta


def lgrad_value(x_g, x_l, k=2.0, eps=1e-8):
    """与仓库 grad_consistency_loss 等价。"""
    sx = SOBEL_X.to(x_g.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sy = SOBEL_Y.to(x_g.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gxg = F.conv2d(x_g, sx, padding=1, groups=3)
    gyg = F.conv2d(x_g, sy, padding=1, groups=3)
    gxl = F.conv2d(x_l, sx, padding=1, groups=3)
    gyl = F.conv2d(x_l, sy, padding=1, groups=3)
    mag = torch.sqrt(gxg ** 2 + gyg ** 2)
    tau = mag.mean() + k * mag.std()
    mask = (mag > tau).float()
    dot = gxg * gxl + gyg * gyl
    cos = dot / (torch.sqrt(gxg ** 2 + gyg ** 2) + eps) / (torch.sqrt(gxl ** 2 + gyl ** 2) + eps)
    loss = ((1.0 - cos) * mask).sum() / (mask.sum() + eps)
    return float(loss), float(mask.mean())


def strata_stats(x_g, x_l, eps=1e-8):
    """按参考梯度强度分层，统计方向偏差与幅值比。

    分层依据 |∇x_g| 的分位数：
      弱 (0-50%) / 中 (50-90%) / 强 (90-99%) / 极强 (>99%)
    现有 L_grad 的 mask 只覆盖"极强"附近（mean+2std ≈ 前 3-4%），
    因此弱/中层的退化它完全看不到。
    """
    sx = SOBEL_X.to(x_g.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sy = SOBEL_Y.to(x_g.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gxg = F.conv2d(x_g, sx, padding=1, groups=3)
    gyg = F.conv2d(x_g, sy, padding=1, groups=3)
    gxl = F.conv2d(x_l, sx, padding=1, groups=3)
    gyl = F.conv2d(x_l, sy, padding=1, groups=3)

    mag_g = torch.sqrt(gxg ** 2 + gyg ** 2).mean(dim=1)   # 按通道平均，便于分层
    mag_l = torch.sqrt(gxl ** 2 + gyl ** 2).mean(dim=1)
    dot = (gxg * gxl + gyg * gyl).mean(dim=1)
    denom = (torch.sqrt(gxg ** 2 + gyg ** 2) * torch.sqrt(gxl ** 2 + gyl ** 2)).mean(dim=1) + eps
    cos = (dot / denom).flatten()
    ratio = (mag_l / (mag_g + eps)).flatten()
    ref = mag_g.flatten()

    qs = torch.quantile(ref, torch.tensor([0.5, 0.9, 0.99]).to(ref.device))
    out = {}
    for name, lo, hi in [('弱(0-50%)', -1.0, qs[0]), ('中(50-90%)', qs[0], qs[1]),
                         ('强(90-99%)', qs[1], qs[2]), ('极强(>99%)', qs[2], 1e9)]:
        m = (ref >= lo) & (ref < hi)
        if m.sum() > 100:
            out[name] = (float((1.0 - cos[m]).mean()), float(ratio[m].mean()))
    return out


def lgrad_full(x_g, x_l, eps=1e-8, min_mag=1e-4):
    """新口径：全图有效像素的方向一致性（仅以 min_mag 作数值下限）。"""
    sx = SOBEL_X.to(x_g.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sy = SOBEL_Y.to(x_g.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    gxg = F.conv2d(x_g, sx, padding=1, groups=3)
    gyg = F.conv2d(x_g, sy, padding=1, groups=3)
    gxl = F.conv2d(x_l, sx, padding=1, groups=3)
    gyl = F.conv2d(x_l, sy, padding=1, groups=3)
    mag = torch.sqrt(gxg ** 2 + gyg ** 2)
    mask = (mag > min_mag).float()
    dot = gxg * gxl + gyg * gyl
    cos = dot / (mag + eps) / (torch.sqrt(gxl ** 2 + gyl ** 2) + eps)
    return float(((1.0 - cos) * mask).sum() / (mask.sum() + eps)), float(mask.mean())


def main():
    files = sorted((f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')),
                   key=lambda s: int(s.split('.')[0]))[:8]
    print('== 新旧两种掩码口径下的 L_grad 值（同一批模型、同一批图像）==')
    print('%-16s %12s %12s %10s %10s' % ('模型', '旧(edge)', '新(full)', '倍数', '新口径覆盖率'))
    rows = {}
    for name, ckpt, flags in MODELS:
        bb = build(flags)
        bb, miss, unexp = load(bb, ckpt)
        old_v, new_v, cov, strat = [], [], [], {}
        with torch.no_grad():
            for f in files:
                x_i, meta = inputs(os.path.join(NPZ_DIR, f))
                mat = bb.joint_mat(meta)
                dmat, _, _ = bb.params_predictor(x_i, mat)
                x_g, _ = color_transform(x_i, mat, dmat=dmat)
                x_g = x_g.clip(0, 1)
                x_l = bb.nonlinear(x_g)
                old_v.append(lgrad_value(x_g, x_l)[0])
                v2, c2 = lgrad_full(x_g, x_l)
                new_v.append(v2)
                cov.append(c2)
                for k, val in strata_stats(x_g, x_l).items():
                    strat.setdefault(k, []).append(val)
        o, n = np.mean(old_v), np.mean(new_v)
        print('%-16s %12.5f %12.5f %9.1fx %9.1f%%' % (name, o, n, n / max(o, 1e-9), 100 * np.mean(cov)))
        rows[name] = {k: (np.mean([v[0] for v in vs]), np.mean([v[1] for v in vs]))
                      for k, vs in strat.items()}

    print('\n== 分层：(1-cos) 方向偏差 / |∇x_l|/|∇x_g| 幅值比 ==')
    heads = list(next(iter(rows.values())).keys())
    print('%-16s' % '模型' + ''.join('%20s' % h for h in heads))
    for name, d in rows.items():
        cells = ''.join('%20s' % ('%.4f / %.2fx' % d[h]) for h in heads)
        print('%-16s' % name + cells)



if __name__ == '__main__':
    main()
