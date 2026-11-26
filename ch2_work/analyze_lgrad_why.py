"""反推：为什么 L_grad 的效果一般？

对 GGNC v1（方向退化最严重的模型）逐像素分析 (1-cos)，回答三个问题：

1. 弱梯度区的 0.0405 是**整体平移**还是**长尾**？
   —— 若中位数接近 0、只有 p99 很大，说明是数值噪声尾巴，约束它在追噪声。
2. 弱梯度区占多少**梯度能量**？
   —— 若能量占比很低，则该区域的相对误差再大也不影响检测特征。
3. 方向偏差与梯度幅值是否**负相关**？
   —— 若偏差集中在极小梯度处，说明"偏差 ∝ 1/幅值"，本质是数值问题。

必须非受限运行（mmdet 导入会卡在 yapf）。
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
    ('基线', 'checkpoints/2026_06_04_DarkISP_ResNet18/best_pascal_voc_mAP_epoch_13.pth',
     dict(use_grad_guide=False)),
    ('GGNC v1', 'checkpoints/R_Net_GGNC_18/best_pascal_voc_mAP_epoch_8.pth',
     dict(use_grad_guide=True, grad_guide_mode='v1')),
]
SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]) / 4.
SOBEL_Y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]) / 4.


def load(flags, ckpt):
    bb = Dark_ResNet(depth=18, model='our', ISP_version='v2', norm_eval=True, **flags)
    sd = torch.load(ckpt, map_location='cpu')
    sd = sd.get('state_dict', sd)
    st = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    bb.load_state_dict(st, strict=False)
    bb.eval()
    return bb.cuda()


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


def pixel_stats(bb, x_i, meta):
    sx = SOBEL_X.to(x_i.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    sy = SOBEL_Y.to(x_i.device).view(1, 1, 3, 3).repeat(3, 1, 1, 1)
    with torch.no_grad():
        mat = bb.joint_mat(meta)
        dmat, _, _ = bb.params_predictor(x_i, mat)
        x_g, _ = color_transform(x_i, mat, dmat=dmat)
        x_g = x_g.clip(0, 1)
        x_l = bb.nonlinear(x_g)
        gxg = F.conv2d(x_g, sx, padding=1, groups=3)
        gyg = F.conv2d(x_g, sy, padding=1, groups=3)
        gxl = F.conv2d(x_l, sx, padding=1, groups=3)
        gyl = F.conv2d(x_l, sy, padding=1, groups=3)
    mg = torch.sqrt(gxg ** 2 + gyg ** 2 + 1e-12)
    ml = torch.sqrt(gxl ** 2 + gyl ** 2 + 1e-12)
    cos = (gxg * gxl + gyg * gyl) / (mg * ml)
    cos = cos.clamp(-1, 1).mean(dim=1)          # 通道平均
    mg = mg.mean(dim=1)
    return (1.0 - cos).flatten().cpu().numpy(), mg.flatten().cpu().numpy()


def main():
    files = sorted((f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')),
                   key=lambda s: int(s.split('.')[0]))[:8]
    for name, ckpt, flags in MODELS:
        bb = load(flags, ckpt)
        err_all, mag_all = [], []
        for f in files:
            x_i, meta = inputs(os.path.join(NPZ_DIR, f))
            e, m = pixel_stats(bb, x_i, meta)
            err_all.append(e)
            mag_all.append(m)
        err = np.concatenate(err_all)
        mag = np.concatenate(mag_all)

        q = np.quantile(mag, [0.5, 0.9, 0.99])
        strata = [('弱 0-50%', 0, q[0]), ('中 50-90%', q[0], q[1]),
                  ('强 90-99%', q[1], q[2]), ('极强 >99%', q[2], np.inf)]
        print('\n===== %s =====' % name)
        print('%-11s %8s %8s %8s %8s %8s %8s %10s %10s' %
              ('分层', '均值', '中位', 'p90', 'p99', 'max', '>60°占比', '能量占比', '误差×能量'))
        tot_e = mag.sum()
        for lab, lo, hi in strata:
            sel = (mag >= lo) & (mag < hi)
            if sel.sum() < 100:
                continue
            e = err[sel]
            share = mag[sel].sum() / tot_e
            print('%-11s %8.4f %8.4f %8.4f %8.4f %8.2f %8.2f%% %9.1f%% %10.4f'
                  % (lab, e.mean(), np.median(e), np.percentile(e, 90),
                     np.percentile(e, 99), e.max(), 100 * (e > 0.5).mean(),
                     100 * share, float(np.sum(err[sel] * mag[sel]) / tot_e)))
        # 误差与幅值的相关性
        sub = np.random.choice(len(err), size=min(200000, len(err)), replace=False)
        corr = float(np.corrcoef(np.log(mag[sub] + 1e-8), err[sub])[0, 1])
        print('  误差与 log(幅值) 相关系数: %.3f （负相关=误差集中在极小梯度）' % corr)
        # 全图口径的损失值（与 full mask 一致）
        print('  全图 mean(1-cos) = %.4f' % err.mean())


if __name__ == '__main__':
    main()
