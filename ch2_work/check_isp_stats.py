"""诊断：对比 v0（传统 ISP）与 v2（Dark-ISP）送进 ResNet 的输入分布。

加载 v2 基线权重，对同一批 npz 分别算出两条 ISP 分支的输出，统计均值/标准差/分位点
以及饱和比例。用于判断 v2 成绩偏低是否源于输入分布（例如缺少 gamma 编码）。

注意：必须非受限运行（受限进程导入 mmdet 会卡在 yapf）。
"""
import os
import sys

MM = r'E:\!yan\SoftWare\code\Dark-ISP\Dark-ISP\mmdetection_github'
sys.path.insert(0, MM)
os.chdir(MM)

import numpy as np
import torch

import mmdet  # noqa: F401  (注册表)
from mmdet.models.backbones.dark_resnet import Dark_ResNet
from mmdet.models.backbones.Dark_modules.utils import (color_transform,
                                                       default_ISP, normlize)

CKPT = 'checkpoints/2026_06_04_DarkISP_ResNet18/best_pascal_voc_mAP_epoch_13.pth'
NPZ_DIR = os.path.join(os.path.dirname(MM), 'dataset', 'RAWTest', 'RAW_dark_npz')


def load_backbone():
    sd = torch.load(CKPT, map_location='cpu')
    sd = sd.get('state_dict', sd)
    bb = Dark_ResNet(depth=18, model='our', ISP_version='v2', norm_eval=True)
    st = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    missing, unexpected = bb.load_state_dict(st, strict=False)
    print('加载: 缺 %d 键, 多 %d 键' % (len(missing), len(unexpected)))
    bb.eval()          # 注意：Dark_ResNet 重写了 train()，eval() 不返回 self
    return bb.cuda()


def to_input(path):
    d = np.load(path, allow_pickle=True)
    im = torch.from_numpy(d['im'].astype(np.float32))
    bl = torch.from_numpy(d['black_level'].astype(np.float32)).reshape(-1)
    wl = torch.from_numpy(d['white_level'].astype(np.float32)).reshape(-1)
    x = normlize(im.permute(2, 0, 1).unsqueeze(0), torch.tensor(wl).view(1, 4, 1, 1),
                 torch.tensor(bl).view(1, 4, 1, 1))
    ratio = float(np.asarray(d['ratio']).reshape(-1)[0])
    x = (x * ratio).clamp(0, 1)
    meta = {
        'wb': torch.from_numpy(d['wb'].astype(np.float32)).unsqueeze(0).cuda(),
        'ccm': torch.from_numpy(d['ccm'].astype(np.float32)).unsqueeze(0).cuda(),
    }
    return x.cuda(), meta, ratio


def stats(tag, t):
    t = t.detach().float()
    print('%-14s mean=%.4f std=%.4f  p1=%.4f p50=%.4f p99=%.4f  >0.99 占比=%.1f%%'
          % (tag, t.mean(), t.std(), np.percentile(t.cpu().numpy(), 1),
             np.percentile(t.cpu().numpy(), 50), np.percentile(t.cpu().numpy(), 99),
             100.0 * (t > 0.99).float().mean()))


def main():
    bb = load_backbone()
    files = sorted((f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')),
                   key=lambda s: int(s.split('.')[0]))[:4]
    print('%-10s %8s %10s %10s' % ('file', 'ratio', 'v0_mean', 'v2_mean'))
    for f in files:
        x_i, meta, ratio = to_input(os.path.join(NPZ_DIR, f))
        with torch.no_grad():
            v0_out, _ = default_ISP(x_i, meta)
            mat = bb.joint_mat(meta)
            dmat, _, _ = bb.params_predictor(x_i, mat)
            x_g, _ = color_transform(x_i, mat, dmat=dmat)
            x_l = bb.nonlinear(x_g.clip(0, 1))
        print('\n--- %s (ratio=%.1f) ---' % (f, ratio))
        stats('v0 输出(带gamma)', v0_out)
        stats('v2 线性 x_g', x_g)
        stats('v2 输出 x_l', x_l)


if __name__ == '__main__':
    main()
