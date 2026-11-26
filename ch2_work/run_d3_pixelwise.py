"""D3 补充：量化影响的像素级分析。

对每个暗区像素：
  1) 理论：按该像素的线性亮度 x 计算 gamma 1/2.2 后的 8bit 台阶，
     换算回线性域的"最小可分辨增量" thr(x)；
     若该像素的梯度幅值 < thr(x)，则量化后这个梯度不可分辨。
  2) 实测：比较浮点 gamma 输出(s5)与 8bit 输出(s6)的梯度，统计相对变化。
"""
import argparse
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ref_isp  # noqa: E402
import metrics   # noqa: E402

NPZ_DIR = os.path.join(ROOT, 'dataset', 'RAWTest', 'RAW_dark_npz')
GAMMA = 2.2


def list_npz(limit):
    files = sorted((f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')),
                   key=lambda s: int(s.split('.')[0]))
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in idx]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--limit', type=int, default=60)
    args = ap.parse_args()

    files = list_npz(args.limit)
    print('样本数: %d' % len(files), flush=True)

    n_dark = 0
    n_invisible = 0
    rel_err = []
    changed = 0
    n_eval = 0
    per_image = []

    for k, f in enumerate(files):
        im, meta = ref_isp.load_npz(os.path.join(NPZ_DIR, f))
        steps = ref_isp.run_pipeline(im, meta, demosaic='bilinear')
        ref = metrics.Reference(steps['s1_blacklevel'])
        x = ref.lum
        g1 = ref.mag

        idx = ref.idx.reshape(x.shape)
        dark = np.isin(idx, [0, 1])

        # 理论：局部最小可分辨增量
        xs = np.clip(x, 1e-4, 1.0)
        dydx = (1.0 / GAMMA) * xs ** (1.0 / GAMMA - 1.0)
        thr = (1.0 / 255.0) / dydx
        invisible = dark & (g1 < thr)

        # 实测：s5(浮点) 与 s6(8bit) 的梯度差异
        g5 = metrics.gradient_magnitude(metrics.luminance(steps['s5_gamma']))
        g6 = metrics.gradient_magnitude(metrics.luminance(steps['s6_quant']))
        rel = np.abs(g6 - g5) / (g5 + 1e-8)

        n_dark += int(dark.sum())
        n_invisible += int(invisible.sum())
        rel_dark = rel[dark]
        if rel_dark.size:
            rel_err.append(float(np.median(rel_dark)))
            changed += int((rel_dark > 0.2).sum())
            n_eval += int(rel_dark.size)

        per_image.append([f, int(dark.sum()), int(invisible.sum()),
                          float(100.0 * invisible.sum() / max(dark.sum(), 1)),
                          float(np.median(rel_dark)) if rel_dark.size else float('nan')])

        if (k + 1) % 10 == 0 or k == 0:
            print('  %d/%d' % (k + 1, len(files)), flush=True)

    rows = [
        ['dark_pixels_total', n_dark],
        ['dark_grad_below_local_step', n_invisible],
        ['pct_below_local_step', '%.2f' % (100.0 * n_invisible / max(n_dark, 1))],
        ['median_relative_grad_change_s5_to_s6', '%.4f' % float(np.median(rel_err)) if rel_err else 'nan'],
        ['pct_grad_changed_over_20pct', '%.2f' % (100.0 * changed / max(n_eval, 1))],
        ['samples', len(files)],
    ]

    os.makedirs(os.path.join(args.out, 'tables'), exist_ok=True)
    with open(os.path.join(args.out, 'tables', 'd3_pixelwise_quant.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'value'])
        w.writerows(rows)

    with open(os.path.join(args.out, 'tables', 'd3_pixelwise_per_image.csv'), 'w',
              newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['file', 'dark_pixels', 'below_step', 'pct_below',
                    'median_rel_change'])
        w.writerows(per_image)

    for r in rows:
        print('  %-42s %s' % (r[0], r[1]), flush=True)

    # 图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        pct = np.array([r[3] for r in per_image], dtype=float)
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        ax.hist(pct[np.isfinite(pct)], bins=30)
        ax.set_xlabel('dark gradients below local 8-bit step (%)')
        ax.set_ylabel('images')
        ax.set_title('D3: fraction of unresolvable dark gradients per image')
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, 'figures', 'd3_pixelwise_hist.png'), dpi=160)
        plt.close(fig)
    except Exception as exc:
        print('作图失败: %r' % (exc,), flush=True)


if __name__ == '__main__':
    main()
