"""用已有 CSV 重绘 D1/D2 图（不重跑分析）。

修正：原 D1 图把"暗区梯度保留率"当结构损耗，但幅值其实在上升；
改为双面板——(a) 幅值上升 (b) 方向一致性下降，才是真实结论。
"""
import csv
import os
import sys

import numpy as np


def read_csv(path):
    with open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'out')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    rows = read_csv(os.path.join(out, 'tables', 'd1_stage_summary.csv'))
    labels = [r['label'] for r in rows]
    grad = np.array([float(r['dark_grad']) for r in rows])
    cos = np.array([float(r['dark_cos']) for r in rows])
    weak = np.array([float(r['dark_weak_cos']) for r in rows])
    x = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(x, 100.0 * grad / grad[0], 'o-')
    ax.set_ylabel('dark gradient retained (%)')
    ax.set_title('(a) gradient magnitude: RISES (gamma stretches darks)')
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(x, cos, 'o-', label='all dark pixels')
    ax.plot(x, weak, 's--', ms=3, label='weak-gradient pixels')
    ax.set_ylabel('directional agreement with linear reference')
    ax.set_title('(b) structural direction: FALLS')
    ax.set_ylim(0.4, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    fig.suptitle('D1: along the ISP chain, amplitude rises while structure degrades', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'd1_dark_gradient_retention.png'),
                dpi=160, bbox_inches='tight')
    plt.close(fig)

    # D2：双面板（梯度变化 + 方向一致性）
    rows2 = read_csv(os.path.join(out, 'tables', 'd2_leave_one_out.csv'))
    variants = [r['variant'] for r in rows2 if r['variant'] != 'full']
    delta = [float(r['delta_vs_full_pct']) for r in rows2 if r['variant'] != 'full']
    cos2 = [float(r['dark_cos']) for r in rows2 if r['variant'] != 'full']
    full_cos = [float(r['dark_cos']) for r in rows2 if r['variant'] == 'full'][0]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    names = [v.replace('no_', '') for v in variants]
    axes[0].bar(range(len(delta)), delta)
    axes[0].set_xticks(range(len(delta)))
    axes[0].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    axes[0].set_ylabel('dark gradient change (%)')
    axes[0].set_title('(a) magnitude: denoise removes most')
    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].grid(alpha=0.3, axis='y')

    axes[1].bar(range(len(cos2)), cos2)
    axes[1].axhline(full_cos, color='r', ls='--', lw=1.0,
                    label='full chain (%.3f)' % full_cos)
    axes[1].set_xticks(range(len(cos2)))
    axes[1].set_xticklabels(names, rotation=20, ha='right', fontsize=9)
    axes[1].set_ylabel('directional agreement (dark)')
    axes[1].set_title('(b) direction: removing denoise/CCM restores it')
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis='y')
    fig.suptitle('D2: leave-one-out attribution', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'figures', 'd2_leave_one_out.png'),
                dpi=160, bbox_inches='tight')
    plt.close(fig)
    print('重绘完成')


if __name__ == '__main__':
    main()
