"""第 2 章实验驱动：D1 逐级累积 / D2 留一法归因 / D3 量化步长 / D4 可逆性探针。

只依赖 numpy / cv2 / matplotlib，不导入 mmdet 或 torch。

用法（在仓库根目录）：
    python ch2_work/run_ch2.py --out ch2_work/out --limit 200
"""
import argparse
import csv
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ref_isp  # noqa: E402
import metrics  # noqa: E402

ROOT = os.path.dirname(HERE)
NPZ_DIR = os.path.join(ROOT, 'dataset', 'RAWTest', 'RAW_dark_npz')
STAGES = ['s1_blacklevel', 's2_wb', 's3_ccm', 's4_denoise', 's5_gamma', 's6_quant']
STAGE_LABEL = {
    's1_blacklevel': 'black level',
    's2_wb': '+ white balance',
    's3_ccm': '+ CCM',
    's4_denoise': '+ denoise',
    's5_gamma': '+ gamma',
    's6_quant': '+ 8-bit quant',
}
ALL_OPS = ('wb', 'ccm', 'denoise', 'gamma', 'quant')
LOO_VARIANTS = ['no_%s' % o for o in ALL_OPS]


# ---------------------------------------------------------------- 工具

def list_npz(limit=None, every=None):
    files = sorted((f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')),
                   key=lambda s: int(s.split('.')[0]))
    if every:
        files = files[::every]
    if limit and len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in idx]
    return files


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def dark_weight(count, nb, n_dark=2):
    w = np.zeros(nb)
    w[:n_dark] = count[:n_dark]
    s = w.sum()
    return w / s if s > 0 else w


def weighted(x, w):
    m = np.isfinite(x) & (w > 0)
    if not m.any():
        return float('nan')
    return float((x[m] * w[m]).sum() / w[m].sum())


def new_acc(nb):
    return {'grad': np.zeros(nb), 'cos': np.zeros(nb), 'weak': np.zeros(nb),
            'edge': np.zeros(nb), 'wt': np.zeros(nb), 'psd': 0.0,
            'flat': 0.0, 'crg': 0.0, 'cbg': 0.0, 'n': 0}


def accumulate(acc, m, count):
    acc['grad'] += np.nan_to_num(m['grad_mag']) * count
    acc['cos'] += np.nan_to_num(m['dir_consistency']) * count
    acc['weak'] += np.nan_to_num(m['weak_region_cos']) * count
    acc['edge'] += np.nan_to_num(m['edge_density']) * count
    acc['wt'] += count
    acc['psd'] += m['psd_slope'] - m['ref_psd_slope']
    acc['flat'] += m['flat_std']
    acc['crg'] += m['chroma_rg']
    acc['cbg'] += m['chroma_bg']
    acc['n'] += 1


def quant_curve(gamma=2.2, n=60):
    """线性域输入 x -> 8bit 台阶对应的最小可分辨增量。"""
    xs = np.logspace(-3.2, 0.0, n)
    ys = xs ** (1.0 / gamma)
    dydx = (1.0 / gamma) * xs ** (1.0 / gamma - 1.0)
    dx_min = (1.0 / 255.0) / np.maximum(dydx, 1e-12)
    return xs, ys, dx_min


def hist_median(edges, counts):
    total = counts.sum()
    if total <= 0:
        return float('nan')
    cum = np.cumsum(counts)
    i = int(np.searchsorted(cum, total / 2.0))
    i = min(i, len(counts) - 1)
    return float(np.sqrt(edges[i] * edges[i + 1]))


def hist_frac_below(edges, counts, thresholds):
    total = counts.sum()
    if total <= 0:
        return np.zeros(len(thresholds))
    cum = np.cumsum(counts)
    out = []
    for t in thresholds:
        i = int(np.searchsorted(edges, t)) - 1
        i = int(np.clip(i, -1, len(counts) - 1))
        out.append(0.0 if i < 0 else float(cum[i] / total))
    return np.array(out)


# ---------------------------------------------------------------- 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(HERE, 'out'))
    ap.add_argument('--limit', type=int, default=200)
    ap.add_argument('--every', type=int, default=None)
    args = ap.parse_args()

    out = args.out
    for sub in ('tables', 'figures'):
        os.makedirs(os.path.join(out, sub), exist_ok=True)

    files = list_npz(args.limit, args.every)
    nb = len(metrics.DEFAULT_EDGES) - 1
    print('样本数: %d, 亮度桶数: %d' % (len(files), nb), flush=True)

    acc_stage = {s: new_acc(nb) for s in STAGES}
    acc_loo = {v: new_acc(nb) for v in LOO_VARIANTS}
    acc_loo['full'] = new_acc(nb)
    acc_d4 = {v: new_acc(nb) for v in ['linear_s1', 'gamma_s5', 'quant_s6', 'inverse_s6']}

    hist_edges = np.logspace(-5, 0, 61)
    hist_ref = np.zeros(len(hist_edges) - 1)
    hist_quant = np.zeros(len(hist_edges) - 1)

    t0 = time.time()
    for k, fname in enumerate(files):
        im, meta = ref_isp.load_npz(os.path.join(NPZ_DIR, fname))
        steps = ref_isp.run_pipeline(im, meta, demosaic='bilinear')
        ref = metrics.Reference(steps['s1_blacklevel'])
        count = ref.count.astype(np.float64)

        # ---- D1 逐级
        for s in STAGES:
            accumulate(acc_stage[s], ref.metrics(steps[s]), count)
        m_full = ref.metrics(steps['s6_quant'])
        accumulate(acc_loo['full'], m_full, count)   # 复用 D1 的完整链路结果

        # ---- D2 留一法（完整链路由上面复用）
        for variant in LOO_VARIANTS:
            op = variant[3:]
            enable = tuple(o for o in ALL_OPS if o != op)
            st = ref_isp.run_pipeline(im, meta, demosaic='bilinear', enable=enable)
            last = [s for s in STAGES if s in st][-1]
            accumulate(acc_loo[variant], ref.metrics(st[last]), count)

        # ---- D4 可逆性
        s1, s5, s6 = steps['s1_blacklevel'], steps['s5_gamma'], steps['s6_quant']
        inv = np.clip(np.clip(s6, 0.0, 1.0) ** 2.2, 0.0, 1.0)
        for name, img in [('linear_s1', s1), ('gamma_s5', s5),
                          ('quant_s6', s6), ('inverse_s6', inv)]:
            accumulate(acc_d4[name], ref.metrics(img), count)

        # ---- 暗区梯度直方图（抽样，控制开销）
        dark = np.isin(ref.idx.reshape(ref.lum.shape), [0, 1])
        hist_ref += np.histogram(ref.mag[dark][::16], bins=hist_edges)[0]
        g_q = metrics.gradient_magnitude(metrics.luminance(s6))
        hist_quant += np.histogram(g_q[dark][::16], bins=hist_edges)[0]

        if (k + 1) % 10 == 0 or k == 0:
            el = time.time() - t0
            eta = el / (k + 1) * (len(files) - k - 1)
            print('  %d/%d  已用 %.0fs  预计剩余 %.0fs' % (k + 1, len(files), el, eta),
                  flush=True)

    # ------------------------------------------------------------ D1 导出
    rows = []
    for s in STAGES:
        a = acc_stage[s]
        w = np.maximum(a['wt'], 1)
        for b in range(nb):
            rows.append([s, STAGE_LABEL[s], b,
                         '%.6f' % (a['grad'][b] / w[b]),
                         '%.6f' % (a['cos'][b] / w[b]),
                         '%.6f' % (a['weak'][b] / w[b]),
                         '%.6f' % (a['edge'][b] / w[b]),
                         int(a['wt'][b])])
    write_csv(os.path.join(out, 'tables', 'd1_stage_by_bucket.csv'),
              ['stage', 'label', 'bucket', 'grad_mag', 'dir_consistency',
               'weak_region_cos', 'edge_density', 'pixels'], rows)

    summary, dark_idx = [], {}
    for s in STAGES:
        a = acc_stage[s]
        w = np.maximum(a['wt'], 1)
        n = max(a['n'], 1)
        dw = dark_weight(a['wt'], nb)
        grad_dark = weighted(a['grad'] / w, dw)
        dark_idx[s] = grad_dark
        summary.append([s, STAGE_LABEL[s], '%.6f' % grad_dark,
                        '%.4f' % weighted(a['cos'] / w, dw),
                        '%.6f' % weighted(a['weak'] / w, dw),
                        '%.4f' % (a['psd'] / n), '%.6f' % (a['flat'] / n),
                        '%.4f' % (a['crg'] / n), '%.4f' % (a['cbg'] / n)])
    write_csv(os.path.join(out, 'tables', 'd1_stage_summary.csv'),
              ['stage', 'label', 'dark_grad', 'dark_cos', 'dark_weak_cos',
               'psd_slope_delta', 'flat_std', 'chroma_rg', 'chroma_bg'], summary)

    # ------------------------------------------------------------ D2 导出
    base = acc_loo['full']
    b_w = np.maximum(base['wt'], 1)
    b_dark = dark_weight(b_w, nb)
    base_grad = weighted(base['grad'] / b_w, b_dark)
    rows = []
    for v in ['full'] + LOO_VARIANTS:
        a = acc_loo[v]
        w = np.maximum(a['wt'], 1)
        n = max(a['n'], 1)
        dg = weighted(a['grad'] / w, b_dark)
        rows.append([v, '%.6f' % dg,
                     '%.2f' % (100.0 * (dg - base_grad) / max(base_grad, 1e-9)),
                     '%.4f' % weighted(a['cos'] / w, b_dark),
                     '%.4f' % (a['psd'] / n), '%.6f' % (a['flat'] / n),
                     '%.4f' % (a['crg'] / n), '%.4f' % (a['cbg'] / n)])
    write_csv(os.path.join(out, 'tables', 'd2_leave_one_out.csv'),
              ['variant', 'dark_grad', 'delta_vs_full_pct', 'dark_cos',
               'psd_slope_delta', 'flat_std', 'chroma_rg', 'chroma_bg'], rows)

    # ------------------------------------------------------------ D4 导出
    rows = []
    for v in ['linear_s1', 'gamma_s5', 'quant_s6', 'inverse_s6']:
        a = acc_d4[v]
        w = np.maximum(a['wt'], 1)
        n = max(a['n'], 1)
        rows.append([v, '%.6f' % weighted(a['grad'] / w, b_dark),
                     '%.4f' % weighted(a['cos'] / w, b_dark),
                     '%.4f' % (a['psd'] / n), '%.6f' % (a['flat'] / n)])
    write_csv(os.path.join(out, 'tables', 'd4_invertibility.csv'),
              ['variant', 'dark_grad', 'dark_cos', 'psd_slope_delta', 'flat_std'], rows)

    # ------------------------------------------------------------ D3 导出
    xs, ys, dx_min = quant_curve()
    frac = hist_frac_below(hist_edges, hist_ref, dx_min)
    write_csv(os.path.join(out, 'tables', 'd3_quant_curve.csv'),
              ['linear_value', 'gamma_value', 'min_resolvable_step', 'dark_frac_below'],
              [['%.6g' % x, '%.6g' % y, '%.6g' % d, '%.4f' % f]
               for x, y, d, f in zip(xs, ys, dx_min, frac)])

    write_csv(os.path.join(out, 'tables', 'd3_dark_grad_hist.csv'),
              ['bin_left', 'ref_count', 'quant_count'],
              [['%.6g' % hist_edges[i], int(hist_ref[i]), int(hist_quant[i])]
               for i in range(len(hist_edges) - 1)])

    med_ref = hist_median(hist_edges, hist_ref)
    med_q = hist_median(hist_edges, hist_quant)

    # ------------------------------------------------------------ 作图
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        centers = np.sqrt(hist_edges[:-1] * hist_edges[1:])
        labels = [STAGE_LABEL[s] for s in STAGES]

        fig, ax = plt.subplots(figsize=(7.0, 4.2))
        vals = [100.0 * dark_idx[s] / max(dark_idx[STAGES[0]], 1e-12) for s in STAGES]
        ax.plot(range(len(vals)), vals, 'o-')
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
        ax.set_ylabel('dark gradient retained (%)')
        ax.set_title('D1: structural loss accumulates along the ISP chain')
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', 'd1_dark_gradient_retention.png'), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        edges_mid = np.asarray(metrics.DEFAULT_EDGES)[:-1]
        for s in STAGES:
            a = acc_stage[s]
            w = np.maximum(a['wt'], 1)
            ax.plot(edges_mid, a['grad'] / w, 'o-', ms=3, label=STAGE_LABEL[s])
        ax.set_xlabel('reference luminance (bucket left edge)')
        ax.set_ylabel('gradient magnitude')
        ax.set_yscale('log')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', 'd1_gradient_by_bucket.png'), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.8, 4.0))
        vals = []
        for v in LOO_VARIANTS:
            a = acc_loo[v]
            w = np.maximum(a['wt'], 1)
            dg = weighted(a['grad'] / w, b_dark)
            vals.append(100.0 * (dg - base_grad) / max(base_grad, 1e-9))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(['w/o ' + o for o in ALL_OPS], fontsize=9)
        ax.set_ylabel('dark gradient change vs full chain (%)')
        ax.set_title('D2: leave-one-out attribution')
        ax.axhline(0, color='k', lw=0.8)
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', 'd2_leave_one_out.png'), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.loglog(xs, dx_min, label='min resolvable increment (8-bit)')
        ax.loglog(xs, xs, 'k--', lw=0.8, label='dx = x')
        ax.axhline(med_ref, color='r', ls=':', lw=1.0,
                   label='median dark gradient (measured %.4f)' % med_ref)
        ax.set_xlabel('linear input value x')
        ax.set_ylabel('min resolvable increment dx')
        ax.set_title('D3: 8-bit quantization step after gamma 1/2.2')
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', 'd3_quant_step.png'), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.6, 4.0))
        ax.loglog(centers, hist_ref.clip(1), label='reference (s1)')
        ax.loglog(centers, hist_quant.clip(1), label='after 8-bit (s6)')
        ax.set_xlabel('gradient magnitude (dark region)')
        ax.set_ylabel('count')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which='both')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', 'd3_dark_grad_hist.png'), dpi=160)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.0, 3.8))
        names = ['linear_s1', 'gamma_s5', 'quant_s6', 'inverse_s6']
        vals = []
        for v in names:
            a = acc_d4[v]
            w = np.maximum(a['wt'], 1)
            vals.append(weighted(a['grad'] / w, b_dark))
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(['linear (s1)', 'gamma (s5)', '8-bit (s6)', 'inverse (s6)'],
                           fontsize=9)
        ax.set_ylabel('dark gradient')
        ax.set_title('D4: quantization irreversibility')
        ax.grid(alpha=0.3, axis='y')
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'figures', 'd4_invertibility.png'), dpi=160)
        plt.close(fig)
    except Exception as exc:
        print('作图失败: %r' % (exc,), flush=True)

    print('\n暗区梯度中位数: 参考 %.5f -> 8bit 后 %.5f (保留 %.1f%%)'
          % (med_ref, med_q, 100.0 * med_q / max(med_ref, 1e-12)), flush=True)
    print('完成，输出目录: %s' % out, flush=True)


if __name__ == '__main__':
    main()
