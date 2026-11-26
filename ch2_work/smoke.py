"""冒烟测试：与参考 PNG 对齐、检查亮度分布、计时。"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ref_isp  # noqa: E402
import metrics   # noqa: E402

NPZ_DIR = os.path.join(ROOT, 'dataset', 'RAWTest', 'RAW_dark_npz')
PNG_DIR = os.path.join(ROOT, 'vis_isp_output')


def main():
    files = sorted(os.listdir(NPZ_DIR), key=lambda s: int(s.split('.')[0]))[:3]
    print('样本:', files, flush=True)

    # 1) 与原版 torch default_ISP 的产物（vis_common_isp.py 生成的 PNG）对齐
    cases = [(100, 'bicycle_100_common.png'), (10, 'car_10_common.png'),
             (140, 'bus_140_common.png')]
    print('\n[校验] 与 vis_isp_output 参考图对比（binning + 不截断 WB + 不降噪 + 不乘 ratio）')
    for img_id, png in cases:
        npz = os.path.join(NPZ_DIR, '%d.npz' % img_id)
        png_path = os.path.join(PNG_DIR, png)
        if not (os.path.exists(npz) and os.path.exists(png_path)):
            print('  跳过', png, '(缺文件)')
            continue
        mx, mean, _ = ref_isp.validate_against_png(npz, png_path)
        flag = 'OK' if mx < 2.0 / 255.0 else '不一致'
        print('  %-24s 最大误差 %.4f  平均误差 %.5f  -> %s' % (png, mx, mean, flag), flush=True)

    # 2) 亮度分布：决定分桶边界
    print('\n[亮度分布] s1 参考图分位点（乘 ratio，模拟检测器输入）')
    for p in files:
        im, meta = ref_isp.load_npz(os.path.join(NPZ_DIR, p))
        steps = ref_isp.run_pipeline(im, meta, demosaic='bilinear')
        lum = metrics.luminance(steps['s1_blacklevel'])
        qs = np.percentile(lum, [1, 10, 25, 50, 75, 90, 99])
        print('  %-10s ' % p, ' '.join('%.3f' % q for q in qs), flush=True)

    # 3) 计时
    im, meta = ref_isp.load_npz(os.path.join(NPZ_DIR, files[0]))
    t0 = time.time()
    steps = ref_isp.run_pipeline(im, meta, demosaic='bilinear')
    t_isp = time.time() - t0
    ref_rgb = steps['s1_blacklevel']
    t0 = time.time()
    m = metrics.stage_metrics(steps['s6_quant'], ref_rgb)
    t_met = time.time() - t0
    print('\n[计时] ISP 全链路 %.3f s, 单阶段指标 %.3f s' % (t_isp, t_met), flush=True)
    print('[指标] 梯度幅值/桶 :', np.round(m['grad_mag'], 5))
    print('[指标] 像素数/桶   :', m['bucket_count'])
    print('[指标] 方向一致性  :', np.round(m['dir_consistency'], 3))
    print('[指标] PSD 斜率 %.3f (参考 %.3f)' % (m['psd_slope'], m['ref_psd_slope']))
    print('[指标] 平坦区标准差 %.5f' % m['flat_std'])
    print('[指标] 色度残差 R/G %.4f B/G %.4f' % (m['chroma_rg'], m['chroma_bg']))


if __name__ == '__main__':
    main()
