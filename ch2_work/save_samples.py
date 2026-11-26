"""导出参考 ISP 各阶段的样例图（供论文配图 / 人工核对）。

输出：
    samples/stage_strip_<id>.png   单张图的 6 段横向拼图
    samples/pair_<id>.png          参考图(线性) vs 8bit 输出的两联大图
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import ref_isp  # noqa: E402

NPZ_DIR = os.path.join(ROOT, 'dataset', 'RAWTest', 'RAW_dark_npz')
STAGES = ['s1_blacklevel', 's2_wb', 's3_ccm', 's4_denoise', 's5_gamma', 's6_quant']
LABEL = ['black level', '+ WB', '+ CCM', '+ denoise', '+ gamma', '+ 8-bit']


def to_u8(rgb):
    return (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def strip(stages, panel_w=380):
    panels = []
    for s, lab in zip(STAGES, LABEL):
        img = Image.fromarray(to_u8(stages[s]))
        h = int(img.height * panel_w / img.width)
        img = img.resize((panel_w, h), Image.BILINEAR)
        canvas = Image.new('RGB', (panel_w, h + 18), (255, 255, 255))
        canvas.paste(img, (0, 18))
        ImageDraw.Draw(canvas).text((4, 3), lab, fill=(0, 0, 0))
        panels.append(canvas)
    out = Image.new('RGB', (panel_w * len(panels), panels[0].height), (255, 255, 255))
    for i, p in enumerate(panels):
        out.paste(p, (i * panel_w, 0))
    return out


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'out', 'samples')
    os.makedirs(out_dir, exist_ok=True)

    # 挑几张：中位亮度、最暗、最亮
    files = sorted((f for f in os.listdir(NPZ_DIR) if f.endswith('.npz')),
                   key=lambda s: int(s.split('.')[0]))
    picks = []
    for f in files[::max(len(files) // 3, 1)][:3]:
        picks.append(f)
    picks = picks[:3]

    for f in picks:
        im, meta = ref_isp.load_npz(os.path.join(NPZ_DIR, f))
        steps = ref_isp.run_pipeline(im, meta, demosaic='bilinear')
        img_id = f.split('.')[0]
        strip(steps).save(os.path.join(out_dir, 'stage_strip_%s.png' % img_id))

        pair = Image.new('RGB', (900, 620), (255, 255, 255))
        for i, (s, lab) in enumerate([('s1_blacklevel', 'linear (reference)'),
                                      ('s6_quant', 'full chain (8-bit)')]):
            p = Image.fromarray(to_u8(steps[s]))
            w = 440
            h = int(p.height * w / p.width)
            p = p.resize((w, h), Image.BILINEAR)
            pair.paste(p, (10 + i * 460, 24))
            ImageDraw.Draw(pair).text((10 + i * 460, 6), lab, fill=(0, 0, 0))
        pair.save(os.path.join(out_dir, 'pair_%s.png' % img_id))
        print('导出', img_id, flush=True)

    print('样例图目录:', out_dir, flush=True)


if __name__ == '__main__':
    main()
