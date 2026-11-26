"""可分解参考 ISP（方案 B）。

对应目录 2.2 的五个环节 + 量化：
    s1 黑电平减法与截断
    s2 白平衡增益（可含饱和截断）
    s3 色彩校正矩阵
    s4 降噪
    s5 gamma 编码
    s6 8 比特量化

与仓库 `Dark_modules/utils.py::default_ISP` 的关系：
    demosaic='binning' + clip_wb=False + denoise=False 时，s3/s6 应与
    `default_ISP` 的 image_g / image 完全一致（见 validate_against_default_isp）。
"""
import numpy as np
import cv2

_VALUE2LETTER = {0: 'R', 1: 'G', 2: 'B', 3: 'G'}
_GAMMA = 2.2


# ---------------------------------------------------------------- 基础工具

def bayer_code(raw_pattern):
    letters = ''.join(_VALUE2LETTER[int(raw_pattern[r, c])]
                      for r in range(2) for c in range(2))
    code = getattr(cv2, 'COLOR_Bayer%s2BGR' % letters, None)
    if code is None:
        raise ValueError('unsupported bayer pattern: %s' % letters)
    return code, letters


def black_level_norm(im, meta):
    """(H,W,4) 原始码值 -> [0,1] 归一化 4 通道（逐通道黑/白电平）。"""
    im = im.astype(np.float32)
    bl = np.asarray(meta['black_level'], dtype=np.float32).reshape(1, 1, 4)
    wl = np.asarray(meta['white_level'], dtype=np.float32).reshape(1, 1, 4)
    return np.clip((im - bl) / np.maximum(wl - bl, 1.0), 0.0, 1.0)


def demosaic_binning(img4):
    """RGBG 四通道直接合并为三通道（现有 default_ISP 的做法）。"""
    return np.stack([img4[..., 0],
                     (img4[..., 1] + img4[..., 3]) * 0.5,
                     img4[..., 2]], axis=-1)


def demosaic_bilinear(img4, raw_pattern):
    """还原整幅 mosaic 后走 cv2 双线性去马赛克（输出 RGB）。"""
    h, w, _ = img4.shape
    mosaic = np.zeros((h * 2, w * 2), dtype=np.float32)
    for ch in range(4):
        pos = np.where(raw_pattern == ch)
        if len(pos[0]) != 1:
            raise ValueError('pattern 通道 %d 出现 %d 次' % (ch, len(pos[0])))
        r0, c0 = int(pos[0][0]), int(pos[1][0])
        mosaic[r0::2, c0::2] = img4[..., ch]

    code, _ = bayer_code(raw_pattern)
    m16 = np.clip(mosaic * 65535.0, 0, 65535).astype(np.uint16)
    bgr = cv2.cvtColor(m16, code).astype(np.float32) / 65535.0
    rgb = bgr[..., ::-1]                      # cv2 给的是 BGR
    return np.clip(rgb, 0.0, 1.0)


def apply_wb(img4, wb):
    wb = np.asarray(wb, dtype=np.float32).reshape(1, 1, 4)
    return img4 * wb


def apply_ccm(rgb, ccm):
    """与原实现一致：out_r = Σ_c rgb_c * ccm[r, c]。"""
    ccm = np.asarray(ccm, dtype=np.float32).reshape(3, 3)
    return np.einsum('hwc,rc->hwr', rgb, ccm)


def denoise_bilateral(rgb, d=5, sigma_color=0.05, sigma_space=5):
    return np.clip(cv2.bilateralFilter(rgb.astype(np.float32), d,
                                       sigma_color, sigma_space), 0.0, 1.0)


def gamma_encode(rgb, gamma=_GAMMA):
    return np.clip(np.clip(rgb, 1e-8, 1.0) ** (1.0 / gamma), 0.0, 1.0)


def quantize8(rgb):
    return np.clip(np.round(rgb * 255.0) / 255.0, 0.0, 1.0)


# ---------------------------------------------------------------- 主链路

def run_pipeline(im, meta, demosaic='bilinear', apply_ratio=True, clip_wb=True,
                 enable=('wb', 'ccm', 'denoise', 'gamma', 'quant')):
    """按 2.2 的环节顺序跑一遍，返回每一步之后的中间结果。

    参数
    ----
    im : (H, W, 4) uint16，npz 里的 im
    meta : dict，含 black_level / white_level / wb / ccm / ratio / raw_pattern
    enable : 需要启用的环节，留一法时传入去掉某一项后的元组
    """
    steps = {}

    x = black_level_norm(im, meta)
    if apply_ratio:
        x = np.clip(x * float(np.asarray(meta['ratio']).reshape(-1)[0]), 0.0, 1.0)
    x = np.clip(x, 0.0, 1.0)
    pattern = np.asarray(meta['raw_pattern'])

    def to_rgb(four):
        if demosaic == 'binning':
            return demosaic_binning(four)
        return demosaic_bilinear(four, pattern)

    # s1：黑电平（+ratio 重曝光）后重建为三通道，白平衡取单位增益
    steps['s1_blacklevel'] = to_rgb(x)

    # s2：白平衡增益（可选饱和截断）
    if 'wb' in enable:
        xw = apply_wb(x, meta['wb'])
        if clip_wb:
            xw = np.clip(xw, 0.0, 1.0)
        rgb = to_rgb(xw)
    else:
        rgb = steps['s1_blacklevel']
    steps['s2_wb'] = rgb

    # s3：色彩校正
    if 'ccm' in enable:
        rgb = np.clip(apply_ccm(rgb, meta['ccm']), 0.0, 1.0)
    steps['s3_ccm'] = rgb

    # s4：降噪
    if 'denoise' in enable:
        rgb = denoise_bilateral(rgb)
    steps['s4_denoise'] = rgb

    # s5：gamma 编码
    if 'gamma' in enable:
        rgb = gamma_encode(rgb)
    steps['s5_gamma'] = rgb

    # s6：8 比特量化
    if 'quant' in enable:
        rgb = quantize8(rgb)
    steps['s6_quant'] = rgb

    return steps


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    meta = {k: d[k] for k in ('black_level', 'white_level', 'wb', 'ccm',
                              'ratio', 'raw_pattern')}
    return d['im'], meta


# ---------------------------------------------------------------- 校验

def validate_against_png(npz_path, png_path, tol=2.0 / 255.0):
    """与仓库里 `vis_common_isp.py` 生成的参考图对比。

    那批 PNG 由原版 torch `default_ISP`（WB→通道合并→CCM→gamma→8bit，**未乘 ratio**）
    生成，因此这里用 binning + 不截断白平衡 + 不降噪 + 不乘 ratio 复现。
    返回 (最大误差, 平均误差, 我方结果)。
    """
    im, meta = load_npz(npz_path)
    steps = run_pipeline(im, meta, demosaic='binning', clip_wb=False,
                         apply_ratio=False,
                         enable=('wb', 'ccm', 'gamma', 'quant'))
    mine = steps['s6_quant']

    bgr = cv2.imread(png_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(png_path)
    ref = bgr[..., ::-1].astype(np.float32) / 255.0

    if ref.shape[:2] != mine.shape[:2]:
        h = min(ref.shape[0], mine.shape[0])
        w = min(ref.shape[1], mine.shape[1])
        ref, mine = ref[:h, :w], mine[:h, :w]

    diff = np.abs(ref - mine)
    return float(diff.max()), float(diff.mean()), mine
