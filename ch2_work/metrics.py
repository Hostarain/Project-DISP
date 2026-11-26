"""结构 / 色度指标库（第 2–4 章共用，定义不得随意改动）。

约定
----
* 所有输入为 (H, W, 3) float RGB，范围 [0, 1]；
* 亮度一律用 Rec.601 权重；
* 亮度分桶一律基于**参考图（s1，黑电平后、无白平衡）**的亮度，
  避免不同 ISP 之间桶边界漂移；
* 梯度幅值用同一套 Sobel 核，跨阶段可比。
"""
import numpy as np
import cv2

LUMA_W = np.array([0.299, 0.587, 0.114], dtype=np.float32)
DEFAULT_EDGES = (0.0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.0001)


def luminance(rgb):
    return rgb.astype(np.float32) @ LUMA_W


def gradients(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    return gx, gy


def gradient_magnitude(gray):
    gx, gy = gradients(gray)
    return np.sqrt(gx * gx + gy * gy)


def bucket_index(ref_lum, edges=DEFAULT_EDGES):
    """返回 0..nb-1 的桶编号；越界归入首/末桶。"""
    edges = np.asarray(edges, dtype=np.float32)
    idx = np.digitize(ref_lum, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def bucket_mean(values, idx, nb):
    out = np.full(nb, np.nan, dtype=np.float64)
    for b in range(nb):
        m = idx == b
        if m.any():
            out[b] = float(values[m].mean())
    return out


def bucket_count(idx, nb):
    return np.array([int((idx == b).sum()) for b in range(nb)], dtype=np.int64)


def direction_consistency(ref_gray, img_gray, mask=None, eps=1e-8):
    """参考图与目标图梯度方向的余弦相似度均值（方向对单调色调映射不变）。"""
    rx, ry = gradients(ref_gray)
    ix, iy = gradients(img_gray)
    dot = rx * ix + ry * iy
    nr = np.sqrt(rx * rx + ry * ry) + eps
    ni = np.sqrt(ix * ix + iy * iy) + eps
    cos = np.clip(dot / (nr * ni), -1.0, 1.0)
    if mask is None:
        return float(cos.mean())
    if not mask.any():
        return float('nan')
    return float(cos[mask].mean())


def edge_density(mag, tau):
    return float((mag > tau).mean())


def adaptive_tau(mag, k=2.0):
    """与 L_grad 一致的阈值约定：均值 + k 倍标准差。"""
    return float(mag.mean() + k * mag.std())


def flat_block_std(gray, ref_mag, block=8, grad_frac=0.2):
    """平坦区噪声底：取块内参考梯度最低的若干块，返回块内标准差的中位数。"""
    h, w = gray.shape
    hb, wb = h // block, w // block
    if hb < 2 or wb < 2:
        return float('nan')
    g = gray[:hb * block, :wb * block].reshape(hb, block, wb, block)
    m = ref_mag[:hb * block, :wb * block].reshape(hb, block, wb, block)
    block_std = g.std(axis=(1, 3))
    block_mag = m.mean(axis=(1, 3))
    thr = np.quantile(block_mag, grad_frac)
    sel = block_mag <= thr
    if not sel.any():
        return float('nan')
    return float(np.median(block_std[sel]))


def radial_psd_slope(gray, size=512, rmin=4, rmax_frac=0.25):
    """中心裁剪后做二维功率谱，径向量化后拟合 log-log 斜率（高频衰减速度）。"""
    h, w = gray.shape
    s = min(size, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    patch = gray[y0:y0 + s, x0:x0 + s].astype(np.float32)
    win = np.outer(np.hanning(s), np.hanning(s)).astype(np.float32)
    f = np.fft.fftshift(np.fft.fft2((patch - patch.mean()) * win))
    psd = (f.real ** 2 + f.imag ** 2).astype(np.float64)

    yy, xx = np.mgrid[0:s, 0:s]
    r = np.sqrt((yy - s // 2) ** 2 + (xx - s // 2) ** 2).astype(np.int32)
    rmax = int(s * rmax_frac)
    rr = np.arange(rmin, rmax + 1)
    prof = np.array([psd[r == k].mean() if (r == k).any() else np.nan for k in rr])
    good = np.isfinite(prof) & (prof > 0)
    if good.sum() < 4:
        return float('nan')
    lx = np.log(rr[good].astype(np.float64))
    ly = np.log(prof[good])
    slope = np.polyfit(lx, ly, 1)[0]
    return float(slope)


def chroma_residual(rgb, lum, top_frac=0.02, eps=1e-6):
    """最亮区域的 R/G、B/G 偏离（白平衡残差）；理想白平衡下接近 0。"""
    thr = np.quantile(lum, 1.0 - top_frac)
    m = lum >= thr
    if m.sum() < 16:
        return float('nan'), float('nan')
    r = rgb[..., 0][m].mean()
    g = rgb[..., 1][m].mean() + eps
    b = rgb[..., 2][m].mean()
    return float(r / g - 1.0), float(b / g - 1.0)


class Reference(object):
    """参考图（s1：黑电平后、白平衡取单位增益）的缓存。

    分桶、参考梯度、阈值、参考谱斜率都只与参考图有关，所有阶段共用，
    避免重复计算。
    """

    def __init__(self, rgb, edges=DEFAULT_EDGES):
        self.edges = np.asarray(edges, dtype=np.float32)
        self.nb = len(self.edges) - 1
        self.lum = luminance(rgb)
        self.gx, self.gy = gradients(self.lum)
        self.mag = np.sqrt(self.gx * self.gx + self.gy * self.gy)
        self.idx = bucket_index(self.lum, self.edges).ravel().astype(np.int64)
        self.tau = adaptive_tau(self.mag)
        w = self.mag.ravel().astype(np.float64)
        self.count = np.bincount(self.idx, minlength=self.nb)
        self.grad_sum = np.bincount(self.idx, weights=w, minlength=self.nb)
        self.psd_slope = radial_psd_slope(self.lum)

    def _bucket_mean(self, total):
        out = np.full(self.nb, np.nan)
        ok = self.count > 256
        out[ok] = total[ok] / self.count[ok]
        return out

    def metrics(self, rgb):
        """目标图相对参考图的结构 / 色度指标。"""
        lum = luminance(rgb)
        gx, gy = gradients(lum)
        mag = np.sqrt(gx * gx + gy * gy)

        dot = self.gx * gx + self.gy * gy
        cos_map = np.clip(dot / (self.mag * mag + 1e-8), -1.0, 1.0)

        flat_idx = self.idx
        m_cnt = np.maximum(self.count, 1)
        grad_mean = self._bucket_mean(
            np.bincount(flat_idx, weights=mag.ravel().astype(np.float64),
                        minlength=self.nb))
        cos_mean = self._bucket_mean(
            np.bincount(flat_idx, weights=cos_map.ravel().astype(np.float64),
                        minlength=self.nb))
        edge_frac = self._bucket_mean(
            np.bincount(flat_idx,
                        weights=(mag > self.tau).ravel().astype(np.float64),
                        minlength=self.nb))

        # 弱梯度区（参考梯度低于阈值 1/4）的方向一致性：考察平坦区是否被造出结构
        weak = self.mag.ravel() <= self.tau * 0.25
        weak_cos = np.full(self.nb, np.nan)
        if weak.any():
            w_cnt = np.bincount(self.idx[weak], minlength=self.nb)
            w_sum = np.bincount(self.idx[weak],
                                weights=cos_map.ravel()[weak].astype(np.float64),
                                minlength=self.nb)
            ok = w_cnt > 256
            weak_cos[ok] = w_sum[ok] / w_cnt[ok]

        chroma_rg, chroma_bg = chroma_residual(rgb, lum)
        return {
            'grad_mag': grad_mean,
            'ref_grad_mag': self.grad_sum / m_cnt,
            'dir_consistency': cos_mean,
            'weak_region_cos': weak_cos,
            'edge_density': edge_frac,
            'bucket_count': self.count,
            'psd_slope': radial_psd_slope(lum),
            'ref_psd_slope': self.psd_slope,
            'flat_std': flat_block_std(lum, self.mag),
            'chroma_rg': chroma_rg,
            'chroma_bg': chroma_bg,
        }


def stage_metrics(rgb, ref_rgb_or_reference, edges=DEFAULT_EDGES):
    """兼容接口：ref 可以是参考图或已构造的 Reference 对象。"""
    ref = (ref_rgb_or_reference if isinstance(ref_rgb_or_reference, Reference)
           else Reference(ref_rgb_or_reference, edges))
    return ref.metrics(rgb)
