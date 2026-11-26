"""D5 基线：传统 ISP（默认 ISP，v0）× RetinaNet-R18。

与 `R_Net_DarkISP_18.py` 完全同一条数据管线（npz + RAWRetinaNet + 相同增强/尺寸/
优化器），唯一差别是 `ISP_version='v0'`：
RAW → 黑电平归一化 → 白平衡 → RGBG 通道合并 → CCM → gamma → 8bit，
且输出 detach（检测损失不回传 RAW）。

这样 D5 与 61.90 的 Dark-ISP 基线是**同协议对照**。

注意：不要用同目录下旧的 `R_Net_defaultISP_18.py`——那是给预渲染 RGB 图目录
（`RGB_dark/` + `RAW-dark-Annotations/`）用的，本地没有那批数据。
"""
_base_ = ['R_Net_DarkISP_18.py']

model = dict(
    backbone=dict(ISP_version='v0'),
)
