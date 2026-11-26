"""D6/R2 对照：仅去马赛克（v1）× RetinaNet-R18。

`ISP_version='v1'`：不做白平衡、不做 CCM、不做 gamma，只把 RGBG 四通道按
绿通道平均合并为三通道。用于把"颜色处理"的整体贡献从"空间重建 + 量化"里剥出来。

与 `R_Net_DarkISP_18.py` 同一条数据管线，只换 ISP 版本。

注意：同目录下旧的 `R_Net_demosaic_18.py` 依赖预渲染图目录（`RAW_dark/`），
本地没有该数据，不可直接用。
"""
_base_ = ['R_Net_DarkISP_18.py']

model = dict(
    backbone=dict(ISP_version='v1'),
)
