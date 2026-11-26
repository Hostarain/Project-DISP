"""GGNC v1 + L_grad（能量加权口径）。

相对 `R_Net_GGNC_v1_18.py`（65.27）的唯一差别是多一项梯度方向一致性正则，
且把损失从"只看最强 3.7% 像素"改为**按参考梯度幅值加权**：

    L = Σ (1−cos)·||∇x_g|| / Σ ||∇x_g||

改动依据（实测，见 ch2_work/analyze_lgrad_why.py）：
  * 旧 'edge' 口径：mask 内方向偏差仅 0.0017，正则无事可做；
  * 'full' 口径：弱梯度区均值 0.0568，但中位仅 0.0008、p99=1.0，误差与 log 幅值
    相关系数 −0.82 —— 主要是近零梯度向量间的无意义夹角，训练被噪声尾巴主导；
  * 能量加权后：基线 0.0054 / GGNC v1 0.0114（2.1 倍），信号落在中段
    （50~90% 分位，占约一半梯度能量），且无需人工阈值。

`lambda_grad=0.1` 暂沿用。若前几轮 mAP 明显落后于 65.27 的同期轨迹，下调到 0.05。
"""
_base_ = ['R_Net_GGNC_v1_18.py']

model = dict(
    backbone=dict(
        use_grad_reg=True,
        lambda_grad=0.1,
        grad_mask='weighted',
    ),
)

# 损失级 + 梯度级双重 NaN 保护（见 mmdet/engine/hooks/nan_skip_hook.py）
custom_hooks = [dict(type='NaNSkipHook')]
