"""GGNC v1 + L_grad（新掩码口径）。

与 `R_Net_GGNC_v1_18.py`（65.27）的唯一差别是多一项梯度方向一致性正则，
且把 mask 从"仅强梯度像素"改为"全图有效像素"（`grad_mask='full'`）。

改动依据（实测，见 ch2_work/measure_lgrad.py）：
  旧口径的 mask 只覆盖 |grad| 前 3.7% 的像素，该处 GGNC v1 的方向偏差仅 0.0017，
  正则无事可做；而它在弱梯度区（0~50% 分位）的偏差达 0.0405，是基线的 22 倍。

`lambda_grad` 暂取 0.1。注意：新口径下 loss 量级约为旧口径的 5~10 倍，
因此该项的实际影响显著强于旧配置，若训练早期 mAP 明显落后于 65.27 的同期轨迹，
应下调到 0.02~0.05。
"""
_base_ = ['R_Net_GGNC_v1_18.py']

model = dict(
    backbone=dict(
        use_grad_reg=True,
        lambda_grad=0.1,
        grad_mask='full',
    ),
)

# 非有限损失保护：2026-10-30 那次训练在第 10 轮 NaN 崩溃，加此 hook 后
# 单批发散只会被跳过，不会污染权重。
custom_hooks = [dict(type='NaNSkipHook')]
