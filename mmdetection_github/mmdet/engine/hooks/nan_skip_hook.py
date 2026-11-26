# Copyright (c) OpenMMLab. All rights reserved.
"""非有限损失 / 非有限梯度时跳过参数更新，避免一次发散毁掉整次训练。

背景（2026-10-30/31 两次 15 轮训练均在 epoch 10 崩溃）：
  * 第一次：`1-cos` 里 `sqrt(0)` 的反向为 inf，梯度裁剪在 `total_norm=inf` 时
    缩放系数为 0，`inf*0=NaN`，一步把权重全打成 NaN。
  * 第二次：加过 eps 修复后仍在同一处崩溃。日志显示首个 NaN 损失出现在
    iter 24302，说明 **iter 24301 的 loss 是有限的、但梯度已非有限**——
    只检查 loss 的守卫拦不住这种情况，权重在 24301 那一步就被污染。

因此本 hook 把检查放在 `OptimWrapper.step` 之前：此时反向传播与梯度裁剪都已完成，
只要任一参数的梯度不是有限值，就丢弃该批（清梯度、不更新），训练继续。
"""
import functools

import torch
from mmengine.hooks import Hook
from mmengine.optim import OptimWrapper
from mmdet.registry import HOOKS

_STATS = {'step': 0, 'skipped': 0, 'loss_nan': 0, 'grad_nan': 0}


@HOOKS.register_module()
class NaNSkipHook(Hook):
    """跳过非有限梯度（及非有限损失）的参数更新。

    Args:
        max_warn (int): 最多打印多少条告警，避免刷屏。
    """

    priority = 'VERY_HIGH'

    def __init__(self, max_warn: int = 8):
        self.max_warn = max_warn

    def before_run(self, runner):
        if getattr(OptimWrapper.step, '_dsh_nan_guard', False):
            return

        # ① 梯度级守卫：step 之前检查所有参数的梯度
        orig_step = OptimWrapper.step

        @functools.wraps(orig_step)
        def patched_step(wrapper, **kwargs):
            _STATS['step'] += 1
            for group in wrapper.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        _STATS['skipped'] += 1
                        _STATS['grad_nan'] += 1
                        wrapper.optimizer.zero_grad(set_to_none=True)
                        if _STATS['skipped'] <= self.max_warn:
                            runner.logger.warning(
                                'NaNSkipHook: 检测到非有限梯度，已跳过该次更新 '
                                '(累计跳过 %d)', _STATS['skipped'])
                        return None
            return orig_step(wrapper, **kwargs)

        patched_step._dsh_nan_guard = True
        OptimWrapper.step = patched_step

        # ② 损失级守卫：NaN 损失直接丢弃，省掉一次无用的反向
        orig_update = OptimWrapper.update_params

        @functools.wraps(orig_update)
        def patched_update(wrapper, loss, **kwargs):
            if isinstance(loss, torch.Tensor) and not torch.isfinite(loss).all():
                _STATS['loss_nan'] += 1
                wrapper.optimizer.zero_grad(set_to_none=True)
                if hasattr(wrapper, '_inner_count'):
                    wrapper._inner_count += 1
                return None
            return orig_update(wrapper, loss, **kwargs)

        patched_update._dsh_nan_guard = True
        OptimWrapper.update_params = patched_update

        runner.logger.info('NaNSkipHook: 已启用（损失级 + 梯度级双重保护）')

    def after_run(self, runner):
        runner.logger.info(
            'NaNSkipHook: 共 %d 次 step，跳过 %d 次（其中梯度非有限 %d 次、'
            '损失非有限 %d 次）', _STATS['step'], _STATS['skipped'],
            _STATS['grad_nan'], _STATS['loss_nan'])
