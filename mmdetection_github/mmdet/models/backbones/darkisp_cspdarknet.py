# Dark-ISP + CSPDarknet backbone
# ISP processes RAW → RGB, then CSPDarknet extracts features

import warnings
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from .csp_darknet import CSPDarknet, Focus, SPPBottleneck
from mmdet.models.backbones.Dark_modules.noise_predictor import (
    Matrix_Predictor, Color_Level_Process, NILUT, LUT3D)
from mmdet.models.backbones.Dark_modules.utils import color_transform, default_ISP


@MODELS.register_module()
class DarkISP_CSPDarknet(BaseModule):
    """
    Dark-ISP as ISP frontend + CSPDarknet as backbone.
    
    Pipeline:
        RAW (B,4,H,W) → Dark-ISP (Linear+Nonlinear+Self-Boost) → RGB (B,3,H,W)
        → CSPDarknet → multi-scale features
    
    Returns (features, loss_mat) for compatibility with SingleStageRAWDetector.
    """

    def __init__(self,
                 ISP_version='v2',
                 use_grad_guide=False,
                 # --- CSPDarknet params ---
                 arch='P5',
                 deepen_factor=1.0,
                 widen_factor=1.0,
                 out_indices=(2, 3, 4),
                 frozen_stages=-1,
                 use_depthwise=False,
                 spp_kernal_sizes=(5, 9, 13),
                 norm_cfg=dict(type='BN', momentum=0.03, eps=0.001),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 cspdarknet_init_cfg=dict(
                     type='Kaiming', layer='Conv2d', a=2.236,
                     distribution='uniform', mode='fan_in',
                     nonlinearity='leaky_relu'),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)

        # ═══════════ ISP Modules ═══════════
        self.version = ISP_version
        self.params_predictor = Matrix_Predictor(
            attn_drop=0.1, proj_drop=0.1, head_drop=0.2, data='LOD')
        self.nonlinear = Color_Level_Process(use_grad_guide=use_grad_guide)

        # ═══════════ CSPDarknet Backbone ═══════════
        self.cspdarknet = CSPDarknet(
            init_cfg=cspdarknet_init_cfg,
            arch=arch,
            deepen_factor=deepen_factor,
            widen_factor=widen_factor,
            out_indices=out_indices,
            frozen_stages=frozen_stages,
            use_depthwise=use_depthwise,
            spp_kernal_sizes=spp_kernal_sizes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            norm_eval=norm_eval,
        )

    # ────────── ISP helpers (same as Dark_ResNet) ──────────

    def joint_mat(self, metainfo, increment=None):
        ccm = metainfo['ccm']
        second = ccm[:, :, 1].unsqueeze(-1)
        ccm = torch.cat((ccm, second), dim=2)
        wb = metainfo['wb'].unsqueeze(1)
        if increment is not None:
            dccm = increment[0]
            second = dccm[:, :, 1].unsqueeze(-1)
            dccm = torch.cat((dccm, second), dim=2)
            dr1, db1 = increment[1], increment[2]
            ccm += dccm
            wb[:, :, 0] += dr1
            wb[:, :, 2] += db1
        wb[:, :, 1] *= 0.5
        wb[:, :, 3] *= 0.5
        return wb * ccm

    def mat_loss2(self, x_i, x_l, P):
        b, c, h, w = x_i.shape
        x_i = x_i.view(b, c, -1)
        x_l = x_l.view(b, 3, -1)
        x_iT = x_i.permute(0, 2, 1)
        result1 = torch.bmm(x_l, x_iT)
        result2 = torch.bmm(x_i, x_iT)
        det = torch.linalg.det(result2)
        if torch.any(det.abs() < 1e-8):
            return torch.tensor(0.0, device=x_i.device, requires_grad=True)
        result2 = torch.linalg.inv(result2)
        A = torch.bmm(result1, result2)
        P_mean = torch.mean(P, dim=(1, 2))
        B_, mh, mw = A.shape
        P_mean = P_mean.view(B_, mh, mw)
        dot_product = (A * P_mean).sum(dim=2)
        norm1 = torch.norm(P_mean, dim=-1) + 1e-5
        norm2 = torch.norm(A, dim=-1) + 1e-5
        similarity = dot_product / (norm1 * norm2)
        loss_mat = (1 - similarity).sum(dim=-1).mean()
        return loss_mat * 0.05

    # ────────── Forward ──────────

    def forward(self, x, metainfo, cur_epoch=None, batch_data_samples=None):
        x_i = x.clip(0, 1)

        if self.version == 'v0':
            x_l, x_g = default_ISP(x_i, metainfo)
            x_l = x_l.clip(0, 1).detach()
            loss_mat = torch.zeros_like(x_l)
        elif self.version == 'v1':
            x_l = torch.stack([x_i[:, 0, ...],
                               torch.mean(x_i[:, [1, 3], ...], dim=1),
                               x_i[:, 2, ...]], dim=1)
            x_l = x_l.clip(0, 1)
            loss_mat = torch.zeros_like(x_l)
        elif self.version == 'v2':
            mat = self.joint_mat(metainfo)
            dmat, glo, local = self.params_predictor(x_i, mat)
            x_g, mat = color_transform(x_i, mat, dmat=dmat)
            x_g = x_g.clip(0, 1)
            x_l = self.nonlinear(x_g)
            if self.training:
                loss_mat = self.mat_loss2(x_i, x_l, mat)
            else:
                loss_mat = torch.zeros_like(x_l)
        else:
            raise ValueError(f'Unknown ISP_version: {self.version}')

        # ─── RGB → CSPDarknet ───
        features = self.cspdarknet(x_l)
        return features, loss_mat

    def train(self, mode=True):
        super().train(mode)
        self.cspdarknet.train(mode)
        if mode and self.cspdarknet.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
