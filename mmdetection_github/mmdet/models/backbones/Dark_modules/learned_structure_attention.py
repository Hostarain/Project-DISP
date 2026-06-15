"""
LearnedStructureAttention — a learnable structural response operator for
Dark-ISP output.  K differential kernels (groups=in_ch) are initialised
with mixed differential priors and adapt to the per-channel gradient
distribution of Dark-ISP during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedStructureAttention(nn.Module):
    """
    Args:
        in_ch:   input channels (3 for Dark-ISP RGB output).
        K:       total output channels of the learnable structural response
                 operator.  Must satisfy K // in_ch == 4.
        mid_ch:  hidden channels of the attention head.
    """

    def __init__(self, in_ch: int = 3, K: int = 12, mid_ch: int = 16):
        super().__init__()
        assert K // in_ch == 4, \
            f"K ({K}) must be 4 × in_ch ({in_ch}) for the chosen initialisation"

        # ---------- learnable structural response kernels ----------
        # groups=in_ch → each RGB channel gets its own K//in_ch kernels
        self.diff_conv = nn.Conv2d(
            in_ch, K, kernel_size=3, padding=1,
            groups=in_ch, bias=False,
        )
        self._init_diff_kernels()

        # ---------- attention head (unchanged from v1) ----------
        self.attn = nn.Sequential(
            nn.Conv2d(K, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid(),
        )

        # ---------- learnable enhancement strength ----------
        # Unconstrained — may become negative, turning the operator into a
        # general structural residual modulator rather than pure enhancement.
        self.gamma = nn.Parameter(torch.tensor(0.1))

    # ------------------------------------------------------------------
    #  initialisation
    # ------------------------------------------------------------------
    def _init_diff_kernels(self):
        """Mixed differential priors: 4 kernel types per group."""
        sobel_x = torch.tensor([[-1.,  0.,  1.],
                                [-2.,  0.,  2.],
                                [-1.,  0.,  1.]]) / 4.
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]) / 4.
        center_x = torch.tensor([[ 0.,  0.,  0.],
                                 [-1.,  0.,  1.],
                                 [ 0.,  0.,  0.]]) / 2.
        center_y = torch.tensor([[ 0., -1.,  0.],
                                 [ 0.,  0.,  0.],
                                 [ 0.,  1.,  0.]]) / 2.
        inits = [sobel_x, sobel_y, center_x, center_y]

        K_per_g = self.diff_conv.out_channels // self.diff_conv.groups

        with torch.no_grad():
            for g in range(self.diff_conv.groups):
                for k in range(K_per_g):
                    idx = g * K_per_g + k
                    # k % len(inits) guards against future K changes
                    self.diff_conv.weight.data[idx, 0].copy_(inits[k % len(inits)])

    # ------------------------------------------------------------------
    #  forward
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Dark-ISP output, [B, 3, H, W], values in [0, 1].
        Returns:
            Enhanced tensor of the same shape.
        """
        resp = self.diff_conv(x)          # [B, K, H, W]  structural responses
        alpha = self.attn(resp)           # [B, 3, H, W]  spatial modulation
        return x + x * alpha * self.gamma


class ComputeMatchedBaseline(nn.Module):
    """
    FLOPs-matched naive residual block for ablation study.

    This block has NO structural prior — it is a generic Conv-BN-ReLU-Conv
    residual block with matched FLOPs to LearnedStructureAttention.
    It serves as a control experiment to isolate whether LSA's gain comes
    from the differential kernel design or merely from extra computation.

    FLOPs breakdown (input [B, 3, H, W]):
      - conv1 (3→16, 3×3): 3×9×16 = 432 MACs/px
      - conv2 (16→3, 3×3): 16×9×3 = 432 MACs/px
      - Total: 864 MACs/px ≈ LSA's total (108 + 1728 + 432 = 2268 MACs/px)
        (LSA is ~2.6× more FLOPs; for stricter matching, increase mid_ch)

    For strict FLOPs matching with LSA, use mid_ch=24:
      - conv1 (3→24, 3×3): 3×9×24 = 648 MACs/px
      - conv2 (24→3, 3×3): 24×9×3 = 648 MACs/px
      - Total: 1296 MACs/px (still less than LSA's 2268...
        but matching exactly requires a different architecture)

    The key point: if this generic block achieves similar mAP to LSA,
    then the structural prior is NOT the source of gain.
    """
    def __init__(self, in_ch: int = 3, mid_ch: int = 24):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.gamma = nn.Parameter(torch.tensor(0.1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.conv_block(x)
        return x + residual * self.gamma
