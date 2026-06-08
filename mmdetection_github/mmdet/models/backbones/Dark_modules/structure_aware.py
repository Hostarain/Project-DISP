import torch
import torch.nn as nn
import torch.nn.functional as F


class StructureAwareBlock(nn.Module):
    """
    Gradient-aware stem preprocessor for Dark-ISP output.

    Extracts per-channel spatial gradients (Sobel x/y) from the 3-channel RGB
    output of Dark-ISP, learns a pixel-wise attention map, and injects it
    back via residual connection.  The module is designed to amplify the weak
    edges that Dark-ISP's non-convex polynomial bases stretch out of dark
    regions, without disturbing the original signal.

    Args:
        in_ch (int):  input channels (3 for RGB).
        mid_ch (int): hidden channels of the attention sub-network.
        scale (float): residual multiplier to keep the enhancement mild
                       at initialisation and let the optimizer decide how
                       much structural boost is needed.
    """

    def __init__(self, in_ch: int = 3, mid_ch: int = 16, scale: float = 0.1):
        super().__init__()
        self.scale = scale

        # ---------- fixed Sobel kernels (not learned) ----------
        sobel_x = torch.tensor([[-1., 0., 1.],
                                [-2., 0., 2.],
                                [-1., 0., 1.]]) / 4.0   # normalised
        sobel_y = torch.tensor([[-1., -2., -1.],
                                [ 0.,  0.,  0.],
                                [ 1.,  2.,  1.]]) / 4.0

        # shape: [out_ch, in_ch/groups, kh, kw] → [3, 1, 3, 3] with groups=3
        self.register_buffer('sobel_x',
                             sobel_x.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))
        self.register_buffer('sobel_y',
                             sobel_y.view(1, 1, 3, 3).repeat(in_ch, 1, 1, 1))

        # ---------- learnable attention head ----------
        # 6-channel input: 3 ch grad_x + 3 ch grad_y
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, mid_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Dark-ISP output, shape [B, 3, H, W], values in [0, 1].

        Returns:
            Enhanced tensor with same shape.
        """
        # per-channel Sobel gradients (groups=3 → each RGB channel independently)
        gx = F.conv2d(x, self.sobel_x, padding=1, groups=x.shape[1])
        gy = F.conv2d(x, self.sobel_y, padding=1, groups=x.shape[1])
        grad = torch.cat([gx, gy], dim=1)          # [B, 6, H, W]

        alpha = self.attn(grad)                     # [B, 3, H, W] in [0, 1]

        # residual injection — at init alpha≈0.5 → boost≈5 % of original
        return x + x * alpha * self.scale
