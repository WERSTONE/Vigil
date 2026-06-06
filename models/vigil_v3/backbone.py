"""AttentionCSPDarkNet — CNN 前端 + Area Attention 后端.

YOLOv12 风格: 前两阶段保留 CNN (低层特征受益于卷积归纳偏置)，
后两阶段使用 R-ELAN + Area Attention (高层语义建模)。
"""

import torch.nn as nn
from models.common import Conv, C2f, SPPF
from models.vigil_v3.attention import RELAN


class AttentionCSPDarkNet(nn.Module):
    """混合 CNN-Attention backbone.

    输出 P3/P4/P5 (stride 8/16/32)。

    Args:
        w: 宽度系数 (1.0=完整, 0.5=轻量)
        max_ch: 通道上限
    """

    def __init__(self, w=1.0, max_ch=512):
        super().__init__()
        ch = lambda x: min(int(x * w), max_ch)

        # ── Stem: /2 → /4 (纯 CNN) ──
        self.stem = nn.Sequential(
            Conv(3, ch(64), 3, stride=2),
            Conv(ch(64), ch(128), 3, stride=2),
            C2f(ch(128), ch(128), n=2),
        )

        # ── Stage 3: /8 (CNN) ──
        self.stage3 = nn.Sequential(
            Conv(ch(128), ch(256), 3, stride=2),
            C2f(ch(256), ch(256), n=2),
        )

        # ── Stage 4: /16 (Attention) ──
        self.stage4 = nn.Sequential(
            Conv(ch(256), ch(512), 3, stride=2),
            RELAN(ch(512), ch(512), num_heads=8, area=4),
        )

        # ── Stage 5: /32 (Attention + SPPF) ──
        self.stage5 = nn.Sequential(
            Conv(ch(512), ch(512), 3, stride=2),
            RELAN(ch(512), ch(512), num_heads=8, area=2),
            SPPF(ch(512), ch(512)),
        )

        self.out_channels = [ch(128), ch(256), ch(512), ch(512)]

    def forward(self, x):
        p2 = self.stem(x)       # /4
        p3 = self.stage3(p2)    # /8
        p4 = self.stage4(p3)    # /16
        p5 = self.stage5(p4)    # /32
        return p2, p3, p4, p5
