"""CSPDarkNet Backbone: 输出 P1-P5，支持宽度缩放."""

import torch.nn as nn
from models.common import Conv, C2f, SPPF


class CSPDarkNet(nn.Module):
    """CSPDarkNet with C2f blocks.

    Args:
        w: 宽度系数 (0.25/0.5/0.75/1.0), 0.5 约 2.8M 参数
        max_ch: 通道上限
    """

    def __init__(self, w=0.5, max_ch=512):
        super().__init__()
        ch = lambda x: min(int(x * w), max_ch)

        # Stem: /2 → /4
        self.stem = nn.Sequential(
            Conv(3, ch(32), 3, stride=2),          # /2, 320x320
            Conv(ch(32), ch(64), 3, stride=2),      # /4, 160x160
            C2f(ch(64), ch(64), n=1),
        )

        # Stage 3: /8, 80x80
        self.stage3 = nn.Sequential(
            Conv(ch(64), ch(128), 3, stride=2),
            C2f(ch(128), ch(128), n=2),
        )

        # Stage 4: /16, 40x40
        self.stage4 = nn.Sequential(
            Conv(ch(128), ch(256), 3, stride=2),
            C2f(ch(256), ch(256), n=2),
        )

        # Stage 5: /32, 20x20
        self.stage5 = nn.Sequential(
            Conv(ch(256), ch(512), 3, stride=2),
            C2f(ch(512), ch(512), n=1),
            SPPF(ch(512), ch(512)),
        )

        self.out_channels = [ch(32), ch(64), ch(128), ch(256), ch(512)]

    def forward(self, x):
        # P1: stem 第一层输出 (stride=2)
        p1 = self.stem[0](x)
        p2 = self.stem[1:](p1)                       # /4, 160x160
        p3 = self.stage3(p2)                         # /8,  80x80
        p4 = self.stage4(p3)                         # /16, 40x40
        p5 = self.stage5(p4)                         # /32, 20x20
        return p1, p2, p3, p4, p5
