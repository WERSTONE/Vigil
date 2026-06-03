from models.common import Conv, C2f, SPPF
import torch.nn as nn


class CSPDarkNet(nn.Module):
    """YOLOv8n-compatible backbone. Outputs P2(/4), P3(/8), P4(/16), P5(/32)."""
    def __init__(self, w=0.25, d=0.33, max_ch=1024):
        super().__init__()
        ch = lambda x: min(int(x * w), max_ch)
        n = lambda x: max(1, int(x * d))
        sm = ch(max_ch)

        self.stem = nn.Sequential(Conv(3, ch(64), 3, 2), Conv(ch(64), ch(128), 3, 2))
        self.stage2 = C2f(ch(128), ch(128), n(3))
        self.down3 = Conv(ch(128), ch(256), 3, 2)
        self.stage3 = C2f(ch(256), ch(256), n(6))
        self.down4 = Conv(ch(256), ch(512), 3, 2)
        self.stage4 = C2f(ch(512), ch(512), n(6))
        self.down5 = Conv(ch(512), sm, 3, 2)
        self.stage5 = C2f(sm, sm, n(3))
        self.sppf = SPPF(sm, sm)
        self.feat_channels = [ch(128), ch(256), ch(512), sm]

    def forward(self, x):
        x = self.stem(x)
        p2 = self.stage2(x)
        p3 = self.stage3(self.down3(p2))
        p4 = self.stage4(self.down4(p3))
        p5 = self.sppf(self.stage5(self.down5(p4)))
        return [p2, p3, p4, p5]
