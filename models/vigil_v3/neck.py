"""增强 Gather-Distribute Neck — SE 通道注意力注入.

延续 Gold-YOLO 的 Gather→Inject→PAN 结构，但注入步骤加入 SE 通道注意力，
让各级自适应地选择全局信息中最相关的通道。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, C2f


class SE(nn.Module):
    """Squeeze-and-Excitation 通道注意力."""

    def __init__(self, ch, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(ch, ch // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch // reduction, ch, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.fc(x)


class InjectWithSE(nn.Module):
    """注入块: lateral + global_info → SE 加权 → C2f 融合."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.se = SE(in_ch * 2)
        self.fuse = C2f(in_ch * 2, out_ch, n=1)

    def forward(self, lateral, global_info):
        x = torch.cat([lateral, global_info], dim=1)
        x = self.se(x)
        return self.fuse(x)


class AttentionGDNeck(nn.Module):
    """增强 Gather-Distribute Neck.

    Args:
        in_channels: [c3, c4, c5] backbone 输出通道
        out_ch: 统一输出通道
    """

    def __init__(self, in_channels, out_ch=256):
        super().__init__()
        c3, c4, c5 = in_channels

        # Lateral convs
        self.lat_p5 = Conv(c5, out_ch, 1)
        self.lat_p4 = Conv(c4, out_ch, 1)
        self.lat_p3 = Conv(c3, out_ch, 1)

        # Gather: 汇聚到统一尺度
        self.gather = Conv(out_ch * 3, out_ch, 1)

        # Inject with SE
        self.inject_p3 = InjectWithSE(out_ch, out_ch)
        self.inject_p4 = InjectWithSE(out_ch, out_ch)
        self.inject_p5 = InjectWithSE(out_ch, out_ch)

        # PAN bottom-up
        self.down_p3 = Conv(out_ch, out_ch, 3, stride=2)
        self.down_p4 = Conv(out_ch, out_ch, 3, stride=2)
        self.fuse_p4 = C2f(out_ch * 2, out_ch, n=1)
        self.fuse_p5 = C2f(out_ch * 2, out_ch, n=1)

        self.out_channels = [out_ch] * 3

    def forward(self, feats):
        p3, p4, p5 = feats

        # ── Lateral ──
        n3 = self.lat_p3(p3)   # [B, C, 80, 80]
        n4 = self.lat_p4(p4)   # [B, C, 40, 40]
        n5 = self.lat_p5(p5)   # [B, C, 20, 20]

        # ── Gather at 40×40 ──
        target_size = n4.shape[2:]
        g3 = F.interpolate(n3, size=target_size, mode="bilinear", align_corners=False)
        g5 = F.interpolate(n5, size=target_size, mode="bilinear", align_corners=False)
        global_info = self.gather(torch.cat([g3, n4, g5], dim=1))

        # ── Inject with SE ──
        gi3 = F.interpolate(global_info, size=n3.shape[2:], mode="bilinear", align_corners=False)
        gi5 = F.interpolate(global_info, size=n5.shape[2:], mode="bilinear", align_corners=False)

        m3 = self.inject_p3(n3, gi3)
        m4 = self.inject_p4(n4, global_info)
        m5 = self.inject_p5(n5, gi5)

        # ── PAN bottom-up ──
        p3_out = m3
        p4_out = self.fuse_p4(torch.cat([m4, self.down_p3(p3_out)], dim=1))
        p5_out = self.fuse_p5(torch.cat([m5, self.down_p4(p4_out)], dim=1))

        return [p3_out, p4_out, p5_out]
