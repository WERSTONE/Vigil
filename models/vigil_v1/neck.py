"""FPN+PAN Neck: 双向特征融合 (YOLOv8 style)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, C2f


class FPNPANNeck(nn.Module):
    """4 级 FPN+PAN, 输入 P2-P5, 输出 4 级融合特征.

    Args:
        in_channels: [c2, c3, c4, c5] backbone 输出
        out_ch: 统一输出通道数
    """

    def __init__(self, in_channels, out_ch=128):
        super().__init__()
        c2, c3, c4, c5 = in_channels

        # FPN: lateral convs
        self.lat_p5 = Conv(c5, out_ch, 1)
        self.lat_p4 = Conv(c4, out_ch, 1)
        self.lat_p3 = Conv(c3, out_ch, 1)
        self.lat_p2 = Conv(c2, out_ch, 1)

        # FPN: top-down fusion
        self.fpn_p4 = C2f(out_ch * 2, out_ch, n=1)
        self.fpn_p3 = C2f(out_ch * 2, out_ch, n=1)
        self.fpn_p2 = C2f(out_ch * 2, out_ch, n=1)

        # PAN: downsample
        self.down_p2 = Conv(out_ch, out_ch, 3, stride=2)
        self.down_p3 = Conv(out_ch, out_ch, 3, stride=2)
        self.down_p4 = Conv(out_ch, out_ch, 3, stride=2)

        # PAN: bottom-up fusion
        self.pan_p3 = C2f(out_ch * 2, out_ch, n=1)
        self.pan_p4 = C2f(out_ch * 2, out_ch, n=1)
        self.pan_p5 = C2f(out_ch * 2, out_ch, n=1)

        self.out_channels = [out_ch] * 4

    def forward(self, feats):
        p2, p3, p4, p5 = feats

        # FPN top-down
        n5 = self.lat_p5(p5)
        n4 = self.fpn_p4(torch.cat(
            [self.lat_p4(p4), F.interpolate(n5, size=p4.shape[2:], mode="nearest")], dim=1))
        n3 = self.fpn_p3(torch.cat(
            [self.lat_p3(p3), F.interpolate(n4, size=p3.shape[2:], mode="nearest")], dim=1))
        n2 = self.fpn_p2(torch.cat(
            [self.lat_p2(p2), F.interpolate(n3, size=p2.shape[2:], mode="nearest")], dim=1))

        # PAN bottom-up
        m2 = n2
        m3 = self.pan_p3(torch.cat([n3, self.down_p2(m2)], dim=1))
        m4 = self.pan_p4(torch.cat([n4, self.down_p3(m3)], dim=1))
        m5 = self.pan_p5(torch.cat([n5, self.down_p4(m4)], dim=1))

        return [m2, m3, m4, m5]
