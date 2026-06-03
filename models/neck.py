import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common import Conv, C2f


class FPNPANNeck(nn.Module):
    """FPN+PAN with P2-P5 four-level fusion. P2 added for tiny objects."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        c2, c3, c4, c5 = in_channels
        o2, o3, o4, o5 = out_channels

        self.lat_p5 = Conv(c5, o4, 1)
        self.lat_p4 = Conv(c4, o4, 1)
        self.lat_p3 = Conv(c3, o3, 1)
        self.lat_p2 = Conv(c2, o2, 1)
        self.fuse_p4 = C2f(o4 * 2, o4)
        self.fuse_p3 = C2f(o4 + o3, o3)
        self.fuse_p2 = C2f(o3 + o2, o2)
        self.down_n2 = Conv(o2, o3, 3, 2)
        self.down_n3 = Conv(o3, o4, 3, 2)
        self.down_n4 = Conv(o4, o5, 3, 2)
        self.fuse_n3 = C2f(o3 * 2, o3)
        self.fuse_n4 = C2f(o4 * 2, o4)
        self.fuse_n5 = C2f(o4 + o5, o5)
        self.out_channels = out_channels

    def forward(self, features):
        p2, p3, p4, p5 = features
        p5_lat = self.lat_p5(p5)
        p4_lat = self.lat_p4(p4)
        p3_lat = self.lat_p3(p3)
        p2_lat = self.lat_p2(p2)

        up_p5 = F.interpolate(p5_lat, size=p4.shape[2:], mode="nearest")
        n4_td = self.fuse_p4(torch.cat([p4_lat, up_p5], dim=1))
        up_n4 = F.interpolate(n4_td, size=p3.shape[2:], mode="nearest")
        n3_td = self.fuse_p3(torch.cat([p3_lat, up_n4], dim=1))
        up_n3 = F.interpolate(n3_td, size=p2.shape[2:], mode="nearest")
        n2_td = self.fuse_p2(torch.cat([p2_lat, up_n3], dim=1))

        d2 = self.down_n2(n2_td)
        n3 = self.fuse_n3(torch.cat([n3_td, d2], dim=1))
        d3 = self.down_n3(n3)
        n4 = self.fuse_n4(torch.cat([n4_td, d3], dim=1))
        d4 = self.down_n4(n4)
        n5 = self.fuse_n5(torch.cat([p5_lat, d4], dim=1))
        return [n2_td, n3, n4, n5]
