import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding or kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ch, shortcut=True, groups=1, e=0.5):
        super().__init__()
        h = int(ch * e)
        self.cv1 = Conv(ch, h, 1)
        self.cv2 = Conv(h, ch, 3, groups=groups)
        self.add = shortcut

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C2f(nn.Module):
    def __init__(self, in_ch, out_ch, n=1, shortcut=True, groups=1, e=0.5):
        super().__init__()
        self.c = int(out_ch * e)
        self.cv1 = Conv(in_ch, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, out_ch, 1)
        self.m = nn.ModuleList([Bottleneck(self.c, shortcut, groups, e=1.0) for _ in range(n)])

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, dim=1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, dim=1))


class SPPF(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5):
        super().__init__()
        h = in_ch // 2
        self.cv1 = Conv(in_ch, h, 1)
        self.cv2 = Conv(h * 4, out_ch, 1)
        self.k = kernel

    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = F.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = F.max_pool2d(y2, self.k, 1, self.k // 2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))
