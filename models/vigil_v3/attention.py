"""Area Attention + R-ELAN — YOLOv12 核心模块.

Area Attention: 将特征图沿 H/W 方向切分为 region，区域内独立自注意力。
  计算量 O(HW·d²·HW/l²) vs 全局 O(H²W²·d)，降低约 l 倍。
  - Conv2d+BN 替代 Linear+LN (部署友好)
  - ReLU 预激活 (YOLOv12 发现这比 post-activation 更稳定)
  - 7×7 depthwise conv 隐式编码位置信息 (无需显式位置编码)

R-ELAN: 残差高效层聚合网络，解决注意力模块在大模型中的梯度阻塞问题。
  - 瓶颈结构 → 多分支 → concat → 1×1 融合
  - 残差连接 + 缩放因子 0.01 (类 LayerScale)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════════════════════
# Area Attention
# ═══════════════════════════════════════════════════════════════

class AreaAttention(nn.Module):
    """YOLOv12 Area Attention.

    将特征图沿一个维度切分为 ``area`` 个区域，每个区域内独立执行多头自注意力。
    相邻 block 交替使用 vertical (切 H) 和 horizontal (切 W) 分区。

    Args:
        dim: 输入/输出通道数
        num_heads: 注意力头数
        area: 区域切分数 (实际取 min(area, H, W))
        partition: "vertical" (切 H) 或 "horizontal" (切 W)
    """

    def __init__(self, dim, num_heads=8, area=4, partition="vertical"):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} 必须被 num_heads {num_heads} 整除"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.area = area
        self.partition = partition
        self.scale = self.head_dim ** -0.5

        # 位置感知: 7×7 depthwise conv 隐式注入位置信息
        self.pos = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
        )

        # QKV 投影: Conv2d+BN (替代 Linear+LN)
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.qkv_bn = nn.BatchNorm2d(dim * 3)

        # 输出投影
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.proj_bn = nn.BatchNorm2d(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        area = min(self.area, H if self.partition == "vertical" else W)
        area = max(area, 1)

        # ── 位置感知 + 残差 ──
        identity = x
        pos = self.pos(x)
        pos = F.relu(pos)
        x = x + pos

        # ── QKV 投影 + ReLU 预激活 ──
        qkv = self.qkv(x)
        qkv = self.qkv_bn(qkv)
        qkv = F.relu(qkv)
        q, k, v = qkv.chunk(3, dim=1)                  # each [B, C, H, W]

        # ── 重塑为 [B, heads, N, head_dim] ──
        N = H * W
        q = q.reshape(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        k = k.reshape(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        v = v.reshape(B, self.num_heads, self.head_dim, N).transpose(2, 3)
        # shape: [B, heads, N, head_dim]

        # ── 区域切分 (沿 H 或 W) ──
        if self.partition == "vertical":
            # 将 H 切为 area 份: [B, heads, H*W, d] → [B*area, heads, (H/area)*W, d]
            h_per = H // area
            def divide(t):
                t = t.reshape(B, self.num_heads, area, h_per, W, self.head_dim)
                t = t.permute(0, 2, 1, 3, 4, 5)
                return t.reshape(B * area, self.num_heads, h_per * W, self.head_dim)

            def merge(t):
                t = t.reshape(B, area, self.num_heads, h_per, W, self.head_dim)
                t = t.permute(0, 2, 1, 3, 4, 5)
                return t.reshape(B, self.num_heads, H * W, self.head_dim)
        else:  # horizontal
            w_per = W // area
            def divide(t):
                t = t.reshape(B, self.num_heads, H, area, w_per, self.head_dim)
                t = t.permute(0, 1, 3, 2, 4, 5)
                return t.reshape(B * area, self.num_heads, H * w_per, self.head_dim)

            def merge(t):
                t = t.reshape(B, area, self.num_heads, H, w_per, self.head_dim)
                t = t.permute(0, 2, 1, 3, 4, 5)
                return t.reshape(B, self.num_heads, H * W, self.head_dim)

        q, k, v = divide(q), divide(k), divide(v)

        # ── 区域内注意力 ──
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B*area, heads, N_per, N_per]
        attn = attn.softmax(dim=-1)
        out = attn @ v                                    # [B*area, heads, N_per, d]

        out = merge(out)                                  # [B, heads, N, d]

        # ── 重塑回 [B, C, H, W] ──
        out = out.transpose(2, 3).reshape(B, C, H, W)

        # ── 输出投影 ──
        out = self.proj(out)
        out = self.proj_bn(out)
        out = F.relu(out)

        return out + identity


# ═══════════════════════════════════════════════════════════════
# R-ELAN (Residual Efficient Layer Aggregation Network)
# ═══════════════════════════════════════════════════════════════

class RELAN(nn.Module):
    """R-ELAN block — 集成 Area Attention 的残差高效层聚合网络。

    结构:
        identity = Transition(input)          # 1×1 对齐通道
        b1 = Conv → AreaAttn(vert) → Conv     # 分支 1
        b2 = Conv → AreaAttn(horiz) → Conv    # 分支 2
        out = Fuse(concat(identity, b1, b2))  # 拼接融合
        out = out + scale * identity           # 残差缩放 (类 LayerScale)

    Args:
        in_ch: 输入通道
        out_ch: 输出通道
        num_heads: 注意力头数
        area: 区域切分数
        scale: 残差缩放因子 (YOLOv12 默认 0.01)
    """

    def __init__(self, in_ch, out_ch, num_heads=8, area=4, scale=0.01):
        super().__init__()
        mid_ch = out_ch // 2
        self.scale_val = scale

        # 过渡层
        self.transition = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

        # 分支 1: vertical area attention
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            AreaAttention(mid_ch, num_heads, area, partition="vertical"),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )

        # 分支 2: horizontal area attention
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
            AreaAttention(mid_ch, num_heads, area, partition="horizontal"),
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_ch),
            nn.SiLU(inplace=True),
        )

        # 融合
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch + mid_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        identity = self.transition(x)
        b1 = self.branch1(identity)
        b2 = self.branch2(identity)
        fused = self.fuse(torch.cat([identity, b1, b2], dim=1))
        return fused + self.scale_val * identity
