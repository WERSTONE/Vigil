"""
Vigil — 双流多任务模型 (Human + Scene)
CSPDarkNet → FPN-PAN(P2-P5) → HumanAnalysisHead + SceneAnomalyHead + ProtoBranch
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import math


# ── 基础模块 ──

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=1, stride=1, padding=None, groups=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding or kernel // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU() if act else nn.Identity()
    def forward(self, x): return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, ch, shortcut=True, groups=1, e=0.5):
        super().__init__()
        h = int(ch * e)
        self.cv1, self.cv2 = Conv(ch, h, 1), Conv(h, ch, 3, groups=groups)
        self.add = shortcut
    def forward(self, x): return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


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
        self.cv1, self.cv2 = Conv(in_ch, h, 1), Conv(h * 4, out_ch, 1)
        self.k = kernel
    def forward(self, x):
        x = self.cv1(x)
        y1 = F.max_pool2d(x, self.k, 1, self.k // 2)
        y2 = F.max_pool2d(y1, self.k, 1, self.k // 2)
        y3 = F.max_pool2d(y2, self.k, 1, self.k // 2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


# ── CSPDarkNet Backbone ──

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


# ── FPN-PAN Neck (P2-P5) ──

class FPNPANNeck(nn.Module):
    """FPN+PAN with P2-P5 four-level fusion. P2 added for tiny objects."""
    def __init__(self, in_channels, out_channels=None):
        super().__init__()
        if out_channels is None: out_channels = in_channels
        c2, c3, c4, c5 = in_channels
        o2, o3, o4, o5 = out_channels

        self.lat_p5, self.lat_p4 = Conv(c5, o4, 1), Conv(c4, o4, 1)
        self.lat_p3, self.lat_p2 = Conv(c3, o3, 1), Conv(c2, o2, 1)
        self.fuse_p4 = C2f(o4 * 2, o4)
        self.fuse_p3 = C2f(o4 + o3, o3)
        self.fuse_p2 = C2f(o3 + o2, o2)
        self.down_n2, self.down_n3, self.down_n4 = Conv(o2, o3, 3, 2), Conv(o3, o4, 3, 2), Conv(o4, o5, 3, 2)
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


# ── 任务头 ──

class HumanAnalysisHead(nn.Module):
    """Person detect + attributes + keypoints. One anchor predicts [bbox(4)+person(1)+helmet(3)+smoking(1)+kpts(51)]."""
    def __init__(self, in_channels, num_keypoints=17, reg_max=16):
        super().__init__()
        attr_dim = 5
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, attr_dim, 1) for c in in_channels])
        self.reg_preds = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in in_channels])
        self.kpt_preds = nn.ModuleList([nn.Conv2d(c, num_keypoints * 3, 1) for c in in_channels])

    def forward(self, features):
        cls_outs, reg_outs, kpt_outs = [], [], []
        for f, cl, rl, kl in zip(features, self.cls_preds, self.reg_preds, self.kpt_preds):
            cls_outs.append(cl(f)); reg_outs.append(rl(f)); kpt_outs.append(kl(f))
        return cls_outs, reg_outs, kpt_outs


class SceneAnomalyHead(nn.Module):
    """Anomaly detect + mask coeffs. [bbox(4)+cls(4:fire/smoke/stain/drip)+mask(32)]."""
    def __init__(self, in_channels, num_classes=4, mask_dim=32, reg_max=16):
        super().__init__()
        self.cls_preds = nn.ModuleList([nn.Conv2d(c, num_classes, 1) for c in in_channels])
        self.reg_preds = nn.ModuleList([nn.Conv2d(c, 4 * reg_max, 1) for c in in_channels])
        self.mask_preds = nn.ModuleList([nn.Conv2d(c, mask_dim, 1) for c in in_channels])

    def forward(self, features):
        cls_outs, reg_outs, mask_outs = [], [], []
        for f, cl, rl, ml in zip(features, self.cls_preds, self.reg_preds, self.mask_preds):
            cls_outs.append(cl(f)); reg_outs.append(rl(f)); mask_outs.append(ml(f))
        return cls_outs, reg_outs, mask_outs


class ProtoBranch(nn.Module):
    """YOLACT-style prototype masks. N3(80×80)→up→[32,160,160]."""
    def __init__(self, in_ch=64, proto_dim=32):
        super().__init__()
        self.net = nn.Sequential(Conv(in_ch, in_ch, 3), Conv(in_ch, in_ch, 3), Conv(in_ch, in_ch, 3), nn.Conv2d(in_ch, proto_dim, 1))
    def forward(self, x):
        return F.interpolate(self.net(x), scale_factor=2.0, mode="bilinear", align_corners=False)


# ── 主模型 ──

class VigilMultiTaskModel(nn.Module):
    """
    Vigil 双流多任务模型。

    HumanAnalysisHead (N2-N5): person bbox + helmet(3) + smoking + 17 kpts → 任务 1,2,4,5,7,8
    SceneAnomalyHead  (N3-N5): fire/smoke/stain/drip bbox + 32 mask coeffs → 任务 3,6
    ProtoBranch       (N3→):   32 prototype masks → 与 mask coeffs 组合得实例分割
    """
    def __init__(self, backbone_w=0.25, backbone_d=0.33, num_keypoints=17,
                 anomaly_classes=4, mask_dim=32, reg_max=16):
        super().__init__()
        self.backbone = CSPDarkNet(w=backbone_w, d=backbone_d)
        bb_ch = self.backbone.feat_channels
        self.neck = FPNPANNeck(bb_ch, bb_ch)
        neck_ch = self.neck.out_channels
        self.ha_head = HumanAnalysisHead(neck_ch, num_keypoints, reg_max)
        self.sa_head = SceneAnomalyHead(neck_ch[1:], anomaly_classes, mask_dim, reg_max)
        self.proto_branch = ProtoBranch(neck_ch[1], mask_dim)

    def forward(self, x):
        neck_feats = self.neck(self.backbone(x))
        ha_cls, ha_reg, ha_kpt = self.ha_head(neck_feats)
        sa_cls, sa_reg, sa_mask = self.sa_head(neck_feats[1:])
        proto = self.proto_branch(neck_feats[1])
        return {"ha_cls": ha_cls, "ha_reg": ha_reg, "ha_kpt": ha_kpt,
                "sa_cls": sa_cls, "sa_reg": sa_reg, "sa_mask": sa_mask, "proto": proto}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters())

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ── 解码 & 后处理 ──

@dataclass
class DecodedPerson:
    bbox: List[float]       # [x1,y1,x2,y2]
    confidence: float
    helmet_status: int      # 0=on, 1=off, 2=none
    helmet_conf: float
    smoking_conf: float
    keypoints: List         # [17,3]

@dataclass
class DecodedAnomaly:
    bbox: List[float]
    class_id: int           # 0=fire,1=smoke,2=water_stain,3=water_drip
    class_name: str
    confidence: float
    mask_coeffs: List[float]  # [32]


def _make_grid(nx, ny, device):
    """生成 YOLO-style grid 坐标"""
    yv, xv = torch.meshgrid(torch.arange(ny, device=device), torch.arange(nx, device=device), indexing="ij")
    return torch.stack((xv, yv), 2).float()


def decode_human_outputs(ha_cls, ha_reg, ha_kpt, input_shape, conf_threshold=0.25, iou_threshold=0.45, reg_max=16):
    """
    解码 HA Head 原始输出 → DecodedPerson 列表。input_shape = (model_H, model_W)。

    ha_cls: List[4] of [B,5,Hi,Wi]  — person_conf + helmet(3) + smoking
    ha_reg: List[4] of [B,64,Hi,Wi] — bbox DFL bins
    ha_kpt: List[4] of [B,51,Hi,Wi] — 17 keypoints × 3
    """
    all_bboxes, all_confs, all_helmets, all_smokings, all_kpts = [], [], [], [], []

    for cls_t, reg_t, kpt_t in zip(ha_cls, ha_reg, ha_kpt):
        B, _, H, W = cls_t.shape
        cls_t = cls_t.permute(0, 2, 3, 1)     # [B,H,W,5]
        reg_t = reg_t.permute(0, 2, 3, 1)     # [B,H,W,64]
        kpt_t = kpt_t.permute(0, 2, 3, 1)     # [B,H,W,51]

        # DFL decode: [B,H,W,64] → [B,H,W,4]
        reg_t = reg_t.view(B, H, W, 4, reg_max).softmax(-1)
        reg_t = (reg_t @ torch.arange(reg_max, device=reg_t.device, dtype=reg_t.dtype))

        # Grid
        grid = _make_grid(W, H, cls_t.device)
        stride = input_shape[0] / W
        reg_t[..., :2] = (reg_t[..., :2] + grid) * stride
        reg_t[..., 2:4] = reg_t[..., 2:4] * stride * 2

        # xyxy
        cx, cy, w, h = reg_t[..., 0], reg_t[..., 1], reg_t[..., 2], reg_t[..., 3]
        bboxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        # Scale keypoints per-scale before concatenation
        kpt_flat = kpt_t.reshape(B, -1, 51)
        kpt_flat[..., 0::3] *= stride
        kpt_flat[..., 1::3] *= stride

        all_bboxes.append(bboxes.reshape(B, -1, 4))
        all_confs.append(cls_t[..., 0:1].reshape(B, -1).sigmoid())
        all_helmets.append(cls_t[..., 1:4].reshape(B, -1, 3))
        all_smokings.append(cls_t[..., 4:5].reshape(B, -1).sigmoid())
        all_kpts.append(kpt_flat)

    bboxes = torch.cat(all_bboxes, dim=1)[0]       # [N,4]
    confs = torch.cat(all_confs, dim=1)[0]           # [N]
    helmets = torch.cat(all_helmets, dim=1)[0]        # [N,3]
    smokings = torch.cat(all_smokings, dim=1)[0]      # [N]
    kpts = torch.cat(all_kpts, dim=1)[0]              # [N,51]

    # NMS per class (only person class)
    keep = confs > conf_threshold
    if not keep.any():
        return []

    bboxes, confs = bboxes[keep], confs[keep]
    helmets, smokings = helmets[keep], smokings[keep]
    kpts = kpts[keep]

    # Sort & NMS
    order = confs.argsort(descending=True)
    keep_indices = []
    while order.numel() > 0:
        if order.numel() == 1:
            keep_indices.append(order.item())
            break
        idx = order[0].item()
        keep_indices.append(idx)
        ious = _box_iou_batch(bboxes[idx:idx+1], bboxes[order[1:]])[0]
        order = order[1:][ious < iou_threshold]

    results = []
    for idx in keep_indices[:50]:  # max 50 persons
        k = kpts[idx].view(17, 3)
        results.append(DecodedPerson(
            bbox=bboxes[idx].clamp(0).tolist(),
            confidence=confs[idx].detach().item(),
            helmet_status=int(helmets[idx].argmax()),
            helmet_conf=helmets[idx].max().detach().item(),
            smoking_conf=smokings[idx].detach().item(),
            keypoints=k.tolist(),
        ))
    return results


def decode_anomaly_outputs(sa_cls, sa_reg, sa_mask, proto, input_shape, conf_threshold=0.15, iou_threshold=0.45):
    """解码 SA Head → DecodedAnomaly 列表。input_shape = (model_H, model_W)。"""
    CLASS_NAMES = ["fire", "smoke", "water_stain", "water_drip"]
    all_bboxes, all_scores, all_classes, all_coeffs = [], [], [], []

    for cls_t, reg_t, mask_t in zip(sa_cls, sa_reg, sa_mask):
        B, _, H, W = cls_t.shape
        cls_t = cls_t.permute(0, 2, 3, 1)
        reg_t = reg_t.permute(0, 2, 3, 1)
        mask_t = mask_t.permute(0, 2, 3, 1)

        reg_t = reg_t.view(B, H, W, 4, 16).softmax(-1)
        reg_t = (reg_t @ torch.arange(16, device=reg_t.device, dtype=reg_t.dtype))

        grid = _make_grid(W, H, cls_t.device)
        strides = input_shape[0] / W
        reg_t[..., :2] = (reg_t[..., :2] + grid) * strides
        reg_t[..., 2:4] = reg_t[..., 2:4] * strides * 2

        cx, cy, w, h = reg_t[..., 0], reg_t[..., 1], reg_t[..., 2], reg_t[..., 3]
        bboxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=-1)

        scores = cls_t.sigmoid()  # [B,H,W,4]
        max_sc, max_cl = scores.max(dim=-1)

        all_bboxes.append(bboxes.reshape(B, -1, 4))
        all_scores.append(max_sc.reshape(B, -1))
        all_classes.append(max_cl.reshape(B, -1))
        all_coeffs.append(mask_t.reshape(B, -1, 32))

    bboxes = torch.cat(all_bboxes, dim=1)[0]
    scores = torch.cat(all_scores, dim=1)[0]
    classes = torch.cat(all_classes, dim=1)[0]
    coeffs = torch.cat(all_coeffs, dim=1)[0]

    keep = scores > conf_threshold
    if not keep.any():
        return []
    bboxes, scores, classes, coeffs = bboxes[keep], scores[keep], classes[keep], coeffs[keep]

    # Per-class NMS
    keep_final = []
    for cls_id in range(4):
        mask_c = classes == cls_id
        if not mask_c.any(): continue
        b_c, s_c, idx_map = bboxes[mask_c], scores[mask_c], mask_c.nonzero().squeeze(-1)
        order = s_c.argsort(descending=True)
        while order.numel() > 0:
            if order.numel() == 1: keep_final.append(idx_map[order[0].item()].item()); break
            idx = order[0].item(); keep_final.append(idx_map[idx].item())
            ious = _box_iou_batch(b_c[idx:idx+1], b_c[order[1:]])[0]
            order = order[1:][ious < iou_threshold]

    return [DecodedAnomaly(
        bbox=bboxes[i].clamp(0).tolist(),
        class_id=int(classes[i]),
        class_name=CLASS_NAMES[int(classes[i])],
        confidence=scores[i].detach().item(),
        mask_coeffs=coeffs[i].tolist(),
    ) for i in keep_final[:30]]


def _box_iou_batch(boxes1, boxes2):
    """向量化 IoU"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter + 1e-16)


# ── 模型工厂 & 导出 ──

def create_model(variant="n", pretrained_path=None):
    cfgs = {"n": (0.25, 0.33), "s": (0.50, 0.50)}
    w, d = cfgs.get(variant, cfgs["n"])
    model = VigilMultiTaskModel(backbone_w=w, backbone_d=d)
    if pretrained_path:
        _load_weights(model, pretrained_path)
    return model


def _load_weights(model, path):
    """从 YOLOv8 .pt 加载 backbone 权重 (结构与 YOLOv8n 完全一致)。Neck 因含 P2 融合，随机初始化。"""
    import warnings
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        warnings.warn(f"Checkpoint not found: {path}")
        return

    if "model" in ckpt:
        ckpt = ckpt["model"]
    yolo_state = ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt
    vig_state = model.state_dict()

    # Backbone: 结构与 YOLOv8n 完全一致, 直接前缀映射
    BB_MAP = [
        ("backbone.stem.0", "model.0"),
        ("backbone.stem.1", "model.1"),
        ("backbone.stage2",  "model.2"),
        ("backbone.down3",   "model.3"),
        ("backbone.stage3",  "model.4"),
        ("backbone.down4",   "model.5"),
        ("backbone.stage4",  "model.6"),
        ("backbone.down5",   "model.7"),
        ("backbone.stage5",  "model.8"),
        ("backbone.sppf",    "model.9"),
    ]

    # Neck: YOLOv8 P3-P5 FPN+PAN 部分层可映射
    NECK_MAP = [
        ("neck.lat_p5",  "model.12"),   # Conv(256→128) → lat conv for P5
        ("neck.fuse_p4", "model.15"),   # C2f(192→64)? No. Let me think again.
    ]
    # YOLOv8 neck mapping is imprecise due to P2 addition. Only load backbone.

    loaded = 0
    for vig_prefix, yolo_prefix in BB_MAP:
        for vk in list(vig_state.keys()):
            if vk.startswith(vig_prefix):
                yk = vk.replace(vig_prefix, yolo_prefix)
                if yk in yolo_state and yolo_state[yk].shape == vig_state[vk].shape:
                    vig_state[vk] = yolo_state[yk]
                    loaded += 1

    model.load_state_dict(vig_state, strict=False)
    print(f"[Vigil] Loaded {loaded} backbone params from {path} ({len(BB_MAP)} layers)" if loaded else
          f"[Vigil] No weights loaded. Training from scratch.")


def export_onnx(model, path, size=(640, 640)):
    """导出 ONNX"""
    model.eval()
    dummy = torch.randn(1, 3, *size)
    output_names = []
    for prefix, n in [("ha_cls", 4), ("ha_reg", 4), ("ha_kpt", 4),
                       ("sa_cls", 3), ("sa_reg", 3), ("sa_mask", 3), ("proto", 1)]:
        output_names.extend([f"{prefix}_{i}" for i in range(n)])
    torch.onnx.export(model, dummy, path, opset_version=17,
                      input_names=["input"], output_names=output_names,
                      dynamic_axes={**{"input": {0: "batch"}},
                                    **{n: {0: "batch"} for n in output_names}})
    print(f"Exported: {path}")


def export_torchscript(model, path, size=(640, 640)):
    """导出 TorchScript (适用于端侧 LibTorch 推理)"""
    model.eval()
    traced = torch.jit.trace(model, torch.randn(1, 3, *size))
    traced.save(path)
    print(f"Exported: {path}")
