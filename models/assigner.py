"""中心点分配器: GT → FPN 层级 → 格点."""

import torch


class CenterAssigner:
    """FCOS-style 中心点分配.

    分配规则:
        - GT 中心落在格点 radius 范围内 → 正样本
        - 不同尺度 GT 分配到不同 FPN 层级 (按 max(l,t,r,b))
        - 返回每层的正样本索引用于损失计算
    """

    def __init__(self, strides, radius=1.5):
        self.strides = strides
        self.radius = radius
        self.num_levels = len(strides)
        # 各层级负责的 ltrb 范围 (原图像素)
        # P2 负责 0~64, P3 负责 64~128, ...
        self.level_ranges = []
        for i, s in enumerate(strides):
            if i == 0:
                self.level_ranges.append((0, s * 8))
            else:
                self.level_ranges.append(
                    (strides[i-1] * 8, s * 8))

    def _get_level(self, ltrb_max):
        """根据 max(l,t,r,b) 选择 FPN 层级."""
        level = torch.zeros_like(ltrb_max, dtype=torch.long)
        for i, (lo, hi) in enumerate(self.level_ranges):
            if i == self.num_levels - 1:
                level[ltrb_max >= lo] = i
            else:
                level[(ltrb_max >= lo) & (ltrb_max < hi)] = i
        return level

    def __call__(self, gt_boxes, gt_classes, gt_attrs, feat_sizes):
        """
        Args:
            gt_boxes:   List[[M_i, 4]]      xyxy 像素坐标
            gt_classes: List[[M_i]]          0=person, 1=fire, 2=water
            gt_attrs:   List[dict]           kpts/helmet/smoking (可为 None)
            feat_sizes: List[(H, W)]         各 FPN 层特征图尺寸

        Returns:
            targets_per_level: List[dict or None], 每层包含:
                grid_xy, gt_boxes, gt_classes, gt_kpts, gt_helmet, gt_smoking, batch_idx
        """
        B = len(gt_boxes)
        targets = [{
            "grid_xy": [], "gt_boxes": [], "gt_classes": [],
            "gt_kpts": [], "gt_helmet": [], "gt_smoking": [], "batch_idx": [],
        } for _ in range(self.num_levels)]

        for b in range(B):
            if len(gt_boxes[b]) == 0:
                continue

            boxes = gt_boxes[b]        # [M, 4]
            classes = gt_classes[b]     # [M]
            attrs = gt_attrs[b] if b < len(gt_attrs) else {}

            w = boxes[:, 2] - boxes[:, 0]
            h = boxes[:, 3] - boxes[:, 1]
            # 用 max(w, h) 近似 max(ltrb)，作为尺度代理
            max_side = torch.max(w, h)
            levels = self._get_level(max_side)

            for lvl in range(self.num_levels):
                mask = levels == lvl
                if not mask.any():
                    continue

                lvl_boxes = boxes[mask]       # [K, 4]
                lvl_cls = classes[mask]        # [K]
                H, W = feat_sizes[lvl]
                stride = self.strides[lvl]

                # 中心点 → 格点
                cx = (lvl_boxes[:, 0] + lvl_boxes[:, 2]) / 2 / stride
                cy = (lvl_boxes[:, 1] + lvl_boxes[:, 3]) / 2 / stride
                gx = cx.floor().long().clamp(0, W - 1)
                gy = cy.floor().long().clamp(0, H - 1)

                for dx in range(-int(self.radius), int(self.radius) + 1):
                    for dy in range(-int(self.radius), int(self.radius) + 1):
                        nx = (gx + dx).clamp(0, W - 1)
                        ny = (gy + dy).clamp(0, H - 1)

                        targets[lvl]["grid_xy"].append(torch.stack([nx, ny], dim=1))
                        targets[lvl]["gt_boxes"].append(lvl_boxes)
                        targets[lvl]["gt_classes"].append(lvl_cls)
                        targets[lvl]["batch_idx"].append(
                            torch.full((len(lvl_cls),), b, dtype=torch.long))

                        # 人体属性
                        k = attrs.get("kpts")
                        targets[lvl]["gt_kpts"].append(k[mask] if k is not None else None)
                        hlm = attrs.get("helmet")
                        targets[lvl]["gt_helmet"].append(hlm[mask] if hlm is not None else None)
                        smk = attrs.get("smoking")
                        targets[lvl]["gt_smoking"].append(smk[mask] if smk is not None else None)

        # 合并每层
        merged = []
        for lvl in range(self.num_levels):
            t = targets[lvl]
            if len(t["grid_xy"]) == 0:
                merged.append(None)
                continue
            merged.append({
                "grid_xy": torch.cat(t["grid_xy"], dim=0),
                "gt_boxes": torch.cat(t["gt_boxes"], dim=0),
                "gt_classes": torch.cat(t["gt_classes"], dim=0),
                "gt_kpts": torch.cat([x for x in t["gt_kpts"] if x is not None], dim=0)
                    if any(x is not None for x in t["gt_kpts"]) else None,
                "gt_helmet": torch.cat([x for x in t["gt_helmet"] if x is not None], dim=0)
                    if any(x is not None for x in t["gt_helmet"]) else None,
                "gt_smoking": torch.cat([x for x in t["gt_smoking"] if x is not None], dim=0)
                    if any(x is not None for x in t["gt_smoking"]) else None,
                "batch_idx": torch.cat(t["batch_idx"], dim=0),
            })
        return merged
