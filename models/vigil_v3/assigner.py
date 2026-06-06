"""TaskAlignedAssigner v3 — 降低 IoU 权重 + 可选 beta 预热.

v3 改进: beta 从 6.0 降到 2.0，IoU² 替代 IoU⁶，让训练初期框回归尚未收敛时
alignment 更多由 cls 分数主导，避免正样本选择退化为随机噪声。

可选的 beta_warmup: 训练初期 beta 从 1.0 线性爬升到目标值，
进一步缓解冷启动。
"""

import torch


def _box_iou(box1, box2):
    lt = torch.max(box1[:, None, :2], box2[None, :, :2])
    rb = torch.min(box1[:, None, 2:], box2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    return inter / (area1[:, None] + area2[None, :] - inter + 1e-16)


class TaskAlignedAssignerV3:
    """动态 top-k 正样本分配器 (beta 降低版 + 可选预热).

    alignment = cls^α × IoU^β
    beta=2.0 (vs v2 的 6.0) → IoU 的影响更温和，冷启动友好。

    Args:
        topk: 每个 GT 的正样本数
        alpha: 分类权重指数
        beta: IoU 权重指数 (v3 默认 2.0)
        center_radius: 中心约束半径, -1=框内严格模式
        beta_warmup_epochs: beta 从 1.0 爬到 beta 的 epoch 数, 0=不预热
    """

    def __init__(self, topk=13, alpha=1.0, beta=2.0, center_radius=-1,
                 beta_warmup_epochs=0):
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.center_radius = center_radius
        self.beta_warmup_epochs = beta_warmup_epochs
        self._current_epoch = 0

    @property
    def current_beta(self):
        if self.beta_warmup_epochs <= 0 or self._current_epoch >= self.beta_warmup_epochs:
            return self.beta
        progress = self._current_epoch / max(1, self.beta_warmup_epochs)
        return 1.0 + (self.beta - 1.0) * progress

    def set_epoch(self, epoch):
        self._current_epoch = epoch

    def _build_level_mask(self, num_levels, strides):
        ranges = []
        for i, s in enumerate(strides):
            if i == 0:
                ranges.append((0, s * 8))
            else:
                ranges.append((strides[i - 1] * 8, s * 8))
        return ranges

    def __call__(self, pred_scores, pred_boxes,
                 gt_boxes, gt_classes, gt_attrs,
                 feat_sizes, strides):
        device = gt_boxes.device
        num_levels = len(feat_sizes)
        num_gts = len(gt_boxes)
        beta = self.current_beta

        if num_gts == 0:
            return [None] * num_levels

        level_ranges = self._build_level_mask(num_levels, strides)

        targets = [{
            "grid_xy": [], "gt_boxes": [], "gt_classes": [],
            "gt_kpts": [], "gt_helmet": [], "gt_smoking": [], "batch_idx": [],
        } for _ in range(num_levels)]

        # ── 拼接所有 level ──
        offsets, level_W = [], []
        total_N = 0
        for lvl, (H, W) in enumerate(feat_sizes):
            offsets.append(total_N)
            level_W.append(W)
            total_N += H * W

        all_scores = torch.cat([s.view(1, -1, 3) for s in pred_scores], dim=1)
        all_boxes = torch.cat([b.view(1, -1, 4) for b in pred_boxes], dim=1)

        # ── 全局格点中心 ──
        all_centers = []
        for lvl, (H, W) in enumerate(feat_sizes):
            stride = strides[lvl]
            yv, xv = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device), indexing="ij")
            cx = (xv.float() + 0.5) * stride
            cy = (yv.float() + 0.5) * stride
            all_centers.append(torch.stack([cx.flatten(), cy.flatten()], dim=1))
        all_centers = torch.cat(all_centers, dim=0)

        gt_cls = gt_classes.long()
        n_person = (gt_cls == 0).sum().item()

        person_count = 0
        for gt_i in range(num_gts):
            gt_box = gt_boxes[gt_i]
            gt_cls_i = gt_cls[gt_i]

            # ── Center-radius 约束 ──
            if self.center_radius >= 0:
                gt_cx = (gt_box[0] + gt_box[2]) / 2
                gt_cy = (gt_box[1] + gt_box[3]) / 2
                max_dist = self.center_radius * strides[0]
                dist = (all_centers - torch.tensor([gt_cx, gt_cy], device=device)).norm(dim=1)
                center_mask = dist <= max_dist
            else:
                center_mask = (
                    (all_centers[:, 0] >= gt_box[0]) &
                    (all_centers[:, 0] <= gt_box[2]) &
                    (all_centers[:, 1] >= gt_box[1]) &
                    (all_centers[:, 1] <= gt_box[3])
                )

            # ── 尺寸级别过滤 ──
            gt_w, gt_h = (gt_box[2] - gt_box[0]).item(), (gt_box[3] - gt_box[1]).item()
            max_side = max(gt_w, gt_h)

            valid_levels = set()
            for lvl, (lo, hi) in enumerate(level_ranges):
                if lvl == num_levels - 1:
                    if max_side >= lo:
                        valid_levels.add(lvl)
                elif lo <= max_side < hi:
                    valid_levels.add(lvl)
            adjacent = set()
            for lvl in valid_levels:
                if lvl > 0: adjacent.add(lvl - 1)
                if lvl < num_levels - 1: adjacent.add(lvl + 1)
            valid_levels |= adjacent
            if not valid_levels:
                valid_levels.add(num_levels - 1 if max_side >= level_ranges[-1][0] else 0)

            level_mask = torch.zeros(total_N, dtype=torch.bool, device=device)
            for lvl in valid_levels:
                start = offsets[lvl]
                end = start + feat_sizes[lvl][0] * feat_sizes[lvl][1]
                level_mask[start:end] = True

            valid_mask = center_mask & level_mask
            if valid_mask.sum().item() == 0:
                valid_mask = center_mask
                if valid_mask.sum().item() == 0:
                    continue

            # ── Alignment: cls^α × IoU^β (beta 降低, 冷启动友好) ──
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            cls_valid = all_scores[0, valid_indices, gt_cls_i]
            ious_valid = _box_iou(all_boxes[0, valid_indices], gt_box.unsqueeze(0)).squeeze(-1)
            align_valid = cls_valid.pow(self.alpha) * ious_valid.pow(beta)

            topk = min(self.topk, valid_indices.numel())
            _, topk_local = align_valid.topk(topk)

            # ── 提取属性 ──
            gt_kpt_val = gt_helm_val = gt_smoke_val = None
            if gt_cls_i == 0 and n_person > 0:
                if gt_attrs and "kpts" in gt_attrs:
                    gt_kpt_val = gt_attrs["kpts"][person_count]
                if gt_attrs and "helmet" in gt_attrs:
                    gt_helm_val = gt_attrs["helmet"][person_count]
                if gt_attrs and "smoking" in gt_attrs:
                    gt_smoke_val = gt_attrs["smoking"][person_count]
                person_count += 1

            for k in range(topk):
                global_idx = valid_indices[topk_local[k]].item()
                for lvl in range(num_levels - 1, -1, -1):
                    if global_idx >= offsets[lvl]:
                        break
                W_l = level_W[lvl]
                local_idx = global_idx - offsets[lvl]
                gx, gy = local_idx % W_l, local_idx // W_l

                t = targets[lvl]
                t["grid_xy"].append(torch.tensor([gx, gy], device=device))
                t["gt_boxes"].append(gt_box)
                t["gt_classes"].append(gt_cls_i)
                t["batch_idx"].append(0)
                t["gt_kpts"].append(gt_kpt_val)
                t["gt_helmet"].append(gt_helm_val)
                t["gt_smoking"].append(gt_smoke_val)

        # ── 合并每层 ──
        merged = []
        for lvl in range(num_levels):
            t = targets[lvl]
            if len(t["grid_xy"]) == 0:
                merged.append(None)
                continue
            merged.append({
                "grid_xy": torch.stack(t["grid_xy"], dim=0),
                "gt_boxes": torch.stack(t["gt_boxes"], dim=0),
                "gt_classes": torch.stack(t["gt_classes"], dim=0),
                "gt_kpts": (torch.stack([x for x in t["gt_kpts"] if x is not None])
                            if any(x is not None for x in t["gt_kpts"]) else None),
                "gt_helmet": (torch.stack([x for x in t["gt_helmet"] if x is not None])
                              if any(x is not None for x in t["gt_helmet"]) else None),
                "gt_smoking": (torch.stack([x for x in t["gt_smoking"] if x is not None])
                               if any(x is not None for x in t["gt_smoking"]) else None),
                "batch_idx": torch.tensor(t["batch_idx"], device=device),
            })
        return merged
