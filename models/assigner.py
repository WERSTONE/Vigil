import torch
import torch.nn as nn


def box_iou(boxes1, boxes2):
    """向量化 IoU [M,4] × [N,4] → [M,N]"""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter + 1e-16)


class TaskAlignedAssigner(nn.Module):
    """
    YOLOv8-style TaskAlignedAssigner.
    对每个 GT 框，综合 cls_score 和 IoU 选出最匹配的预测作为正样本。
    """
    def __init__(self, topk=13, eps=1e-9):
        super().__init__()
        self.topk = topk
        self.eps = eps

    @torch.no_grad()
    def forward(self, pred_boxes, pred_scores, gt_boxes):
        """
        pred_boxes: [N, 4]  xyxy, 所有尺度拼接后的预测框
        pred_scores: [N]     person confidence (已 sigmoid)
        gt_boxes: [M, 4]    GT 框 xyxy

        Returns:
            fg_mask: [N]         正样本 mask
            matched_gt_idx: [N]  每个正样本对应的 GT 索引 (负样本为 -1)
            target_boxes: [N, 4] 分配的目标框 (负样本为零)
        """
        N, M = pred_boxes.shape[0], gt_boxes.shape[0]

        fg_mask = torch.zeros(N, dtype=torch.bool)
        matched_gt_idx = torch.full((N,), -1, dtype=torch.long)
        target_boxes = torch.zeros_like(pred_boxes)

        if M == 0:
            return fg_mask, matched_gt_idx, target_boxes

        ious = box_iou(gt_boxes, pred_boxes)                     # [M, N]
        topk_iou, topk_idx = ious.topk(min(self.topk, N), dim=1) # [M, topk]

        topk_scores = pred_scores[topk_idx]                      # [M, topk]
        align_metrics = (topk_scores * topk_iou).sqrt()          # [M, topk]
        threshold = align_metrics.mean()                         # 动态阈值

        for gt_i in range(M):
            mask = align_metrics[gt_i] > threshold
            for k in range(topk_idx.shape[1]):
                if not mask[k]:
                    continue
                pred_i = topk_idx[gt_i, k].item()
                if fg_mask[pred_i]:
                    continue  # 已被其他 GT 占用
                fg_mask[pred_i] = True
                matched_gt_idx[pred_i] = gt_i
                target_boxes[pred_i] = gt_boxes[gt_i]

        return fg_mask, matched_gt_idx, target_boxes
