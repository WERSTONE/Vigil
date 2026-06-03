import cv2
import numpy as np
import torch
from typing import List, Tuple, Optional


MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def letterbox(img, target_size=640, fill=114):
    """等比缩放 + 填充到固定 target_size × target_size。"""
    h, w = img.shape[:2]
    r = target_size / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = target_size - new_w
    dh = target_size - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return img, r, (left, top)


def random_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """HSV 随机扰动，增加颜色不变性。"""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    rng = np.random
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-1, 1) * hgain * 180) % 360
    hsv[..., 1] *= 1 + rng.uniform(-1, 1) * sgain
    hsv[..., 2] *= 1 + rng.uniform(-1, 1) * vgain
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def random_flip_lr(img, bboxes, prob=0.5):
    """随机水平翻转图像和 bbox。"""
    if np.random.random() > prob:
        return img, bboxes
    img = np.ascontiguousarray(img[:, ::-1])
    w = img.shape[1]
    for b in bboxes:
        x1, x2 = b[0], b[2]
        b[0], b[2] = w - x2, w - x1
    return img, bboxes


def normalize(img):
    """归一化 → [C,H,W] tensor，值域约 [-2, 2]。"""
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1))


def adjust_bboxes(bboxes, scale, pad):
    """将原始坐标映射到 letterbox 后的坐标。"""
    out = []
    for b in bboxes:
        b = [v * scale + pad[i % 2] for i, v in enumerate(b)]
        out.append(b)
    return out


def adjust_keypoints(keypoints, scale, pad):
    """将原始关键点坐标映射到 letterbox 后的坐标。"""
    out = []
    for kpts in keypoints:
        adj = []
        for kx, ky, kv in kpts:
            adj.append([kx * scale + pad[0], ky * scale + pad[1], kv])
        out.append(adj)
    return out


class TrainTransform:
    def __init__(self, input_size=640, hsv_cfg=None, flip_prob=0.5):
        self.size = input_size
        self.hsv = hsv_cfg or {}
        self.flip_prob = flip_prob

    def __call__(self, img, human_boxes, human_kpts, anomaly_boxes, head_boxes=None):
        img, scale, pad = letterbox(img, self.size)

        if human_boxes:
            human_boxes = adjust_bboxes(human_boxes, scale, pad)
        if human_kpts:
            human_kpts = adjust_keypoints(human_kpts, scale, pad)
        if anomaly_boxes:
            anomaly_boxes = adjust_bboxes(anomaly_boxes, scale, pad)
        if head_boxes:
            head_boxes = adjust_bboxes(head_boxes, scale, pad)

        img = random_hsv(img, **self.hsv)

        # flip 必须统一: person 和 head 同一次随机决定
        do_flip = np.random.random() < self.flip_prob
        if do_flip:
            img = np.ascontiguousarray(img[:, ::-1])
            w = img.shape[1]
            if human_boxes:
                for b in human_boxes:
                    b[0], b[2] = w - b[2], w - b[0]
            if anomaly_boxes:
                for b in anomaly_boxes:
                    b[0], b[2] = w - b[2], w - b[0]
            if head_boxes:
                for b in head_boxes:
                    b[0], b[2] = w - b[2], w - b[0]

        return normalize(img), human_boxes, human_kpts, anomaly_boxes, head_boxes


class ValTransform:
    def __init__(self, input_size=640):
        self.size = input_size

    def __call__(self, img, human_boxes, human_kpts, anomaly_boxes, head_boxes=None):
        img, scale, pad = letterbox(img, self.size)
        if human_boxes:
            human_boxes = adjust_bboxes(human_boxes, scale, pad)
        if human_kpts:
            human_kpts = adjust_keypoints(human_kpts, scale, pad)
        if anomaly_boxes:
            anomaly_boxes = adjust_bboxes(anomaly_boxes, scale, pad)
        if head_boxes:
            head_boxes = adjust_bboxes(head_boxes, scale, pad)
        return normalize(img), human_boxes, human_kpts, anomaly_boxes, head_boxes
