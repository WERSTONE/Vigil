"""数据增强: letterbox + HSV + flip + normalize."""

import cv2
import numpy as np
import torch

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# COCO 17 点水平翻转交换对: (L, R)
KPT_FLIP_PAIRS = [(0, 0), (1, 2), (3, 4), (5, 6), (7, 8), (9, 10),
                   (11, 12), (13, 14), (15, 16)]


def letterbox(img, target_size=640, fill=114):
    """等比缩放 + 居中填充."""
    h, w = img.shape[:2]
    r = target_size / max(h, w)
    new_w, new_h = int(w * r), int(h * r)
    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    dw = target_size - new_w
    dh = target_size - new_h
    top, bottom = dh // 2, dh - dh // 2
    left, right = dw // 2, dw - dw // 2
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=(fill, fill, fill))
    return img, r, (left, top)


def random_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    """HSV 随机扰动."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    rng = np.random
    hsv[..., 0] = (hsv[..., 0] + rng.uniform(-1, 1) * hgain * 180) % 360
    hsv[..., 1] *= 1 + rng.uniform(-1, 1) * sgain
    hsv[..., 2] *= 1 + rng.uniform(-1, 1) * vgain
    hsv = np.clip(hsv, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def normalize(img):
    """归一化 → [C,H,W] tensor."""
    img = img.astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return torch.from_numpy(img.transpose(2, 0, 1))


def _adjust_boxes(boxes, scale, pad_l, pad_t):
    """原始坐标 → letterbox 坐标."""
    for b in boxes:
        b[0] = b[0] * scale + pad_l
        b[1] = b[1] * scale + pad_t
        b[2] = b[2] * scale + pad_l
        b[3] = b[3] * scale + pad_t


def _adjust_keypoints(keypoints, scale, pad_l, pad_t):
    """原始关键点 → letterbox 坐标."""
    for kpts in keypoints:
        for kp in kpts:
            kp[0] = kp[0] * scale + pad_l
            kp[1] = kp[1] * scale + pad_t


def _flip_boxes(boxes, img_w):
    """水平翻转 bbox."""
    for b in boxes:
        b[0], b[2] = img_w - b[2], img_w - b[0]


def _flip_keypoints(keypoints, img_w):
    """水平翻转关键点 — 交换 x 坐标 AND 交换左右配对."""
    for kpts in keypoints:
        for kp in kpts:
            kp[0] = img_w - kp[0]
        # 交换左右配对 (如左肩↔右肩)
        for l_idx, r_idx in KPT_FLIP_PAIRS:
            if l_idx != r_idx and l_idx < len(kpts) and r_idx < len(kpts):
                kpts[l_idx], kpts[r_idx] = kpts[r_idx].copy(), kpts[l_idx].copy()


class TrainTransform:
    def __init__(self, input_size=640, flip_prob=0.5,
                 hsv_cfg=None):
        self.size = input_size
        self.flip_prob = flip_prob
        self.hsv_cfg = hsv_cfg or {}

    def __call__(self, img, person_boxes=None, person_kpts=None,
                 helmet_boxes=None, smoking_boxes=None, anomaly_boxes=None,
                 cached_meta=None):
        """cached_meta: (scale, pad_l, pad_t) — 非 None 则跳过 letterbox, 直接调整坐标."""
        if cached_meta is not None:
            scale, pad_l, pad_t = cached_meta
        else:
            img, scale, (pad_l, pad_t) = letterbox(img, self.size)

        all_boxes = {
            "person": person_boxes or [],
            "helmet": helmet_boxes or [],
            "smoking": smoking_boxes or [],
            "anomaly": anomaly_boxes or [],
        }
        all_kpts = {"person": person_kpts or []}

        for boxes in all_boxes.values():
            if boxes:
                _adjust_boxes(boxes, scale, pad_l, pad_t)
        for kpts in all_kpts.values():
            if kpts:
                _adjust_keypoints(kpts, scale, pad_l, pad_t)

        img = random_hsv(img, **self.hsv_cfg)

        if np.random.random() < self.flip_prob:
            img = np.ascontiguousarray(img[:, ::-1])
            w = img.shape[1]
            for boxes in all_boxes.values():
                if boxes:
                    _flip_boxes(boxes, w)
            for kpts in all_kpts.values():
                if kpts:
                    _flip_keypoints(kpts, w)

        return (normalize(img),
                all_boxes["person"], all_kpts["person"],
                all_boxes["helmet"], all_boxes["smoking"],
                all_boxes["anomaly"])


class ValTransform:
    def __init__(self, input_size=640):
        self.size = input_size

    def __call__(self, img, person_boxes=None, person_kpts=None,
                 helmet_boxes=None, smoking_boxes=None, anomaly_boxes=None):
        img, scale, (pad_l, pad_t) = letterbox(img, self.size)
        for boxes in [person_boxes, helmet_boxes, smoking_boxes, anomaly_boxes]:
            if boxes:
                _adjust_boxes(boxes, scale, pad_l, pad_t)
        if person_kpts:
            _adjust_keypoints(person_kpts, scale, pad_l, pad_t)
        return (normalize(img),
                person_boxes or [], person_kpts or [],
                helmet_boxes or [], smoking_boxes or [],
                anomaly_boxes or [])
