"""
时序后处理 — 跌倒(≥5s) + 挥手(≥2s) + 闯入/安全帽/越界/烟火/漏水事件生成
"""
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
import time


@dataclass
class TemporalEvent:
    event_type: str
    start_time: float
    duration: float
    person_id: int
    confidence: float
    keypoint_history: List[np.ndarray] = field(default_factory=list)


class PoseTemporalBuffer:
    def __init__(self, window_size=150, fps=15):
        self.keypoints_deque: deque = deque(maxlen=window_size)
        self.timestamps: deque = deque(maxlen=window_size)

    def append(self, keypoints, timestamp):
        self.keypoints_deque.append(keypoints)
        self.timestamps.append(timestamp)

    def get_window(self, seconds):
        if not self.timestamps: return []
        cutoff = self.timestamps[-1] - seconds
        return [kp for kp, ts in zip(self.keypoints_deque, self.timestamps) if ts >= cutoff]


class FallDetector:
    LEFT_HIP, RIGHT_HIP = 11, 12
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    def __init__(self, drop_ratio=0.15, duration=5.0, fps=15):
        self.drop_threshold = drop_ratio
        self.duration_threshold = duration
        self.fps = fps
        self.tracked: dict = {}
        self.active: dict = {}

    def _center_y(self, kp):
        pts = kp[[self.LEFT_HIP, self.RIGHT_HIP, self.LEFT_SHOULDER, self.RIGHT_SHOULDER], :]
        valid = [p[1] for p in pts if p[2] > 0.3]
        return float(np.mean(valid)) if len(valid) >= 2 else None

    def update(self, pid, kp, img_h=640):
        now = time.time()
        if pid not in self.tracked:
            self.tracked[pid] = PoseTemporalBuffer(fps=self.fps)
        buf = self.tracked[pid]
        buf.append(kp, now)

        if pid in self.active:
            cy = self._center_y(kp)
            if cy is not None:
                ev = self.active[pid]
                sy = np.mean(kp[[self.LEFT_SHOULDER, self.RIGHT_SHOULDER], 1])
                hy = np.mean(kp[[self.LEFT_HIP, self.RIGHT_HIP], 1])
                if (hy - sy) > 0.05 * img_h:
                    ev.duration = now - ev.start_time
                    del self.active[pid]
                    return ev if ev.duration >= self.duration_threshold else None
            return None

        recent = buf.get_window(1.0)
        if len(recent) >= 3:
            s_y, e_y = self._center_y(recent[0]), self._center_y(recent[-1])
            if s_y and e_y and (s_y - e_y) > self.drop_threshold * img_h:
                self.active[pid] = TemporalEvent("fall", now, 0.0, pid, 0.8)
        return None


class WaveDetector:
    LEFT_WRIST, RIGHT_WRIST = 9, 10
    LEFT_SHOULDER, RIGHT_SHOULDER = 5, 6

    def __init__(self, freq_th=1.5, amp_th=0.08, duration=2.0, fps=15):
        self.freq_th = freq_th
        self.amp_th = amp_th
        self.duration = duration
        self.fps = fps
        self.tracked: dict = {}

    def _wrist_pos(self, kp):
        if kp[self.LEFT_WRIST, 2] > 0.3 and kp[self.LEFT_SHOULDER, 2] > 0.3:
            return (kp[self.LEFT_WRIST, 0] - kp[self.LEFT_SHOULDER, 0],
                    kp[self.LEFT_WRIST, 1] - kp[self.LEFT_SHOULDER, 1])
        if kp[self.RIGHT_WRIST, 2] > 0.3 and kp[self.RIGHT_SHOULDER, 2] > 0.3:
            return (kp[self.RIGHT_WRIST, 0] - kp[self.RIGHT_SHOULDER, 0],
                    kp[self.RIGHT_WRIST, 1] - kp[self.RIGHT_SHOULDER, 1])
        return None

    def update(self, pid, kp, img_h=640):
        now = time.time()
        if pid not in self.tracked:
            self.tracked[pid] = PoseTemporalBuffer(fps=self.fps)
        buf = self.tracked[pid]
        buf.append(kp, now)

        window = buf.get_window(self.duration)
        if len(window) < self.fps * self.duration * 0.5:
            return None
        positions = [p for k in window if (p := self._wrist_pos(k)) is not None]
        if len(positions) < 5: return None
        positions = np.array(positions)
        std_x, std_y = np.std(positions[:, 0]), np.std(positions[:, 1])
        amp = (std_x + std_y) / (2 * img_h)
        yc = positions[:, 1] - np.mean(positions[:, 1])
        zc = np.sum(np.diff(np.sign(yc)) != 0)
        freq = zc / (len(positions) / self.fps) / 2
        if amp > self.amp_th and freq > self.freq_th:
            return TemporalEvent("wave", now - self.duration, self.duration, pid,
                                 min(0.95, (amp / self.amp_th) * 0.7))
        return None


class PostProcessor:
    def __init__(self, config: dict):
        self.fall_detector = FallDetector(drop_ratio=config.get("fall_drop_ratio", 0.15),
                                          duration=config.get("fall_duration", 5.0))
        self.wave_detector = WaveDetector(duration=config.get("wave_duration", 2.0))
        self.roi_zones = config.get("roi_zones", [])
        self.boundary_lines = config.get("boundary_lines", [])

    def process_frame_v2(self, persons: List, anomalies: List, frame_h: int, frame_w: int) -> List[dict]:
        events = []
        for p in persons:
            bbox = np.array(p.bbox) if not isinstance(p.bbox, list) else p.bbox
            conf = float(p.confidence)

            if self.roi_zones:
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                for zone in self.roi_zones:
                    if self._point_in_polygon([cx, cy], zone):
                        events.append({"type": "intrusion", "task_id": 1, "bbox": bbox if isinstance(bbox, list) else bbox.tolist(), "confidence": conf})

            if int(p.helmet_status) == 1:
                events.append({"type": "helmet_violation", "task_id": 2, "bbox": bbox if isinstance(bbox, list) else bbox.tolist(), "confidence": float(p.helmet_conf)})

            if float(p.smoking_conf) > 0.5:
                events.append({"type": "smoking", "task_id": 4, "bbox": bbox if isinstance(bbox, list) else bbox.tolist(), "confidence": float(p.smoking_conf)})

            if self.boundary_lines:
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                for line in self.boundary_lines:
                    if self._cross_line([cx, cy], line):
                        events.append({"type": "boundary", "task_id": 5, "bbox": bbox if isinstance(bbox, list) else bbox.tolist(), "confidence": conf})

            if p.keypoints is not None and len(p.keypoints) > 0:
                kp = np.array(p.keypoints)
                pid = hash(tuple(bbox)) if isinstance(bbox, list) else hash(bbox.tobytes())
                fe = self.fall_detector.update(pid, kp, frame_h)
                if fe: events.append(fe.__dict__)
                we = self.wave_detector.update(pid, kp, frame_h)
                if we: events.append(we.__dict__)

        for a in anomalies:
            bbox = np.array(a.bbox) if not isinstance(a.bbox, list) else a.bbox
            events.append({"type": a.class_name, "task_id": 3 if a.class_id < 2 else 6,
                           "bbox": bbox if isinstance(bbox, list) else bbox.tolist(),
                           "confidence": float(a.confidence)})
        return events

    @staticmethod
    def _point_in_polygon(point, polygon):
        x, y = point; n = len(polygon); inside = False; j = n - 1
        for i in range(n):
            xi, yi = polygon[i]; xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    @staticmethod
    def _cross_line(point, line):
        return False
