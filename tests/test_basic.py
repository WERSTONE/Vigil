"""
Vigil 测试套件 — 架构验证 + 端到端推理 + 损失函数
运行: python main.py test  或  pytest tests/test_basic.py -v
"""
import pytest
import numpy as np
import torch

from models.model import (
    VigilMultiTaskModel, create_model, CSPDarkNet, FPNPANNeck,
    HumanAnalysisHead, SceneAnomalyHead, ProtoBranch,
    decode_human_outputs, decode_anomaly_outputs, DecodedPerson, DecodedAnomaly,
)


class TestArchitecture:
    """模型架构形状验证"""

    def test_backbone(self):
        bb = CSPDarkNet(w=0.25, d=0.33)
        bb.eval()
        feats = bb(torch.randn(1, 3, 640, 640))
        assert len(feats) == 4
        assert feats[0].shape == (1, 32, 160, 160)
        assert feats[1].shape == (1, 64, 80, 80)
        assert feats[2].shape == (1, 128, 40, 40)
        assert feats[3].shape == (1, 256, 20, 20)

    def test_neck(self):
        bb = CSPDarkNet(w=0.25, d=0.33)
        neck = FPNPANNeck([32, 64, 128, 256], [32, 64, 128, 256])
        bb.eval(); neck.eval()
        feats = neck(bb(torch.randn(1, 3, 640, 640)))
        for i, (c, h, w) in enumerate([(32, 160, 160), (64, 80, 80), (128, 40, 40), (256, 20, 20)]):
            assert feats[i].shape == (1, c, h, w), f"N{i+2} shape mismatch"

    def test_ha_head(self):
        ha = HumanAnalysisHead([32, 64, 128, 256])
        ha.eval()
        feats = [torch.randn(1, c, h, w) for c, h, w in [(32, 160, 160), (64, 80, 80), (128, 40, 40), (256, 20, 20)]]
        cls_o, reg_o, kpt_o = ha(feats)
        assert all(o.shape[1] == 5 for o in cls_o)   # person + helmet + smoking
        assert all(o.shape[1] == 64 for o in reg_o)   # 4 × 16 DFL
        assert all(o.shape[1] == 51 for o in kpt_o)   # 17 × 3

    def test_sa_head(self):
        sa = SceneAnomalyHead([64, 128, 256])
        sa.eval()
        feats = [torch.randn(1, c, h, w) for c, h, w in [(64, 80, 80), (128, 40, 40), (256, 20, 20)]]
        cls_o, reg_o, mask_o = sa(feats)
        assert all(o.shape[1] == 4 for o in cls_o)
        assert all(o.shape[1] == 64 for o in reg_o)
        assert all(o.shape[1] == 32 for o in mask_o)

    def test_proto(self):
        p = ProtoBranch(64, 32); p.eval()
        assert p(torch.randn(1, 64, 80, 80)).shape == (1, 32, 160, 160)

    def test_full_model(self):
        m = VigilMultiTaskModel(); m.eval()
        out = m(torch.randn(1, 3, 640, 640))
        assert len(out["ha_cls"]) == 4 and len(out["sa_cls"]) == 3
        assert out["proto"].shape == (1, 32, 160, 160)

    def test_nano_small(self):
        for w, d in [(0.25, 0.33), (0.50, 0.50)]:
            m = VigilMultiTaskModel(w, d); m.eval()
            m(torch.randn(1, 3, 640, 640))

    def test_factory(self):
        m = create_model("n", pretrained_path=None)
        assert isinstance(m, VigilMultiTaskModel)
        m.eval(); m(torch.randn(1, 3, 640, 640))

    def test_param_count(self):
        m = VigilMultiTaskModel()
        n = m.count_parameters()
        assert 1_500_000 < n < 4_000_000, f"Params: {n:,} (expected 1.5-4M)"


class TestDecode:
    """端到端解码验证"""

    def test_decode_human_empty(self):
        """无高置信度检测时返回空列表"""
        feats = [torch.randn(1, c, h, w) for c, h, w in [(64, 80, 80), (128, 40, 40), (256, 20, 20)]]
        ha = HumanAnalysisHead([64, 128, 256]); ha.eval()
        cls_o, reg_o, kpt_o = ha(feats)
        # 高阈值 → 应该有 0 个 detection
        results = decode_human_outputs(cls_o, reg_o, kpt_o, (640, 640), conf_threshold=0.99)
        assert results == []

    def test_decode_anomaly_empty(self):
        feats = [torch.randn(1, c, h, w) for c, h, w in [(64, 80, 80), (128, 40, 40), (256, 20, 20)]]
        sa = SceneAnomalyHead([64, 128, 256]); sa.eval()
        cls_o, reg_o, mask_o = sa(feats)
        proto = ProtoBranch(64, 32)(feats[0])
        results = decode_anomaly_outputs(cls_o, reg_o, mask_o, proto, (640, 640), conf_threshold=0.99)
        assert results == []

    def test_full_pipeline(self):
        """完整 pipeline: 输入 → 模型 → decode → 输出 (验证不报错)"""
        m = VigilMultiTaskModel(); m.eval()
        out = m(torch.randn(1, 3, 640, 640))
        persons = decode_human_outputs(out["ha_cls"], out["ha_reg"], out["ha_kpt"], (640, 640), 0.5)
        anomalies = decode_anomaly_outputs(out["sa_cls"], out["sa_reg"], out["sa_mask"], out["proto"], (640, 640), 0.3)
        for p in persons:
            assert isinstance(p, DecodedPerson)
            assert len(p.bbox) == 4
            assert 0 <= p.helmet_status <= 2
            assert len(p.keypoints) == 17
        for a in anomalies:
            assert isinstance(a, DecodedAnomaly)
            assert a.class_name in ["fire", "smoke", "water_stain", "water_drip"]

    def test_postprocessor(self):
        from postprocess.temporal import PostProcessor
        pp = PostProcessor({"roi_zones": [[[0, 0], [100, 0], [100, 100], [0, 100]]]})
        # 模拟一个在 ROI 内的 person
        p = DecodedPerson(bbox=[50, 50, 80, 80], confidence=0.9, helmet_status=0,
                          helmet_conf=0.7, smoking_conf=0.1, keypoints=[[0]*3]*17)
        events = pp.process_frame([p], [], 640, 640)
        assert any(e["type"] == "intrusion" for e in events)

        # 未戴安全帽
        p2 = DecodedPerson(bbox=[10, 10, 50, 50], confidence=0.9, helmet_status=1,
                           helmet_conf=0.8, smoking_conf=0.1, keypoints=[[0]*3]*17)
        events2 = pp.process_frame([p2], [], 640, 640)
        assert any(e["type"] == "helmet_violation" for e in events2)

        # 场景异常 (需连续3帧确认 — fire/smoke 时序过滤)
        a = DecodedAnomaly(bbox=[100, 100, 200, 200], class_id=0, class_name="fire",
                           confidence=0.85, mask_coeffs=[0.0]*32)
        for _ in range(3):
            events3 = pp.process_frame([], [a], 640, 640)
        assert any(e["type"] == "fire" for e in events3)


class TestTemporal:
    """时序后处理"""

    def test_fall_detector(self):
        from postprocess.temporal import FallDetector
        fd = FallDetector(); assert fd.duration_threshold == 5.0

    def test_wave_detector(self):
        from postprocess.temporal import WaveDetector
        wd = WaveDetector(); assert wd.duration == 2.0

    def test_buffer(self):
        from postprocess.temporal import PoseTemporalBuffer
        buf = PoseTemporalBuffer(window_size=30, fps=15)
        kp = np.random.randn(17, 3)
        for i in range(50): buf.append(kp, i / 15.0)
        assert len(buf.keypoints_deque) == 30
        assert len(buf.get_window(1.0)) <= 16


class TestLoss:
    """损失函数"""

    def test_human_loss(self):
        from models.loss import HumanLoss
        hl = HumanLoss()
        assert hl.box_w == 7.5 and hl.kpt_w == 12.0

    def test_anomaly_loss(self):
        from models.loss import AnomalyLoss
        al = AnomalyLoss()
        assert al.focal_gamma == 2.0

    def test_multitask_loss(self):
        from models.loss import VigilMultiTaskLoss
        mtl = VigilMultiTaskLoss(balance="manual")
        assert mtl.h_w == 1.0 and mtl.a_w == 0.5

    def test_uncertainty(self):
        from models.loss import UncertaintyWeighting
        uw = UncertaintyWeighting(2)
        total = uw([torch.tensor(1.0), torch.tensor(0.5)])
        assert total.ndim == 0


class TestPipeline:
    def test_mock_pipeline(self):
        from pipeline.gst_pipeline import MockPipeline
        p = MockPipeline()
        assert p.callbacks == []
        frames = []; p.add_callback(lambda f: frames.append(f))
        assert len(p.callbacks) == 1


class TestInference:
    def test_engine_create(self):
        from inference.engine import create_inference_engine
        engine = create_inference_engine(config_path="config/config.yaml", device="cpu")
        assert engine.model is not None

    def test_engine_infer(self):
        from inference.engine import create_inference_engine
        engine = create_inference_engine(config_path="config/config.yaml", device="cpu")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = engine.infer(frame)
        assert result.frame_id == 1
        assert result.latency_ms > 0
        assert isinstance(result.persons, list)
        assert isinstance(result.anomalies, list)
        assert isinstance(result.events, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
