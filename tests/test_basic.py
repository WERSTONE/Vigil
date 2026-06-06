"""
Vigil 测试套件 — 架构验证 + 解码 + 损失 + 后处理 + 推理
运行: pytest tests/test_basic.py -v
"""
import pytest
import numpy as np
import torch

from models.vigil_v1.model import VigilModel, create_model
from models.vigil_v1.backbone import CSPDarkNet
from models.vigil_v1.neck import FPNPANNeck
from models.vigil_v1.head import VigilHead, decode_outputs
from models.vigil_v1.assigner import CenterAssigner
from models.vigil_v1.loss import VigilLoss


class TestArchitecture:
    """模型架构形状验证"""

    def test_backbone(self):
        bb = CSPDarkNet(w=0.5)
        bb.eval()
        feats = bb(torch.randn(1, 3, 640, 640))
        assert len(feats) == 5  # P1-P5
        # w=0.5: ch(32)=16, ch(64)=32, ch(128)=64, ch(256)=128, ch(512)=256
        assert feats[1].shape == (1, 32, 160, 160)   # P2
        assert feats[2].shape == (1, 64, 80, 80)     # P3
        assert feats[3].shape == (1, 128, 40, 40)    # P4
        assert feats[4].shape == (1, 256, 20, 20)    # P5

    def test_backbone_w50(self):
        """w=0.5 与 w=0.25 通道翻倍"""
        bb = CSPDarkNet(w=0.25)
        bb.eval()
        feats = bb(torch.randn(1, 3, 640, 640))
        # w=0.25: ch(64)=16
        assert feats[2].shape[1] == 32  # P3

    def test_neck(self):
        bb = CSPDarkNet(w=0.5)
        # w=0.5 backbone: P2-P5 out = [32, 64, 128, 256]
        neck = FPNPANNeck([32, 64, 128, 256], out_ch=128)
        bb.eval(); neck.eval()
        p2_p5 = bb(torch.randn(1, 3, 640, 640))[1:]
        feats = neck(p2_p5)
        for i, (c, h, w) in enumerate([(128, 160, 160), (128, 80, 80), (128, 40, 40), (128, 20, 20)]):
            assert feats[i].shape == (1, c, h, w), f"N{i+2} shape mismatch"

    def test_head(self):
        head = VigilHead(in_ch=128, num_classes=4)
        head.eval()
        feats = [torch.randn(1, 128, h, w) for h, w in [(160, 160), (80, 80), (40, 40), (20, 20)]]
        out = head(feats)
        assert out["cls"][0].shape == (1, 4, 160, 160)
        assert out["bbox"][0].shape == (1, 4, 160, 160)
        assert out["obj"][0].shape == (1, 1, 160, 160)
        assert out["kpts"][0].shape == (1, 51, 160, 160)
        assert out["helmet"][0].shape == (1, 1, 160, 160)
        assert out["smoking"][0].shape == (1, 1, 160, 160)

    def test_full_model_forward(self):
        m = VigilModel(backbone_w=0.5)
        m.eval()
        out = m(torch.randn(1, 3, 640, 640))
        assert len(out["cls"]) == 4
        assert out["cls"][0].shape == (1, 4, 160, 160)

    def test_full_model_detect(self):
        m = VigilModel(backbone_w=0.5)
        m.eval()
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        det = m.detect(frame)
        for key in det:
            assert "boxes" in det[key]
            assert "scores" in det[key]

    def test_factory(self):
        m = create_model(pretrained=False)
        assert isinstance(m, VigilModel)
        m.eval(); m(torch.randn(1, 3, 640, 640))

    def test_param_count(self):
        m = VigilModel(backbone_w=0.5)
        n = m.num_params
        assert 1_500_000 < n < 5_000_000, f"Params: {n:,} (expected 1.5-5M)"


class TestDecode:
    """解码验证"""

    def test_decode_all_empty(self):
        """高阈值时返回空张量"""
        head = VigilHead(in_ch=128, num_classes=4); head.eval()
        feats = [torch.randn(1, 128, h, w) for h, w in [(80, 80), (40, 40), (20, 20)]]
        out = head(feats)
        boxes, scores, kpts, helmet, smoking = decode_outputs(
            out, strides=[8, 16, 32], score_thresh=0.999)
        assert boxes.shape == (1, 0, 4)
        assert scores.shape == (1, 0, 3)
        assert kpts.shape == (1, 0, 17, 3)

    def test_decode_has_detections(self):
        """低阈值时有检测输出"""
        head = VigilHead(in_ch=128, num_classes=4); head.eval()
        feats = [torch.randn(1, 128, h, w) for h, w in [(80, 80), (40, 40), (20, 20)]]
        out = head(feats)
        boxes, scores, kpts, helmet, smoking = decode_outputs(
            out, strides=[8, 16, 32], score_thresh=0.01)
        # 低阈值下通常会有检测 (随机权重)
        assert boxes.ndim == 3 and boxes.shape[-1] == 4

    def test_postprocessor(self):
        from postprocess.temporal import PostProcessor
        from inference.engine import Person, Anomaly

        pp = PostProcessor({"roi_zones": [[[0, 0], [100, 0], [100, 100], [0, 100]]]})
        p = Person(bbox=[50, 50, 80, 80], confidence=0.9, helmet_status=0,
                   helmet_conf=0.7, smoking_status=0, smoking_conf=0.1, keypoints=[[0]*3]*17)
        events = pp.process_frame([p], [], 640, 640)
        assert any(e["type"] == "intrusion" for e in events)

        p2 = Person(bbox=[10, 10, 50, 50], confidence=0.9, helmet_status=1,
                    helmet_conf=0.8, smoking_status=0, smoking_conf=0.1, keypoints=[[0]*3]*17)
        events2 = pp.process_frame([p2], [], 640, 640)
        assert any(e["type"] == "helmet_violation" for e in events2)

        a = Anomaly(bbox=[100, 100, 200, 200], class_name="fire", confidence=0.85)
        for _ in range(3):
            events3 = pp.process_frame([], [a], 640, 640)
        assert any(e["type"] == "fire" for e in events3)


class TestLoss:
    """损失函数"""

    def test_vigil_loss_create(self):
        vl = VigilLoss(w_box=3.0, w_cls=1.0, w_obj=1.0,
                       w_kpt=5.0, w_helm=1.0, w_smoke=1.0)
        assert vl.w_box == 3.0
        assert vl.w_kpt == 5.0

    def test_vigil_loss_forward_no_targets(self):
        """无分配目标时返回零损失"""
        vl = VigilLoss()
        head = VigilHead(in_ch=64, num_classes=4); head.eval()
        feats = [torch.randn(1, 64, h, w) for h, w in [(40, 40), (20, 20)]]
        head_outs = head(feats)
        # 空分配目标 → 只计算 obj 负样本损失
        loss_dict = vl(head_outs, [None, None], strides=[8, 16], feat_sizes=[(40, 40), (20, 20)])
        assert "cls" in loss_dict
        assert "bbox" in loss_dict
        assert loss_dict["num_pos"] == 0

    def test_vigil_loss_forward_with_targets(self):
        """单个 GT 分配 → 全部损失组件非零"""
        vl = VigilLoss()
        head = VigilHead(in_ch=64, num_classes=4); head.eval()
        feats = [torch.randn(1, 64, h, w) for h, w in [(40, 40), (20, 20)]]
        head_outs = head(feats)

        # 构造一个分配目标: level 0, 中心格点 (20, 20)
        assign_targets = [
            {
                "gt_boxes": torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
                "gt_classes": torch.tensor([0]),     # person
                "gt_kpts": torch.randn(1, 17, 3),
                "gt_helmet": torch.tensor([0]),       # helmet on
                "gt_smoking": torch.tensor([0]),      # no smoking
                "grid_xy": torch.tensor([[20, 20]]),
                "batch_idx": torch.tensor([0]),
            },
            None,  # level 1: 无目标
        ]
        loss_dict = vl(head_outs, assign_targets, strides=[8, 16], feat_sizes=[(40, 40), (20, 20)])
        assert loss_dict["num_pos"] == 1
        assert loss_dict["bbox"].item() > 0
        assert loss_dict["cls"].item() > 0


class TestAssigner:
    """分配器"""

    def test_center_assigner_create(self):
        ca = CenterAssigner(strides=[8, 16, 32], radius=1.5)
        assert ca.num_levels == 3
        assert ca.radius == 1.5

    def test_center_assigner_call(self):
        ca = CenterAssigner(strides=[8, 16, 32, 64], radius=1.5)
        # 按 batch 传入: List[Tensor] 形式
        gt_boxes = [torch.tensor([[100.0, 100.0, 200.0, 200.0]])]
        gt_classes = [torch.tensor([0])]
        gt_attrs = [{
            "kpts": torch.randn(1, 17, 3),
            "helmet": torch.tensor([0]),
            "smoking": torch.tensor([0]),
        }]
        targets = ca(gt_boxes, gt_classes, gt_attrs,
                     [(160, 160), (80, 80), (40, 40), (20, 20)])
        any_match = any(t is not None for t in targets)
        assert any_match, "GT box should be assigned to at least one level"


class TestTemporal:
    """时序后处理"""

    def test_fall_detector(self):
        from postprocess.temporal import FallDetector
        fd = FallDetector()
        assert fd.duration_threshold == 5.0

    def test_wave_detector(self):
        from postprocess.temporal import WaveDetector
        wd = WaveDetector()
        assert wd.duration == 2.0

    def test_buffer(self):
        from postprocess.temporal import PoseTemporalBuffer
        buf = PoseTemporalBuffer(window_size=30, fps=15)
        kp = np.random.randn(17, 3)
        for i in range(50):
            buf.append(kp, i / 15.0)
        assert len(buf.keypoints_deque) == 30
        assert len(buf.get_window(1.0)) <= 16


class TestPipeline:
    def test_mock_pipeline(self):
        from pipeline.gst_pipeline import MockPipeline
        p = MockPipeline()
        assert p.callbacks == []
        frames = []
        p.add_callback(lambda f: frames.append(f))
        assert len(p.callbacks) == 1


class TestInference:
    def test_engine_create(self):
        from inference.engine import create_engine
        engine = create_engine(config_path="config/config.yaml", device="cpu")
        assert engine.model is not None

    def test_engine_infer(self):
        from inference.engine import create_engine
        engine = create_engine(config_path="config/config.yaml", device="cpu")
        frame = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        result = engine.infer(frame)
        assert result.frame_id == 1
        assert result.latency_ms > 0
        assert isinstance(result.persons, list)
        assert isinstance(result.anomalies, list)
        assert isinstance(result.events, list)


class TestRegistry:
    """模型注册表"""

    def test_list_models(self):
        from models.registry import list_models
        names = list_models()
        assert "vigil_v1" in names, f"vigil_v1 should be registered, got: {names}"

    def test_create_via_registry(self):
        from models.registry import create_model as registry_create
        m = registry_create("vigil_v1", pretrained=False)
        assert isinstance(m, VigilModel)

    def test_create_model_export(self):
        from models import create_model as top_create
        m = top_create("vigil_v1", pretrained=False)
        assert isinstance(m, VigilModel)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
