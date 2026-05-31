"""
Vigil — 泵房多任务监控系统
用法:
    python main.py demo <图片> --weights checkpoints/yolov8n.pt
    python main.py live --cam 0 --weights checkpoints/yolov8n.pt --show
    python main.py live --video test.mp4 --weights checkpoints/yolov8n.pt --show
"""
import argparse
import sys
import time
import cv2
import numpy as np
import yaml
from loguru import logger


def _build_engine(args):
    from models.model import create_model
    from inference.engine import InferenceEngine

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    model = create_model(variant=args.variant, pretrained_path=args.weights)
    return InferenceEngine(model, config, device=args.device)


def _draw_results(frame, result, latency_ms):
    """在画面上绘制检测框和事件"""
    h, w = frame.shape[:2]

    for p in result.persons:
        bx = [int(p.bbox[0]), int(p.bbox[1]), int(p.bbox[2]), int(p.bbox[3])]
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), (0, 255, 0), 2)
        label = f"P {p.confidence:.2f}"
        if int(p.helmet_status) == 1:
            label += " NO_HELMET"
        if float(p.smoking_conf) > 0.5:
            label += " SMOKE"
        cv2.putText(frame, label, (bx[0], bx[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # 画关键点
        if p.keypoints and len(p.keypoints) == 17:
            for kx, ky, kc in p.keypoints:
                if kc > 0.3:
                    cv2.circle(frame, (int(kx), int(ky)), 2, (0, 0, 255), -1)

    for a in result.anomalies:
        bx = [int(a.bbox[0]), int(a.bbox[1]), int(a.bbox[2]), int(a.bbox[3])]
        color = (0, 0, 255) if a.class_id < 2 else (255, 128, 0)
        cv2.rectangle(frame, (bx[0], bx[1]), (bx[2], bx[3]), color, 2)
        cv2.putText(frame, f"{a.class_name} {a.confidence:.2f}",
                    (bx[0], bx[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # 延迟 & FPS
    cv2.putText(frame, f"{latency_ms:.0f}ms", (w - 90, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # 告警事件计数
    y0 = h - 20
    for ev in result.events:
        txt = f"[{ev.get('type', '?')}] t{ev.get('task_id', '?')}"
        cv2.putText(frame, txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        y0 -= 16
        if y0 < h - 120: break


def cmd_demo(args):
    engine = _build_engine(args)
    img = cv2.imread(args.image)
    if img is None:
        logger.error(f"Cannot read: {args.image}")
        sys.exit(1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = engine.infer(rgb)

    logger.info(f"Frame: {len(result.persons)} persons, {len(result.anomalies)} anomalies, "
                f"{len(result.events)} events, latency={result.latency_ms:.1f}ms")
    for p in result.persons:
        logger.info(f"  Person: conf={p.confidence:.3f} helmet={p.helmet_status} smoke={p.smoking_conf:.3f}")
    for a in result.anomalies:
        logger.info(f"  Anomaly: {a.class_name} conf={a.confidence:.3f}")
    for ev in result.events:
        logger.info(f"  Event: [{ev.get('type','?')}] task={ev.get('task_id','?')} conf={ev.get('confidence',0):.3f}")

    if args.show:
        _draw_results(img, result, result.latency_ms)
        cv2.imshow("Vigil — Demo", img)
        logger.info("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def cmd_live(args):
    engine = _build_engine(args)

    if args.cam is not None:
        cap = cv2.VideoCapture(int(args.cam))
    elif args.video:
        cap = cv2.VideoCapture(args.video)
    else:
        from pipeline.gst_pipeline import VideoPipeline
        logger.info("RTSP mode (no --video/--cam)")
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        def on_frame(frame):
            result = engine.infer(frame)
            for ev in result.events:
                logger.info(f"[{ev.get('type','?')}] conf={ev.get('confidence',0):.3f}")

        pipeline = VideoPipeline(source_uri=config["pipeline"]["source"],
                                  inference_callback=on_frame,
                                  fps=config["pipeline"]["fps"])
        pipeline.start()
        try:
            while pipeline.running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pipeline.stop()
        return

    if not cap.isOpened():
        logger.error("Cannot open video source")
        sys.exit(1)

    logger.info("Running... press ESC to stop")
    frame_count = 0
    total_latency = 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = engine.infer(rgb)
        frame_count += 1
        total_latency += result.latency_ms

        if args.show:
            _draw_results(frame, result, result.latency_ms)
            cv2.imshow("Vigil — Live", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count}: {len(result.persons)} persons, "
                        f"{len(result.anomalies)} anomalies, "
                        f"avg latency={total_latency/frame_count:.1f}ms")

    cap.release()
    cv2.destroyAllWindows()
    logger.info(f"Done. {frame_count} frames, avg latency={total_latency/max(frame_count,1):.1f}ms")


def main():
    parser = argparse.ArgumentParser(description="Vigil 泵房监控系统")
    sub = parser.add_subparsers(dest="cmd")

    p_demo = sub.add_parser("demo", help="单图推理")
    p_demo.add_argument("image")
    p_demo.add_argument("--show", action="store_true", help="显示检测结果画面")

    p_live = sub.add_parser("live", help="实时推理")
    src = p_live.add_mutually_exclusive_group()
    src.add_argument("--video", default=None, help="本地视频文件")
    src.add_argument("--cam", default=None, help="摄像头索引 (0=默认摄像头)")
    p_live.add_argument("--show", action="store_true", help="显示实时画面")

    for p in [p_demo, p_live]:
        p.add_argument("--config", default="config/config.yaml")
        p.add_argument("--weights", default=None)
        p.add_argument("--device", default="cpu")
        p.add_argument("--variant", default="n")

    args = parser.parse_args()
    if args.cmd == "demo": cmd_demo(args)
    elif args.cmd == "live": cmd_live(args)
    else: parser.print_help()


if __name__ == "__main__":
    main()
