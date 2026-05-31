"""
推理性能基准 — 本地 & 端侧设备通用
用法: python scripts/benchmark_inference.py                    # PyTorch CPU
      python scripts/benchmark_inference.py --device cuda       # GPU
      python scripts/benchmark_inference.py --onnx              # ONNX 导出+测试
      python scripts/benchmark_inference.py --torchscript       # TorchScript 导出
"""
import argparse
import time
import numpy as np
import torch
from loguru import logger


def benchmark_pytorch(model, config, args):
    size = tuple(config["model"].get("input_size", [640, 640]))
    dummy = torch.randn(1, 3, *size)
    if args.device == "cuda" and torch.cuda.is_available():
        model = model.cuda().half()
        dummy = dummy.cuda().half()
    else:
        model = model.cpu().eval()

    for _ in range(10):
        with torch.no_grad(): model(dummy)

    latencies = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        with torch.no_grad(): model(dummy)
        latencies.append((time.perf_counter() - t0) * 1000)

    lat = np.array(latencies)
    params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {params:,}")
    logger.info(f"Mean: {lat.mean():.1f}ms  P50: {np.percentile(lat, 50):.1f}ms  "
                f"P95: {np.percentile(lat, 95):.1f}ms  FPS: {1000/lat.mean():.1f}")
    return model, dummy


def benchmark_onnx(config, args, dummy):
    from models.model import export_onnx
    size = tuple(config["model"].get("input_size", [640, 640]))
    onnx_path = args.onnx if args.onnx != "auto" else f"vigil_{args.variant}.onnx"

    if args.onnx == "auto":
        from models.model import create_model
        model = create_model(variant=args.variant, pretrained_path=args.weights)
        model.eval()
        export_onnx(model, onnx_path, size)

    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        iname = sess.get_inputs()[0].name
        d = dummy.cpu().numpy() if args.device == "cuda" else dummy.numpy()
        for _ in range(10): sess.run(None, {iname: d})
        latencies = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            sess.run(None, {iname: d})
            latencies.append((time.perf_counter() - t0) * 1000)
        lat = np.array(latencies)
        logger.info(f"ONNX — Mean: {lat.mean():.1f}ms  P50: {np.percentile(lat, 50):.1f}ms  "
                    f"P95: {np.percentile(lat, 95):.1f}ms  FPS: {1000/lat.mean():.1f}")
    except ImportError:
        logger.warning("onnxruntime not installed — skip ONNX benchmark")


def benchmark_torchscript(config, args):
    from models.model import create_model, export_torchscript
    size = tuple(config["model"].get("input_size", [640, 640]))
    ts_path = args.torchscript if args.torchscript != "auto" else f"vigil_{args.variant}.pt"

    if args.torchscript == "auto":
        model = create_model(variant=args.variant, pretrained_path=args.weights)
        model.eval()
        export_torchscript(model, ts_path, size)

    logger.info(f"TorchScript exported: {ts_path}")


def main():
    parser = argparse.ArgumentParser(description="Vigil 推理性能基准")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--weights", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--variant", default="n")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--onnx", nargs="?", const="auto", default=None)
    parser.add_argument("--torchscript", nargs="?", const="auto", default=None)
    args = parser.parse_args()

    import yaml
    from models.model import create_model

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"=== PyTorch Benchmark ({args.device}) ===")
    model = create_model(variant=args.variant, pretrained_path=args.weights)
    model, dummy = benchmark_pytorch(model, config, args)

    if args.onnx:
        logger.info(f"=== ONNX Benchmark ({args.device}) ===")
        benchmark_onnx(config, args, dummy)

    if args.torchscript:
        logger.info("=== TorchScript Export ===")
        benchmark_torchscript(config, args)


if __name__ == "__main__":
    main()
