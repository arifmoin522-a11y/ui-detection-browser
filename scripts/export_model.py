#!/usr/bin/env python3
"""
Export foduucom/web-form-ui-field-detection model to ONNX and TensorFlow.js formats
for browser-based inference.

Usage:
    pip install ultralyticsplus==0.0.28 ultralytics==8.0.43 tensorflowjs
    python scripts/export_model.py [onnx|tfjs|both]
"""

import os
import sys
import time

MODEL_ID = "foduucom/web-form-ui-field-detection"
SUPPORTED_FORMATS = ["onnx", "tfjs"]
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def validate_environment():
    """Check that required packages are installed."""
    missing = []
    try:
        import ultralyticsplus  # noqa: F401
    except ImportError:
        missing.append("ultralyticsplus")
    try:
        import ultralytics  # noqa: F401
    except ImportError:
        missing.append("ultralytics")
    if missing:
        print(f"[ERROR] Missing packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def export_onnx():
    """Export model to ONNX format for ONNX Runtime Web."""
    from ultralyticsplus import YOLO

    print(f"[*] Loading model: {MODEL_ID}")
    model = YOLO(MODEL_ID)

    model.overrides["conf"] = 0.25
    model.overrides["iou"] = 0.45
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"] = 1000

    os.makedirs(EXPORT_DIR, exist_ok=True)
    onnx_path = os.path.join(EXPORT_DIR, "ui-detection.onnx")

    print(f"[*] Exporting to ONNX: {onnx_path}")
    t0 = time.time()
    model.export(format="onnx", imgsz=640, simplify=True)
    elapsed = time.time() - t0
    size_mb = os.path.getsize(onnx_path) / 1024 / 1024 if os.path.exists(onnx_path) else 0
    print(f"[+] ONNX export complete in {elapsed:.1f}s ({size_mb:.1f} MB)")
    return onnx_path


def export_tfjs():
    """Export model to TensorFlow.js format."""
    from ultralyticsplus import YOLO

    print(f"[*] Loading model: {MODEL_ID}")
    model = YOLO(MODEL_ID)

    model.overrides["conf"] = 0.25
    model.overrides["iou"] = 0.45
    model.overrides["agnostic_nms"] = False
    model.overrides["max_det"] = 1000

    print("[*] Exporting to TensorFlow.js")
    t0 = time.time()
    model.export(format="tfjs", imgsz=640)
    elapsed = time.time() - t0
    print(f"[+] TFJS export complete in {elapsed:.1f}s")


if __name__ == "__main__":
    validate_environment()

    fmt = sys.argv[1] if len(sys.argv) > 1 else "onnx"
    if fmt not in SUPPORTED_FORMATS:
        print(f"Unknown format: {fmt}. Use {', '.join(SUPPORTED_FORMATS)}")
        sys.exit(1)

    if fmt == "both":
        export_onnx()
        export_tfjs()
    else:
        export_onnx() if fmt == "onnx" else export_tfjs()