#!/usr/bin/env python3
"""
Export foduucom/web-form-ui-field-detection model to ONNX and TensorFlow.js formats
for browser-based inference.

Usage:
    pip install ultralyticsplus==0.0.28 ultralytics==8.0.43 tensorflowjs
    python scripts/export_model.py
"""

import os
import sys

MODEL_ID = "foduucom/web-form-ui-field-detection"
EXPORT_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


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
    model.export(format="onnx", imgsz=640, simplify=True)
    print("[+] ONNX export complete")


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
    model.export(format="tfjs", imgsz=640)
    print("[+] TFJS export complete")


if __name__ == "__main__":
    fmt = sys.argv[1] if len(sys.argv) > 1 else "onnx"
    if fmt == "onnx":
        export_onnx()
    elif fmt == "tfjs":
        export_tfjs()
    elif fmt == "both":
        export_onnx()
        export_tfjs()
    else:
        print(f"Unknown format: {fmt}. Use 'onnx', 'tfjs', or 'both'")
        sys.exit(1)
