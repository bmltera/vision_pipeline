"""
Convert a YOLO model (Ultralytics .pt) to ONNX and then to TensorRT FP16.

usage:
    python export_to_trt.py --weights ./models/bestyolo.pt
                            --img 640
                            --batch 1
"""
import argparse
from pathlib import Path
import subprocess
import sys

from ultralytics import YOLO

def export(weights: Path, img: int, batch: int):
    weights = Path(weights).expanduser().resolve()
    onnx_path = weights.with_suffix(".onnx")
    engine_path = weights.with_suffix(".trt")

    # --- 1.  PyTorch → ONNX -------------------------------------------------
    if not onnx_path.exists():
        print(f"[1/2] Exporting ONNX to {onnx_path}")
        YOLO(str(weights)).export(
            format="onnx",
            imgsz=img,
            batch=batch,
            opset=17,           # ONNX opset; 17 is safe for TensorRT 8.6+
            dynamic=True,       # dynamic input shapes
            simplify=True,
        )
    else:
        print(f"[✓] ONNX already exists: {onnx_path}")

    # --- 2.  ONNX → TensorRT engine ----------------------------------------
    if engine_path.exists():
        print(f"[✓] TensorRT engine already exists: {engine_path}")
        return engine_path

    print(f"[2/2] Building TensorRT FP16 engine at {engine_path}")
    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        "--fp16",                 # use half precision
        f"--minShapes=input:{batch}x3x{img}x{img}",
        f"--optShapes=input:{batch}x3x{img}x{img}",
        f"--maxShapes=input:{batch}x3x{img}x{img}",
        "--workspace=4096"        # MB of GPU memory to allow for build
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print("[✓] Done!")
    return engine_path

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="path/to/best.pt")
    ap.add_argument("--img", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    try:
        export(Path(args.weights), args.img, args.batch)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
