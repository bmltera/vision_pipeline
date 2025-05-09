#!/usr/bin/env python3
"""
2-Stage Computer Vision Pipeline
--------------------------------
Detects graffiti and trash in video frames using YOLO, then refines each detection
with CLIP image–text similarity. Outputs original and annotated frames, an annotated
video, plus a run report including timing and confidence metrics.

Usage example:
    python vision_pipeline.py --input_video ./demo.mp4 \
        --yolo_weights ./models/bestyolo.pt --clip_model ViT-B-32 --conf_thres 0.50
"""
import argparse
import datetime as dt
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import torch
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from ultralytics import YOLO
except ImportError as e:
    sys.exit("[ERROR] ultralytics not found. Install with `pip install ultralytics`.\n" + str(e))

try:
    import open_clip
except ImportError as e:
    sys.exit("[ERROR] open_clip_torch not found. Install with `pip install open_clip_torch`.\n" + str(e))

def make_run_dirs(root: Path) -> dict[str, Path]:
    ts = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = root / f"run_{ts}"
    orig_dir = run_dir / "original_frames"
    ann_dir = run_dir / "annotated_frames"
    orig_dir.mkdir(parents=True, exist_ok=False)
    ann_dir.mkdir(parents=True, exist_ok=False)
    return {"run": run_dir, "orig": orig_dir, "ann": ann_dir}

def load_clip(model_name: str = "ViT-B-32", device: str | torch.device | None = None):
    model_name = model_name.replace("/", "-")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained="openai")
    model.eval().requires_grad_(False)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device

def draw_box(image: Image.Image, xyxy, label_text: str, color=(255, 0, 0)) -> Image.Image:
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = map(int, xyxy)
    draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 28)
    except IOError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x1, y1), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
    draw.text((x1, y1 - text_h), label_text, fill=(255, 255, 255), font=font)
    return image

def main(opts: argparse.Namespace):
    start_time = time.perf_counter()
    dirs = make_run_dirs(Path(opts.output_dir))
    run_dir, orig_dir, ann_dir = dirs.values()

    cap = cv2.VideoCapture(str(opts.input_video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {opts.input_video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    total_yolo_time = total_clip_time = 0.0
    total_yolo_conf = total_clip_conf = 0.0
    detection_frames = []

    model = YOLO(opts.yolo_weights)
    model.fuse()

    # Load CLIP
    clip_model, clip_preprocess, clip_tokenizer, device = load_clip(opts.clip_model)
    clip_labels = [
        "graffiti on public objects",
        "a photo of trash on the street",
        "clean walls and streets",
    ]
    text_inputs = clip_tokenizer(clip_labels)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_inputs.to(device))
        text_features /= text_features.norm(dim=-1, keepdim=True)

    frames_with_det = 0
    total_dets = 0

    for idx in tqdm(range(frame_count), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        # Save original frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_frame = Image.fromarray(frame_rgb)
        orig_path = orig_dir / f"frame_{idx:06d}.jpg"
        pil_frame.save(orig_path)

        # YOLO inference with explicit conf threshold
        yolo_start = time.perf_counter()
        results = model.predict(
            source=frame_rgb,
            conf=opts.conf_thres,         # explicit YOLO conf
            verbose=False
        )[0]
        yolo_end = time.perf_counter()
        total_yolo_time += (yolo_end - yolo_start)

        # Extract and manually filter detections
        dets = results.boxes.data.cpu().numpy()
        if dets.size == 0:
            shutil.copy(orig_path, ann_dir / orig_path.name)
            continue

        # safety-net threshold
        mask = dets[:, 4] >= opts.conf_thres
        dets = dets[mask]

        if dets.shape[0] == 0:
            shutil.copy(orig_path, ann_dir / orig_path.name)
            continue

        detection_frames.append(idx)
        frames_with_det += 1
        annot_img = pil_frame.copy()

        # CLIP refinement + drawing
        for x1, y1, x2, y2, conf, cls in dets.tolist():
            total_yolo_conf += conf
            cls = int(cls)

            patch = pil_frame.crop((x1, y1, x2, y2))
            clip_start = time.perf_counter()
            patch_t = clip_preprocess(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                image_feat = clip_model.encode_image(patch_t)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                sim = (100.0 * image_feat @ text_features.T).softmax(dim=-1).squeeze(0)
            clip_end = time.perf_counter()
            total_clip_time += (clip_end - clip_start)

            top_idx = sim.argmax().item()
            clip_label = clip_labels[top_idx]
            clip_score = sim[top_idx].item()
            total_clip_conf += clip_score
            total_dets += 1

            label_text = f"{clip_label}: y{conf:.2f} c{clip_score:.2f}"
            color = (255, 0, 0) if cls == 0 else (0, 255, 0)
            annot_img = draw_box(annot_img, (x1, y1, x2, y2), label_text, color)

        annot_img.save(ann_dir / orig_path.name)

    cap.release()

    # Reassemble annotated video
    annotated_video_path = run_dir / "annotated_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_vid = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))
    for idx in range(frame_count):
        frame_file = ann_dir / f"frame_{idx:06d}.jpg"
        img = cv2.imread(str(frame_file))
        out_vid.write(img)
    out_vid.release()

    # Summarize metrics
    end_time = time.perf_counter()
    total_time = end_time - start_time
    avg_yolo_ms = (total_yolo_time / frame_count) * 1000 if frame_count else 0
    avg_clip_ms = (total_clip_time / frame_count) * 1000 if frame_count else 0
    avg_inf_ms = ((total_yolo_time + total_clip_time) / frame_count) * 1000 if frame_count else 0
    avg_yolo_conf = (total_yolo_conf / total_dets) if total_dets else 0
    avg_clip_conf = (total_clip_conf / total_dets) if total_dets else 0

    details = {
        "input_video": str(opts.input_video),
        "yolo_weights": str(opts.yolo_weights),
        "clip_model": opts.clip_model,
        "conf_threshold": opts.conf_thres,
        "frames_total": frame_count,
        "frames_with_detections": frames_with_det,
        "total_detections": total_dets,
        "detection_frames": detection_frames,
        "avg_yolo_confidence": avg_yolo_conf,
        "avg_clip_confidence": avg_clip_conf,
        "annotated_video": str(annotated_video_path),
        "total_time_s": total_time,
        "avg_inference_ms_per_frame": avg_inf_ms,
        "avg_yolo_ms_per_frame": avg_yolo_ms,
        "avg_clip_ms_per_frame": avg_clip_ms,
        "datetime": dt.datetime.now().isoformat(),
    }
    with open(run_dir / "details.txt", "w", encoding="utf-8") as f:
        json.dump(details, f, indent=4)

    print(f"[✓] Finished. Results saved to: {run_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="2-Stage graffiti/trash detector")
    parser.add_argument("--input_video", type=Path, required=True, help="Path to input video")
    parser.add_argument("--yolo_weights", type=Path, required=True, help="YOLO weights .pt file")
    parser.add_argument("--clip_model", type=str, default="ViT-B-32", help="CLIP model name (dash, not slash)")
    parser.add_argument("--conf_thres", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--output_dir", type=Path, default="./runs", help="Root output directory")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        print("Interrupted by user")
