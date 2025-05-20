import tempfile
import uuid
from pathlib import Path
import os

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("[ERROR] ultralytics not found. Install with `pip install ultralytics`.\n" + str(e))

try:
    import open_clip
except ImportError as e:
    raise RuntimeError("[ERROR] open_clip_torch not found. Install with `pip install open_clip_torch`.\n" + str(e))

# ---------------------------- Helpers ---------------------------------

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
        font = ImageFont.truetype("DejaVuSans.ttf", 24)
    except IOError:
        font = ImageFont.load_default()
    bbox = draw.textbbox((x1, y1), label_text, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    draw.rectangle([x1, y1 - text_h, x1 + text_w, y1], fill=color)
    draw.text((x1, y1 - text_h), label_text, fill=(255, 255, 255), font=font)
    return image

# ----------------------------------------------------------------------
# Global (lazy) caches --------------------------------------------------
_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_TOKENIZER = None
_CLIP_DEVICE = None
_TEXT_FEATURES = None

YOLO_WEIGHTS = Path("./models/bestyolo.pt")  # Update to your weights path
CLIP_MODEL_NAME = "ViT-B-32"
LABELS = [
    "graffiti on public objects",
    "a photo of trash on the street",
    "clean walls and streets",
]


def _lazy_load_models():
    global _YOLO_MODEL, _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE, _TEXT_FEATURES

    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(str(YOLO_WEIGHTS))
        _YOLO_MODEL.fuse()

    if _CLIP_MODEL is None:
        _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_TOKENIZER, _CLIP_DEVICE = load_clip(CLIP_MODEL_NAME)
        text_inputs = _CLIP_TOKENIZER(LABELS)
        with torch.no_grad():
            _TEXT_FEATURES = _CLIP_MODEL.encode_text(text_inputs.to(_CLIP_DEVICE))
            _TEXT_FEATURES /= _TEXT_FEATURES.norm(dim=-1, keepdim=True)


# ----------------------------------------------------------------------
# Core processing -------------------------------------------------------

def process_video(video_input, yolo_conf: float, clip_conf: float, frame_interval: int):
    # Handle Gradio file input variants (path str or dict)
    if isinstance(video_input, dict):
        video_path = video_input.get("name")
    else:
        video_path = video_input

    if not video_path or not Path(video_path).exists():
        raise ValueError("Invalid or missing video input")

    _lazy_load_models()

    # Ensure outputs/ exists and build unique file name
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)
    annotated_video_path = outputs_dir / f"annotated_{uuid.uuid4().hex[:8]}.mp4"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    idx = 0
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        if idx % frame_interval == 0:
            results = _YOLO_MODEL.predict(source=pil_frame, conf=yolo_conf, verbose=False)[0]
            dets = results.boxes.data.cpu().numpy()
            if dets.size:
                dets = dets[dets[:, 4] >= yolo_conf]

            if dets.size:
                for x1, y1, x2, y2, y_conf, cls in dets:
                    patch = pil_frame.crop((x1, y1, x2, y2))
                    patch_t = _CLIP_PREPROCESS(patch).unsqueeze(0).to(_CLIP_DEVICE)
                    with torch.no_grad():
                        img_feat = _CLIP_MODEL.encode_image(patch_t)
                        img_feat /= img_feat.norm(dim=-1, keepdim=True)
                        sim = (100.0 * img_feat @ _TEXT_FEATURES.T).softmax(dim=-1).squeeze(0)
                    top_idx = int(sim.argmax())
                    c_score = float(sim[top_idx])

                    if c_score < clip_conf:
                        continue

                    label = LABELS[top_idx]
                    color = (255, 0, 0) if int(cls) == 0 else (0, 255, 0)
                    pil_frame = draw_box(
                        pil_frame,
                        (x1, y1, x2, y2),
                        f"{label}: y{y_conf:.2f} c{c_score:.2f}",
                        color,
                    )

        writer.write(cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR))
        idx += 1

    cap.release()
    writer.release()

    # Return plain str path; Gradio Video accepts this
    return str(annotated_video_path)

# ----------------------------------------------------------------------
# Gradio UI -------------------------------------------------------------

demo = gr.Blocks(title="Graffiti & Trash Detector")

with demo:
    gr.Markdown("""# Graffiti & Trash Detector 🖌️🗑️\nUpload a video, set your thresholds, and click **Run** to download an annotated MP4 (saved in ./outputs).""")

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Input video", sources=["upload"], interactive=True)
            run_btn = gr.Button("Run pipeline 🚀", variant="primary", size="sm")
        with gr.Column(scale=1):
            yolo_slider = gr.Slider(0.05, 1.0, step=0.05, value=0.25, label="YOLO confidence threshold")
            clip_slider = gr.Slider(0.05, 1.0, step=0.05, value=0.30, label="CLIP confidence threshold")
            interval_slider = gr.Slider(1, 10, step=1, value=1, label="Frame sampling interval (N)")

    video_output = gr.Video(label="Annotated output", interactive=False)

    run_btn.click(
        fn=process_video,
        inputs=[video_input, yolo_slider, clip_slider, interval_slider],
        outputs=video_output,
    )

if __name__ == "__main__":
    demo.launch()