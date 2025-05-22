import os
import uuid
import time
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

# ---------------------------- PyAV + NVDEC -----------------------------
try:
    import av
except ImportError as e:
    raise RuntimeError(
        "[ERROR] PyAV not found. Install with `pip install av`.\n" + str(e)
    )

# ---------------------------- ONNX Runtime -----------------------------
try:
    import onnxruntime as ort

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print(
            "[WARN] ONNX Runtime does not have GPU support. "
            "Install with: pip install onnxruntime-gpu"
        )
except ImportError as e:
    raise RuntimeError(
        "[ERROR] onnxruntime-gpu not found. Install with `pip install onnxruntime-gpu`.\n"
        + str(e)
    )

# ---------------------------- Ultralytics YOLO -----------------------------
try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError(
        "[ERROR] ultralytics not found. Install with `pip install ultralytics`.\n" + str(e)
    )

# ---------------------------- open_clip -----------------------------
try:
    import open_clip
except ImportError as e:
    raise RuntimeError(
        "[ERROR] open_clip_torch not found. Install with `pip install open_clip_torch`.\n"
        + str(e)
    )

# ---------------------------- NVDEC Loader -----------------------------
class NVDecoder:
    """GPU-accelerated video frame generator via FFmpeg NVDEC (cuvid) through PyAV."""

    def __init__(self, path):
        self.container = av.open(
            str(path),
            options={
                "hwaccel": "cuda",  # request NVDEC
                "hwaccel_output_format": "cuda",  # keep frames on GPU until to_ndarray()
            },
        )
        self.stream = self.container.streams.video[0]
        self.fps = float(self.stream.average_rate) if self.stream.average_rate else 25.0
        self.width = self.stream.codec_context.width
        self.height = self.stream.codec_context.height
        self.decode_time = 0.0

    def __iter__(self):
        for frame in self.container.decode(self.stream):
            start = time.time()
            arr = frame.to_rgb().to_ndarray()  # decode + copy
            self.decode_time += time.time() - start
            yield arr

    def close(self):
        self.container.close()


# ---------------------------- Helpers ---------------------------------
def load_clip(model_name="ViT-B-32", device=None):
    model_name = model_name.replace("/", "-")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained="openai"
    )
    model.eval().requires_grad_(False)
    model = model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device


def draw_box(
    image: Image.Image, xyxy, label_text: str, color=(255, 0, 0)
) -> Image.Image:
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


# ---------------------------- Globals & Labels -----------------------------
_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_TEXT_FEATURES = None

YOLO_WEIGHTS = Path("./models/bestyolo.onnx")
CLIP_MODEL_NAME = "ViT-B-32"
LABELS = [
    "graffiti on public objects",
    "a photo of trash on the street",
    "clean walls and streets",
]


# ---------------------------- Lazy-load models -----------------------------
def _lazy_load_models():
    global _YOLO_MODEL, _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE, _TEXT_FEATURES

    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(str(YOLO_WEIGHTS))  # ONNX loaded via ORT

    if _CLIP_MODEL is None:
        _CLIP_MODEL, _CLIP_PREPROCESS, tokenizer, _CLIP_DEVICE = load_clip(
            CLIP_MODEL_NAME
        )
        with torch.no_grad():
            text_inputs = tokenizer(LABELS).to(_CLIP_DEVICE)
            _TEXT_FEATURES = _CLIP_MODEL.encode_text(text_inputs)
            _TEXT_FEATURES /= _TEXT_FEATURES.norm(dim=-1, keepdim=True)


# ---------------------------- Core processing -----------------------------
def process_video(
    video_input,
    yolo_conf: float,
    clip_conf: float,
    frame_interval: int,
    clip_batch: bool,
):
    # Resolve video path
    video_path = video_input.get("name") if isinstance(video_input, dict) else video_input
    if not video_path or not Path(video_path).exists():
        raise ValueError("Invalid or missing video input")

    _lazy_load_models()

    # Prepare output folder
    base_out = Path("outputs")
    base_out.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = base_out / timestamp
    folder.mkdir(exist_ok=True)
    annotated_video_path = folder / f"annotated_{uuid.uuid4().hex[:8]}.mp4"
    metrics_path = folder / "metrics.txt"

    # Decoder & writer
    decoder = NVDecoder(video_path)
    fps, width, height = decoder.fps, decoder.width, decoder.height
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(annotated_video_path), fourcc, fps, (width, height))

    # Metrics
    yolo_time = clip_time = encoding_time = 0.0
    yolo_calls = clip_calls = detected_objects = frame_count = 0
    device_arg = 0 if torch.cuda.is_available() else "cpu"

    # ------------- frame loop -------------
    for idx, frame_rgb in enumerate(decoder):
        frame_count += 1
        pil_frame = Image.fromarray(frame_rgb)

        if idx % frame_interval == 0:
            # YOLO
            t0 = time.time()
            results = _YOLO_MODEL.predict(
                source=pil_frame, conf=yolo_conf, device=device_arg, verbose=False
            )[0]
            yolo_time += time.time() - t0
            yolo_calls += 1

            dets = results.boxes.data.cpu().numpy()
            if dets.size:
                dets = dets[dets[:, 4] >= yolo_conf]

            if dets.size:
                if clip_batch:
                    # -------- batch CLIP -----------
                    patches = [
                        pil_frame.crop((int(x1), int(y1), int(x2), int(y2)))
                        for x1, y1, x2, y2, *_ in dets
                    ]
                    batch = torch.stack(
                        [_CLIP_PREPROCESS(p) for p in patches]
                    ).to(_CLIP_DEVICE)

                    t1 = time.time()
                    with torch.no_grad():
                        img_feat = _CLIP_MODEL.encode_image(batch)
                        img_feat /= img_feat.norm(dim=-1, keepdim=True)
                        sims = (100.0 * img_feat @ _TEXT_FEATURES.T).softmax(dim=-1)
                    clip_time += time.time() - t1
                    clip_calls += 1  # one batched call

                    sims_np = sims.cpu().numpy()

                    for (x1, y1, x2, y2, y_conf, cls), sim_vec in zip(dets, sims_np):
                        top_idx = int(sim_vec.argmax())
                        c_score = float(sim_vec[top_idx])
                        if c_score < clip_conf:
                            continue
                        detected_objects += 1
                        label = LABELS[top_idx]
                        color = (255, 0, 0) if int(cls) == 0 else (0, 255, 0)
                        pil_frame = draw_box(
                            pil_frame,
                            (x1, y1, x2, y2),
                            f"{label}: y{y_conf:.2f} c{c_score:.2f}",
                            color,
                        )
                else:
                    # -------- per-crop CLIP -------
                    for x1, y1, x2, y2, y_conf, cls in dets:
                        patch = pil_frame.crop((int(x1), int(y1), int(x2), int(y2)))
                        patch_t = _CLIP_PREPROCESS(patch).unsqueeze(0).to(_CLIP_DEVICE)

                        t1 = time.time()
                        with torch.no_grad():
                            img_feat = _CLIP_MODEL.encode_image(patch_t)
                            img_feat /= img_feat.norm(dim=-1, keepdim=True)
                            sim = (
                                100.0 * img_feat @ _TEXT_FEATURES.T
                            ).softmax(dim=-1).squeeze(0)
                        clip_time += time.time() - t1
                        clip_calls += 1

                        top_idx = int(sim.argmax())
                        c_score = float(sim[top_idx])
                        if c_score < clip_conf:
                            continue
                        detected_objects += 1
                        label = LABELS[top_idx]
                        color = (255, 0, 0) if int(cls) == 0 else (0, 255, 0)
                        pil_frame = draw_box(
                            pil_frame,
                            (x1, y1, x2, y2),
                            f"{label}: y{y_conf:.2f} c{c_score:.2f}",
                            color,
                        )

        # Encode / write
        t2 = time.time()
        writer.write(cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR))
        encoding_time += time.time() - t2

    # ------------ cleanup --------------
    decoder.close()
    writer.release()

    # Metrics summary
    decode_time = decoder.decode_time
    avg_yolo = (yolo_time / yolo_calls) if yolo_calls else 0.0
    avg_clip = (clip_time / clip_calls) if clip_calls else 0.0
    total_inference = yolo_time + clip_time

    metrics_text = "\n".join(
        [
            f"Frame dimensions: {width}x{height}",
            f"Total frames processed: {frame_count}",
            f"Objects detected: {detected_objects}",
            f"Average YOLO inference time: {avg_yolo:.4f} s over {yolo_calls} calls",
            f"Average CLIP inference time: {avg_clip:.4f} s over {clip_calls} calls",
            f"Total inference time (YOLO+CLIP): {total_inference:.4f} s",
            f"Total decode time: {decode_time:.4f} s",
            f"Total encode time: {encoding_time:.4f} s",
            f"Batch CLIP: {clip_batch}",
            f"Output folder: {folder.resolve()}",
        ]
    )

    with open(metrics_path, "w") as f:
        f.write(metrics_text)

    return str(annotated_video_path), metrics_text


# ---------------------------- Gradio UI -----------------------------
demo = gr.Blocks(title="Graffiti & Trash Detector (Profiling Edition)")

with demo:
    gr.Markdown(
        "## Graffiti & Trash Detector 🖌️🗑️\nUpload a video and profile GPU/CPU timings."
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(
                label="Input video", sources=["upload"], interactive=True
            )
            run_btn = gr.Button("Run pipeline 🚀", variant="primary", size="sm")
        with gr.Column(scale=1):
            yolo_slider = gr.Slider(
                0.05, 1.0, step=0.05, value=0.25, label="YOLO confidence threshold"
            )
            clip_slider = gr.Slider(
                0.05, 1.0, step=0.05, value=0.30, label="CLIP confidence threshold"
            )
            interval_slider = gr.Slider(
                1, 10, step=1, value=1, label="Frame sampling interval (N)"
            )
            batch_toggle = gr.Checkbox(
                value=False, label="Batch CLIP inference (per-frame)"
            )

    video_output = gr.Video(label="Annotated output", interactive=False)
    metrics_output = gr.Textbox(
        label="Performance metrics (s)", lines=12, interactive=False
    )

    run_btn.click(
        fn=process_video,
        inputs=[
            video_input,
            yolo_slider,
            clip_slider,
            interval_slider,
            batch_toggle,
        ],
        outputs=[video_output, metrics_output],
    )

if __name__ == "__main__":
    demo.launch(share=True)
