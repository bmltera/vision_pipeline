"""
app.py – Flask webcam inference with live FPS stats, tunable sliders,
         and an optional batch CLIP-inference mode.
"""

import json
import time
from collections import deque
from pathlib import Path
from typing import Deque, List, Tuple

import cv2
import numpy as np
import torch
from flask import Flask, Response, render_template_string, request
from PIL import Image, ImageDraw, ImageFont

# ---------------- Dependency checks ----------------
try:
    import onnxruntime as ort
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        print("[WARN] ONNX Runtime built without GPU support – falling back to CPU")
except ImportError as e:
    raise RuntimeError("[ERROR] onnxruntime-gpu missing. Install with `pip install onnxruntime-gpu`.\n" + str(e))

try:
    from ultralytics import YOLO
except ImportError as e:
    raise RuntimeError("[ERROR] ultralytics not found. Install with `pip install ultralytics`.\n" + str(e))

try:
    import open_clip
except ImportError as e:
    raise RuntimeError("[ERROR] open_clip_torch not found. Install with `pip install open_clip_torch`.\n" + str(e))

# ---------------- Helper functions ----------------
def load_clip(model_name="ViT-B-32", device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = model_name.replace("/", "-")
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

# ---------------- Configuration -------------------
YOLO_WEIGHTS = Path("./models/bestyolo.onnx")
CLIP_MODEL_NAME = "ViT-B-32"
LABELS: List[str] = [
    "graffiti on public objects",
    "a photo of trash on the street",
    "clean walls and streets",
]

# Values changed at runtime by the UI
CONFIG = {
    "frame_interval": 1,   # process every Nth frame
    "yolo_conf": 0.25,
    "clip_conf": 0.30,
    "clip_batch": False,   # NEW – batch CLIP inference toggle
}

# ---------------- Lazy-loaded models --------------
_YOLO_MODEL = None
_CLIP_MODEL = None
_CLIP_PREPROCESS = None
_CLIP_DEVICE = None
_TEXT_FEATURES = None


def _lazy_load_models():
    global _YOLO_MODEL, _CLIP_MODEL, _CLIP_PREPROCESS, _CLIP_DEVICE, _TEXT_FEATURES
    if _YOLO_MODEL is None:
        _YOLO_MODEL = YOLO(str(YOLO_WEIGHTS))
    if _CLIP_MODEL is None:
        _CLIP_MODEL, _CLIP_PREPROCESS, tokenizer, _CLIP_DEVICE = load_clip(CLIP_MODEL_NAME)
        with torch.no_grad():
            text_inputs = tokenizer(LABELS)
            _TEXT_FEATURES = _CLIP_MODEL.encode_text(text_inputs.to(_CLIP_DEVICE))
            _TEXT_FEATURES /= _TEXT_FEATURES.norm(dim=-1, keepdim=True)

# ---------------- Frame inference -----------------
def run_inference(frame_bgr: np.ndarray, frame_idx: int) -> np.ndarray:
    interval   = CONFIG["frame_interval"]
    yolo_conf  = CONFIG["yolo_conf"]
    clip_conf  = CONFIG["clip_conf"]
    clip_batch = CONFIG["clip_batch"]

    if frame_idx % interval != 0:
        return frame_bgr

    _lazy_load_models()

    pil_frame = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    device_arg = 0 if torch.cuda.is_available() else "cpu"
    results = _YOLO_MODEL.predict(
        source=pil_frame,
        conf=yolo_conf,
        device=device_arg,
        verbose=False,
    )[0]

    dets = results.boxes.data.cpu().numpy()
    if dets.size:
        dets = dets[dets[:, 4] >= yolo_conf]

    if dets.size == 0:
        return frame_bgr  # nothing to classify / draw

    # ------- CLIP classification -------
    if clip_batch:
        # collect all patches -> batch tensor
        patches = [pil_frame.crop((x1, y1, x2, y2)) for x1, y1, x2, y2, *_ in dets]
        batch = torch.stack([_CLIP_PREPROCESS(p) for p in patches]).to(_CLIP_DEVICE)
        with torch.no_grad():
            img_feat = _CLIP_MODEL.encode_image(batch)
            img_feat /= img_feat.norm(dim=-1, keepdim=True)
            sims = (100.0 * img_feat @ _TEXT_FEATURES.T).softmax(dim=-1)
        sims_cpu = sims.cpu().numpy()  # shape (N, #labels)
    else:
        sims_cpu = []

    # ------- draw boxes & labels -------
    for det_idx, (x1, y1, x2, y2, y_conf, cls) in enumerate(dets):
        if clip_batch:
            sim_vec = sims_cpu[det_idx]
            top_idx = int(sim_vec.argmax())
            c_score = float(sim_vec[top_idx])
        else:
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

    return cv2.cvtColor(np.array(pil_frame), cv2.COLOR_RGB2BGR)

# ---------------- Flask app & UI ------------------
app = Flask(__name__)

HTML_PAGE = """
<!doctype html>
<title>Graffiti & Trash Detector – Live</title>
<style>
  body { font-family: Arial, sans-serif; margin: 20px; }
  label { display: block; margin-top: 10px; }
  input[type=range] { width: 300px; }
</style>

<h2>Graffiti & Trash Detector (webcam)</h2>

<p>
  Avg FPS: <span id="fps_avg">0</span> |
  Avg FPS (5&nbsp;s): <span id="fps_avg5">0</span> |
  1&nbsp;% Lows: <span id="fps_low1">0</span>
</p>

<img src="{{ url_for('video_feed') }}" width="720" />

<h3>Inference settings</h3>

<label>
  Frame interval (process every Nth frame):
  <input type="range" id="interval" min="1" max="10" value="1">
  <span id="interval_val">1</span>
</label>

<label>
  YOLO confidence threshold:
  <input type="range" id="yolo" min="0.05" max="1.0" step="0.05" value="0.25">
  <span id="yolo_val">0.25</span>
</label>

<label>
  CLIP confidence threshold:
  <input type="range" id="clip" min="0.05" max="1.0" step="0.05" value="0.30">
  <span id="clip_val">0.30</span>
</label>

<label>
  <input type="checkbox" id="batch"> Batch CLIP inference
</label>

<script>
const sse = new EventSource("{{ url_for('fps_sse') }}");
sse.onmessage = (e) => {
  const d = JSON.parse(e.data);
  document.getElementById("fps_avg").textContent  = d.fps_avg.toFixed(2);
  document.getElementById("fps_avg5").textContent = d.fps_avg5.toFixed(2);
  document.getElementById("fps_low1").textContent = d.fps_1pct.toFixed(2);
};

function postConfig() {
  fetch("{{ url_for('config') }}", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      frame_interval: parseInt(document.getElementById("interval").value),
      yolo_conf:      parseFloat(document.getElementById("yolo").value),
      clip_conf:      parseFloat(document.getElementById("clip").value),
      clip_batch:     document.getElementById("batch").checked
    })
  });
}

["interval","yolo","clip","batch"].forEach(id => {
  const el = document.getElementById(id);
  const out = document.getElementById(id+"_val");
  if(el.type==="range"){
      el.addEventListener("input", () => { out.textContent = el.value; postConfig(); });
  }else{
      el.addEventListener("change", postConfig);
  }
});
</script>
"""

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

# ---------------- Video feed & metrics ------------
FRAME_TIMES: Deque[float] = deque(maxlen=600)    # ~20 s @ 30 FPS
TIME_WINDOW: Deque[Tuple[float, float]] = deque(maxlen=1800)

def make_stats() -> dict:
    """Return dict with fps_avg, fps_avg5, fps_1pct."""
    fps_avg = 1.0 / (sum(FRAME_TIMES) / len(FRAME_TIMES)) if FRAME_TIMES else 0.0

    now = time.time()
    last5 = [ft for ts, ft in TIME_WINDOW if now - ts <= 5.0]
    fps_avg5 = (len(last5) / sum(last5)) if last5 else 0.0

    if FRAME_TIMES:
        worst_cnt = max(1, len(FRAME_TIMES) // 100)
        worst = sorted(FRAME_TIMES, reverse=True)[:worst_cnt]
        fps_1pct = 1.0 / (sum(worst) / len(worst)) if worst else 0.0
    else:
        fps_1pct = 0.0

    return {"fps_avg": fps_avg, "fps_avg5": fps_avg5, "fps_1pct": fps_1pct}

def gen_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    frame_idx = 0
    while True:
        success, frame = cap.read()
        if not success:
            break

        start = time.time()
        annotated = run_inference(frame, frame_idx)
        ft = time.time() - start

        FRAME_TIMES.append(ft)
        TIME_WINDOW.append((time.time(), ft))
        frame_idx += 1

        fps_now = 1.0 / ft if ft > 0 else 0.0
        cv2.putText(
            annotated,
            f"FPS: {fps_now:5.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        ret, buf = cv2.imencode(".jpg", annotated)
        if not ret:
            continue
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"

    cap.release()

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/fps_sse")
def fps_sse():
    def stream():
        while True:
            time.sleep(1.0)
            yield f"data: {json.dumps(make_stats())}\n\n"
    return Response(stream(), mimetype="text/event-stream")

# --------------- Config endpoint ------------------
@app.route("/config", methods=["GET", "POST"])
def config():
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        CONFIG["frame_interval"] = max(1, int(data.get("frame_interval", CONFIG["frame_interval"])))
        CONFIG["yolo_conf"]      = float(data.get("yolo_conf", CONFIG["yolo_conf"]))
        CONFIG["clip_conf"]      = float(data.get("clip_conf", CONFIG["clip_conf"]))
        CONFIG["clip_batch"]     = bool(data.get("clip_batch", CONFIG["clip_batch"]))
        return ("", 204)
    return CONFIG

# ---------------- Main entry ----------------------
if __name__ == "__main__":
    # python app.py
    app.run(host="0.0.0.0", port=5000, threaded=True)
