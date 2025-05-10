import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np
import time
import os
import argparse  # <-- Add this

# -------------------------------
# Argument parser
# -------------------------------
parser = argparse.ArgumentParser(description="Run OWLv2 Object Detection on a video.")
parser.add_argument("video_path", type=str, help="Path to the input video file")
args = parser.parse_args()
video_path = args.video_path
# -------------------------------

# Load the model and processor
print("Loading model ...")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")
model.to(device).eval()

# Open video
base_name = os.path.splitext(os.path.basename(video_path))[0]
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
if not ret:
    raise ValueError("Failed to read the first frame. Check video path.")
height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(f"{base_name}_out.avi", fourcc, fps, (width, height))

frame_idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Processing video...")
total_inference_time = 0
last_boxes = []
last_scores = []
last_labels = []
start_processing_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % 3 == 0:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompts = [["graffiti letters", "painted graffiti drawings", "graffiti"]]
        inputs = processor(images=image, text=prompts, return_tensors="pt").to(device)
        start_time = time.time()
        with torch.no_grad():
            outputs = model(**inputs)
        end_time = time.time()

        total_inference_time += (end_time - start_time)

        target_sizes = torch.tensor([image.size[::-1]]).to(device)
        results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]
        last_boxes = results["boxes"]
        last_scores = results["scores"]
        last_labels = results["labels"]
    else:
        # Use the last known detections
        results = {"boxes": last_boxes, "scores": last_scores, "labels": last_labels}

    # Draw detections (whether new or reused)
    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{prompts[0][label]}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    out.write(frame)
end_processing_time=time.time()
avg_infer_time = total_inference_time / (frame_idx // 3)
total_processing_time = end_processing_time-start_processing_time
print(f"Total Processing Time: {total_processing_time:.4f}s")
print(f"Average Inference Time: {avg_infer_time:.4f}s")
print(f"FPS: {1 / avg_infer_time:.2f}")
cap.release()
out.release()
print("Finished inferencing!")
