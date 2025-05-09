import cv2
import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
import numpy as np
import time

# load the model and processor
print("Loading model ...")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device).eval()

# Open video
cap = cv2.VideoCapture("clipped_2.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
ret, frame = cap.read()
height, width = frame.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter("output_owlv2.avi", fourcc, fps, (width, height))

frame_idx = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

print("Processing video...")
start_time = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1

    if frame_idx % 3 != 0:
        out.write(frame)
        continue

    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    prompts = [["graffiti letters", "painted graffiti drawings", "graffiti"]]
    inputs = processor(images=image, text=prompts, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(outputs, target_sizes=target_sizes, threshold=0.2)[0]

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = map(int, box.tolist())
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{prompts[0][label]}: {score:.2f}"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    out.write(frame)

end_time = time.time()

print(f"Inference Time: {end_time-start_time}")
cap.release()
out.release()
print("Finished inferencing!")