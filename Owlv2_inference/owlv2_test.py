import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Paths
image_dir = "path/to/test/images"
annotation_file = "path/to/annotations.json"  # COCO format
prompt_classes = ["graffiti letters", "painted graffiti drawings", "graffiti"]

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(device).eval()

# Load annotations
coco_gt = COCO(annotation_file)
image_ids = coco_gt.getImgIds()

detections = []

for image_id in tqdm(image_ids):
    info = coco_gt.loadImgs(image_id)[0]
    path = os.path.join(image_dir, info['file_name'])

    image = Image.open(path).convert("RGB")
    width, height = image.size

    prompts = [prompt_classes]
    inputs = processor(images=image, text=prompts, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]]).to(device)
    results = processor.post_process_grounded_object_detection(
        outputs, target_sizes=target_sizes, threshold=0.2
    )[0]

    for box, score, label in zip(results["boxes"], results["scores"], results["labels"]):
        x1, y1, x2, y2 = box.tolist()
        w, h = x2 - x1, y2 - y1
        detections.append({
            "image_id": image_id,
            "category_id": label.item() + 1,  # COCO category IDs usually start at 1
            "bbox": [x1, y1, w, h],
            "score": score.item()
        })

# Save detections to file
with open("owlv2_detections.json", "w") as f:
    json.dump(detections, f)

# Evaluate
coco_dt = coco_gt.loadRes("owlv2_detections.json")
coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()