# YOLO + CLIP Vision Pipeline
A multi-model vision inference pipeline integrating YOLO, and CLIP, with evaluation tools and an interactive Gradio UI.

## Contents
**model_evaluation**: evaluation of different detection models

**yolo_clip**: YOLO+CLIP python implementation

**gradio**: demo Gradio deployment

**yolo11_configs**: YOLOv11 training results

**Owlv2_inference**: OWLv2 video inference

## Running the Gradio Demo
### Prerequisites
Ensure you are in the gradio directory. Install the required dependencies:

```pip install -r requirements.txt```

Run the Gradio batch inference demo with:

```python gradio_app_onnx_batchinference.py```

## Video Demo
[Watch Demo on YouTube](https://www.youtube.com/watch?v=z7P4RkwmXOI)

## Dataset
[Link to training dataset](https://drive.google.com/drive/folders/1DGGo6FSN5iE1alCq1SkLNr73ki-p-laC?usp=sharing)
