# Real-Time Bottle Defect Detection System

Automated visual inspection system for identifying defects in Kirkland plastic bottles using YOLO-based computer vision.

## Overview

This system uses YOLOv8 object detection and Bytetrack object tracking to automatically detect and classify defective bottles in real-time. Designed for quality control in manufacturing line environments.

[![Demo video](assets/thumbnail.png)](https://www.youtube.com/watch?v=Nh-hHzurprs)
*Click to Watch Demo*


## Training (image collection)

- images were collected by me and annotated in YOLO format using Roboflow
- dataset config: `dataset/data.yaml`
- classes are defined by the trained model and `dataset/data.yaml`
- trained on external Google Colab for GPU access

## Defect classes

- `good`
- `low_water`
- `no_cap`
- `no_label`

## Preliminary Results (Controlled Environment)

The custom YOLOv8n model achieves >99% mAP@0.5 on an 80/20 stratified train/val split. Dataset consists of ~300 images captured with varied camera angles, zoom levels, lighting, and positions. While these metrics demonstrate the model's capability to learn defect patterns, the small dataset size and single-environment capture may limit generalization.

![Training Results](my_model/train/results.png)
*Training metrics over 100 epochs showing convergence*

**Important Limitations:**
- Small dataset size increases risk of overfitting
- Single capture environment may not generalize to diverse production settings
- Model performance on real-world manufacturing data remains to be validated

**WIP:**
- Expand dataset across multiple environments and bottle types
- K-fold cross-validation to better assess model robustness

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## How to run it

Tkinter GUI:

```bash
python app.py
```

CLI:

```bash
python scripts/detect.py --model my_model/train/weights/best.pt --source 0
```

## How to log data

Logging is automatic:

- sqlite database: `database/defects.db`
- defect image crops: `detections/`

Export CSV:

```bash
python scripts/utils.py export
```

## Weights

- trained weights: `my_model/train/weights/best.pt` (and `last.pt`)
- additional weights: `my_model/my_model.pt`
