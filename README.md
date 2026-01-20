# Real-Time Bottle Defect Detection System

Automated visual inspection system for identifying defects in plastic bottles using YOLO-based computer vision.

## Overview

This system uses YOLOv8 object detection to automatically detect and classify defective bottles in real-time. Designed for quality control in manufacturing environments.

Fine-tuning + accuracy WIP

## Training (image collection)

- images were collected by me and annotated in YOLO format
- dataset config: `dataset/data.yaml`
- classes are defined by the trained model and `dataset/data.yaml`

## Defect classes
- `good`
- `low_water`
- `no_cap`
- `no_label`

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
