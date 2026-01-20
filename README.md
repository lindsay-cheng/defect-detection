# 🔍 Real-Time Bottle Defect Detection System

Automated visual inspection system for identifying defects in plastic bottles using YOLO-based computer vision.

## 📋 Overview

This system uses YOLOv8 object detection to automatically detect and classify defective bottles in real-time. Designed for quality control in manufacturing environments.

### Key Features

- ✅ Real-time bottle inspection using webcam
- ✅ Defect classification into 4 categories:
  - Water level issues
  - Bottle/plastic damage (cracks)
  - Label problems (damaged/missing)
  - Wrong bottle types
- ✅ Unique ID assignment per bottle (object tracking)
- ✅ Automated image capture of defective items
- ✅ SQLite database logging
- ✅ Tkinter GUI for real-time monitoring

## 🛠️ Tech Stack

- **Computer Vision**: OpenCV
- **Deep Learning**: YOLOv8 (Ultralytics)
- **Object Tracking**: Custom centroid tracker
- **Database**: SQLite
- **GUI**: Tkinter (native desktop app)
- **Language**: Python 3.10+

## 📁 Project Structure

```
defect-detection/
│
├── dataset/                      # Training data
│   ├── images/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── data.yaml                 # Dataset configuration
│
├── models/
│   ├── yolov8n.pt               # Base pretrained model
│   └── best.pt                   # Trained model (after training)
│
├── scripts/
│   ├── train.py                  # Model training script
│   ├── detect.py                 # Real-time detection
│   └── utils.py                  # Utility functions
│
├── database/
│   └── defects.db               # SQLite database
│
├── app.py                        # Tkinter GUI application
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### 1. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd defect-detection

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Collection

Collect images of bottles using your iPhone or webcam:

1. **Normal bottles** (150+ images)
2. **Cracked bottles** (100+ images)
3. **Low water level** (100+ images)
4. **Damaged labels** (100+ images)

Place images in `dataset/images/` (will be organized during labeling).

### 3. Data Labeling

Use [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/heartexlabs/labelImg) to label your images:

1. Draw bounding boxes around bottles
2. Assign class labels (normal, crack, low_water, label_damage)
3. Export in YOLO format
4. Place in `dataset/` following the structure above

### 4. Training

```bash
# Initialize database
python -c "from scripts.utils import init_database; init_database()"

# Train model
python scripts/train.py
```

Training will save:
- Best model: `runs/detect/train/weights/best.pt`
- Training metrics and plots in `runs/detect/train/`

Copy best model to `models/`:
```bash
cp runs/detect/train/weights/best.pt models/best.pt
```

### 5. Run GUI Application

**Launch the Tkinter GUI:**
```bash
python app.py
```

The GUI provides:
- Live video feed with real-time detection
- Detection statistics and counters
- Model loading and configuration
- Database viewer

## 🖥️ GUI Features

The Tkinter application provides:

**Main Window:**
- Live video feed with bounding boxes
- Real-time FPS counter
- Detection statistics by type
- Recent detections log
- Model configuration panel

**Controls:**
- Start/Stop detection
- Adjust confidence threshold
- View database records
- Reset statistics

**Why Tkinter Instead of Streamlit:**
- ✅ Actually real-time (30+ FPS)
- ✅ Native desktop performance
- ✅ Direct OpenCV integration
- ✅ Low latency
- ✅ Perfect for control systems

Streamlit is good for data dashboards, but terrible for real-time video processing.

## 🎯 Usage Examples

### Real-time Detection
```python
from scripts.detect import detect_live

detect_live(
    model_path='models/best.pt',
    source=0,  # 0 for webcam
    conf_threshold=0.5,
    save_detections=True,
    log_to_db=True
)
```

### Process Video File
```python
detect_live(
    model_path='models/best.pt',
    source='test_video.mp4',
    conf_threshold=0.5
)
```

## 📈 Model Performance

(Add your metrics after training)

- **mAP50**: TBD
- **mAP50-95**: TBD
- **Inference Speed**: ~30 FPS on MacBook M4
- **Model Size**: ~6MB (YOLOv8n)

## 🔧 Configuration

Edit `dataset/data.yaml` to modify classes or dataset paths:

```yaml
path: ../dataset
train: images/train
val: images/val

nc: 4
names:
  0: normal
  1: crack
  2: low_water
  3: label_damage
```

## 🐛 Troubleshooting

**Issue**: Model not found
```bash
# Make sure you've trained and copied the model
cp runs/detect/train/weights/best.pt models/best.pt
```

**Issue**: No detections showing
- Check confidence threshold (lower it to 0.3)
- Verify model is trained on similar data
- Ensure good lighting conditions

**Issue**: Database errors
```bash
# Reinitialize database
python -c "from scripts.utils import init_database; init_database()"
```

## 📝 TODO

- [ ] Collect training data (500+ images)
- [ ] Label dataset in Roboflow
- [ ] Train initial model
- [ ] Test and iterate on model performance
- [ ] Add confidence-based rejection queue
- [ ] Benchmark multiple YOLO versions
- [ ] Deploy on production hardware

## 🤝 Contributing

This is a portfolio project. Feedback and suggestions welcome!

## 📄 License

MIT License

## 🙏 Acknowledgments

- Based on YOLO architecture by Ultralytics
- Inspired by industrial defect detection systems
- Built for quality control in manufacturing

---

**Author**: Your Name  
**Contact**: your.email@example.com  
**GitHub**: github.com/yourusername