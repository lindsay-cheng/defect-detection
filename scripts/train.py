"""
model training script for yolov8 defect detection
run this after collecting and labeling your dataset
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def train_model(
    data_yaml: str = "dataset/data.yaml",
    model_size: str = "n",
    epochs: int = 100,
    img_size: int = 640,
    batch_size: int = 16,
    device: str = "mps"
):
    """train yolov8 model on bottle defect dataset
    
    args:
        data_yaml: path to dataset configuration
        model_size: yolo model size (n/s/m/l/x) - n is fastest
        epochs: number of training epochs
        img_size: input image size
        batch_size: training batch size
        device: device to use (mps/cuda/cpu)
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: ultralytics not installed")
        print("install with: pip install ultralytics")
        return
    
    if not os.path.exists(data_yaml):
        print(f"error: dataset config not found at {data_yaml}")
        print("make sure you've created and labeled your dataset")
        return
    model_name = f"yolov8{model_size}.pt"
    print(f"loading pretrained model: {model_name}")
    model = YOLO(model_name)
    print(f"\nstarting training...")
    print(f"epochs: {epochs}")
    print(f"image size: {img_size}")
    print(f"batch size: {batch_size}")
    print(f"device: {device}")
    print("-" * 50)
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=img_size,
        batch=batch_size,
        device=device,
        project="runs/detect",
        name="train",
        exist_ok=True,
        patience=10,
        save=True,
        plots=True
    )
    
    print("\ntraining complete!")
    print(f"best model saved to: runs/detect/train/weights/best.pt")
    print(f"copy to models directory: cp runs/detect/train/weights/best.pt models/best.pt")
    
    print("\nrunning validation...")
    metrics = model.val()
    
    print(f"\nvalidation results:")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    
    return model, results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="train yolo bottle defect detector")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", 
                       help="path to data.yaml")
    parser.add_argument("--model", type=str, default="n", 
                       help="model size (n/s/m/l/x)")
    parser.add_argument("--epochs", type=int, default=100, 
                       help="number of epochs")
    parser.add_argument("--img-size", type=int, default=640, 
                       help="input image size")
    parser.add_argument("--batch", type=int, default=16, 
                       help="batch size")
    parser.add_argument("--device", type=str, default="mps",
                       help="device (mps/cuda/cpu)")
    
    args = parser.parse_args()
    
    train_model(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch,
        device=args.device
    )
