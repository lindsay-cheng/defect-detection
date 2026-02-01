"""
analyze validation set composition and class distribution
helps identify if validation set has class imbalance
"""
import os
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def analyze_validation_split(data_yaml_path: str = "dataset/data.yaml", 
                             images_dir: str = "dataset/data/images",
                             labels_dir: str = "dataset/data/labels"):
    """
    analyze the validation split to check for class imbalance
    
    args:
        data_yaml_path: path to data.yaml config
        images_dir: directory containing images
        labels_dir: directory containing label files
    """
    
    # check if data.yaml exists
    if not os.path.exists(data_yaml_path):
        print(f"warning: {data_yaml_path} not found")
        print("yolo may have auto-split your data without a config file")
        print("\nanalyzing all images in dataset...")
    
    # get all image files
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)
    
    if not images_path.exists():
        print(f"error: images directory not found at {images_dir}")
        return
    
    if not labels_path.exists():
        print(f"error: labels directory not found at {labels_dir}")
        return
    
    # get all image and label files
    image_files = sorted(list(images_path.glob("*.jpg")) + list(images_path.glob("*.png")))
    label_files = sorted(list(labels_path.glob("*.txt")))
    
    print(f"\ntotal images: {len(image_files)}")
    print(f"total labels: {len(label_files)}")
    
    # analyze class distribution across all data
    class_counts = Counter()
    images_per_class = {0: [], 1: [], 2: [], 3: []}  # assuming 4 classes
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            
        image_name = label_file.stem
        
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                class_counts[class_id] += 1
                
                # track which images contain each class
                if class_id not in images_per_class:
                    images_per_class[class_id] = []
                if image_name not in images_per_class[class_id]:
                    images_per_class[class_id].append(image_name)
    
    # class names (update based on your dataset)
    class_names = {
        0: "good",
        1: "low_water", 
        2: "no_cap",
        3: "no_label"
    }
    
    print("\n" + "="*60)
    print("OVERALL CLASS DISTRIBUTION")
    print("="*60)
    
    total_instances = sum(class_counts.values())
    for class_id in sorted(class_counts.keys()):
        count = class_counts[class_id]
        percentage = (count / total_instances) * 100
        class_name = class_names.get(class_id, f"class_{class_id}")
        print(f"{class_name:15} (class {class_id}): {count:4} instances ({percentage:5.1f}%)")
    
    print(f"\ntotal instances: {total_instances}")
    
    # show images per class
    print("\n" + "="*60)
    print("IMAGES CONTAINING EACH CLASS")
    print("="*60)
    
    for class_id in sorted(images_per_class.keys()):
        if images_per_class[class_id]:
            class_name = class_names.get(class_id, f"class_{class_id}")
            count = len(images_per_class[class_id])
            percentage = (count / len(label_files)) * 100
            print(f"{class_name:15} (class {class_id}): {count:4} images ({percentage:5.1f}%)")
    
    # estimate typical yolo split (80/20 train/val)
    print("\n" + "="*60)
    print("ESTIMATED VALIDATION SET (assuming 20% split)")
    print("="*60)
    
    val_size = int(len(image_files) * 0.2)
    print(f"estimated val images: ~{val_size}")
    
    # yolo typically uses the last 20% of sorted files for validation
    # let's check the class distribution in the last 20%
    val_start_idx = len(label_files) - val_size
    val_label_files = label_files[val_start_idx:]
    
    val_class_counts = Counter()
    for label_file in val_label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            if parts:
                class_id = int(parts[0])
                val_class_counts[class_id] += 1
    
    print("\nclass distribution in estimated val set:")
    val_total = sum(val_class_counts.values())
    
    if val_total > 0:
        for class_id in sorted(val_class_counts.keys()):
            count = val_class_counts[class_id]
            percentage = (count / val_total) * 100
            class_name = class_names.get(class_id, f"class_{class_id}")
            print(f"{class_name:15} (class {class_id}): {count:4} instances ({percentage:5.1f}%)")
        
        print(f"\ntotal val instances: {val_total}")
        
        # check for severe imbalance
        print("\n" + "="*60)
        print("IMBALANCE ANALYSIS")
        print("="*60)
        
        if val_class_counts:
            max_count = max(val_class_counts.values())
            min_count = min(val_class_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            print(f"imbalance ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 10:
                print("⚠️  WARNING: severe class imbalance detected!")
                print("   this can inflate accuracy metrics and learning curves")
            elif imbalance_ratio > 5:
                print("⚠️  moderate class imbalance detected")
            else:
                print("✓ class distribution is relatively balanced")
    
    # list validation images
    print("\n" + "="*60)
    print("ESTIMATED VALIDATION IMAGES (first 20)")
    print("="*60)
    
    val_images = sorted([f.name for f in image_files])[val_start_idx:]
    for i, img in enumerate(val_images[:20]):
        print(f"{i+1:3}. {img}")
    
    if len(val_images) > 20:
        print(f"... and {len(val_images) - 20} more")
    
    return {
        'total_images': len(image_files),
        'val_size': val_size,
        'class_counts': class_counts,
        'val_class_counts': val_class_counts,
        'val_images': val_images
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="analyze validation set composition")
    parser.add_argument("--data", type=str, default="dataset/data.yaml",
                       help="path to data.yaml")
    parser.add_argument("--images", type=str, default="dataset/data/images",
                       help="path to images directory")
    parser.add_argument("--labels", type=str, default="dataset/data/labels",
                       help="path to labels directory")
    
    args = parser.parse_args()
    
    analyze_validation_split(args.data, args.images, args.labels)
