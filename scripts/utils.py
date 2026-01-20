"""
utility functions for the defect detection system
"""
import os
import csv
from datetime import datetime
from typing import List, Dict, Any

from backend.database import DefectDatabase


def init_database(db_path: str = "database/defects.db"):
    """initialize database with proper schema
    
    args:
        db_path: path to database file
    """
    from backend.database import init_database as _init_db
    return _init_db(db_path)


def export_to_csv(
    output_path: str = "defect_report.csv",
    db_path: str = "database/defects.db",
    limit: int = 1000
):
    """export defect records to csv file
    
    args:
        output_path: path to output csv file
        db_path: path to database file
        limit: max number of records to export
    """
    db = DefectDatabase(db_path)
    records = db.get_defects(limit=limit)
    db.close()
    
    if not records:
        print("no records to export")
        return
    
    # write to csv
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = records[0].keys()
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        writer.writerows(records)
    
    print(f"exported {len(records)} records to {output_path}")


def get_database_stats(db_path: str = "database/defects.db", hours: int = 24):
    """print database statistics
    
    args:
        db_path: path to database file
        hours: time window for statistics
    """
    db = DefectDatabase(db_path)
    stats = db.get_statistics(hours=hours)
    db.close()
    
    print(f"\n=== defect statistics (last {hours} hours) ===")
    print(f"total defects: {stats['total_defects']}")
    print("\ndefects by type:")
    for defect_type, count in stats['defects_by_type'].items():
        print(f"  {defect_type}: {count}")
    print()


def clear_database(db_path: str = "database/defects.db"):
    """clear all records from database
    
    args:
        db_path: path to database file
    """
    response = input("are you sure you want to delete all records? (yes/no): ")
    if response.lower() == 'yes':
        db = DefectDatabase(db_path)
        db.clear_all_records()
        db.close()
        print("all records deleted")
    else:
        print("operation cancelled")


def create_project_structure():
    """create necessary project directories"""
    directories = [
        "dataset/images/train",
        "dataset/images/val",
        "dataset/images/test",
        "dataset/labels/train",
        "dataset/labels/val",
        "dataset/labels/test",
        "models",
        "database",
        "detections",
        "backend"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"created: {directory}")
    
    print("project structure created")


def validate_dataset(data_yaml_path: str = "dataset/data.yaml"):
    """validate dataset structure and files
    
    args:
        data_yaml_path: path to data.yaml configuration
    """
    import yaml
    
    if not os.path.exists(data_yaml_path):
        print(f"error: {data_yaml_path} not found")
        return False
    
    with open(data_yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    
    print("\n=== dataset validation ===")
    print(f"dataset path: {config.get('path', 'N/A')}")
    print(f"number of classes: {config.get('nc', 'N/A')}")
    print(f"class names: {config.get('names', [])}")
    
    # check if image directories exist
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(config.get('path', ''), 'images', split)
        label_dir = os.path.join(config.get('path', ''), 'labels', split)
        
        if os.path.exists(img_dir):
            num_images = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))])
            print(f"\n{split} images: {num_images}")
        else:
            print(f"\n{split} images: directory not found")
        
        if os.path.exists(label_dir):
            num_labels = len([f for f in os.listdir(label_dir) if f.endswith('.txt')])
            print(f"{split} labels: {num_labels}")
        else:
            print(f"{split} labels: directory not found")
    
    print()
    return True


def format_detection_summary(detections: List[Dict[str, Any]]) -> str:
    """format detection results as readable string
    
    args:
        detections: list of detection dictionaries
    
    returns:
        formatted string
    """
    if not detections:
        return "no detections"
    
    summary = []
    for det in detections:
        bottle_id = det.get('bottle_id', 'N/A')
        defect_type = det.get('defect_type', 'unknown')
        confidence = det.get('confidence', 0.0)
        
        summary.append(
            f"{bottle_id}: {defect_type} ({confidence:.2f})"
        )
    
    return " | ".join(summary)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("usage: python utils.py [init|export|stats|clear|validate|structure]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "init":
        init_database()
    elif command == "export":
        export_to_csv()
    elif command == "stats":
        get_database_stats()
    elif command == "clear":
        clear_database()
    elif command == "validate":
        validate_dataset()
    elif command == "structure":
        create_project_structure()
    else:
        print(f"unknown command: {command}")
