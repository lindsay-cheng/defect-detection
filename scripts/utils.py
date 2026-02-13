"""
utility functions for the defect detection system
"""
import csv
from typing import List, Dict, Any

from backend.database import DefectDatabase


def export_to_csv(
    output_path: str = "defect_report.csv",
    db_path: str = "database/defects.db",
    limit: int = 1000
):
    """export defect records from the database to a csv file
    
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
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    
    print(f"exported {len(records)} records to {output_path}")


def get_database_stats(db_path: str = "database/defects.db", hours: int = 24):
    """print defect statistics for the last n hours"""
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
    """clear all records from database (prompts for confirmation)"""
    response = input("are you sure you want to delete all records? (yes/no): ")
    if response.lower() == 'yes':
        db = DefectDatabase(db_path)
        db.clear_all_records()
        db.close()
        print("all records deleted")
    else:
        print("operation cancelled")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("usage: python utils.py [export|stats|clear]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "export":
        export_to_csv()
    elif command == "stats":
        get_database_stats()
    elif command == "clear":
        clear_database()
    else:
        print(f"unknown command: {command}")
