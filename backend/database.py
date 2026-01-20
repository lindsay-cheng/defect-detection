"""
database module for storing defect detection records
"""
import sqlite3
import os
from datetime import datetime
from typing import Optional, List, Dict, Any


class DefectDatabase:
    """manages sqlite database for defect logging"""
    
    def __init__(self, db_path: str = "database/defects.db"):
        """initialize database connection
        
        args:
            db_path: path to sqlite database file
        """
        self.db_path = db_path
        
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
        
        self.connection = None
        self.cursor = None
        self._connect()
        self._create_tables()
    
    def _connect(self):
        """establish database connection"""
        self.connection = sqlite3.connect(self.db_path, check_same_thread=False)
        self.cursor = self.connection.cursor()
    
    def _create_tables(self):
        """create bottles and defect tables if they don't exist"""
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS bottles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_bottle TEXT NOT NULL UNIQUE,
                production_lot TEXT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL
            )
        """)
        
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS defect (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                id_bottle INTEGER NOT NULL,
                defect_type TEXT NOT NULL,
                confidence REAL,
                image_path TEXT,
                timestamp TEXT NOT NULL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                FOREIGN KEY (id_bottle) REFERENCES bottles (id)
            )
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_bottles_id_bottle ON bottles(id_bottle)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_defect_id_bottle ON defect(id_bottle)
        """)
        
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_defect_timestamp ON defect(timestamp)
        """)
        
        self.connection.commit()
    
    def insert_bottle(
        self,
        bottle_id: str,
        production_lot: str = None,
        status: str = "PASS"
    ) -> int:
        """insert or get existing bottle record
        
        args:
            bottle_id: unique bottle identifier (e.g., BTL_00001)
            production_lot: production lot number
            status: PASS or FAIL
        
        returns:
            bottle's primary key id
        """
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute(
            "SELECT id FROM bottles WHERE id_bottle = ?",
            (bottle_id,)
        )
        result = self.cursor.fetchone()
        
        if result:
            if status == "FAIL":
                self.cursor.execute(
                    "UPDATE bottles SET status = ? WHERE id_bottle = ?",
                    (status, bottle_id)
                )
                self.connection.commit()
            return result[0]
        self.cursor.execute("""
            INSERT INTO bottles (id_bottle, production_lot, timestamp, status)
            VALUES (?, ?, ?, ?)
        """, (bottle_id, production_lot, timestamp, status))
        
        self.connection.commit()
        return self.cursor.lastrowid
    
    def insert_defect(
        self,
        bottle_id: str,
        defect_type: str,
        confidence: float = None,
        image_path: str = None,
        production_lot: str = None,
        bbox: tuple = None
    ) -> int:
        """insert a defect record into the database
        
        args:
            bottle_id: unique identifier for the bottle (e.g., BTL_00001)
            defect_type: type of defect
            confidence: detection confidence score (0-1)
            image_path: path to saved defect image
            production_lot: production lot number
            bbox: bounding box coordinates (x, y, w, h)
        
        returns:
            inserted defect record id
        """
        timestamp = datetime.now().isoformat()
        
        bottle_pk = self.insert_bottle(bottle_id, production_lot, status="FAIL")
        
        bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (None, None, None, None)
        
        self.cursor.execute("""
            INSERT INTO defect (
                id_bottle, defect_type, confidence, image_path,
                timestamp, bbox_x, bbox_y, bbox_w, bbox_h
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            bottle_pk, defect_type, confidence, image_path,
            timestamp, bbox_x, bbox_y, bbox_w, bbox_h
        ))
        
        self.connection.commit()
        return self.cursor.lastrowid
    
    def get_defects(
        self,
        limit: int = 100,
        defect_type: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict[str, Any]]:
        """retrieve defect records with optional filtering
        
        args:
            limit: maximum number of records to return
            defect_type: filter by specific defect type
            start_date: filter records after this date (ISO format)
            end_date: filter records before this date (ISO format)
        
        returns:
            list of defect records as dictionaries
        """
        query = """
            SELECT 
                defect.id,
                bottles.id_bottle,
                bottles.production_lot,
                defect.defect_type,
                defect.confidence,
                defect.image_path,
                defect.timestamp,
                defect.bbox_x,
                defect.bbox_y,
                defect.bbox_w,
                defect.bbox_h
            FROM defect
            JOIN bottles ON defect.id_bottle = bottles.id
            WHERE 1=1
        """
        params = []
        
        if defect_type:
            query += " AND defect.defect_type = ?"
            params.append(defect_type)
        
        if start_date:
            query += " AND defect.timestamp >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND defect.timestamp <= ?"
            params.append(end_date)
        
        query += " ORDER BY defect.timestamp DESC LIMIT ?"
        params.append(limit)
        
        self.cursor.execute(query, params)
        
        columns = [desc[0] for desc in self.cursor.description]
        results = []
        
        for row in self.cursor.fetchall():
            results.append(dict(zip(columns, row)))
        
        return results
    
    def get_defect_by_bottle_id(self, bottle_id: str) -> Optional[Dict[str, Any]]:
        """get defect record for a specific bottle
        
        args:
            bottle_id: unique bottle identifier (e.g., BTL_00001)
        
        returns:
            defect record or None if not found
        """
        self.cursor.execute("""
            SELECT 
                defect.id,
                bottles.id_bottle,
                bottles.production_lot,
                defect.defect_type,
                defect.confidence,
                defect.image_path,
                defect.timestamp
            FROM defect
            JOIN bottles ON defect.id_bottle = bottles.id
            WHERE bottles.id_bottle = ?
            ORDER BY defect.timestamp DESC
            LIMIT 1
        """, (bottle_id,))
        
        row = self.cursor.fetchone()
        if not row:
            return None
        
        columns = [desc[0] for desc in self.cursor.description]
        return dict(zip(columns, row))
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """get defect statistics for the last n hours
        
        args:
            hours: number of hours to look back
        
        returns:
            statistics dictionary
        """
        # calculate start time
        from datetime import timedelta
        start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        # total bottles inspected
        self.cursor.execute(
            "SELECT COUNT(*) FROM bottles WHERE timestamp >= ?",
            (start_time,)
        )
        total_bottles = self.cursor.fetchone()[0]
        
        # total defects
        self.cursor.execute(
            "SELECT COUNT(*) FROM defect WHERE timestamp >= ?",
            (start_time,)
        )
        total_defects = self.cursor.fetchone()[0]
        
        # defects by type
        self.cursor.execute("""
            SELECT defect_type, COUNT(*) as count
            FROM defect
            WHERE timestamp >= ?
            GROUP BY defect_type
        """, (start_time,))
        
        defects_by_type = {row[0]: row[1] for row in self.cursor.fetchall()}
        
        return {
            "total_bottles": total_bottles,
            "total_defects": total_defects,
            "defects_by_type": defects_by_type,
            "time_window_hours": hours
        }
    
    def clear_all_records(self):
        """delete all records from the database (use with caution)"""
        self.cursor.execute("DELETE FROM defect")
        self.cursor.execute("DELETE FROM bottles")
        self.connection.commit()
    
    def close(self):
        """close database connection"""
        if self.connection:
            self.connection.close()
    
    def __enter__(self):
        """context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """context manager exit"""
        self.close()


# helper function for quick database initialization
def init_database(db_path: str = "database/defects.db"):
    """initialize database with proper schema
    
    args:
        db_path: path to database file
    """
    db = DefectDatabase(db_path)
    db.close()
    print(f"database initialized at: {db_path}")
    return db_path


if __name__ == "__main__":
    # test database creation
    init_database()
