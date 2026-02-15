"""
database module for storing defect detection records
thread-safe implementation using a dedicated DB worker thread
"""
import os
import sqlite3
import threading
from queue import Queue, Empty
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any


class _DefectDatabaseCore:
    """core sqlite operations (runs only on DB worker thread)"""
    
    def __init__(self, db_path: str):
        """initialize database connection and ensure schema exists
        
        args:
            db_path: path to sqlite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        
        # use default check_same_thread=True since this runs on single thread
        self.connection = sqlite3.connect(db_path)
        self.cursor = self.connection.cursor()
        self._create_tables()
    
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
        
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_bottles_id_bottle ON bottles(id_bottle)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_id_bottle ON defect(id_bottle)")
        self.cursor.execute("CREATE INDEX IF NOT EXISTS idx_defect_timestamp ON defect(timestamp)")
        self.connection.commit()
    
    def insert_bottle(self, bottle_id: str, production_lot: str = None, status: str = "PASS") -> int:
        """insert or get existing bottle record
        
        args:
            bottle_id: unique bottle identifier (e.g. BTL_00001)
            production_lot: production lot number
            status: PASS or FAIL
        
        returns:
            bottle's primary key id
        """
        self.cursor.execute("SELECT id FROM bottles WHERE id_bottle = ?", (bottle_id,))
        result = self.cursor.fetchone()
        
        if result:
            # update status to FAIL if needed (never downgrade FAIL -> PASS)
            if status == "FAIL":
                self.cursor.execute("UPDATE bottles SET status = ? WHERE id_bottle = ?", (status, bottle_id))
                self.connection.commit()
            return result[0]
        
        timestamp = datetime.now().isoformat()
        self.cursor.execute(
            "INSERT INTO bottles (id_bottle, production_lot, timestamp, status) VALUES (?, ?, ?, ?)",
            (bottle_id, production_lot, timestamp, status)
        )
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
        """insert a defect record (also upserts the bottle as FAIL)
        
        args:
            bottle_id: unique identifier for the bottle (e.g. BTL_00001)
            defect_type: type of defect
            confidence: detection confidence score (0-1)
            image_path: path to saved defect image
            production_lot: production lot number
            bbox: bounding box coordinates (x, y, w, h)
        
        returns:
            inserted defect record id
        """
        bottle_pk = self.insert_bottle(bottle_id, production_lot, status="FAIL")
        
        bbox_x, bbox_y, bbox_w, bbox_h = bbox if bbox else (None, None, None, None)
        timestamp = datetime.now().isoformat()
        
        self.cursor.execute("""
            INSERT INTO defect (
                id_bottle, defect_type, confidence, image_path,
                timestamp, bbox_x, bbox_y, bbox_w, bbox_h
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (bottle_pk, defect_type, confidence, image_path,
              timestamp, bbox_x, bbox_y, bbox_w, bbox_h))
        
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
                defect.id, bottles.id_bottle, bottles.production_lot,
                defect.defect_type, defect.confidence, defect.image_path,
                defect.timestamp, defect.bbox_x, defect.bbox_y,
                defect.bbox_w, defect.bbox_h
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
        return [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    
    def get_defect_by_bottle_id(self, bottle_id: str) -> Optional[Dict[str, Any]]:
        """get the most recent defect record for a specific bottle"""
        self.cursor.execute("""
            SELECT 
                defect.id, bottles.id_bottle, bottles.production_lot,
                defect.defect_type, defect.confidence, defect.image_path,
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
        """get defect statistics for the last n hours"""
        start_time = (datetime.now() - timedelta(hours=hours)).isoformat()
        
        self.cursor.execute("SELECT COUNT(*) FROM bottles WHERE timestamp >= ?", (start_time,))
        total_bottles = self.cursor.fetchone()[0]
        
        self.cursor.execute("SELECT COUNT(*) FROM defect WHERE timestamp >= ?", (start_time,))
        total_defects = self.cursor.fetchone()[0]
        
        self.cursor.execute("""
            SELECT defect_type, COUNT(*) as count
            FROM defect WHERE timestamp >= ?
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
        """delete all records from the database"""
        self.cursor.execute("DELETE FROM defect")
        self.cursor.execute("DELETE FROM bottles")
        self.connection.commit()
    
    def close(self):
        """close database connection"""
        if self.connection:
            self.connection.close()


class DefectDatabase:
    """thread-safe facade for defect database operations"""
    
    _SENTINEL = object()
    
    def __init__(self, db_path: str = "database/defects.db"):
        """initialize thread-safe database wrapper
        
        args:
            db_path: path to sqlite database file
        """
        self.db_path = db_path
        self._request_queue = Queue()
        self._worker_thread_id = None
        self._stopped = False
        
        # start the worker thread
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="DBWorker"
        )
        self._worker_thread.start()
    
    def _worker_loop(self):
        """worker thread main loop (processes DB requests)"""
        self._worker_thread_id = threading.get_ident()
        core = _DefectDatabaseCore(self.db_path)
        
        try:
            while True:
                task = self._request_queue.get()
                
                # check for shutdown sentinel
                if task is self._SENTINEL:
                    break
                
                method_name, args, kwargs, response_queue = task
                
                try:
                    result = getattr(core, method_name)(*args, **kwargs)
                    response_queue.put((True, result))
                except Exception as e:
                    response_queue.put((False, e))
        finally:
            core.close()
    
    def _execute(self, method_name: str, *args, **kwargs):
        """execute a DB operation via the worker thread
        
        args:
            method_name: name of the core method to call
            *args, **kwargs: arguments to pass to the method
        
        returns:
            result from the DB operation
        """
        # reentrancy guard: if called from worker thread, execute directly
        if threading.get_ident() == self._worker_thread_id:
            raise RuntimeError("cannot call DB methods from within DB worker thread")
        
        if self._stopped:
            raise RuntimeError("database has been closed")
        
        response_queue = Queue(maxsize=1)
        self._request_queue.put((method_name, args, kwargs, response_queue))
        
        ok, value = response_queue.get()
        if not ok:
            raise value
        return value
    
    def insert_bottle(self, bottle_id: str, production_lot: str = None, status: str = "PASS") -> int:
        """insert or get existing bottle record (thread-safe)"""
        return self._execute("insert_bottle", bottle_id, production_lot, status)
    
    def insert_defect(
        self,
        bottle_id: str,
        defect_type: str,
        confidence: float = None,
        image_path: str = None,
        production_lot: str = None,
        bbox: tuple = None
    ) -> int:
        """insert a defect record (thread-safe)"""
        return self._execute("insert_defect", bottle_id, defect_type, confidence, image_path, production_lot, bbox)
    
    def get_defects(
        self,
        limit: int = 100,
        defect_type: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> List[Dict[str, Any]]:
        """retrieve defect records with optional filtering (thread-safe)"""
        return self._execute("get_defects", limit, defect_type, start_date, end_date)
    
    def get_defect_by_bottle_id(self, bottle_id: str) -> Optional[Dict[str, Any]]:
        """get the most recent defect record for a specific bottle (thread-safe)"""
        return self._execute("get_defect_by_bottle_id", bottle_id)
    
    def get_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """get defect statistics for the last n hours (thread-safe)"""
        return self._execute("get_statistics", hours)
    
    def clear_all_records(self):
        """delete all records from the database (thread-safe)"""
        return self._execute("clear_all_records")
    
    def close(self):
        """shutdown the worker thread and close database connection"""
        if self._stopped:
            return
        
        self._stopped = True
        self._request_queue.put(self._SENTINEL)
        self._worker_thread.join(timeout=5.0)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def init_database(db_path: str = "database/defects.db"):
    """initialize database with proper schema (creates tables if needed)"""
    db = DefectDatabase(db_path)
    db.close()
    print(f"database initialized at: {db_path}")
    return db_path
