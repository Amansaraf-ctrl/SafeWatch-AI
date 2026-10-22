import os
import sqlite3
import datetime
from pathlib import Path

# Optional MongoDB Atlas support
try:
    import certifi
    from pymongo import MongoClient
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

SQLITE_PATH = Path(r"C:\ppe detection\safewatch.db")
SNAPSHOTS_DIR = Path(r"C:\ppe detection\static\snapshots")
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_URI = "mongodb+srv://<username>:<password>@<cluster-url>.mongodb.net/?retryWrites=true&w=majority"
MONGO_URI = os.environ.get("MONGO_URI", DEFAULT_URI)

_mongo_client = None
_mongo_collection = None


def get_sqlite_conn():
    """Initializes and returns a SQLite connection with WAL mode enabled."""
    conn = sqlite3.connect(str(SQLITE_PATH), timeout=10.0)
    conn.row_factory = sqlite3.Row
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                camera_id TEXT NOT NULL,
                worker_name TEXT NOT NULL,
                status TEXT NOT NULL,
                missing_ppe TEXT NOT NULL,
                snapshot_file TEXT,
                face_crop_file TEXT,
                confidence REAL,
                resolved INTEGER DEFAULT 0
            )
        """)
    return conn


def get_mongo_collection():
    """Lazily initializes the MongoDB Atlas connection with TLS certifi support."""
    global _mongo_client, _mongo_collection
    if not HAS_PYMONGO:
        return None
    if _mongo_collection is None:
        try:
            if "<username>" in MONGO_URI or "<password>" in MONGO_URI:
                return None
            _mongo_client = MongoClient(
                MONGO_URI,
                tlsCAFile=certifi.where(),
                serverSelectionTimeoutMS=3000
            )
            db = _mongo_client["safewatch_db"]
            _mongo_collection = db["ppe_violations"]
        except Exception as e:
            print(f"[DB CONNECTION ERROR] MongoDB Atlas: {e}")
            return None
    return _mongo_collection


def log_violation_to_db(status, missing_items, camera_id="Cam_01", worker_name="Unidentified Worker",
                        snapshot_file=None, face_crop_file=None, confidence=0.0):
    """
    Logs violation event to local SQLite and optionally syncs with MongoDB Atlas.
    """
    now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
    missing_str = ", ".join(missing_items) if isinstance(missing_items, (list, tuple)) else str(missing_items)

    violation_id = None
    # 1. Local SQLite storage
    try:
        conn = get_sqlite_conn()
        with conn:
            cur = conn.execute("""
                INSERT INTO violations (
                    timestamp, camera_id, worker_name, status, 
                    missing_ppe, snapshot_file, face_crop_file, confidence, resolved
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """, (now_iso, camera_id, worker_name, status, missing_str, snapshot_file, face_crop_file, confidence))
            violation_id = cur.lastrowid
        conn.close()
    except Exception as e:
        print(f"[SQLITE ERROR] Failed to insert violation: {e}")

    # 2. MongoDB Atlas Sync (if connected)
    coll = get_mongo_collection()
    if coll is not None:
        doc = {
            "timestamp": datetime.datetime.now(datetime.timezone.utc),
            "camera_id": camera_id,
            "worker_name": worker_name,
            "status": status,
            "missing_ppe": missing_items,
            "snapshot_file": snapshot_file,
            "face_crop_file": face_crop_file,
            "confidence": confidence,
            "resolved": False
        }
        try:
            coll.insert_one(doc)
        except Exception as e:
            print(f"[MONGO ERROR] Failed to push to Atlas: {e}")

    return violation_id


def get_recent_violations(limit=50, resolved_filter=None):
    """Retrieves recent violation records for the frontend dashboard."""
    conn = get_sqlite_conn()
    try:
        query = "SELECT * FROM violations"
        params = []
        if resolved_filter is not None:
            query += " WHERE resolved = ?"
            params.append(1 if resolved_filter else 0)
        query += " ORDER BY id DESC LIMIT ?"
        params.append(limit)

        cur = conn.execute(query, params)
        rows = cur.fetchall()
        results = [dict(r) for r in rows]
        return results
    finally:
        conn.close()


def resolve_violation_in_db(violation_id):
    """Marks a violation as resolved."""
    conn = get_sqlite_conn()
    try:
        with conn:
            conn.execute("UPDATE violations SET resolved = 1 WHERE id = ?", (violation_id,))
        return True
    except Exception as e:
        print(f"[SQLITE ERROR] Resolve violation {violation_id}: {e}")
        return False
    finally:
        conn.close()


def get_analytics_summary():
    """Computes real-time analytics for the dashboard metrics bar."""
    conn = get_sqlite_conn()
    try:
        total = conn.execute("SELECT COUNT(*) FROM violations").fetchone()[0]
        active = conn.execute("SELECT COUNT(*) FROM violations WHERE resolved = 0").fetchone()[0]
        resolved = conn.execute("SELECT COUNT(*) FROM violations WHERE resolved = 1").fetchone()[0]
        
        # Missing PPE counts
        no_helmets = conn.execute("SELECT COUNT(*) FROM violations WHERE missing_ppe LIKE '%Helmet%'").fetchone()[0]
        no_shoes = conn.execute("SELECT COUNT(*) FROM violations WHERE missing_ppe LIKE '%Shoe%'").fetchone()[0]
        
        compliance_rate = 100.0 if total == 0 else max(0.0, round(((total - active) / max(total, 1)) * 100, 1))
        
        return {
            "total_incidents": total,
            "active_violations": active,
            "resolved_incidents": resolved,
            "compliance_rate": compliance_rate,
            "breakdown": {
                "no_helmet": no_helmets,
                "no_safety_shoes": no_shoes
            }
        }
    finally:
        conn.close()


if __name__ == "__main__":
    print("Testing db_logger...")
    conn = get_sqlite_conn()
    conn.close()
    print("SQLite ready at:", SQLITE_PATH)
