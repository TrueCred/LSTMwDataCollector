"""Reset database and recreate templates table."""
from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Optional


def reset_database(db_path: str | Path) -> None:
    """
    Connect to database, drop tables, and recreate templates table.
    
    Args:
        db_path: Path to sentinel_lab.db
    """
    db_path = Path(db_path)
    
    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    if not db_path.exists():
        print(f"Database not found at {db_path}, creating new one...")
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        # Drop existing tables
        cursor.execute("DROP TABLE IF EXISTS templates")
        cursor.execute("DROP TABLE IF EXISTS sessions")
        cursor.execute("DROP TABLE IF EXISTS live_sessions")
        print("Dropped existing tables")
        
        # Create templates table
        cursor.execute("""
            CREATE TABLE templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT UNIQUE NOT NULL,
                key_dna TEXT NOT NULL,
                scroll_dna TEXT NOT NULL,
                imu_stats TEXT NOT NULL,
                enrolled_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        print("Created templates table")
        
        conn.commit()
        print("Database reset complete")
    finally:
        conn.close()


if __name__ == "__main__":
    # Path relative to ml_pipeline folder: ../backend/sentinel_lab.db
    db_path = Path(__file__).parent.parent.parent / "backend" / "sentinel_lab.db"
    reset_database(db_path)
