import sqlite3
from pathlib import Path

db_path = Path("backend/sentinel_lab.db")
print(f"✓ Database file exists: {db_path.exists()}")
print(f"✓ Database size: {db_path.stat().st_size} bytes")

conn = sqlite3.connect(str(db_path))
cursor = conn.cursor()

# Check templates table
cursor.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='templates'")
schema = cursor.fetchone()[0]
print(f"\n✓ Templates table schema:\n{schema}")

# Check table info
cursor.execute("PRAGMA table_info(templates)")
columns = cursor.fetchall()
print(f"\n✓ Table columns ({len(columns)}):")
for col_id, name, col_type, notnull, default, pk in columns:
    print(f"  - {name} ({col_type})")

conn.close()
