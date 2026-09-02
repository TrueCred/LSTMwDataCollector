# migrate_add_gaussian.py — One-shot migration to add gaussian_profile column
#
# Run: python migrate_add_gaussian.py
# Safe to run multiple times — checks if column exists first.

import sqlite3
from config import DB_URL

def migrate():
    # Extract file path from sqlite URL
    db_path = DB_URL.replace("sqlite:///./", "./")
    print(f"Migrating database: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Check if column already exists
    cursor.execute("PRAGMA table_info(templates)")
    columns = [row[1] for row in cursor.fetchall()]

    if "gaussian_profile" in columns:
        print("Column 'gaussian_profile' already exists. Nothing to do.")
    else:
        cursor.execute("ALTER TABLE templates ADD COLUMN gaussian_profile TEXT")
        conn.commit()
        print("Added 'gaussian_profile' column to templates table.")

    conn.close()
    print("Migration complete.")


if __name__ == "__main__":
    migrate()
