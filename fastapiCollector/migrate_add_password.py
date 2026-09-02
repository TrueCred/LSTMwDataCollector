import sqlite3

conn = sqlite3.connect('sentinel_lab.db')
c = conn.cursor()

# Add password_hash column if missing
c.execute("PRAGMA table_info(users)")
columns = [row[1] for row in c.fetchall()]

if "password_hash" not in columns:
    c.execute("ALTER TABLE users ADD COLUMN password_hash TEXT")
    conn.commit()
    print("Added 'password_hash' column to users table.")
else:
    print("Column 'password_hash' already exists.")

conn.close()
print("Migration complete.")
