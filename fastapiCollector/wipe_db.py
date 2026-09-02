import sqlite3

conn = sqlite3.connect('sentinel_lab.db')
c = conn.cursor()
c.execute('DELETE FROM raw_enrollment')
c.execute('DELETE FROM templates')  
c.execute('DELETE FROM users')
conn.commit()

users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
tmpls = c.execute("SELECT COUNT(*) FROM templates").fetchone()[0]
raws = c.execute("SELECT COUNT(*) FROM raw_enrollment").fetchone()[0]
print(f"Remaining: {users} users, {tmpls} templates, {raws} enrollments")
conn.close()
print("Database wiped clean. Ready for fresh enrollment.")
