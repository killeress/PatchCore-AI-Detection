import sqlite3, os

db = "test_results.db"
print(f"DB exists: {os.path.exists(db)}, Size: {os.path.getsize(db)} bytes")

conn = sqlite3.connect(db)
cur = conn.cursor()

# List tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print(f"Tables: {tables}")

for t in tables:
    count = cur.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    print(f"  {t}: {count} rows")

# Check inference_records
if "inference_records" in tables:
    cur.execute("SELECT id, glass_id, ai_judgment, created_at FROM inference_records ORDER BY id DESC LIMIT 5")
    rows = cur.fetchall()
    print(f"\nRecent records:")
    for r in rows:
        print(f"  id={r[0]} glass={r[1]} judgment={r[2]} created={r[3]}")
else:
    print("\nNo inference_records table found!")

conn.close()
