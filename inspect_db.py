import sqlite3
import tempfile
from pathlib import Path

db_path = Path(tempfile.gettempdir()) / "pulseai.db"
print("DB path:", db_path)
print("Exists:", db_path.exists(), "Size:", db_path.stat().st_size if db_path.exists() else 0)

conn = sqlite3.connect(str(db_path))
cur = conn.cursor()
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
print("Tables:", cur.fetchall())
cur.execute("SELECT id, email FROM users")
users = cur.fetchall()
print("Users:", users)
if users:
    uid = users[0][0]
    cur.execute("SELECT id, user_id, title FROM conversations")
    all_convs = cur.fetchall()
    print("ALL conversations:", all_convs)
    cur.execute("SELECT id, user_id, title FROM conversations WHERE user_id=?", (uid,))
    user_convs = cur.fetchall()
    print("Conversations for user", uid[:8], ":", user_convs)
conn.close()
