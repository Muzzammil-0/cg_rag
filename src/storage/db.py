import sqlite3
from pathlib import Path

import os
DB_PATH = Path("/tmp/rag.db") if os.getenv("SPACE_ID") else Path("data/rag.db")

def get_connection():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(DB_PATH)

def init_db():
    conn = get_connection()
    cur = conn.cursor()
    cur.executescript("""
        CREATE TABLE IF NOT EXISTS queries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT,
            citations TEXT,
            confidence REAL,
            retrieved_top_source TEXT,
            retrieved_top_section TEXT,
            match_type TEXT,
            latency_ms INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS eval_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            recall_at_1 REAL,
            recall_at_3 REAL,
            recall_at_5 REAL,
            mrr REAL,
            not_found_passed INTEGER,
            not_found_total INTEGER,
            total_questions INTEGER,
            notes TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print(f"Initialized DB at {DB_PATH}")