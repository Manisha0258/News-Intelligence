"""
database.py
SQLite storage for swipe interactions (like/dislike/save).
"""

import sqlite3
import datetime
import os

DB_FILE = os.path.join(os.path.dirname(__file__), "news.db")


def init_db():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            article_link TEXT,
            action TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def log_interaction(article_link, action):
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO interactions (article_link, action, created_at) VALUES (?, ?, ?)",
        (article_link, action, datetime.datetime.now().isoformat())
    )
    conn.commit()
    conn.close()


def get_seen_links():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT article_link FROM interactions")
    seen = {row[0] for row in cur.fetchall()}
    conn.close()
    return seen


def get_liked_links():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT article_link FROM interactions WHERE action = 'like'")
    liked = [row[0] for row in cur.fetchall()]
    conn.close()
    return liked


def get_all_interactions():
    conn = sqlite3.connect(DB_FILE)
    cur = conn.cursor()
    cur.execute("SELECT article_link, action, created_at FROM interactions ORDER BY created_at DESC")
    rows = cur.fetchall()
    conn.close()
    return rows
