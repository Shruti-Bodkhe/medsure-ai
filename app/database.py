import sqlite3
import hashlib
from pathlib import Path
from typing import List, Dict, Optional

DB_PATH = Path("data/medsure.db")


def get_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def hash_password(password: str) -> str:
    salt = b"medsure_clinical_vault_v3_salt"
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000).hex()


def init_db():
    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            age INTEGER,
            gender TEXT,
            chronic_conditions TEXT DEFAULT '',
            allergies TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS chat_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            action TEXT DEFAULT 'ANSWER',
            risk_level TEXT DEFAULT 'low',
            sources_json TEXT DEFAULT '[]',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES chat_sessions (id)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS medical_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            filename TEXT NOT NULL,
            extracted_insight TEXT NOT NULL,
            governed_summary TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    """)

    # Ensure a default guest user (id=1) exists so unauthenticated queries work seamlessly
    cur.execute("SELECT id FROM users WHERE id = 1")
    if not cur.fetchone():
        cur.execute(
            "INSERT INTO users (id, email, password_hash, full_name, age, chronic_conditions, allergies) "
            "VALUES (1, 'guest@medsure.ai', ?, 'Guest Patient', 30, 'None reported', 'None reported')",
            (hash_password("guest123"),)
        )

    conn.commit()
    conn.close()


def build_patient_longitudinal_context(user_id: int, session_id: Optional[str] = None) -> str:
    """Builds a longitudinal summary of the patient's profile, past OCR reports, and recent conversation."""
    conn = get_db()
    parts = []

    user = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    if user and user["email"] != "guest@medsure.ai":
        profile_items = [f"Name: {user['full_name']}"]
        if user["age"]:
            profile_items.append(f"Age: {user['age']}")
        if user["gender"]:
            profile_items.append(f"Gender: {user['gender']}")
        if user["chronic_conditions"]:
            profile_items.append(f"Known Conditions: {user['chronic_conditions']}")
        if user["allergies"]:
            profile_items.append(f"Known Allergies: {user['allergies']}")
        parts.append("PATIENT HEALTH PROFILE: " + " | ".join(profile_items))

    records = conn.execute(
        "SELECT filename, extracted_insight, created_at FROM medical_records "
        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 3",
        (user_id,)
    ).fetchall()
    if records:
        rec_lines = [f"- [{r['filename']}]: {r['extracted_insight'][:350]}" for r in records]
        parts.append("SAVED MEDICAL REPORTS / PRESCRIPTIONS (OCR VAULT):\n" + "\n".join(rec_lines))

    if session_id:
        msgs = conn.execute(
            "SELECT role, content FROM chat_messages WHERE session_id = ? "
            "ORDER BY id DESC LIMIT 4",
            (session_id,)
        ).fetchall()
        if msgs:
            history_lines = [f"{m['role'].upper()}: {m['content'][:250]}" for m in reversed(msgs)]
            parts.append("RECENT CONVERSATION CONTEXT:\n" + "\n".join(history_lines))

    conn.close()
    return "\n\n".join(parts).strip()