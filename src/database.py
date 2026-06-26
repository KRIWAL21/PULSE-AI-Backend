"""
Database configuration and session management for SQLite backend.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker

# Check for cloud PostgreSQL URL (e.g. Supabase on Render)
DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL:
    # Render / Supabase sometimes provide postgres:// instead of postgresql://
    if DATABASE_URL.startswith("postgres://"):
        DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
    
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
else:
    # Fallback to local SQLite for offline dev
    DATABASE_PATH = Path(__file__).resolve().parent.parent / "pulseai.db"
    DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """Dependency for FastAPI to inject database session into routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables in the database (idempotent — safe to call on every startup)."""
    try:
        Base.metadata.create_all(bind=engine)
        print(f"[OK] Database tables verified and ready.")
    except OperationalError as exc:
        # Handle stale journal files on SQLite Windows
        message = str(exc).lower()
        if "disk i/o error" not in message or os.getenv("DATABASE_URL"):
            raise

        engine.dispose()
        db_path = Path(__file__).resolve().parent.parent / "pulseai.db"
        for suffix in ("-journal", "-wal", "-shm"):
            lock_file = Path(f"{db_path}{suffix}")
            if lock_file.exists():
                lock_file.unlink()

        Base.metadata.create_all(bind=engine)
        print("[OK] Database ready (after clearing stale SQLite journal).")
