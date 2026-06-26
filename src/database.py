"""
Database configuration and session management for SQLite backend.
"""

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

# Store database in the backend project directory
DATABASE_PATH = Path(__file__).resolve().parent.parent / "pulseai.db"
DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
DATABASE_URL = f"sqlite:///{DATABASE_PATH.as_posix()}"

# Use default connection pool (QueuePool) to allow proper concurrent sessions
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
        print(f"[OK] Database ready at: {DATABASE_PATH}")
    except OperationalError as exc:
        # Handle stale journal files on Windows
        message = str(exc).lower()
        if "disk i/o error" not in message:
            raise

        engine.dispose()
        for suffix in ("-journal", "-wal", "-shm"):
            lock_file = Path(f"{DATABASE_PATH}{suffix}")
            if lock_file.exists():
                lock_file.unlink()

        Base.metadata.create_all(bind=engine)
        print(f"[OK] Database ready at: {DATABASE_PATH} (after clearing stale journal)")
