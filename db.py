import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Text
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class PracticeSession(Base):
    __tablename__ = "sessions"
    id          = Column(Integer, primary_key=True)
    timestamp   = Column(String,  nullable=False)
    script_name = Column(String,  nullable=False)
    script_text = Column(String,  nullable=False)
    audio_path  = Column(String,  nullable=False)
    # Allow empty values until scoring is run
    transcript  = Column(String,  nullable=True)
    wer         = Column(Float,   nullable=True)
    clarity     = Column(Float,   nullable=True)
    score       = Column(Float,   nullable=True)
    # JSON string of Whisper segments (optional)
    segments    = Column(Text,    nullable=True)


def get_engine(db_path: str = "sessions.db"):
    full = os.path.abspath(db_path)
    return create_engine(f"sqlite:///{full}", echo=False)


def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    # Lightweight migration: add 'segments' column if missing (SQLite)
    try:
        # Use a transaction and driver SQL for compatibility with SQLAlchemy 1.4/2.0
        with engine.begin() as conn:
            cols = conn.exec_driver_sql("PRAGMA table_info(sessions)").fetchall()
            names = {row[1] for row in cols}  # row[1] is the column name
            if "segments" not in names:
                conn.exec_driver_sql("ALTER TABLE sessions ADD COLUMN segments TEXT")
    except Exception:
        # best-effort; ignore if migration not applicable
        pass


def get_session(db_path: str = "sessions.db"):
    engine = get_engine(db_path)
    init_db(engine)
    return sessionmaker(bind=engine)()


def get_all_sessions(db):
    return db.query(PracticeSession).order_by(PracticeSession.timestamp.desc()).all()


def get_session_by_id(db, sess_id: int):
    return db.query(PracticeSession).get(sess_id)


def add_session(
    db,
    script_name: str,
    script_text: str,
    audio_path: str,
    transcript: str | None = None,
    wer: float | None = None,
    clarity: float | None = None,
    score: float | None = None,
    segments_json: str | None = None,
):
    ts = datetime.now().isoformat(timespec="seconds")
    sess = PracticeSession(
        timestamp=ts,
        script_name=script_name,
        script_text=script_text,
        audio_path=audio_path,
        transcript=transcript,
        wer=wer,
        clarity=clarity,
        score=score,
        segments=segments_json,
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def update_session_scores(
    db,
    sess_id: int,
    transcript: str,
    wer: float,
    clarity: float,
    score: float,
    segments_json: str | None = None,
):
    sess = db.query(PracticeSession).get(sess_id)
    if not sess:
        return None
    sess.transcript = transcript
    sess.wer = wer
    sess.clarity = clarity
    sess.score = score
    if segments_json is not None:
        sess.segments = segments_json
    db.commit()
    db.refresh(sess)
    return sess


def delete_session(db, sess_id: int):
    sess = db.query(PracticeSession).get(sess_id)
    if not sess:
        return
    # delete WAV file if no other session references it
    exists = db.query(PracticeSession).filter(
        PracticeSession.audio_path == sess.audio_path,
        PracticeSession.id != sess.id
    ).first()
    if not exists:
        try:
            os.remove(sess.audio_path)
        except Exception:
            pass
    db.delete(sess)
    db.commit()