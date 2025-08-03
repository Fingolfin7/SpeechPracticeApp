import os
from datetime import datetime

from sqlalchemy import (
    create_engine, Column, Integer, String, Float
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
    transcript  = Column(String,  nullable=False)
    wer         = Column(Float,   nullable=False)
    clarity     = Column(Float,   nullable=False)
    score       = Column(Float, nullable=False)


def get_engine(db_path: str = "sessions.db"):
    full = os.path.abspath(db_path)
    return create_engine(f"sqlite:///{full}", echo=False)


def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)


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
    transcript: str,
    wer: float,
    clarity: float,
    score: float,
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
    )
    db.add(sess)
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