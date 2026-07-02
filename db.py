from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import create_engine, Column, Integer, String, Float, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()


class PracticeSession(Base):
    __tablename__ = "sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True, index=True)
    timestamp = Column(String, nullable=False)
    script_name = Column(String, nullable=False)
    script_text = Column(String, nullable=False)
    audio_path = Column(String, nullable=False)
    # Allow empty values until scoring is run
    transcript = Column(String, nullable=True)
    wer = Column(Float, nullable=True)
    clarity = Column(Float, nullable=True)
    score = Column(Float, nullable=True)
    # JSON string of Whisper segments (optional)
    segments = Column(Text, nullable=True)
    # New metrics
    cer = Column(Float, nullable=True)
    artic_rate = Column(Float, nullable=True)  # words/minute (speech time)
    pause_ratio = Column(Float, nullable=True)  # fraction of total time
    filled_pauses = Column(Float, nullable=True)  # count (float for simplicity)
    avg_conf = Column(Float, nullable=True)  # 0..1 normalized from avg_logprob


class SessionError(Base):
    __tablename__ = "session_errors"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=True, index=True)
    session_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(String, nullable=False, index=True)
    script_name = Column(String, nullable=True)
    ref_token = Column(String, nullable=True, index=True)
    hyp_token = Column(String, nullable=True)
    op = Column(String, nullable=False, index=True)  # sub | del | ins
    error_kind = Column(String, nullable=False, index=True)
    ref_start = Column(Integer, nullable=True)
    ref_end = Column(Integer, nullable=True)
    hyp_start = Column(Integer, nullable=True)
    hyp_end = Column(Integer, nullable=True)
    ref_local_start = Column(Integer, nullable=True)
    ref_local_end = Column(Integer, nullable=True)
    hyp_local_start = Column(Integer, nullable=True)
    hyp_local_end = Column(Integer, nullable=True)
    ref_token_len = Column(Integer, nullable=True)
    hyp_token_len = Column(Integer, nullable=True)
    confidence = Column(Float, nullable=True)
    segment_start = Column(Float, nullable=True)
    segment_end = Column(Float, nullable=True)


def get_engine(db_path: str = "sessions.db"):
    full = os.path.abspath(db_path)
    return create_engine(f"sqlite:///{full}", echo=False)


def init_db(engine=None):
    if engine is None:
        engine = get_engine()
    Base.metadata.create_all(engine)
    # Lightweight migration: add columns if missing (SQLite)
    try:
        with engine.begin() as conn:
            cols = conn.exec_driver_sql(
                "PRAGMA table_info(sessions)"
            ).fetchall()
            names = {row[1] for row in cols}  # row[1] is the column name

            def add_col(name: str, ddl: str) -> None:
                if name not in names:
                    conn.exec_driver_sql(
                        f"ALTER TABLE sessions ADD COLUMN {name} {ddl}"
                    )

            add_col("segments", "TEXT")
            add_col("user_id", "INTEGER")
            add_col("cer", "FLOAT")
            add_col("artic_rate", "FLOAT")
            add_col("pause_ratio", "FLOAT")
            add_col("filled_pauses", "FLOAT")
            add_col("avg_conf", "FLOAT")

            # session_errors migration for older DBs
            err_cols = conn.exec_driver_sql(
                "PRAGMA table_info(session_errors)"
            ).fetchall()
            err_names = {row[1] for row in err_cols}

            def add_err_col(name: str, ddl: str) -> None:
                if name not in err_names:
                    conn.exec_driver_sql(
                        f"ALTER TABLE session_errors ADD COLUMN {name} {ddl}"
                    )

            add_err_col("ref_local_start", "INTEGER")
            add_err_col("user_id", "INTEGER")
            add_err_col("ref_local_end", "INTEGER")
            add_err_col("hyp_local_start", "INTEGER")
            add_err_col("hyp_local_end", "INTEGER")
            add_err_col("ref_token_len", "INTEGER")
            add_err_col("hyp_token_len", "INTEGER")
    except Exception:
        # best-effort; ignore if migration not applicable
        pass


def get_session(db_path: str = "sessions.db"):
    engine = get_engine(db_path)
    init_db(engine)
    return sessionmaker(bind=engine)()


def get_all_sessions(db, user_id: int | None = None):
    query = db.query(PracticeSession)
    if user_id is not None:
        query = query.filter(PracticeSession.user_id == int(user_id))
    return query.order_by(PracticeSession.timestamp.desc(), PracticeSession.id.desc()).all()


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
    cer: float | None = None,
    artic_rate: float | None = None,
    pause_ratio: float | None = None,
    filled_pauses: float | None = None,
    avg_conf: float | None = None,
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
        cer=cer,
        artic_rate=artic_rate,
        pause_ratio=pause_ratio,
        filled_pauses=filled_pauses,
        avg_conf=avg_conf,
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return sess


def replace_session_errors(
    db,
    sess_id: int,
    timestamp: str,
    script_name: str | None,
    events: list[dict],
):
    """
    Replace all error events for a session.
    """
    db.query(SessionError).filter(SessionError.session_id == sess_id).delete()
    if not events:
        db.commit()
        return 0

    rows = []
    for ev in events:
        rows.append(
            SessionError(
                session_id=sess_id,
                timestamp=timestamp,
                script_name=script_name,
                ref_token=ev.get("ref_token"),
                hyp_token=ev.get("hyp_token"),
                op=ev.get("op", ""),
                error_kind=ev.get("error_kind", ""),
                ref_start=ev.get("ref_start"),
                ref_end=ev.get("ref_end"),
                hyp_start=ev.get("hyp_start"),
                hyp_end=ev.get("hyp_end"),
                ref_local_start=ev.get("ref_local_start"),
                ref_local_end=ev.get("ref_local_end"),
                hyp_local_start=ev.get("hyp_local_start"),
                hyp_local_end=ev.get("hyp_local_end"),
                ref_token_len=ev.get("ref_token_len"),
                hyp_token_len=ev.get("hyp_token_len"),
                confidence=ev.get("confidence"),
                segment_start=ev.get("segment_start"),
                segment_end=ev.get("segment_end"),
            )
        )
    db.add_all(rows)
    db.commit()
    return len(rows)
