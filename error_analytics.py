from __future__ import annotations

import re
from collections import Counter
from datetime import datetime, timedelta
from typing import Dict, List

import db
from alignment_utils import (
    align_tokens,
    char_level_spans_for_substitution,
    tokenize_with_spans,
)
from highlight_theme import (
    REF_CONS_DELETE,
    REF_VOWEL_DELETE,
    REPLACE,
    TRANSCRIPT_CHAR_INSERT,
)

VOWELS = set("aeiou")


def clean_text_for_alignment(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text


def _char_kind_from_color(color: str) -> str:
    if color == REPLACE:
        return "char_replace"
    if color == TRANSCRIPT_CHAR_INSERT:
        return "char_insert"
    if color == REF_VOWEL_DELETE:
        return "vowel_delete"
    if color == REF_CONS_DELETE:
        return "cons_delete"
    return "char_replace"


def extract_error_events(ref_text: str, hyp_text: str) -> List[Dict]:
    """
    Convert alignment output into row-like error events for persistence.
    """
    ref_clean = clean_text_for_alignment(ref_text)
    hyp_clean = clean_text_for_alignment(hyp_text)
    ref_tokens = tokenize_with_spans(ref_clean)
    hyp_tokens = tokenize_with_spans(hyp_clean)

    ref_words = [t[0] for t in ref_tokens]
    hyp_words = [t[0] for t in hyp_tokens]
    ops = align_tokens(ref_words, hyp_words)

    events: List[Dict] = []
    for op, ri, hj in ops:
        if op.startswith("equal"):
            continue

        ref_tok, ref_start, ref_end = (None, None, None)
        hyp_tok, hyp_start, hyp_end = (None, None, None)
        if ri is not None:
            ref_tok, ref_start, ref_end = ref_tokens[ri]
        if hj is not None:
            hyp_tok, hyp_start, hyp_end = hyp_tokens[hj]

        if op == "del":
            events.append(
                {
                    "ref_token": ref_tok,
                    "hyp_token": None,
                    "op": "del",
                    "error_kind": "word_missing",
                    "ref_start": ref_start,
                    "ref_end": ref_end,
                    "hyp_start": None,
                    "hyp_end": None,
                    "ref_local_start": 0 if ref_tok else None,
                    "ref_local_end": len(ref_tok) if ref_tok else None,
                    "hyp_local_start": None,
                    "hyp_local_end": None,
                    "ref_token_len": len(ref_tok) if ref_tok else None,
                    "hyp_token_len": None,
                    "confidence": None,
                    "segment_start": None,
                    "segment_end": None,
                }
            )
            continue

        if op == "ins":
            events.append(
                {
                    "ref_token": None,
                    "hyp_token": hyp_tok,
                    "op": "ins",
                    "error_kind": "word_insert",
                    "ref_start": None,
                    "ref_end": None,
                    "hyp_start": hyp_start,
                    "hyp_end": hyp_end,
                    "ref_local_start": None,
                    "ref_local_end": None,
                    "hyp_local_start": 0 if hyp_tok else None,
                    "hyp_local_end": len(hyp_tok) if hyp_tok else None,
                    "ref_token_len": None,
                    "hyp_token_len": len(hyp_tok) if hyp_tok else None,
                    "confidence": None,
                    "segment_start": None,
                    "segment_end": None,
                }
            )
            continue

        if op == "sub" and ref_tok is not None and hyp_tok is not None:
            ref_spans, hyp_spans = char_level_spans_for_substitution(
                ref_tok, hyp_tok, int(ref_start), int(hyp_start)
            )

            if not ref_spans and not hyp_spans:
                events.append(
                    {
                        "ref_token": ref_tok,
                        "hyp_token": hyp_tok,
                        "op": "sub",
                        "error_kind": "char_replace",
                        "ref_start": ref_start,
                        "ref_end": ref_end,
                        "hyp_start": hyp_start,
                        "hyp_end": hyp_end,
                        "ref_local_start": 0,
                        "ref_local_end": len(ref_tok),
                        "hyp_local_start": 0,
                        "hyp_local_end": len(hyp_tok),
                        "ref_token_len": len(ref_tok),
                        "hyp_token_len": len(hyp_tok),
                        "confidence": None,
                        "segment_start": None,
                        "segment_end": None,
                    }
                )
                continue

            for s, e, color in ref_spans:
                events.append(
                    {
                        "ref_token": ref_tok,
                        "hyp_token": hyp_tok,
                        "op": "sub",
                        "error_kind": _char_kind_from_color(color),
                        "ref_start": s,
                        "ref_end": e,
                        "hyp_start": hyp_start,
                        "hyp_end": hyp_end,
                        "ref_local_start": int(s - ref_start),
                        "ref_local_end": int(e - ref_start),
                        "hyp_local_start": 0,
                        "hyp_local_end": len(hyp_tok),
                        "ref_token_len": len(ref_tok),
                        "hyp_token_len": len(hyp_tok),
                        "confidence": None,
                        "segment_start": None,
                        "segment_end": None,
                    }
                )
            for s, e, color in hyp_spans:
                events.append(
                    {
                        "ref_token": ref_tok,
                        "hyp_token": hyp_tok,
                        "op": "sub",
                        "error_kind": _char_kind_from_color(color),
                        "ref_start": ref_start,
                        "ref_end": ref_end,
                        "hyp_start": s,
                        "hyp_end": e,
                        "ref_local_start": 0,
                        "ref_local_end": len(ref_tok),
                        "hyp_local_start": int(s - hyp_start),
                        "hyp_local_end": int(e - hyp_start),
                        "ref_token_len": len(ref_tok),
                        "hyp_token_len": len(hyp_tok),
                        "confidence": None,
                        "segment_start": None,
                        "segment_end": None,
                    }
                )

    return events


def _parse_timestamp(ts: str) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


def _window_bounds(
    start_dt: datetime | None,
    end_dt: datetime | None,
) -> tuple[datetime, datetime, datetime | None, datetime | None]:
    now = datetime.now()
    recent_end = end_dt or now
    recent_start = start_dt or (recent_end - timedelta(days=30))
    if recent_end < recent_start:
        recent_start, recent_end = recent_end, recent_start
    span = recent_end - recent_start
    if span.total_seconds() <= 0:
        return recent_start, recent_end, None, None
    prev_end = recent_start
    prev_start = recent_start - span
    return recent_start, recent_end, prev_start, prev_end


def _scored_sessions_in_range(
    db_session,
    start_dt: datetime,
    end_dt: datetime,
    script_name: str | None = None,
) -> List[db.PracticeSession]:
    sessions = db.get_all_sessions(db_session)
    out: List[db.PracticeSession] = []
    for sess in sessions:
        if not sess.script_text or not sess.transcript:
            continue
        ts = _parse_timestamp(sess.timestamp)
        if ts is None:
            continue
        if start_dt <= ts <= end_dt:
            if script_name and script_name != "All scripts":
                if (sess.script_name or "") != script_name:
                    continue
            out.append(sess)
    return out


def _events_for_sessions(db_session, sessions: List[db.PracticeSession]):
    sess_ids = [s.id for s in sessions if getattr(s, "id", None) is not None]
    if not sess_ids:
        return []
    return (
        db_session.query(db.SessionError)
        .filter(db.SessionError.session_id.in_(sess_ids))
        .all()
    )


def _word_stats_for_sessions(
    db_session, sessions: List[db.PracticeSession]
) -> Dict[str, Dict[str, float]]:
    stats: Dict[str, Dict[str, float]] = {}
    if not sessions:
        return stats

    attempts = Counter()
    for sess in sessions:
        words = clean_text_for_alignment(sess.script_text).split()
        attempts.update(words)

    errors = Counter()
    for row in _events_for_sessions(db_session, sessions):
        if row.op not in ("sub", "del") or not row.ref_token:
            continue
        errors[row.ref_token.strip().lower()] += 1

    for word, attempts_count in attempts.items():
        if attempts_count <= 0:
            continue
        err_count = int(errors.get(word, 0))
        stats[word] = {
            "word": word,
            "attempts": float(attempts_count),
            "errors": float(err_count),
            "error_rate": float(err_count) / float(attempts_count),
        }
    return stats


def _summarize_recent_vs_prev(
    recent_stats: Dict[str, Dict[str, float]],
    prev_stats: Dict[str, Dict[str, float]],
    top_n: int,
    min_attempts: int,
    item_key: str,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    recent_filtered = {
        k: v
        for k, v in recent_stats.items()
        if int(v.get("attempts", 0)) >= int(min_attempts)
    }
    top_trouble = sorted(
        recent_filtered.values(),
        key=lambda x: (float(x.get("error_rate", 0.0)), float(x.get("errors", 0.0))),
        reverse=True,
    )[:top_n]

    deltas = []
    common = set(recent_filtered.keys()).intersection(prev_stats.keys())
    for k in common:
        prev = prev_stats[k]
        if int(prev.get("attempts", 0)) < int(min_attempts):
            continue
        recent_rate = float(recent_filtered[k].get("error_rate", 0.0))
        prev_rate = float(prev.get("error_rate", 0.0))
        deltas.append(
            {
                item_key: k,
                "recent_rate": recent_rate,
                "previous_rate": prev_rate,
                "delta": recent_rate - prev_rate,
                "recent_errors": float(recent_filtered[k].get("errors", 0.0)),
                "recent_attempts": float(recent_filtered[k].get("attempts", 0.0)),
            }
        )
    most_improved = sorted(deltas, key=lambda x: x["delta"])[:top_n]
    most_regressed = sorted(deltas, key=lambda x: x["delta"], reverse=True)[:top_n]
    return top_trouble, most_improved, most_regressed


def get_word_trend_summary(
    db_session,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    script_name: str | None = None,
    top_n: int = 8,
    min_attempts: int = 3,
) -> Dict[str, List[Dict]]:
    recent_start, recent_end, prev_start, prev_end = _window_bounds(
        start_dt, end_dt
    )
    recent_sessions = _scored_sessions_in_range(
        db_session, recent_start, recent_end, script_name=script_name
    )
    recent_stats = _word_stats_for_sessions(db_session, recent_sessions)

    if prev_start is None or prev_end is None:
        top = sorted(
            [
                v
                for v in recent_stats.values()
                if int(v.get("attempts", 0)) >= int(min_attempts)
            ],
            key=lambda x: (x["error_rate"], x["errors"]),
            reverse=True,
        )[:top_n]
        return {
            "top_trouble_words": top,
            "most_improved_words": [],
            "most_regressed_words": [],
            "recent_session_count": len(recent_sessions),
            "previous_session_count": 0,
        }

    prev_sessions = _scored_sessions_in_range(
        db_session, prev_start, prev_end, script_name=script_name
    )
    prev_stats = _word_stats_for_sessions(db_session, prev_sessions)
    top, improved, regressed = _summarize_recent_vs_prev(
        recent_stats, prev_stats, top_n, min_attempts, "word"
    )
    return {
        "top_trouble_words": top,
        "most_improved_words": improved,
        "most_regressed_words": regressed,
        "recent_session_count": len(recent_sessions),
        "previous_session_count": len(prev_sessions),
    }


def _token_index_for_error(row, ref_tokens: List[tuple]) -> int | None:
    if not ref_tokens:
        return None

    ref_start = getattr(row, "ref_start", None)
    ref_end = getattr(row, "ref_end", None)
    if ref_start is not None:
        start = int(ref_start)
        end = int(ref_end) if ref_end is not None else start
        best_idx = None
        best_overlap = -1
        for idx, (_word, token_start, token_end) in enumerate(ref_tokens):
            if start == end and token_start <= start <= token_end:
                return idx
            overlap = max(0, min(token_end, end) - max(token_start, start))
            if overlap > best_overlap:
                best_idx = idx
                best_overlap = overlap
        if best_idx is not None and best_overlap > 0:
            return best_idx

    ref_token = (getattr(row, "ref_token", None) or "").strip().lower()
    if ref_token:
        for idx, (word, _start, _end) in enumerate(ref_tokens):
            if word == ref_token:
                return idx
    return None


def _phrase_for_token(words: List[str], token_idx: int, phrase_size: int) -> str | None:
    if len(words) < 2:
        return None
    window = min(max(2, phrase_size), len(words))
    start = min(max(0, token_idx - (window // 2)), len(words) - window)
    return " ".join(words[start : start + window])


def _phrase_stats_for_sessions(
    db_session,
    sessions: List[db.PracticeSession],
    phrase_size: int = 3,
) -> Dict[str, Dict[str, float]]:
    if not sessions:
        return {}

    events_by_session: Dict[int, List[db.SessionError]] = {}
    for row in _events_for_sessions(db_session, sessions):
        if row.op not in ("sub", "del") or not row.ref_token:
            continue
        events_by_session.setdefault(int(row.session_id), []).append(row)

    attempts = Counter()
    errors = Counter()
    for sess in sessions:
        ref_tokens = tokenize_with_spans(clean_text_for_alignment(sess.script_text))
        words = [token for token, _start, _end in ref_tokens]
        if len(words) < 2:
            continue

        window = min(max(2, phrase_size), len(words))
        for start in range(0, len(words) - window + 1):
            attempts[" ".join(words[start : start + window])] += 1

        session_error_phrases = set()
        for row in events_by_session.get(int(sess.id), []):
            token_idx = _token_index_for_error(row, ref_tokens)
            if token_idx is None:
                continue
            phrase = _phrase_for_token(words, token_idx, phrase_size=phrase_size)
            if phrase:
                session_error_phrases.add(phrase)
        errors.update(session_error_phrases)

    stats: Dict[str, Dict[str, float]] = {}
    for phrase, attempts_count in attempts.items():
        if attempts_count <= 0:
            continue
        err_count = int(errors.get(phrase, 0))
        stats[phrase] = {
            "phrase": phrase,
            "attempts": float(attempts_count),
            "errors": float(err_count),
            "error_rate": float(err_count) / float(attempts_count),
        }
    return stats


def get_phrase_trend_summary(
    db_session,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    script_name: str | None = None,
    top_n: int = 5,
    min_attempts: int = 1,
    phrase_size: int = 3,
) -> Dict[str, List[Dict]]:
    recent_start, recent_end, prev_start, prev_end = _window_bounds(
        start_dt, end_dt
    )
    recent_sessions = _scored_sessions_in_range(
        db_session, recent_start, recent_end, script_name=script_name
    )
    recent_stats = _phrase_stats_for_sessions(
        db_session, recent_sessions, phrase_size=phrase_size
    )

    if prev_start is None or prev_end is None:
        top = sorted(
            [
                v
                for v in recent_stats.values()
                if int(v.get("attempts", 0)) >= int(min_attempts)
                and float(v.get("errors", 0.0)) > 0
            ],
            key=lambda x: (x["error_rate"], x["errors"]),
            reverse=True,
        )[:top_n]
        return {
            "top_trouble_phrases": top,
            "most_improved_phrases": [],
            "most_regressed_phrases": [],
            "recent_session_count": len(recent_sessions),
            "previous_session_count": 0,
        }

    prev_sessions = _scored_sessions_in_range(
        db_session, prev_start, prev_end, script_name=script_name
    )
    prev_stats = _phrase_stats_for_sessions(
        db_session, prev_sessions, phrase_size=phrase_size
    )
    top, improved, regressed = _summarize_recent_vs_prev(
        recent_stats, prev_stats, top_n, min_attempts, "phrase"
    )
    top = [row for row in top if float(row.get("errors", 0.0)) > 0]
    return {
        "top_trouble_phrases": top,
        "most_improved_phrases": improved,
        "most_regressed_phrases": regressed,
        "recent_session_count": len(recent_sessions),
        "previous_session_count": len(prev_sessions),
    }


def _character_stats_for_sessions(
    db_session, sessions: List[db.PracticeSession]
) -> Dict[str, Dict[str, float]]:
    opportunities = 0
    for sess in sessions:
        opportunities += sum(1 for ch in clean_text_for_alignment(sess.script_text) if ch.isalpha())
    opportunities = max(1, opportunities)

    counts = Counter()
    for row in _events_for_sessions(db_session, sessions):
        kind = (row.error_kind or "").strip()
        if kind in ("char_replace", "char_insert", "vowel_delete", "cons_delete"):
            counts[kind] += 1

    stats: Dict[str, Dict[str, float]] = {}
    for kind in ("char_replace", "char_insert", "vowel_delete", "cons_delete"):
        err = int(counts.get(kind, 0))
        stats[kind] = {
            "kind": kind,
            "attempts": float(opportunities),
            "errors": float(err),
            "error_rate": float(err) / float(opportunities),
        }
    return stats


def _top_char_confusions(db_session, sessions: List[db.PracticeSession], top_n: int) -> List[Dict]:
    pairs = Counter()
    for row in _events_for_sessions(db_session, sessions):
        if row.error_kind != "char_replace":
            continue
        ref_token = (row.ref_token or "").lower()
        hyp_token = (row.hyp_token or "").lower()
        if not ref_token or not hyp_token:
            continue
        rs = int(row.ref_local_start) if row.ref_local_start is not None else 0
        re_ = int(row.ref_local_end) if row.ref_local_end is not None else len(ref_token)
        hs = int(row.hyp_local_start) if row.hyp_local_start is not None else 0
        he = int(row.hyp_local_end) if row.hyp_local_end is not None else len(hyp_token)
        rs = max(0, min(rs, len(ref_token)))
        re_ = max(rs, min(re_, len(ref_token)))
        hs = max(0, min(hs, len(hyp_token)))
        he = max(hs, min(he, len(hyp_token)))
        ref_chunk = ref_token[rs:re_] or ref_token
        hyp_chunk = hyp_token[hs:he] or hyp_token
        pairs[(ref_chunk, hyp_chunk)] += 1
    out = []
    for (r, h), cnt in pairs.most_common(top_n):
        out.append({"from": r, "to": h, "count": cnt})
    return out


def get_character_trend_summary(
    db_session,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    script_name: str | None = None,
    top_n: int = 6,
) -> Dict:
    recent_start, recent_end, prev_start, prev_end = _window_bounds(
        start_dt, end_dt
    )
    recent_sessions = _scored_sessions_in_range(
        db_session, recent_start, recent_end, script_name=script_name
    )
    recent_stats = _character_stats_for_sessions(db_session, recent_sessions)

    top_kinds = sorted(
        recent_stats.values(),
        key=lambda x: x["error_rate"],
        reverse=True,
    )[:top_n]
    confusions = _top_char_confusions(db_session, recent_sessions, top_n=top_n)

    if prev_start is None or prev_end is None:
        return {
            "top_character_kinds": top_kinds,
            "most_improved_kinds": [],
            "most_regressed_kinds": [],
            "top_character_confusions": confusions,
        }

    prev_sessions = _scored_sessions_in_range(
        db_session, prev_start, prev_end, script_name=script_name
    )
    prev_stats = _character_stats_for_sessions(db_session, prev_sessions)
    _, improved, regressed = _summarize_recent_vs_prev(
        recent_stats,
        prev_stats,
        top_n=top_n,
        min_attempts=1,
        item_key="kind",
    )
    return {
        "top_character_kinds": top_kinds,
        "most_improved_kinds": improved,
        "most_regressed_kinds": regressed,
        "top_character_confusions": confusions,
    }


def _position_bucket(local_start: int | None, local_end: int | None, tok_len: int | None) -> str | None:
    if local_start is None or local_end is None or tok_len is None or tok_len <= 0:
        return None
    s = max(0, min(int(local_start), int(tok_len)))
    e = max(s, min(int(local_end), int(tok_len)))
    if s == 0 and e >= tok_len:
        return "whole"
    if s == 0:
        return "start"
    if e >= tok_len:
        return "end"
    return "middle"


def _position_stats_for_sessions(
    db_session, sessions: List[db.PracticeSession]
) -> Dict[str, Dict[str, float]]:
    starts = 0
    middles = 0
    ends = 0
    wholes = 0
    for sess in sessions:
        words = clean_text_for_alignment(sess.script_text).split()
        for w in words:
            L = len(w)
            if L <= 0:
                continue
            wholes += L
            starts += 1
            ends += 1
            middles += max(0, L - 2)

    opportunities = {
        "start": max(1, starts),
        "middle": max(1, middles),
        "end": max(1, ends),
        "whole": max(1, wholes),
    }
    counts = Counter()
    for row in _events_for_sessions(db_session, sessions):
        if row.op not in ("sub", "del"):
            continue
        bucket = _position_bucket(row.ref_local_start, row.ref_local_end, row.ref_token_len)
        if bucket:
            counts[bucket] += 1

    stats = {}
    for b in ("start", "middle", "end", "whole"):
        err = int(counts.get(b, 0))
        opp = int(opportunities[b])
        stats[b] = {
            "bucket": b,
            "attempts": float(opp),
            "errors": float(err),
            "error_rate": float(err) / float(opp),
        }
    return stats


def get_position_trend_summary(
    db_session,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    script_name: str | None = None,
    top_n: int = 4,
) -> Dict:
    recent_start, recent_end, prev_start, prev_end = _window_bounds(
        start_dt, end_dt
    )
    recent_sessions = _scored_sessions_in_range(
        db_session, recent_start, recent_end, script_name=script_name
    )
    recent_stats = _position_stats_for_sessions(db_session, recent_sessions)
    top_pos = sorted(recent_stats.values(), key=lambda x: x["error_rate"], reverse=True)[:top_n]

    if prev_start is None or prev_end is None:
        return {
            "top_position_buckets": top_pos,
            "most_improved_positions": [],
            "most_regressed_positions": [],
        }

    prev_sessions = _scored_sessions_in_range(
        db_session, prev_start, prev_end, script_name=script_name
    )
    prev_stats = _position_stats_for_sessions(db_session, prev_sessions)
    _, improved, regressed = _summarize_recent_vs_prev(
        recent_stats, prev_stats, top_n=top_n, min_attempts=1, item_key="bucket"
    )
    return {
        "top_position_buckets": top_pos,
        "most_improved_positions": improved,
        "most_regressed_positions": regressed,
    }


DIGRAPH_SYMBOLS = [
    ("tion", "SHUN"),
    ("sion", "ZHUN"),
    ("ough", "OH"),
    ("eigh", "AY"),
    ("igh", "AY"),
    ("tch", "CH"),
    ("sch", "SH"),
    ("ch", "CH"),
    ("sh", "SH"),
    ("th", "TH"),
    ("ph", "F"),
    ("ng", "NG"),
    ("ck", "K"),
    ("qu", "KW"),
    ("wh", "W"),
    ("oo", "UW"),
    ("ee", "IY"),
    ("ea", "IY"),
    ("ai", "EY"),
    ("ay", "EY"),
    ("oi", "OY"),
    ("ow", "AW"),
    ("ou", "AW"),
    ("ar", "AR"),
    ("er", "ER"),
    ("or", "OR"),
]

CHAR_SYMBOLS = {
    "a": "AH",
    "e": "EH",
    "i": "IH",
    "o": "OH",
    "u": "UH",
    "y": "Y",
}


def _word_to_phoneme_symbols(word: str) -> List[str]:
    w = clean_text_for_alignment(word)
    out: List[str] = []
    i = 0
    while i < len(w):
        matched = False
        for pat, sym in DIGRAPH_SYMBOLS:
            if w.startswith(pat, i):
                out.append(sym)
                i += len(pat)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        if ch.isalpha():
            out.append(CHAR_SYMBOLS.get(ch, ch.upper()))
        i += 1
    return out


def _phoneme_stats_for_sessions(
    db_session, sessions: List[db.PracticeSession]
) -> tuple[Dict[str, Dict[str, float]], List[Dict]]:
    opportunities = Counter()
    for sess in sessions:
        for w in clean_text_for_alignment(sess.script_text).split():
            opportunities.update(_word_to_phoneme_symbols(w))

    confusion_pairs = Counter()
    for row in _events_for_sessions(db_session, sessions):
        if row.op != "sub" or not row.ref_token or not row.hyp_token:
            continue
        ref_syms = _word_to_phoneme_symbols(row.ref_token)
        hyp_syms = _word_to_phoneme_symbols(row.hyp_token)
        ops = align_tokens(ref_syms, hyp_syms)
        for op, ri, hj in ops:
            if op == "sub" and ri is not None and hj is not None:
                confusion_pairs[(ref_syms[ri], hyp_syms[hj])] += 1
            elif op == "del" and ri is not None:
                confusion_pairs[(ref_syms[ri], "_DEL")] += 1
            elif op == "ins" and hj is not None:
                confusion_pairs[("_INS", hyp_syms[hj])] += 1

    by_ref = Counter()
    for (r, _h), cnt in confusion_pairs.items():
        if r and r != "_INS":
            by_ref[r] += cnt

    stats = {}
    for sym, opp in opportunities.items():
        if opp <= 0:
            continue
        err = int(by_ref.get(sym, 0))
        stats[sym] = {
            "symbol": sym,
            "attempts": float(opp),
            "errors": float(err),
            "error_rate": float(err) / float(opp),
        }
    def _label(sym: str) -> str:
        if sym == "_DEL":
            return "(dropped)"
        if sym == "_INS":
            return "(extra)"
        return sym

    top_pairs = [
        {"from": _label(r), "to": _label(h), "count": cnt}
        for (r, h), cnt in confusion_pairs.most_common(8)
    ]
    return stats, top_pairs


def get_phoneme_trend_summary(
    db_session,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
    script_name: str | None = None,
    top_n: int = 6,
    min_attempts: int = 4,
) -> Dict:
    recent_start, recent_end, prev_start, prev_end = _window_bounds(
        start_dt, end_dt
    )
    recent_sessions = _scored_sessions_in_range(
        db_session, recent_start, recent_end, script_name=script_name
    )
    recent_stats, top_pairs = _phoneme_stats_for_sessions(db_session, recent_sessions)
    top_symbols = sorted(
        [
            v
            for v in recent_stats.values()
            if int(v.get("attempts", 0)) >= int(min_attempts)
        ],
        key=lambda x: x["error_rate"],
        reverse=True,
    )[:top_n]

    if prev_start is None or prev_end is None:
        return {
            "top_trouble_symbols": top_symbols,
            "most_improved_symbols": [],
            "most_regressed_symbols": [],
            "top_symbol_confusions": top_pairs,
        }

    prev_sessions = _scored_sessions_in_range(
        db_session, prev_start, prev_end, script_name=script_name
    )
    prev_stats, _ = _phoneme_stats_for_sessions(db_session, prev_sessions)
    _, improved, regressed = _summarize_recent_vs_prev(
        recent_stats, prev_stats, top_n=top_n, min_attempts=min_attempts, item_key="symbol"
    )
    return {
        "top_trouble_symbols": top_symbols,
        "most_improved_symbols": improved,
        "most_regressed_symbols": regressed,
        "top_symbol_confusions": top_pairs,
    }


def generate_feedback_summary(
    word_summary: Dict,
    char_summary: Dict,
    pos_summary: Dict,
    phon_summary: Dict,
    phrase_summary: Dict | None = None,
) -> List[str]:
    feedback: List[str] = []

    words = word_summary.get("top_trouble_words", [])
    if words:
        focus = ", ".join(w["word"] for w in words[:3])
        feedback.append(f"Word focus: drill these frequently missed words: {focus}.")

    phrases = (phrase_summary or {}).get("top_trouble_phrases", [])
    if phrases:
        focus = "; ".join(f"'{p.get('phrase', '')}'" for p in phrases[:2])
        feedback.append(f"Phrase focus: slow down and repeat these unstable phrases: {focus}.")

    char_kinds = char_summary.get("top_character_kinds", [])
    if char_kinds:
        top = char_kinds[0]
        feedback.append(
            f"Character pattern: highest error type is {top.get('kind', 'unknown')} "
            f"at {float(top.get('error_rate', 0.0)):.1%}."
        )

    pos = pos_summary.get("top_position_buckets", [])
    if pos:
        p = pos[0]
        bucket = p.get("bucket", "unknown")
        feedback.append(
            f"Position pattern: most errors occur at {bucket} of words "
            f"({float(p.get('error_rate', 0.0)):.1%})."
        )

    pairs = phon_summary.get("top_symbol_confusions", [])
    if pairs:
        p0 = pairs[0]
        feedback.append(
            f"Sound pattern: common confusion is {p0.get('from', '?')} -> {p0.get('to', '?')} "
            f"({int(p0.get('count', 0))} times)."
        )

    improved = word_summary.get("most_improved_words", [])
    if improved:
        w = improved[0]
        feedback.append(
            f"Improvement: {w.get('word', '')} improved by {abs(float(w.get('delta', 0.0))):.1%} "
            f"vs prior window."
        )

    if not feedback:
        feedback.append("Not enough data yet. Record and score more sessions to generate trend feedback.")
    return feedback
