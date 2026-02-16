# alignment_utils.py
from __future__ import annotations

import re
import difflib
from typing import List, Tuple

from highlight_theme import (
    SCRIPT_WORD_MISSING,
    TRANSCRIPT_WORD_INSERT,
    TRANSCRIPT_CHAR_INSERT,
    REPLACE,
    REF_VOWEL_DELETE,
    REF_CONS_DELETE,
)


Token = Tuple[str, int, int]  # (text, start_char, end_char)
Span = Tuple[int, int, str]  # (start, end, color_hex)

VOWELS = set("aeiou")
TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:[\u2019'`-][A-Za-z0-9]+)*", flags=re.UNICODE
)


def _normalize_token_text(token: str) -> str:
    t = token.lower()
    # Treat intra-word apostrophes/hyphens as equivalent variants.
    t = re.sub(r"[\u2019'`-]", "", t)
    return t


def tokenize_with_spans(text: str) -> List[Token]:
    tokens: List[Token] = []
    for m in TOKEN_RE.finditer(text):
        norm = _normalize_token_text(m.group(0))
        if norm:
            tokens.append((norm, int(m.start()), int(m.end())))
    return tokens


def _align_tokens(
    ref: List[str], hyp: List[str]
) -> List[Tuple[str, int | None, int | None]]:
    n, m = len(ref), len(hyp)
    inf = 10**9
    dp = [[inf] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Tuple[str, int, int]]] = [[("", -1, -1) for _ in range(m + 1)] for _ in range(n + 1)]
    dp[0][0] = 0

    def relax(ni: int, nj: int, cost: int, op: str, pi: int, pj: int) -> None:
        cand = dp[pi][pj] + cost
        if cand < dp[ni][nj]:
            dp[ni][nj] = cand
            bt[ni][nj] = (op, pi, pj)

    for i in range(n + 1):
        for j in range(m + 1):
            if dp[i][j] >= inf:
                continue
            if i < n:
                relax(i + 1, j, 1, "del", i, j)
            if j < m:
                relax(i, j + 1, 1, "ins", i, j)
            if i < n and j < m:
                same = ref[i] == hyp[j]
                relax(i + 1, j + 1, 0 if same else 1, "equal" if same else "sub", i, j)
            # zero-cost join/split equivalence: one word vs two words
            if i < n and j + 1 < m and ref[i] == (hyp[j] + hyp[j + 1]):
                relax(i + 1, j + 2, 0, "equal_join_hyp", i, j)
            if i + 1 < n and j < m and (ref[i] + ref[i + 1]) == hyp[j]:
                relax(i + 2, j + 1, 0, "equal_join_ref", i, j)

    i, j = n, m
    ops: List[Tuple[str, int | None, int | None]] = []
    while i > 0 or j > 0:
        op, pi, pj = bt[i][j]
        if pi < 0 or pj < 0:
            # Fallback guard for malformed backtracking state.
            if i > 0:
                ops.append(("del", i - 1, None))
                i -= 1
            elif j > 0:
                ops.append(("ins", None, j - 1))
                j -= 1
            continue
        if op in ("equal", "sub"):
            ops.append((op, pi, pj))
        elif op == "del":
            ops.append(("del", i - 1, None))
        elif op == "ins":
            ops.append(("ins", None, j - 1))
        elif op == "equal_join_hyp":
            # ref[pi] matches hyp[pj] + hyp[pj+1]
            ops.append(("equal_join_hyp", pi, pj))
        elif op == "equal_join_ref":
            # ref[pi] + ref[pi+1] matches hyp[pj]
            ops.append(("equal_join_ref", pi, pj))
        i, j = pi, pj
    ops.reverse()
    return ops


def _append_runs(spans: List[Span], start_base: int, indexes: List[int], color: str) -> None:
    if not indexes:
        return
    indexes.sort()
    run_start = indexes[0]
    prev = indexes[0]
    for k in indexes[1:]:
        if k == prev + 1:
            prev = k
            continue
        spans.append((start_base + run_start, start_base + prev + 1, color))
        run_start = k
        prev = k
    spans.append((start_base + run_start, start_base + prev + 1, color))


def _char_level_spans_for_substitution(
    ref_word: str,
    hyp_word: str,
    ref_start: int,
    hyp_start: int,
) -> Tuple[List[Span], List[Span]]:
    s = difflib.SequenceMatcher(a=ref_word.lower(), b=hyp_word.lower())
    ref_spans: List[Span] = []
    hyp_spans: List[Span] = []

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace",):
            ref_spans.append((ref_start + i1, ref_start + i2, REPLACE))
            hyp_spans.append((hyp_start + j1, hyp_start + j2, REPLACE))
        elif tag == "delete":
            vowel_idx, cons_idx = [], []
            for k in range(i1, i2):
                (vowel_idx if ref_word[k].lower() in VOWELS else cons_idx).append(k)
            _append_runs(ref_spans, ref_start, vowel_idx, REF_VOWEL_DELETE)
            _append_runs(ref_spans, ref_start, cons_idx, REF_CONS_DELETE)
        elif tag == "insert":
            hyp_spans.append((hyp_start + j1, hyp_start + j2, TRANSCRIPT_CHAR_INSERT))
    return ref_spans, hyp_spans


def compute_error_spans_for_display(
    ref_text: str, hyp_text: str
) -> Tuple[List[Span], List[Span]]:
    ref_tokens = tokenize_with_spans(ref_text)
    hyp_tokens = tokenize_with_spans(hyp_text)

    ref_words = [t[0] for t in ref_tokens]
    hyp_words = [t[0] for t in hyp_tokens]
    ops = _align_tokens(ref_words, hyp_words)

    script_spans: List[Span] = []
    transcript_spans: List[Span] = []

    for op, ri, hj in ops:
        if op in ("equal", "equal_join_hyp", "equal_join_ref"):
            continue
        if op == "del" and ri is not None:
            _, rs, re = ref_tokens[ri]
            script_spans.append((rs, re, SCRIPT_WORD_MISSING))
        elif op == "ins" and hj is not None:
            _, hs, he = hyp_tokens[hj]
            transcript_spans.append((hs, he, TRANSCRIPT_WORD_INSERT))
        elif op == "sub" and ri is not None and hj is not None:
            rtok, rs, re = ref_tokens[ri]
            htok, hs, he = hyp_tokens[hj]
            rs_list, hs_list = _char_level_spans_for_substitution(rtok, htok, rs, hs)
            script_spans.extend(rs_list)
            transcript_spans.extend(hs_list)

    return script_spans, transcript_spans


def compute_flexible_wer(ref_text: str, hyp_text: str) -> float:
    """
    WER with tolerance for common formatting/segmentation variants:
    - apostrophes/hyphens inside tokens
    - split/join compounds (e.g., battlefleets vs battle fleets)
    """
    ref_tokens = [t[0] for t in tokenize_with_spans(ref_text)]
    hyp_tokens = [t[0] for t in tokenize_with_spans(hyp_text)]
    if not ref_tokens:
        return 0.0 if not hyp_tokens else 1.0
    ops = _align_tokens(ref_tokens, hyp_tokens)
    errors = sum(1 for op, _ri, _hj in ops if op in ("sub", "del", "ins"))
    return float(errors) / float(len(ref_tokens))


def align_tokens(
    ref_words: List[str], hyp_words: List[str]
) -> List[Tuple[str, int | None, int | None]]:
    """
    Public wrapper for token alignment operations.
    """
    return _align_tokens(ref_words, hyp_words)


def char_level_spans_for_substitution(
    ref_word: str,
    hyp_word: str,
    ref_start: int,
    hyp_start: int,
) -> Tuple[List[Span], List[Span]]:
    """
    Public wrapper for character-level substitution span extraction.
    """
    return _char_level_spans_for_substitution(
        ref_word, hyp_word, ref_start, hyp_start
    )
