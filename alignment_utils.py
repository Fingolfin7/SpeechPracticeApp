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


def tokenize_with_spans(text: str) -> List[Token]:
    tokens: List[Token] = []
    for m in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        tokens.append((m.group(0), int(m.start()), int(m.end())))
    return tokens


def _align_tokens(
    ref: List[str], hyp: List[str]
) -> List[Tuple[str, int | None, int | None]]:
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt: List[List[Tuple[str, int, int]]] = [[("", 0, 0)] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = ("del", i - 1, 0)
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = ("ins", 0, j - 1)

    for i in range(1, n + 1):
        ri = ref[i - 1].lower()
        for j in range(1, m + 1):
            hj = hyp[j - 1].lower()
            cost_sub = 0 if ri == hj else 1

            a = dp[i - 1][j] + 1  # del
            b = dp[i][j - 1] + 1  # ins
            c = dp[i - 1][j - 1] + cost_sub  # sub/equal

            best = min(a, b, c)
            dp[i][j] = best
            if best == c:
                bt[i][j] = ("equal" if cost_sub == 0 else "sub", i - 1, j - 1)
            elif best == a:
                bt[i][j] = ("del", i - 1, j)
            else:
                bt[i][j] = ("ins", i, j - 1)

    i, j = n, m
    ops: List[Tuple[str, int | None, int | None]] = []
    while i > 0 or j > 0:
        op, pi, pj = bt[i][j]
        if op == "equal" or op == "sub":
            ops.append((op, pi, pj))
            i, j = i - 1, j - 1
        elif op == "del":
            ops.append(("del", i - 1, None))
            i -= 1
        elif op == "ins":
            ops.append(("ins", None, j - 1))
            j -= 1
        else:
            ops.append(("del", i - 1, None))
            i -= 1
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
        if op == "equal":
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