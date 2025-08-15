# alignment_utils.py
from __future__ import annotations

import re
import difflib
from typing import List, Tuple


Token = Tuple[str, int, int]  # (text, start_char, end_char)
Span = Tuple[int, int, str]  # (start, end, color_hex)


VOWELS = set("aeiou")


def tokenize_with_spans(text: str) -> List[Token]:
    """
    Find word tokens and their character spans in the given display text.
    Words = [A-Za-z0-9_]+ via \w. Keeps exact start/end for highlighting.
    """
    tokens: List[Token] = []
    for m in re.finditer(r"\b\w+\b", text, flags=re.UNICODE):
        tok = m.group(0)
        tokens.append((tok, int(m.start()), int(m.end())))
    return tokens


def _align_tokens(ref: List[str], hyp: List[str]) -> List[Tuple[str, int | None, int | None]]:
    """
    Simple DP alignment: returns list of ops with indices.
    op in {"equal","sub","del","ins"}.
    """
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

    # Reconstruct
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
            # Shouldn't happen; fallback as deletion
            ops.append(("del", i - 1, None))
            i -= 1
    ops.reverse()
    return ops


def _append_runs(spans: List[Span], start_base: int, indexes: List[int], color: str) -> None:
    """
    Merge consecutive char indexes into fewer spans for efficiency.
    """
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
    """
    Produce character-level highlight spans for a substituted word pair.
    - Deletions in reference: red-ish (vowels stronger)
    - Insertions in hypothesis: blue-ish
    - Replacements in both: amber
    """
    s = difflib.SequenceMatcher(a=ref_word.lower(), b=hyp_word.lower())
    ref_spans: List[Span] = []
    hyp_spans: List[Span] = []

    # Colors (semi-transparent)
    COL_VOWEL_DEL = "#ff3b3066"  # strong red on vowels
    COL_CONS_DEL = "#ff9e9e66"  # softer red/orange for other deletes
    COL_INSERT = "#66a3ff66"  # blue for inserts
    COL_REPLACE = "#ffc10766"  # amber for replace ranges

    for tag, i1, i2, j1, j2 in s.get_opcodes():
        if tag == "equal":
            continue
        if tag in ("replace",):
            # Highlight both sides for replaced ranges
            ref_spans.append((ref_start + i1, ref_start + i2, COL_REPLACE))
            hyp_spans.append((hyp_start + j1, hyp_start + j2, COL_REPLACE))
        elif tag == "delete":
            # Specific per-character, use stronger color for vowels
            vowel_idx, cons_idx = [], []
            for k in range(i1, i2):
                (vowel_idx if ref_word[k].lower() in VOWELS else cons_idx).append(k)
            _append_runs(ref_spans, ref_start, vowel_idx, COL_VOWEL_DEL)
            _append_runs(ref_spans, ref_start, cons_idx, COL_CONS_DEL)
        elif tag == "insert":
            hyp_spans.append((hyp_start + j1, hyp_start + j2, COL_INSERT))
    return ref_spans, hyp_spans


def compute_error_spans_for_display(
    ref_text: str, hyp_text: str
) -> Tuple[List[Span], List[Span]]:
    """
    Compute highlight spans for the UI documents (script and transcript).
    Returns (script_spans, transcript_spans).
    """
    ref_tokens = tokenize_with_spans(ref_text)
    hyp_tokens = tokenize_with_spans(hyp_text)

    ref_words = [t[0] for t in ref_tokens]
    hyp_words = [t[0] for t in hyp_tokens]
    ops = _align_tokens(ref_words, hyp_words)

    # Word-level colors
    COL_WORD_DEL = "#ff6b6b55"  # whole missing word (script only)
    COL_WORD_INS = "#6b9bff55"  # whole inserted word (transcript only)

    script_spans: List[Span] = []
    transcript_spans: List[Span] = []

    for op, ri, hj in ops:
        if op == "equal":
            continue
        if op == "del" and ri is not None:
            # Entire missing word in script
            _, rs, re = ref_tokens[ri]
            script_spans.append((rs, re, COL_WORD_DEL))
        elif op == "ins" and hj is not None:
            # Entire inserted word in transcript
            _, hs, he = hyp_tokens[hj]
            transcript_spans.append((hs, he, COL_WORD_INS))
        elif op == "sub" and ri is not None and hj is not None:
            rtok, rs, re = ref_tokens[ri]
            htok, hs, he = hyp_tokens[hj]
            rs_list, hs_list = _char_level_spans_for_substitution(
                rtok, htok, rs, hs
            )
            script_spans.extend(rs_list)
            transcript_spans.extend(hs_list)

    return script_spans, transcript_spans
