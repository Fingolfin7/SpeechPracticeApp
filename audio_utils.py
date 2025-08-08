from __future__ import annotations

import numpy as np


def trim_silence(
    y: np.ndarray,
    sr: int,
    threshold_db: float = -50.0,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
    pre_pad_ms: float = 120.0,
    post_pad_ms: float = 280.0,
) -> np.ndarray:
    """
    Trim leading/trailing silence using smoothed RMS in dB with generous padding.
    Behavior mirrors `SpeechPracticeApp._trim_silence`.
    """
    x = y.astype(np.float32, copy=False)
    n = x.size
    if n == 0:
        return x

    frame_len = max(1, int(sr * (window_ms / 1000.0)))
    hop = max(1, int(sr * (hop_ms / 1000.0)))
    if frame_len <= 2:
        frame_len = 3

    pad = (-(n - frame_len) % hop) if n >= frame_len else (frame_len - n)
    x_pad = np.pad(x, (0, pad), mode="constant", constant_values=0.0)
    num_frames = 1 + max(0, (x_pad.size - frame_len) // hop)
    if num_frames <= 0:
        return x
    strided = np.lib.stride_tricks.as_strided(
        x_pad,
        shape=(num_frames, frame_len),
        strides=(x_pad.strides[0] * hop, x_pad.strides[0]),
        writeable=False,
    )
    rms = np.sqrt(np.maximum(1e-12, (strided * strided).mean(axis=1)))

    smooth_win = max(1, int(20.0 / hop_ms))
    kernel = np.ones(smooth_win, dtype=np.float32) / float(smooth_win)
    rms_smooth = np.convolve(rms, kernel, mode="same")

    rms_db = 20.0 * np.log10(np.maximum(rms_smooth, 1e-8))
    active = rms_db > float(threshold_db)
    if not np.any(active):
        return x
    first_f = int(np.argmax(active))
    last_f = int(len(active) - np.argmax(active[::-1]) - 1)

    pre = int(sr * (pre_pad_ms / 1000.0))
    post = int(sr * (post_pad_ms / 1000.0))
    i1 = max(0, first_f * hop - pre)
    i2 = min(n, last_f * hop + frame_len + post)
    if i2 <= i1:
        return x
    return x[i1:i2]


def envelope(
    y: np.ndarray,
    sr: int,
    max_points: int = 4_000,
    silence_eps: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Down-sample y to ≤ max_points vertices while preserving peaks.
    Mirrors `SpeechPracticeApp._envelope` semantics.
    """
    n = y.size
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    if n <= max_points:
        x = np.arange(n) / sr
        return x, y

    step = int(np.ceil(n / max_points))
    win = y[: (n // step) * step].reshape(-1, step)
    y_min = win.min(axis=1)
    y_max = win.max(axis=1)
    dyn = y_max - y_min

    quiet = dyn < silence_eps
    y_min[quiet] = 0.0
    y_max[quiet] = 0.0

    y_env = np.empty(y_min.size * 2, dtype=y.dtype)
    y_env[0::2] = y_min
    y_env[1::2] = y_max

    centres = (np.arange(y_min.size) * step + step // 2) / sr
    x_env = np.repeat(centres, 2)

    return x_env, y_env


