"""Bounded audio-native assessment with no saved takes or transcripts."""
import base64
import json
import re
import subprocess
import tempfile
import wave
from pathlib import Path

MODEL = "gpt-audio-1.5"
PROMPT = """Listen to this actual recording as a first-time listener. Give thoughtful, useful feedback on clarity, pronunciation, articulation and whether words sound mumbled or run together. Accept natural accents; describe accent features only if audible, and never treat a particular accent as a defect or infer nationality or identity. Do not diagnose or invent words, timestamps or issues. If the recording cannot support a meaningful assessment, return JSON {"status":"abstain","reason":"..."}. Otherwise return only one JSON object with status "ok", summary (string), strengths (up to three strings), observations (up to three specific audible observations as strings, with a short phrase or approximate location only if justified), practice (up to two concrete exercises as strings), and uncertainty (string). Give no scores or numbers. Return JSON only, without Markdown fences."""
MAX_UPLOAD = 10 * 1024 * 1024
MAX_SECONDS = 120


class AudioError(Exception):
    pass


def parse_response(raw):
    text = raw.strip()
    if text.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL | re.IGNORECASE)
        if not match:
            return None
        text = match.group(1)
    try:
        data = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    if data.get("status") == "abstain":
        reason = data.get("reason")
        return {"status": "abstain", "reason": reason[:500]} if isinstance(reason, str) and reason.strip() else None
    if data.get("status") != "ok":
        return None
    for key in ("summary", "uncertainty"):
        if not isinstance(data.get(key), str) or len(data[key]) > 2000 or (key == "summary" and not data[key].strip()):
            return None
    for key, limit in (("strengths", 3), ("observations", 3), ("practice", 2)):
        val = data.get(key)
        if not isinstance(val, list) or len(val) > limit or any(not isinstance(x, str) or not x.strip() or len(x) > 1000 for x in val):
            return None
    return {k: data[k] for k in ("status", "summary", "strengths", "observations", "practice", "uncertainty")}


def to_wav(upload):
    """Keep temporary input/output inside a cleanup-managed folder."""
    with tempfile.TemporaryDirectory(prefix="speech-studio-") as folder:
        source = Path(folder) / "input.audio"
        output = Path(folder) / "converted.wav"
        with source.open("wb") as file:
            for chunk in upload.chunks():
                file.write(chunk)
        try:
            result = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe", "-format_whitelist", "matroska,webm,wav,mp3,mov,ogg", "-i", str(source), "-map", "0:a:0", "-vn", "-t", "121", "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", "-y", str(output)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30, check=False)
            if result.returncode:
                raise AudioError("The recording could not be read. Try recording again.")
            with wave.open(str(output), "rb") as wav:
                duration = wav.getnframes() / wav.getframerate()
            if duration <= 0 or duration > MAX_SECONDS:
                raise AudioError("The recording must be no longer than two minutes.")
            return output.read_bytes()
        except (OSError, subprocess.TimeoutExpired, wave.Error, ZeroDivisionError):
            raise AudioError("Audio conversion failed. Check that ffmpeg is installed, then try again.") from None


def ask_model(api_key, wav):
    from openai import OpenAI
    client = OpenAI(api_key=api_key, max_retries=0, timeout=90.0)
    response = client.chat.completions.create(model=MODEL, modalities=["text"], store=False, max_completion_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav).decode("ascii"), "format": "wav"}},
        ]}])
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice else None
    if not message or getattr(message, "refusal", None) or getattr(choice, "finish_reason", "") != "stop":
        return {"status": "unavailable", "reason": "The provider could not complete this feedback. You can try again."}
    raw = (message.content or "").strip()[:16000]
    result = parse_response(raw)
    return result if result else {"status": "unstructured", "text": raw}
