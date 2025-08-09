from __future__ import annotations

import os
import json
from typing import Dict, Tuple

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QDialogButtonBox,
)


def default_settings() -> Dict:
    return {
        "device": "auto",   # auto | cpu | gpu
        "model_name": os.getenv("WHISPER_MODEL", "base.en"),
        "preset": "balanced_cpu",  # fast_cpu | balanced_cpu | balanced_gpu | accurate_gpu
        "language": "en",
        "timestamps": False,
        "beam_size": 1,
        "temperature": 0.0,
        # advanced decoding controls
        "no_speech_threshold": 0.45,
        "condition_on_previous_text": True,
    }


def settings_path() -> str:
    return os.path.abspath("settings.json")


def load_settings(defaults: Dict, path: str) -> Dict:
    settings = dict(defaults)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, dict):
                    settings.update(data)
    except Exception:
        pass
    return settings


def save_settings(settings: Dict, path: str) -> None:
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
    except Exception:
        pass


def detect_gpu() -> Tuple[bool, str]:
    try:
        import torch
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            name = torch.cuda.get_device_name(0)
            total_vram = int(torch.cuda.get_device_properties(0).total_memory // (1024 ** 2))
            return True, f"{name} ({total_vram} MB VRAM, {count} device(s))"
    except Exception:
        pass
    return False, "No CUDA GPU detected"


def open_settings_dialog(window) -> None:
    """Open the settings dialog bound to the given window instance."""
    dlg = QDialog(window)
    dlg.setWindowTitle("Settings")
    form = QFormLayout(dlg)

    # Device
    device_cb = QComboBox(dlg)
    device_cb.addItems(["auto", "cpu", "gpu"])
    device_cb.setCurrentText(window.settings.get("device", "auto"))
    form.addRow("Device", device_cb)

    # Model name with descriptive labels
    model_cb = QComboBox(dlg)
    model_choices = [
        ("tiny.en",   "fastest, lowest accuracy, minimal RAM"),
        ("base.en",   "fast, decent accuracy, low RAM"),
        ("small.en",  "balanced speed/accuracy, moderate RAM"),
        ("medium.en", "slower, higher accuracy, higher RAM"),
        ("tiny",      "multilingual tiny; fast but lowest accuracy"),
        ("base",      "multilingual base; low RAM, decent accuracy"),
        ("small",     "multilingual small; balanced option"),
        ("medium",    "multilingual medium; slower, more accurate"),
    ]
    for key, desc in model_choices:
        model_cb.addItem(f"{key} ({desc})", userData=key)
    saved_model = window.settings.get("model_name", "base.en")
    found_index = -1
    for i in range(model_cb.count()):
        if model_cb.itemData(i) == saved_model:
            found_index = i
            break
    model_cb.setCurrentIndex(found_index if found_index >= 0 else 1)
    form.addRow("Model", model_cb)

    # Preset
    preset_cb = QComboBox(dlg)
    preset_cb.addItems(["---", "fast_cpu", "balanced_cpu", "balanced_gpu", "accurate_gpu"])
    preset_cb.setCurrentText(window.settings.get("preset", "balanced_cpu"))
    form.addRow("Preset", preset_cb)

    # Beam size
    beam_sb = QSpinBox(dlg)
    beam_sb.setRange(1, 10)
    beam_sb.setValue(int(window.settings.get("beam_size", 1)))
    form.addRow("Beam size", beam_sb)

    # Temperature
    temp_sb = QDoubleSpinBox(dlg)
    temp_sb.setRange(0.0, 1.0)
    temp_sb.setSingleStep(0.1)
    temp_sb.setDecimals(2)
    temp_sb.setValue(float(window.settings.get("temperature", 0.0)))
    form.addRow("Temperature", temp_sb)

    def _apply_preset_visual(preset_text: str) -> None:
        force = preset_text != "---"
        if preset_text == "fast_cpu":
            beam_sb.setValue(1)
            temp_sb.setValue(0.0)
        elif preset_text == "balanced_cpu":
            beam_sb.setValue(2)
            temp_sb.setValue(0.0)
        elif preset_text == "balanced_gpu":
            beam_sb.setValue(3)
            temp_sb.setValue(0.0)
        elif preset_text == "accurate_gpu":
            beam_sb.setValue(5)
            temp_sb.setValue(0.0)
        beam_sb.setEnabled(not force)
        temp_sb.setEnabled(not force)

    # Conditioning on previous text
    cot_ck = QCheckBox("Use previous text context for continuity", dlg)
    cot_ck.setChecked(bool(window.settings.get("condition_on_previous_text", True)))
    form.addRow("Conditioning", cot_ck)

    # Timestamps
    ts_ck = QCheckBox("Generate timestamps", dlg)
    ts_ck.setChecked(bool(window.settings.get("timestamps", False)))
    form.addRow("Timestamps", ts_ck)

    # No-speech threshold
    ns_sb = QDoubleSpinBox(dlg)
    ns_sb.setRange(0.0, 1.0)
    ns_sb.setSingleStep(0.05)
    ns_sb.setDecimals(2)
    ns_sb.setValue(float(window.settings.get("no_speech_threshold", 0.45)))
    form.addRow("No-speech threshold", ns_sb)

    # Language
    lang_cb = QComboBox(dlg)
    lang_cb.addItems(["auto", "en"])  # can be extended
    lang_cb.setCurrentText(window.settings.get("language", "en"))
    form.addRow("Language", lang_cb)

    # GPU detection / recommendation
    gpu_label = QLabel("")
    has_gpu, summary = detect_gpu()
    recommendation = "Use GPU presets (balanced_gpu)." if has_gpu else "Use CPU presets (fast/balanced_cpu)."
    gpu_label.setText(f"Detected: {summary}\nRecommendation: {recommendation}")
    form.addRow("Hardware", gpu_label)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, parent=dlg)
    form.addRow(buttons)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)

    preset_cb.currentTextChanged.connect(_apply_preset_visual)
    _apply_preset_visual(preset_cb.currentText())

    if dlg.exec_() == QDialog.Accepted:
        window.settings["device"] = device_cb.currentText()
        selected_model = model_cb.currentData()
        window.settings["model_name"] = selected_model if selected_model else model_cb.currentText().split(" ")[0]
        window.settings["preset"] = preset_cb.currentText()
        window.settings["beam_size"] = int(beam_sb.value())
        window.settings["temperature"] = float(temp_sb.value())
        window.settings["condition_on_previous_text"] = bool(cot_ck.isChecked())
        window.settings["timestamps"] = bool(ts_ck.isChecked())
        window.settings["no_speech_threshold"] = float(ns_sb.value())
        window.settings["language"] = lang_cb.currentText()
        # Reset model so it will be recreated with new settings
        window.model = None
        save_settings(window.settings, settings_path())


def whisper_options(settings: Dict, free_speak: bool = False) -> Dict:
    language = None if settings.get("language") == "auto" else settings.get("language", "en")
    without_ts = not bool(settings.get("timestamps", False))
    beam_size = int(settings.get("beam_size", 1))
    temperature = float(settings.get("temperature", 0.0))

    opts: Dict = dict(
        language=language,
        task="transcribe",
        temperature=temperature,
        beam_size=beam_size,
        without_timestamps=without_ts,
        condition_on_previous_text=bool(settings.get("condition_on_previous_text", True)),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=float(settings.get("no_speech_threshold", 0.45)),
    )

    # Device/precision hints: whisper python uses internal detection; we control fp16
    preset = settings.get("preset")
    device = settings.get("device")
    use_fp16 = False
    if device == "gpu":
        use_fp16 = True
    elif device == "auto":
        try:
            import torch
            use_fp16 = torch.cuda.is_available()
        except Exception:
            use_fp16 = False
    opts["fp16"] = bool(use_fp16)

    if preset == "fast_cpu":
        opts.update(dict(beam_size=1, temperature=0.0))
    elif preset == "balanced_cpu":
        opts.update(dict(beam_size=2, temperature=0.0))
    elif preset == "balanced_gpu":
        opts.update(dict(beam_size=3, temperature=0.0, fp16=True))
    elif preset == "accurate_gpu":
        opts.update(dict(beam_size=5, temperature=0.0, fp16=True))
    elif preset == "---":
        pass

    return opts


