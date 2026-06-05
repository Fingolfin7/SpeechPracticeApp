from __future__ import annotations

import os
import json
from typing import Dict, Tuple

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import (
    QDialog,
    QFormLayout,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QLabel,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QWidget,
)

from autumn_client import AutumnClient, AutumnError, normalize_base_url


def default_settings() -> Dict:
    return {
        "device": "auto",   # auto | cpu | gpu
        "model_name": os.getenv("WHISPER_MODEL", "base.en"),
        "preset": "balanced_cpu",  # fast_cpu | balanced_cpu | balanced_gpu | accurate_gpu
        "language": "en",
        "timestamps": True,
        "beam_size": 1,
        "temperature": 0.0,
        "transcribe_chunk_seconds": 60,
        # advanced decoding controls
        "no_speech_threshold": 0.30,
        "condition_on_previous_text": True,
        # Autumn timer integration
        "autumn_base_url": "https://autumn-lg0b.onrender.com",
        "autumn_api_key": "",
        "autumn_connected": False,
        "autumn_project": "",
        "autumn_subprojects": [],
        "autumn_active_session_id": None,
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

    # Transcription chunk size
    chunk_sb = QSpinBox(dlg)
    chunk_sb.setRange(10, 600)
    chunk_sb.setSingleStep(10)
    chunk_sb.setSuffix(" sec")
    chunk_sb.setValue(int(window.settings.get("transcribe_chunk_seconds", 60)))
    form.addRow("Chunk size", chunk_sb)

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

    # Autumn integration
    autumn_box = QGroupBox("Autumn", dlg)
    autumn_layout = QFormLayout(autumn_box)

    autumn_base_edit = QLineEdit(
        normalize_base_url(window.settings.get("autumn_base_url")), autumn_box
    )
    autumn_layout.addRow("URL", autumn_base_edit)

    auth_row = QWidget(autumn_box)
    auth_layout = QHBoxLayout(auth_row)
    auth_layout.setContentsMargins(0, 0, 0, 0)
    autumn_user_edit = QLineEdit(auth_row)
    autumn_user_edit.setPlaceholderText("username")
    autumn_pass_edit = QLineEdit(auth_row)
    autumn_pass_edit.setPlaceholderText("password")
    autumn_pass_edit.setEchoMode(QLineEdit.Password)
    autumn_connect_btn = QPushButton("Connect", auth_row)
    autumn_disconnect_btn = QPushButton("Disconnect", auth_row)
    auth_layout.addWidget(autumn_user_edit)
    auth_layout.addWidget(autumn_pass_edit)
    auth_layout.addWidget(autumn_connect_btn)
    auth_layout.addWidget(autumn_disconnect_btn)
    autumn_layout.addRow("Login", auth_row)

    autumn_status = QLabel("", autumn_box)
    autumn_status.setWordWrap(True)
    autumn_layout.addRow("Status", autumn_status)

    project_row = QWidget(autumn_box)
    project_layout = QHBoxLayout(project_row)
    project_layout.setContentsMargins(0, 0, 0, 0)
    autumn_project_cb = QComboBox(project_row)
    autumn_project_cb.setEditable(True)
    autumn_refresh_btn = QPushButton("Refresh", project_row)
    project_layout.addWidget(autumn_project_cb, 1)
    project_layout.addWidget(autumn_refresh_btn)
    autumn_layout.addRow("Project", project_row)

    autumn_subs_list = QListWidget(autumn_box)
    autumn_subs_list.setMaximumHeight(110)
    autumn_layout.addRow("Subprojects", autumn_subs_list)
    form.addRow(autumn_box)

    autumn_token = {"value": str(window.settings.get("autumn_api_key") or "")}
    loading_projects = {"value": False}

    def _set_autumn_status(text: str) -> None:
        autumn_status.setText(text)

    def _selected_subprojects() -> list[str]:
        selected: list[str] = []
        for i in range(autumn_subs_list.count()):
            item = autumn_subs_list.item(i)
            if item.checkState() == QtCore.Qt.Checked:
                selected.append(item.text())
        return selected

    def _set_subprojects(names: list[str], selected: list[str] | None = None) -> None:
        selected_set = set(selected or [])
        autumn_subs_list.clear()
        for name in names:
            item = QListWidgetItem(str(name))
            item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
            item.setCheckState(
                QtCore.Qt.Checked
                if item.text() in selected_set
                else QtCore.Qt.Unchecked
            )
            autumn_subs_list.addItem(item)

    def _client() -> AutumnClient:
        return AutumnClient(autumn_base_edit.text(), autumn_token["value"])

    def _refresh_subprojects() -> None:
        if loading_projects["value"]:
            return
        project = autumn_project_cb.currentText().strip()
        if not autumn_token["value"] or not project:
            _set_subprojects([], [])
            return
        previous = _selected_subprojects() or list(
            window.settings.get("autumn_subprojects") or []
        )
        try:
            subs = _client().list_subprojects(project)
            _set_subprojects(subs, previous)
            _set_autumn_status("Connected")
        except AutumnError as exc:
            _set_autumn_status(str(exc))

    def _refresh_projects() -> None:
        if not autumn_token["value"]:
            _set_autumn_status("Not connected")
            return
        project = autumn_project_cb.currentText().strip() or str(
            window.settings.get("autumn_project") or ""
        )
        try:
            loading_projects["value"] = True
            projects = _client().list_projects()
            autumn_project_cb.clear()
            autumn_project_cb.addItems(projects)
            if project:
                idx = autumn_project_cb.findText(project)
                if idx >= 0:
                    autumn_project_cb.setCurrentIndex(idx)
                else:
                    autumn_project_cb.setEditText(project)
            elif projects:
                autumn_project_cb.setCurrentIndex(0)
            _set_autumn_status("Connected")
        except AutumnError as exc:
            _set_autumn_status(str(exc))
        finally:
            loading_projects["value"] = False
        _refresh_subprojects()

    def _connect_autumn() -> None:
        username = autumn_user_edit.text().strip()
        password = autumn_pass_edit.text()
        if not username or not password:
            QMessageBox.warning(dlg, "Autumn", "Enter your Autumn username and password.")
            return
        try:
            client = AutumnClient(autumn_base_edit.text())
            token = client.authenticate(username, password)
            autumn_token["value"] = token
            _set_autumn_status("Connected")
            autumn_pass_edit.clear()
            _refresh_projects()
        except AutumnError as exc:
            _set_autumn_status(str(exc))

    def _disconnect_autumn() -> None:
        autumn_token["value"] = ""
        autumn_project_cb.clear()
        autumn_subs_list.clear()
        _set_autumn_status("Not connected")

    autumn_connect_btn.clicked.connect(_connect_autumn)
    autumn_disconnect_btn.clicked.connect(_disconnect_autumn)
    autumn_refresh_btn.clicked.connect(_refresh_projects)
    autumn_project_cb.activated.connect(lambda _idx: _refresh_subprojects())
    if autumn_project_cb.lineEdit() is not None:
        autumn_project_cb.lineEdit().editingFinished.connect(_refresh_subprojects)

    saved_project = str(window.settings.get("autumn_project") or "")
    saved_subprojects = window.settings.get("autumn_subprojects") or []
    if isinstance(saved_subprojects, str):
        saved_subprojects = [
            s.strip() for s in saved_subprojects.split(",") if s.strip()
        ]
    loading_projects["value"] = True
    if saved_project:
        autumn_project_cb.setEditText(saved_project)
    _set_subprojects(list(saved_subprojects), list(saved_subprojects))
    loading_projects["value"] = False
    _set_autumn_status("Connected" if autumn_token["value"] else "Not connected")

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
        window.settings["transcribe_chunk_seconds"] = int(chunk_sb.value())
        window.settings["language"] = lang_cb.currentText()
        window.settings["autumn_base_url"] = normalize_base_url(autumn_base_edit.text())
        window.settings["autumn_api_key"] = autumn_token["value"]
        window.settings["autumn_connected"] = bool(autumn_token["value"])
        window.settings["autumn_project"] = autumn_project_cb.currentText().strip()
        window.settings["autumn_subprojects"] = _selected_subprojects()
        if not autumn_token["value"]:
            window.settings["autumn_active_session_id"] = None
        # Reset any loaded Whisper model so it is recreated with new settings.
        # The active model cache is owned by TranscriptionService.
        try:
            if hasattr(window, "transcription_service") and window.transcription_service is not None:
                if hasattr(window.transcription_service, "unload_model"):
                    window.transcription_service.unload_model()
                else:
                    window.transcription_service.model = None
        except Exception:
            pass
        # Backward compatibility for any legacy window-level model attribute.
        try:
            window.model = None
        except Exception:
            pass
        save_settings(window.settings, settings_path())
        try:
            if hasattr(window, "_sync_autumn_ui"):
                window._sync_autumn_ui()
        except Exception:
            pass


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
        best_of=1,
        without_timestamps=without_ts,
        condition_on_previous_text=bool(settings.get("condition_on_previous_text", True)),
        compression_ratio_threshold=2.4,
        logprob_threshold=-1.0,
        no_speech_threshold=float(settings.get("no_speech_threshold", 0.45)),
        _speech_practice_chunk_seconds=int(
            settings.get("transcribe_chunk_seconds", 60)
        ),
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

    # Use greedy decode when beam size is 1; avoids beam-search overhead.
    try:
        if int(opts.get("beam_size", 1)) <= 1:
            opts["beam_size"] = None
            opts["best_of"] = 1
    except Exception:
        pass

    return opts


