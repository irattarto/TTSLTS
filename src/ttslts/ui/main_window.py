"""Main window: Text to Speech, Voice Changer and Settings tabs."""

from __future__ import annotations

import os
import re
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSystemTrayIcon,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ttslts import APP_NAME, paths
from ttslts.audio import devices
from ttslts.controller import Controller
from ttslts.train import prepare
from ttslts.tts.piper_download import (
    HELP_RHYTHM,
    HELP_SPEED,
    HELP_VARIATION,
    HELP_VOICE,
    HINT,
)

DEFAULT_DEVICE = "System default"

PERFORMANCE_HELP = {
    "chunk_ms": (
        "How much microphone sound is converted at a time. Smaller means less delay but needs "
        "a faster computer; raise it if the voice crackles or breaks up."
    ),
    "context_ms": (
        "How much earlier sound the model also hears each time. More is smoother and more "
        "natural but slower; 0 is fastest."
    ),
    "crossfade_ms": "How long neighbouring chunks are blended together to hide the joins.",
}

# Help for controls that are created without a tip of their own (see _apply_tooltips).
TIPS = {
    "text_edit": "The text to read aloud.",
    "tts_rvc_check": (
        "Run the speech through your RVC voice model, so the text is read in your own custom "
        "voice. Needs a Voice model and ContentVec (Settings > Models)."
    ),
    "tts_device_combo": (
        "Where the speech is played. Choose a virtual cable to send speech into Discord, "
        "games or OBS."
    ),
    "speak_btn": "Read the text aloud now.",
    "stop_btn": "Stop the speech that is playing.",
    "save_btn": "Save the speech as a WAV file instead of playing it.",
    "pron_btn": (
        "Words the voice says wrongly: write them the way they sound and the voice uses that "
        "instead. You can also select a word in the text and right-click it."
    ),
    "input_combo": "The microphone whose voice is changed.",
    "output_combo": (
        "Where the changed voice goes. A virtual cable (for example VB-Audio CABLE Input) lets "
        "other apps use it as a microphone; headphones let you hear it yourself."
    ),
    "pitch_spin": (
        "Raises or lowers the voice in semitones (12 = one octave). It brings a voice into the "
        "voice model's range, for example -12 for a female voice into a male voice. It applies "
        "to the voice changer and to text to speech through the voice model."
    ),
    "vc_button": (
        "Start or stop changing your microphone voice in real time. Delay and quality are set "
        "in Settings > Performance."
    ),
    "latency_label": (
        "How long the computer needs for each chunk; it must stay below the chunk size."
    ),
    "train_files": (
        "The recordings to train on (mp3, wav, flac, ogg): one person speaking clearly, "
        "without music or noise."
    ),
    "train_add_btn": "Choose recordings to train on.",
    "train_remove_btn": "Remove the selected recordings from the list.",
    "train_start_btn": (
        "Cut the recordings into clips, analyse them and train the voice. The first time, the "
        "training environment and the pretrained models are downloaded after you accept."
    ),
    "train_stop_btn": (
        "Stop after saving the voice made so far. Start the same name again to continue."
    ),
    "train_use_btn": (
        "Select the voice that was just trained as the voice model for the voice changer and "
        "for text to speech."
    ),
    "train_folder_btn": "Open the folder with the trained voice files (.pth and .onnx).",
    "train_bar": "Progress of the current step.",
    "train_log": (
        "What training is doing, and the result of each epoch. mel is how far the sound is "
        "from your recordings: lower is better."
    ),
    "minimize_check": "Minimizing hides the window to the system tray instead of the taskbar.",
    "close_check": (
        "The close button hides the window to the tray; use Quit in the tray menu to exit."
    ),
    "start_min_check": "Start with only the tray icon showing.",
    "contentvec_edit": (
        "ContentVec reads what is being said. It is needed by the voice changer, by text to "
        "speech through a voice, and by training. The download button below fetches it."
    ),
    "rmvpe_edit": (
        "RMVPE tracks the pitch of the voice more accurately than the built-in estimator. "
        "Optional, but recommended."
    ),
}


class SpeechTextEdit(QPlainTextEdit):
    """The text box; its context menu can open the pronunciation editor for the selected word."""

    fixPronunciation = Signal(str)

    def contextMenuEvent(self, event) -> None:
        menu = self.createStandardContextMenu()
        word = self.textCursor().selectedText().strip()
        if word and len(word) <= 40 and " " not in word:
            menu.addSeparator()
            action = menu.addAction(f"Fix the pronunciation of “{word}”…")
            action.triggered.connect(lambda: self.fixPronunciation.emit(word))
        menu.exec(event.globalPos())


class MainWindow(QMainWindow):
    def __init__(self, controller: Controller, tray_available: bool = True):
        super().__init__()
        self.controller = controller
        self.settings = controller.settings
        self.tray_available = tray_available
        self.tray: QSystemTrayIcon | None = None
        self._quitting = False
        self._tray_hint_shown = False

        self._voice_combos: list[QComboBox] = []
        self.setWindowTitle(APP_NAME)
        self.resize(560, 460)

        tabs = QTabWidget()
        tabs.addTab(self._build_tts_tab(), "Text to Speech")
        tabs.addTab(self._build_vc_tab(), "Voice Changer")
        tabs.addTab(self._build_train_tab(), "Train")
        tabs.addTab(self._build_settings_tab(), "Settings")
        self.tabs = tabs
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("Ready")
        self._apply_tooltips()

        c = controller
        c.status.connect(self.statusBar().showMessage)
        c.error.connect(self._show_error)
        c.ttsBusy.connect(self._on_tts_busy)
        c.vcRunning.connect(self._on_vc_running)
        c.vcLatency.connect(self._on_latency)
        c.rvcBaseDownloaded.connect(self._on_rvc_base_downloaded)
        c.piperDownloaded.connect(self._on_piper_downloaded)
        c.pretrainedDownloaded.connect(self._on_pretrained_downloaded)
        c.trainRuntimeInstalled.connect(self._on_train_runtime_installed)
        c.trainRunning.connect(self._on_train_running)
        c.trainStage.connect(self._on_train_stage)
        c.trainProgress.connect(self._on_train_progress)
        c.trainSaved.connect(self._on_train_saved)
        c.trainFinished.connect(self._on_train_finished)

    # -- tabs ------------------------------------------------------------
    def _build_tts_tab(self) -> QWidget:
        s = self.settings
        w = QWidget()
        layout = QVBoxLayout(w)
        self.text_edit = SpeechTextEdit()
        self.text_edit.fixPronunciation.connect(self._edit_pronunciations)
        self.text_edit.setPlaceholderText("Type text to speak…")
        layout.addWidget(self.text_edit, 1)

        form = QFormLayout()
        self.piper_combo = QComboBox()
        self.piper_combo.currentIndexChanged.connect(
            lambda i: self._set("piper_voice", self.piper_combo.itemData(i) or "")
        )
        self.piper_combo.setToolTip(HELP_VOICE)
        form.addRow("Voice", self.piper_combo)
        self.piper_hint = QLabel(HINT)
        self.piper_hint.setWordWrap(True)
        self.piper_hint.setStyleSheet("color: gray;")
        form.addRow(self.piper_hint)

        self.variation_spin = QDoubleSpinBox(minimum=0.1, maximum=1.5, singleStep=0.05)
        self.variation_spin.setValue(s.piper_variation)
        self.variation_spin.setToolTip(HELP_VARIATION)
        self.variation_spin.valueChanged.connect(lambda v: self._set("piper_variation", float(v)))
        form.addRow("Voice variation", self.variation_spin)
        self.rhythm_spin = QDoubleSpinBox(minimum=0.1, maximum=1.5, singleStep=0.05)
        self.rhythm_spin.setValue(s.piper_rhythm)
        self.rhythm_spin.setToolTip(HELP_RHYTHM)
        self.rhythm_spin.valueChanged.connect(lambda v: self._set("piper_rhythm", float(v)))
        form.addRow("Rhythm variation", self.rhythm_spin)

        self.speed_spin = QDoubleSpinBox(minimum=0.5, maximum=2.0, singleStep=0.05)
        self.speed_spin.setValue(s.tts_speed)
        self.speed_spin.valueChanged.connect(lambda v: self._set("tts_speed", float(v)))
        self.speed_spin.setToolTip(HELP_SPEED)
        form.addRow("Speed", self.speed_spin)

        form.addRow("Voice model", self._voice_combo())
        self.tts_rvc_check = QCheckBox("Speak with my voice model (RVC)")
        self.tts_rvc_check.setChecked(s.tts_through_rvc)
        self.tts_rvc_check.toggled.connect(lambda v: self._set("tts_through_rvc", v))
        form.addRow("", self.tts_rvc_check)
        self.tts_auto_pitch_check = QCheckBox("Match the speech pitch to the voice automatically")
        self.tts_auto_pitch_check.setChecked(s.tts_auto_pitch)
        self.tts_auto_pitch_check.toggled.connect(lambda v: self._set("tts_auto_pitch", v))
        self.tts_auto_pitch_check.setToolTip(
            "A voice keeps the pitch of the speech it converts, so a high text to speech voice "
            "becomes an unnatural, strained version of a deep voice. This shifts the speech to "
            "the pitch range of the recordings the voice was trained on (voices trained in "
            "TTSLTS only). The Pitch shift setting is added on top."
        )
        form.addRow("", self.tts_auto_pitch_check)

        self.tts_device_combo = self._device_combo("output", "tts_output_device")
        form.addRow("Play on", self.tts_device_combo)
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.speak_btn = QPushButton("Speak")
        self.speak_btn.clicked.connect(self._speak)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.controller.stop_speaking)
        self.save_btn = QPushButton("Save WAV…")
        self.save_btn.clicked.connect(self._save_wav)
        self.pron_btn = QPushButton("Pronunciations…")
        self.pron_btn.clicked.connect(lambda: self._edit_pronunciations())
        for b in (self.speak_btn, self.stop_btn, self.save_btn, self.pron_btn):
            buttons.addWidget(b)
        layout.addLayout(buttons)
        self._refresh_piper_voices()
        return w

    def _build_vc_tab(self) -> QWidget:
        s = self.settings
        w = QWidget()
        layout = QVBoxLayout(w)
        form = QFormLayout()
        self.input_combo = self._device_combo("input", "input_device")
        form.addRow("Microphone", self.input_combo)
        self.output_combo = self._device_combo("output", "output_device")
        form.addRow("Output", self.output_combo)
        form.addRow("Voice model", self._voice_combo())
        self.pitch_spin = QSpinBox(minimum=-24, maximum=24, suffix=" semitones")
        self.pitch_spin.setValue(s.pitch_shift)
        self.pitch_spin.valueChanged.connect(lambda v: self._set("pitch_shift", v))
        form.addRow("Pitch shift", self.pitch_spin)
        self.index_spin = QDoubleSpinBox(minimum=0.0, maximum=1.0, singleStep=0.05)
        self.index_spin.setValue(s.rvc_index_rate)
        self.index_spin.valueChanged.connect(lambda v: self._set("rvc_index_rate", float(v)))
        self.index_spin.setToolTip(
            "How much the result leans on the real recordings of the voice (0 = not at all, "
            "1 = fully). Higher sounds more like the real voice and clearer; too high can sound "
            "rough or lose words. Needs a voice trained in TTSLTS (it has an index); applies to "
            "the voice changer and to text to speech through the voice model."
        )
        form.addRow("Index rate", self.index_spin)
        self.protect_spin = QDoubleSpinBox(minimum=0.0, maximum=0.5, singleStep=0.01)
        self.protect_spin.setValue(s.rvc_protect)
        self.protect_spin.valueChanged.connect(lambda v: self._set("rvc_protect", float(v)))
        self.protect_spin.setToolTip(
            "Protects consonants and breath sounds (s, sh, f, h...) from being smeared by the "
            "index. Lower protects more (0 = most), 0.5 = off. Only matters when Index rate is "
            "above 0."
        )
        form.addRow("Protect consonants", self.protect_spin)
        self.speaker_spin = QSpinBox(minimum=0, maximum=255)
        self.speaker_spin.setValue(s.speaker_id)
        self.speaker_spin.valueChanged.connect(lambda v: self._set("speaker_id", v))
        self.speaker_spin.setToolTip(
            "Only for voices that contain several speakers. Voices trained in TTSLTS have one "
            "speaker, number 0, so this is fixed at 0 for them."
        )
        form.addRow("Speaker ID", self.speaker_spin)
        self._apply_speaker_lock()
        layout.addLayout(form)

        hint = QLabel(
            "Tip: to use the changed voice in Discord, games or OBS, set Output to a virtual "
            "cable (e.g. VB-Audio CABLE Input) and pick the cable as the microphone there."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.vc_button = QPushButton("Start voice changer")
        self.vc_button.setCheckable(True)
        self.vc_button.clicked.connect(self._on_vc_clicked)
        layout.addWidget(self.vc_button)
        self.latency_label = QLabel("")
        layout.addWidget(self.latency_label)
        layout.addStretch(1)
        return w

    def _build_train_tab(self) -> QWidget:
        s = self.settings
        w = QWidget()
        layout = QVBoxLayout(w)
        intro = QLabel(
            "Train your own voice from recordings of it. Add mp3 or wav files with one person "
            "speaking clearly, without music or background noise. About 10 minutes gives a good "
            "voice; even 2-3 minutes works. Only use recordings of voices you have permission "
            "to use."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.train_setup_hint = QLabel(
            "The first time, TTSLTS sets up PyTorch for training (a one-time download, "
            "you are asked first)."
        )
        self.train_setup_hint.setWordWrap(True)
        self.train_setup_hint.setStyleSheet("color: gray;")
        self.train_setup_hint.setVisible(not self.controller.training_runtime_ready())
        layout.addWidget(self.train_setup_hint)

        self.train_files = QListWidget()
        self.train_files.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.train_files.setMinimumHeight(70)
        layout.addWidget(self.train_files, 1)
        file_buttons = QHBoxLayout()
        self.train_add_btn = QPushButton("Add recordings…")
        self.train_add_btn.clicked.connect(self._train_add_files)
        self.train_remove_btn = QPushButton("Remove")
        self.train_remove_btn.clicked.connect(self._train_remove_files)
        file_buttons.addWidget(self.train_add_btn)
        file_buttons.addWidget(self.train_remove_btn)
        file_buttons.addStretch(1)
        layout.addLayout(file_buttons)

        form = QFormLayout()
        self.train_name_edit = QLineEdit(s.train_name)
        self.train_name_edit.setToolTip("The voice is saved as <name>.pth and <name>.onnx")
        self.train_name_edit.editingFinished.connect(
            lambda: self._set("train_name", self._train_name())
        )
        form.addRow("Voice name", self.train_name_edit)
        self.train_sr_combo = QComboBox()
        for rate in (32000, 40000, 48000):
            self.train_sr_combo.addItem(f"{rate // 1000} kHz", rate)
        self.train_sr_combo.setCurrentIndex(
            max(0, self.train_sr_combo.findData(s.train_sample_rate))
        )
        self.train_sr_combo.currentIndexChanged.connect(
            lambda i: self._set("train_sample_rate", self.train_sr_combo.itemData(i))
        )
        self.train_sr_combo.setToolTip("40 kHz is the usual choice; 48 kHz sounds slightly clearer")
        form.addRow("Sample rate", self.train_sr_combo)
        self.train_epochs_spin = QSpinBox(minimum=1, maximum=2000)
        self.train_epochs_spin.setValue(s.train_epochs)
        self.train_epochs_spin.valueChanged.connect(lambda v: self._set("train_epochs", v))
        self.train_epochs_spin.setToolTip(
            "How many passes over your recordings. 100-300 suits a few minutes of audio; "
            "too many on little audio makes the voice worse. You can stop at any time."
        )
        form.addRow("Epochs", self.train_epochs_spin)
        self.train_batch_spin = QSpinBox(minimum=1, maximum=64)
        self.train_batch_spin.setValue(s.train_batch_size)
        self.train_batch_spin.valueChanged.connect(lambda v: self._set("train_batch_size", v))
        self.train_batch_spin.setToolTip("Lower this if you run out of GPU memory")
        form.addRow("Batch size", self.train_batch_spin)
        self.train_save_spin = QSpinBox(minimum=1, maximum=500)
        self.train_save_spin.setValue(s.train_save_every)
        self.train_save_spin.valueChanged.connect(lambda v: self._set("train_save_every", v))
        self.train_save_spin.setToolTip(
            "The voice file is updated this often, so you can try it early"
        )
        form.addRow("Save every", self.train_save_spin)
        self.train_save_spin.setSuffix(" epochs")
        layout.addLayout(form)

        buttons = QHBoxLayout()
        self.train_start_btn = QPushButton("Start training")
        self.train_start_btn.clicked.connect(self._train_start)
        self.train_stop_btn = QPushButton("Stop")
        self.train_stop_btn.setEnabled(False)
        self.train_stop_btn.clicked.connect(self.controller.stop_training)
        self.train_use_btn = QPushButton("Use this voice")
        self.train_use_btn.setEnabled(False)
        self.train_use_btn.clicked.connect(self._train_use_voice)
        self.train_folder_btn = QPushButton("Open folder")
        self.train_folder_btn.clicked.connect(self._train_open_folder)
        for b in (
            self.train_start_btn,
            self.train_stop_btn,
            self.train_use_btn,
            self.train_folder_btn,
        ):
            buttons.addWidget(b)
        layout.addLayout(buttons)

        self.train_bar = QProgressBar()
        self.train_bar.setRange(0, 1000)
        layout.addWidget(self.train_bar)
        self.train_status = QLabel("")
        self.train_status.setWordWrap(True)
        layout.addWidget(self.train_status)
        self.train_log = QPlainTextEdit()
        self.train_log.setReadOnly(True)
        self.train_log.setMaximumHeight(90)
        layout.addWidget(self.train_log)
        self._train_result = ""  # the .onnx of the last finished run
        self._train_busy = False
        self._pending_train: tuple | None = None
        self._apply_train_availability()
        return w

    def _build_settings_tab(self) -> QWidget:
        s = self.settings
        w = QWidget()
        layout = QVBoxLayout(w)

        window_box = QGroupBox("Window")
        wl = QVBoxLayout(window_box)
        self.minimize_check = QCheckBox("Minimize to tray")
        self.minimize_check.setChecked(s.minimize_to_tray)
        self.minimize_check.toggled.connect(lambda v: self._set("minimize_to_tray", v))
        self.close_check = QCheckBox("Close to tray")
        self.close_check.setChecked(s.close_to_tray)
        self.close_check.toggled.connect(lambda v: self._set("close_to_tray", v))
        self.start_min_check = QCheckBox("Start minimized to tray")
        self.start_min_check.setChecked(s.start_minimized)
        self.start_min_check.toggled.connect(lambda v: self._set("start_minimized", v))
        for cb in (self.minimize_check, self.close_check, self.start_min_check):
            cb.setEnabled(self.tray_available)
            wl.addWidget(cb)
        if not self.tray_available:
            wl.addWidget(QLabel("No system tray was found, so these options are disabled."))
        layout.addWidget(window_box)

        models_box = QGroupBox("Models")
        ml = QFormLayout(models_box)
        self.rvc_model_edit = self._path_row(
            ml, "RVC v2 voice", "rvc_model_path", "RVC v2 voice (*.pth *.onnx)"
        )
        self.rvc_model_edit.setToolTip(
            "A downloaded RVC v2 voice: the .pth file works directly (converted once, "
            "the first time it is used) and so does an .onnx export"
        )
        self.contentvec_edit = self._path_row(ml, "ContentVec", "contentvec_path", "ONNX (*.onnx)")
        self.rmvpe_edit = self._path_row(ml, "RMVPE (optional)", "rmvpe_path", "ONNX (*.onnx)")
        piper_dl = QPushButton("Download Hungarian voices (Piper, ~190 MB)")
        piper_dl.setToolTip(
            "From the original Piper voice collection; asks you to accept its terms first"
        )
        piper_dl.clicked.connect(self._download_piper)
        ml.addRow("", piper_dl)
        rvc_dl = QPushButton("Download ContentVec + RMVPE (~550 MB)")
        rvc_dl.setToolTip("From the original RVC project; asks you to accept its terms first")
        rvc_dl.clicked.connect(self._download_rvc_base)
        ml.addRow("", rvc_dl)
        sr = QComboBox()
        for rate in (32000, 40000, 48000):
            sr.addItem(f"{rate // 1000} kHz", rate)
        sr.setCurrentIndex(max(0, sr.findData(s.rvc_sample_rate)))
        sr.currentIndexChanged.connect(lambda i: self._set("rvc_sample_rate", sr.itemData(i)))
        sr.setToolTip("Used only when the RVC model file doesn't say its own sample rate")
        ml.addRow("RVC sample rate", sr)
        layout.addWidget(models_box)

        perf_box = QGroupBox("Performance")
        pl = QFormLayout(perf_box)
        gpu = QCheckBox("Use GPU (CUDA / DirectML) when available")
        gpu.setChecked(s.use_gpu)
        gpu.toggled.connect(lambda v: self._set("use_gpu", v))
        gpu.setToolTip(
            "Run the models on the graphics card (CUDA on NVIDIA, DirectML on any Windows "
            "GPU) when one is available. The real-time voice changer needs this; on the CPU "
            "it is too slow. Applies the next time a model is loaded."
        )
        pl.addRow("", gpu)
        retry = QPushButton("Try the GPU again for every model")
        retry.setToolTip(
            "If a model could not run on the graphics card, the app remembers that and uses the "
            "CPU for it. This forgets it, so every model is tried on the GPU again (for example "
            "after a driver update)."
        )
        retry.clicked.connect(self.controller.retry_gpu)
        pl.addRow("", retry)
        for label, key, lo, hi, suffix in (
            ("Chunk size", "chunk_ms", 50, 1000, " ms"),
            ("Context", "context_ms", 0, 3000, " ms"),
            ("Crossfade", "crossfade_ms", 0, 200, " ms"),
        ):
            spin = QSpinBox(minimum=lo, maximum=hi, suffix=suffix, singleStep=10)
            spin.setValue(getattr(s, key))
            spin.valueChanged.connect(lambda v, k=key: self._set(k, v))
            spin.setToolTip(PERFORMANCE_HELP[key])
            pl.addRow(label, spin)
        gate = QDoubleSpinBox(minimum=-90, maximum=0, suffix=" dB", singleStep=1)
        gate.setValue(s.silence_threshold_db)
        gate.valueChanged.connect(lambda v: self._set("silence_threshold_db", float(v)))
        gate.setToolTip(
            "Microphone sound quieter than this is treated as silence and not converted, "
            "which saves processing and keeps background noise out. Raise it if noise comes "
            "through, lower it if quiet speech is cut off."
        )
        pl.addRow("Silence gate", gate)
        layout.addWidget(perf_box)
        layout.addWidget(QLabel("Voice changer changes apply the next time it starts."))
        layout.addStretch(1)
        return w

    # -- helpers ---------------------------------------------------------
    def _apply_tooltips(self) -> None:
        """Give every control a hover help; rows share their field's help with their label."""
        for name, tip in TIPS.items():
            widget = getattr(self, name)
            if not widget.toolTip():
                widget.setToolTip(tip)
        for form in self.findChildren(QFormLayout):
            for row in range(form.rowCount()):
                label = form.itemAt(row, QFormLayout.ItemRole.LabelRole)
                field = form.itemAt(row, QFormLayout.ItemRole.FieldRole)
                if label is None or field is None or label.widget() is None:
                    continue
                widget = field.widget()
                if widget is None and field.layout() is not None and field.layout().count():
                    widget = field.layout().itemAt(0).widget()
                if widget is not None and widget.toolTip():
                    label.widget().setToolTip(widget.toolTip())

    def _set(self, key: str, value) -> None:
        setattr(self.settings, key, value)
        self.controller.save_settings()
        if key == "rvc_model_path":
            self._refresh_voice_combos()

    @staticmethod
    def _same_path(a: str, b: str) -> bool:
        return (
            bool(a)
            and bool(b)
            and os.path.normcase(os.path.abspath(a)) == os.path.normcase(os.path.abspath(b))
        )

    def _voice_combo(self) -> QComboBox:
        """A picker for the voice model, by name: voices trained here plus the chosen file."""
        combo = QComboBox()
        combo.setToolTip(
            "The voice used by the voice changer and by Speak with my voice model. Voices you "
            "train are listed by name; pick a file of your own in Settings > Models."
        )
        combo.activated.connect(lambda i, c=combo: self._on_voice_picked(c.itemData(i) or ""))
        self._voice_combos.append(combo)
        self._refresh_voice_combos()
        return combo

    def _trained_voices(self) -> list[tuple[str, str]]:
        root = paths.voices_dir()
        found = []
        for folder in sorted(root.iterdir()) if root.exists() else []:
            onnx = folder / f"{folder.name}.onnx"
            if onnx.exists():
                found.append((folder.name, str(onnx)))
        return found

    def _refresh_voice_combos(self) -> None:
        voices = self._trained_voices()
        current = self.settings.rvc_model_path
        for combo in self._voice_combos:
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none selected)", "")
            for name, path in voices:
                combo.addItem(name, path)
            if current and not any(self._same_path(current, p) for _n, p in voices):
                combo.addItem(f"{Path(current).stem} (file from Settings)", current)
            index = next(
                (i for i in range(combo.count()) if self._same_path(current, combo.itemData(i))),
                0,
            )
            combo.setCurrentIndex(index)
            combo.blockSignals(False)
        self._apply_speaker_lock()

    def _apply_speaker_lock(self) -> None:
        """A voice trained here has a single speaker (0); other IDs would sound wrong."""
        spin = getattr(self, "speaker_spin", None)
        if spin is None:
            return
        trained = any(
            self._same_path(self.settings.rvc_model_path, p) for _n, p in self._trained_voices()
        )
        if trained:
            spin.setValue(0)  # also saves the setting through its valueChanged handler
        spin.setEnabled(not trained)

    def _on_voice_picked(self, path: str) -> None:
        if hasattr(self, "rvc_model_edit"):
            self.rvc_model_edit.setText(path)
        self._set("rvc_model_path", path)

    def _device_combo(self, kind: str, key: str) -> QComboBox:
        combo = QComboBox()
        combo.addItem(DEFAULT_DEVICE, "")
        for name in devices.list_devices(kind):
            combo.addItem(name, name)
        current = getattr(self.settings, key)
        if current and combo.findData(current) < 0:
            combo.addItem(f"{current} (not connected)", current)
        combo.setCurrentIndex(max(0, combo.findData(current)))
        combo.currentIndexChanged.connect(lambda i: self._set(key, combo.itemData(i)))
        return combo

    def _path_row(self, form: QFormLayout, label: str, key: str, filt: str) -> QLineEdit:
        edit = QLineEdit(getattr(self.settings, key))
        edit.editingFinished.connect(lambda: self._set(key, edit.text().strip()))
        browse = QToolButton(text="…")
        browse.setToolTip(f"Browse for the {label.lower()} file")

        def pick():
            path, _ = QFileDialog.getOpenFileName(self, label, edit.text(), filt)
            if path:
                edit.setText(path)
                self._set(key, path)

        browse.clicked.connect(pick)
        row = QHBoxLayout()
        row.addWidget(edit, 1)
        row.addWidget(browse)
        form.addRow(label, row)
        return edit

    def _save_wav(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save speech", "speech.wav", "WAV (*.wav)")
        if path:
            self.controller.save_wav(self.text_edit.toPlainText(), path)

    def _show_error(self, message: str) -> None:
        self._pending_train = None  # a failed setup step cancels the queued training
        self.train_bar.setRange(0, 1000)
        self._apply_train_availability()
        self.statusBar().showMessage(message)
        if self.isVisible():
            QMessageBox.warning(self, APP_NAME, message)
        elif self.tray is not None:
            self.tray.showMessage(APP_NAME, message, QSystemTrayIcon.MessageIcon.Warning)

    def _on_tts_busy(self, busy: bool) -> None:
        self.speak_btn.setEnabled(not busy)
        self.save_btn.setEnabled(not busy)

    def _on_vc_clicked(self, checked: bool) -> None:
        # disabled until the voice changer reports it is on or off (starting takes seconds)
        self.vc_button.setEnabled(False)
        self.controller.start_vc() if checked else self.controller.stop_vc()

    def _on_vc_running(self, running: bool) -> None:
        self.vc_button.setEnabled(True)
        self.vc_button.setChecked(running)
        self.vc_button.setText("Stop voice changer" if running else "Start voice changer")
        if not running:
            self.latency_label.setText("")

    def _on_latency(self, ms: float) -> None:
        text = f"Processing: {ms:.0f} ms per chunk"
        if ms > self.settings.chunk_ms:
            text += (
                f" - too slow for a {self.settings.chunk_ms} ms chunk, the voice will break up. "
                "Raise the chunk size in Settings or use a GPU."
            )
        self.latency_label.setText(text)
        self.latency_label.setWordWrap(True)

    # -- pronunciations -----------------------------------------------------
    def _edit_pronunciations(self, word: str = "") -> None:
        from ttslts.tts import pronunciation
        from ttslts.ui.pronunciation_dialog import PronunciationDialog

        dialog = PronunciationDialog(
            self,
            pronunciation.load(),
            self.settings.pron_builtin,
            listen=self.controller.speak,
            word=word,
        )
        if dialog.exec():
            try:
                pronunciation.save(dialog.entries())
            except OSError as exc:
                self._show_error(f"Could not save the pronunciations: {exc}")
                return
            self._set("pron_builtin", dialog.use_builtin())

    # -- Hungarian voices (Piper) ---------------------------------------
    def _speak(self) -> None:
        from ttslts.tts import piper_download

        if not piper_download.installed_voices():  # nothing to speak with yet: offer them
            self._download_piper()
            return
        self._on_tts_busy(True)  # at once, so a quick second click cannot start another
        self.controller.speak(self.text_edit.toPlainText())

    def _refresh_piper_voices(self) -> None:
        from ttslts.tts import piper_download

        installed = piper_download.installed_voices()
        self.piper_combo.blockSignals(True)
        self.piper_combo.clear()
        for voice_id in installed:
            self.piper_combo.addItem(piper_download.display_name(voice_id), voice_id)
            self.piper_combo.setItemData(
                self.piper_combo.count() - 1,
                f"{piper_download.display_name(voice_id)}: "
                f"{piper_download.VOICE_NOTES.get(voice_id, '')}",
                Qt.ItemDataRole.ToolTipRole,
            )
        if not installed:
            self.piper_combo.addItem("None installed (Settings > Models > Download)", "")
        self.piper_combo.blockSignals(False)
        idx = self.piper_combo.findData(self.settings.piper_voice)
        self.piper_combo.setCurrentIndex(idx if idx >= 0 else 0)
        # keep the saved choice in step with what is actually installed
        self.settings.piper_voice = self.piper_combo.currentData() or ""

    def _download_piper(self) -> None:
        from ttslts.tts import piper_download

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(piper_download.CONSENT_TITLE)
        box.setText("Download the Hungarian voices?")
        box.setInformativeText(piper_download.CONSENT_TEXT)
        accept = box.addButton("I accept, download", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is accept:
            self.controller.download_piper_voices()

    def _on_piper_downloaded(self, _installed: list) -> None:
        self._refresh_piper_voices()
        self.controller.save_settings()

    def _download_rvc_base(self) -> None:
        from ttslts.vc import rvc_base

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(rvc_base.CONSENT_TITLE)
        box.setText("Download the RVC base models?")
        box.setInformativeText(rvc_base.CONSENT_TEXT)
        accept = box.addButton("I accept, download", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is accept:
            self.controller.download_rvc_base()

    def _on_rvc_base_downloaded(self, contentvec: str, rmvpe: str) -> None:
        self.contentvec_edit.setText(contentvec)
        self.rmvpe_edit.setText(rmvpe)
        self.settings.contentvec_path = contentvec
        self.settings.rmvpe_path = rmvpe
        self.controller.save_settings()

    # -- training --------------------------------------------------------
    def _train_name(self) -> str:
        """The voice name as a safe folder / file name."""
        name = re.sub(r"[^\w\- ]", "", self.train_name_edit.text(), flags=re.UNICODE).strip()
        return name or "my-voice"

    def _apply_train_availability(self) -> None:
        self.train_start_btn.setEnabled(not self._train_busy)

    def _train_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Recordings to train on", "", prepare.AUDIO_FILTER
        )
        have = {self.train_files.item(i).text() for i in range(self.train_files.count())}
        for f in files:
            if f not in have:
                self.train_files.addItem(f)

    def _train_remove_files(self) -> None:
        for item in self.train_files.selectedItems():
            self.train_files.takeItem(self.train_files.row(item))

    def _train_start(self) -> None:
        from ttslts.train.hparams import CHECKPOINT

        files = [self.train_files.item(i).text() for i in range(self.train_files.count())]
        name = self._train_name()
        self.train_name_edit.setText(name)
        self._set("train_name", name)

        resume = False
        if (paths.train_dir() / name / CHECKPOINT).exists():
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Icon.Question)
            box.setWindowTitle(APP_NAME)
            box.setText(f"A voice called “{name}” has been trained here before.")
            box.setInformativeText(
                "Continue training it where it stopped (the recordings and settings from "
                "before are used), or start over with the recordings listed now?"
            )
            cont = box.addButton("Continue", QMessageBox.ButtonRole.AcceptRole)
            fresh = box.addButton("Start over", QMessageBox.ButtonRole.DestructiveRole)
            box.addButton(QMessageBox.StandardButton.Cancel)
            box.exec()
            if box.clickedButton() is cont:
                resume = True
            elif box.clickedButton() is not fresh:
                return
            if not resume:
                import shutil

                shutil.rmtree(paths.train_dir() / name, ignore_errors=True)

        self._train_continue((files, name, resume))

    def _train_continue(self, job: tuple) -> None:
        """Set up what training still needs (environment, pretrained models), then start it."""
        from ttslts.train import pretrained

        if not self.controller.training_runtime_ready():
            self._install_train_runtime(job)
        elif not pretrained.is_installed(self.settings.train_sample_rate):
            self._download_pretrained(job)
        else:
            self.train_log.clear()
            self._train_result = ""
            self.train_use_btn.setEnabled(False)
            self.controller.start_training(*job)

    def _install_train_runtime(self, job: tuple) -> None:
        from ttslts.train import runtime

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(runtime.CONSENT_TITLE)
        box.setText("Set up the training environment?")
        box.setInformativeText(runtime.consent_text())
        accept = box.addButton("Download and set up", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is accept:
            self._pending_train = job
            self.train_start_btn.setEnabled(False)
            self.train_bar.setRange(0, 0)  # busy indicator while pip works
            self.controller.install_training_runtime()

    def _on_train_runtime_installed(self) -> None:
        self.train_bar.setRange(0, 1000)
        self.train_setup_hint.setVisible(False)
        self._apply_train_availability()
        pending, self._pending_train = self._pending_train, None
        if pending is not None:
            self._train_continue(pending)

    def _download_pretrained(self, then_train: tuple | None = None) -> None:
        from ttslts.train import pretrained

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Question)
        box.setWindowTitle(pretrained.CONSENT_TITLE)
        box.setText("Download the pretrained voice models?")
        box.setInformativeText(pretrained.CONSENT_TEXT)
        accept = box.addButton("I accept, download", QMessageBox.ButtonRole.AcceptRole)
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is accept:
            self._pending_train = then_train
            self.train_start_btn.setEnabled(False)
            self.controller.download_pretrained(self.settings.train_sample_rate)

    def _on_pretrained_downloaded(self, _sample_rate: int) -> None:
        self._apply_train_availability()
        pending, self._pending_train = self._pending_train, None
        if pending is not None:
            self._train_continue(pending)

    def _on_train_running(self, running: bool) -> None:
        self._train_busy = running
        self.train_stop_btn.setEnabled(running)
        for widget in (
            self.train_add_btn,
            self.train_remove_btn,
            self.train_files,
            self.train_name_edit,
            self.train_sr_combo,
            self.train_epochs_spin,
            self.train_batch_spin,
            self.train_save_spin,
        ):
            widget.setEnabled(not running)
        self._apply_train_availability()

    def _on_train_stage(self, text: str) -> None:
        self.train_status.setText(text)
        if text:
            self.train_log.appendPlainText(text)

    def _on_train_progress(self, event: dict) -> None:
        self.train_bar.setValue(int(event.get("fraction", 0) * 1000))
        if event.get("stage") != "train":
            return
        elapsed = event.get("elapsed", 0.0)
        epoch, epochs, step, steps = (event[k] for k in ("epoch", "epochs", "step", "steps"))
        done = (epoch - 1 + step / steps) / epochs
        eta = elapsed * (1 - done) / done if done > 0.002 else 0
        text = f"Epoch {epoch}/{epochs}"
        if eta:
            text += f" - about {int(eta // 60)} min left"
        self.train_status.setText(text)
        losses = event.get("losses")
        if losses:
            self.train_log.appendPlainText(
                f"Epoch {epoch}: mel {losses['mel']:.1f}  gen {losses['gen']:.1f}  "
                f"disc {losses['disc']:.2f}"
            )

    def _on_train_saved(self, _pth: str, epoch: int) -> None:
        self.train_log.appendPlainText(f"Saved the voice at epoch {epoch}")

    def _on_train_finished(self, pth: str, onnx: str, finished: bool) -> None:
        if not pth:
            self.train_status.setText("Stopped before anything was saved.")
            return
        self._train_result = onnx
        self._refresh_voice_combos()
        self.train_use_btn.setEnabled(bool(onnx))
        if finished:
            self.train_bar.setValue(1000)
        self.train_status.setText(
            ("Done. " if finished else "Stopped; the last saved voice is ready. ")
            + f"Saved to {Path(pth).parent}"
        )
        self.train_log.appendPlainText(f"Voice files: {pth}\n{onnx}")

    def _train_use_voice(self) -> None:
        if not self._train_result:
            return
        self.rvc_model_edit.setText(self._train_result)
        self._set("rvc_model_path", self._train_result)
        self.statusBar().showMessage(
            "Voice selected: use it on the Voice Changer tab, or tick “Speak with my voice model” "
            "on Text to Speech"
        )

    def _train_open_folder(self) -> None:

        folder = paths.voices_dir() / self._train_name()
        folder.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # -- tray behaviour --------------------------------------------------
    def _use_tray(self, enabled: bool) -> bool:
        return enabled and self.tray_available and not self._quitting

    def _hide_to_tray(self) -> None:
        self.hide()
        if self.tray is not None and not self._tray_hint_shown:
            self._tray_hint_shown = True
            self.tray.showMessage(
                APP_NAME,
                "Still running in the system tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )

    def changeEvent(self, event: QEvent) -> None:
        if (
            event.type() == QEvent.Type.WindowStateChange
            and self.windowState() & Qt.WindowState.WindowMinimized
            and self._use_tray(self.settings.minimize_to_tray)
        ):
            QTimer.singleShot(0, self._hide_to_tray)
        super().changeEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._use_tray(self.settings.close_to_tray):
            event.ignore()
            self._hide_to_tray()
            return
        event.accept()
        self.quit()

    def restore(self) -> None:
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.show()
        self.raise_()
        self.activateWindow()

    def toggle_visible(self) -> None:
        if self.isVisible() and not self.isMinimized():
            self.hide()
        else:
            self.restore()

    def quit(self) -> None:
        self._quitting = True
        if self.tray is not None:
            self.tray.hide()
        app = QApplication.instance()
        if app is not None:
            app.quit()
