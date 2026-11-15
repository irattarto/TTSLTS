"""Check which sentences an own F5 voice will learn from (see ``train.f5_data``).

Shows what is kept and samples of what is left out, by why; every sample can be played. The
strictness re-chooses at once from the measurements. "Use this selection" saves the choice for
the training step.
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ttslts.train import f5_data

SAMPLES = 40  # shown per group (chosen at random, the same ones each time)
STRICTNESS_LABELS = {
    "loose": "Loose (keeps more; some character voices may stay)",
    "normal": "Normal (as the first own voices were made)",
    "strict": "Strict (only the narrator's calmest sentences)",
}
TABS = [
    ("kept", "Kept", "Sentences the voice will learn from. Mark any that is not the narrator's "
     "own voice: then make it stricter."),
    ("other_voice", "Not the narrator", "Left out as another voice (a character the narrator "
     "plays, another person). If many are the narrator's own voice, make it looser."),
    ("trembling", "Trembling", "The narrator, but with the most trembling pitch: the voice would "
     "learn the trembling."),
    ("unsure", "Unsure text", "Two Whisper models wrote these down differently, so the text may "
     "not match what is said."),
]  # fmt: skip


class F5ReviewDialog(QDialog):
    def __init__(self, work: Path, play: Callable[[Path], None], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Check the sentences for the F5 voice")
        self.resize(820, 560)
        self.work = Path(work)
        self.play = play
        self.clips = json.loads((self.work / "measured.json").read_text(encoding="utf-8"))
        saved = f5_data.load_selection(self.work)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Listen to some sentences in each group (double-click one to play it). The voice "
            "learns only from the kept ones, so they should all be the narrator's own, calm voice."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Strictness"))
        self.strictness = QComboBox()
        for key, label in STRICTNESS_LABELS.items():
            self.strictness.addItem(label, key)
        start = saved["strictness"] if saved else "normal"
        self.strictness.setCurrentIndex(max(0, self.strictness.findData(start)))
        self.strictness.setToolTip(
            "How alike to the narrator's usual voice a sentence must be, how near his or her "
            "usual pitch, and how many of the most trembling ones are left out"
        )
        self.strictness.currentIndexChanged.connect(self._choose)
        row.addWidget(self.strictness, 1)
        layout.addLayout(row)
        self.report = QLabel()
        self.report.setWordWrap(True)
        layout.addWidget(self.report)

        self.tabs = QTabWidget()
        self.trees: dict[str, QTreeWidget] = {}
        for key, title, tip in TABS:
            tree = QTreeWidget()
            tree.setHeaderLabels(["Sentence", "Likeness", "Pitch", "Trembling", "Text check"])
            tree.setRootIsDecorated(False)
            tree.setToolTip(tip)
            tree.itemDoubleClicked.connect(lambda item, _col: self._play(item))
            tree.setColumnWidth(0, 430)
            self.trees[key] = tree
            self.tabs.addTab(tree, title)
        layout.addWidget(self.tabs, 1)

        bottom = QHBoxLayout()
        play = QPushButton("Play")
        play.setToolTip("Play the selected sentence")
        play.clicked.connect(self._play_selected)
        bottom.addWidget(play)
        bottom.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.use_btn = buttons.addButton(
            "Use this selection", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.use_btn.setToolTip("Save the choice: the voice is trained on the kept sentences")
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        bottom.addWidget(buttons)
        layout.addLayout(bottom)
        self._choose()

    def strictness_key(self) -> str:
        return self.strictness.currentData()

    def _choose(self) -> None:
        self.result_report = f5_data.select(self.clips, self.strictness_key())
        r = self.result_report
        g = f5_data.groups(self.clips)
        self.report.setText(
            f"Kept {r['kept']} of {r['clips']} sentences ({r['kept_minutes']:.0f} of "
            f"{r['minutes']:.0f} minutes). Left out: {len(g['other_voice'])} not the narrator, "
            f"{len(g['trembling'])} trembling, {len(g['unsure'])} with an unsure text."
            + ("" if r["kept_minutes"] >= 30 else "  Fewer than 30 minutes: add more recordings.")
        )
        rng = random.Random(0)
        for key, _title, _tip in TABS:
            tree = self.trees[key]
            tree.clear()
            group = g[key]
            for c in rng.sample(group, min(SAMPLES, len(group))):
                item = QTreeWidgetItem([
                    c["text"], f"{c.get('sim', 0):.2f}",
                    f"{c['f0']:.0f} Hz" if c.get("f0") else "–",
                    f"{c['wobble']:.0f}" if c.get("wobble") is not None else "–",
                    f"{c.get('agree', 1.0):.2f}",
                ])  # fmt: skip
                item.setData(0, Qt.ItemDataRole.UserRole, c["name"])
                item.setToolTip(0, c.get("text_check", c["text"]))
                tree.addTopLevelItem(item)
            self.tabs.setTabText(
                [k for k, _t, _x in TABS].index(key), f"{_title_of(key)} ({len(group)})"
            )
        self.use_btn.setEnabled(r["kept"] > 0)

    def _play(self, item: QTreeWidgetItem) -> None:
        name = item.data(0, Qt.ItemDataRole.UserRole)
        self.play(self.work / "wavs" / f"{name}.wav")

    def _play_selected(self) -> None:
        tree = self.trees[TABS[self.tabs.currentIndex()][0]]
        if tree.currentItem() is not None:
            self._play(tree.currentItem())

    def _accept(self) -> None:
        f5_data.save_selection(self.work, self.strictness_key(), self.clips, self.result_report)
        self.accept()


def _title_of(key: str) -> str:
    return next(title for k, title, _tip in TABS if k == key)
