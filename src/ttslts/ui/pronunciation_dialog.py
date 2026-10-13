"""Editor for the pronunciation dictionary."""

from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from ttslts.tts.pronunciation import BUILTIN_HU, Entry

HELP = (
    "Write each word the way it sounds in the language of the text, using that language's "
    "spelling. Examples (Hungarian): Windows -> vindóz, Nvidia -> envídia, TTSLTS -> té té esz "
    "el té esz."
)


class PronunciationDialog(QDialog):
    def __init__(
        self,
        parent,
        entries: list[Entry],
        use_builtin: bool,
        listen: Callable[[str], None],
        word: str = "",
    ):
        super().__init__(parent)
        self.setWindowTitle("Pronunciations")
        self.resize(520, 420)
        self._listen = listen
        layout = QVBoxLayout(self)
        intro = QLabel(HELP)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Written as", "Say it as", "Match case"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setColumnWidth(0, 170)
        self.table.setColumnWidth(1, 200)
        self.table.setToolTip(
            "Every time the first column appears as a whole word in the text, the voice says the "
            "second column instead. With Match case on, only that exact capitalisation counts "
            "(IT, not it)."
        )
        layout.addWidget(self.table, 1)
        for e in entries:
            self._add_row(e.word, e.say, e.match_case)
        if word:
            self._add_row(word, "", False)
            self.table.setCurrentCell(self.table.rowCount() - 1, 1)
            self.table.editItem(self.table.item(self.table.rowCount() - 1, 1))

        row = QHBoxLayout()
        add = QPushButton("Add")
        add.setToolTip("Add an empty row")
        add.clicked.connect(lambda: self._add_row("", "", False, edit=True))
        remove = QPushButton("Remove")
        remove.setToolTip("Remove the selected row")
        remove.clicked.connect(self._remove_selected)
        listen_btn = QPushButton("Listen")
        listen_btn.setToolTip(
            "Speak the selected row's second column with the voice chosen on the Text to Speech "
            "tab, to check how it sounds"
        )
        listen_btn.clicked.connect(self._on_listen)
        for b in (add, remove, listen_btn):
            row.addWidget(b)
        row.addStretch(1)
        layout.addLayout(row)

        self.builtin_check = QCheckBox(
            "Use the built-in list for Hungarian speech (" + ", ".join(BUILTIN_HU) + ")"
        )
        self.builtin_check.setChecked(use_builtin)
        self.builtin_check.setToolTip(
            "Abbreviations that Hungarians say with English letter names (FBI = ef bí áj). Only "
            "the exact capital-letter spelling is replaced, and a row of your own for the same "
            "word wins over the built-in one."
        )
        layout.addWidget(self.builtin_check)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _add_row(self, word: str, say: str, match_case: bool, edit: bool = False) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QTableWidgetItem(word))
        self.table.setItem(r, 1, QTableWidgetItem(say))
        case = QTableWidgetItem()
        case.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        case.setCheckState(Qt.CheckState.Checked if match_case else Qt.CheckState.Unchecked)
        self.table.setItem(r, 2, case)
        if edit:
            self.table.setCurrentCell(r, 0)
            self.table.editItem(self.table.item(r, 0))

    def _remove_selected(self) -> None:
        for r in sorted({i.row() for i in self.table.selectedIndexes()}, reverse=True):
            self.table.removeRow(r)

    def _on_listen(self) -> None:
        r = self.table.currentRow()
        item = self.table.item(r, 1) if r >= 0 else None
        if item is not None and item.text().strip():
            self._listen(item.text().strip())

    def entries(self) -> list[Entry]:
        found = []
        for r in range(self.table.rowCount()):
            word, say, case = (self.table.item(r, c) for c in range(3))
            if word and say and word.text().strip() and say.text().strip():
                found.append(
                    Entry(
                        word.text().strip(),
                        say.text().strip(),
                        case.checkState() == Qt.CheckState.Checked,
                    )
                )
        return found

    def use_builtin(self) -> bool:
        return self.builtin_check.isChecked()
