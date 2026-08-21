"""Tab 6 -- Save job and export data.

"Save job" snapshots Tabs 1-5 and clears them for the next job; "Export data"
writes every saved job into one dataset whose class list is the union over all
jobs.
"""

from __future__ import annotations

import os

from PySide6.QtCore import QEventLoop, QObject, QThread, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core.classes import dataset_classes, validate_classes
from ...core.exporter import FORMATS, ExportError, preflight, report_text, run_export
from ...core.state import AppState
from .. import theme


class _ExportWorker(QObject):
    """Runs :func:`run_export` off the GUI thread.

    Composition is seconds to minutes of numpy; on the GUI thread the window
    would stop repainting and Windows would grey it out as "not responding"
    halfway through a legitimate export.

    ``cancel`` is set from the GUI thread and read from the worker thread.  A
    lone bool needs no lock: the worker only ever reads it, and a cancel that
    lands one image late costs one image.
    """

    progressed = Signal(int, int)
    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, jobs, spec) -> None:
        super().__init__()
        self._jobs = jobs
        self._spec = spec
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        def progress(done: int, total: int) -> bool:
            self.progressed.emit(done, total)
            return not self._cancelled

        try:
            self.finished.emit(run_export(self._jobs, self._spec, progress))
        except ExportError as exc:
            self.failed.emit("\n".join("  - " + e for e in exc.errors))
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.failed.emit(str(exc))


class Tab6Export(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state

        # Live only for the duration of one export; the worker's slots read
        # them and do nothing once they are cleared.
        self._dialog: QProgressDialog | None = None
        self._loop: QEventLoop | None = None
        self._result: dict | None = None

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_right())
        splitter.setSizes([760, 700])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        state.jobsChanged.connect(self.refresh)
        state.classesChanged.connect(self.refresh)

        self.refresh()

    # ==================================================================

    def _build_left(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        fmt_box = QGroupBox("Data format")
        fl = QVBoxLayout(fmt_box)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        rl.addWidget(QLabel("Format"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(list(FORMATS))
        self.format_combo.setToolTip(
            "yolo      -- axis-aligned:  <class> cx cy w h\n"
            "yolo-obb  -- oriented:      <class> x1 y1 x2 y2 x3 y3 x4 y4\n\n"
            "Both write the same images, folders and data.yaml; only the shape "
            "of a label line differs. Characters are upright either way, so the "
            "oriented format is what tilts a line's box with the surface it "
            "sits on."
        )
        self.format_combo.setCurrentText(self.state.export.fmt)
        self.format_combo.currentTextChanged.connect(self._set_format)
        rl.addWidget(self.format_combo)

        rl.addWidget(QLabel("Images per job"))
        self.count_spin = QSpinBox()
        self.count_spin.setRange(1, 100000)
        self.count_spin.setValue(self.state.export.images_per_job)
        self.count_spin.valueChanged.connect(
            lambda v: self.state.set_export(images_per_job=v)
        )
        rl.addWidget(self.count_spin)

        rl.addWidget(QLabel("Seed"))
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 2**31 - 1)
        self.seed_spin.setValue(self.state.export.seed)
        self.seed_spin.valueChanged.connect(lambda v: self.state.set_export(seed=v))
        rl.addWidget(self.seed_spin)

        rl.addStretch(1)
        fl.addWidget(row)

        split_row = QWidget()
        sl = QHBoxLayout(split_row)
        sl.setContentsMargins(0, 0, 0, 0)
        sl.addWidget(QLabel("Split  train / val / test"))

        self.split_spins = []

        for value in self.state.export.split:
            s = QDoubleSpinBox()
            s.setRange(0.0, 1.0)
            s.setSingleStep(0.05)
            s.setDecimals(2)
            s.setValue(value)
            s.valueChanged.connect(self._push_split)
            self.split_spins.append(s)
            sl.addWidget(s)

        sl.addStretch(1)
        fl.addWidget(split_row)

        dir_row = QWidget()
        dl = QHBoxLayout(dir_row)
        dl.setContentsMargins(0, 0, 0, 0)

        self.dir_edit = QLineEdit(self.state.export.out_dir)
        self.dir_edit.setPlaceholderText("Output directory...")
        self.dir_edit.textChanged.connect(lambda v: self.state.set_export(out_dir=v))
        dl.addWidget(self.dir_edit, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        dl.addWidget(browse)

        fl.addWidget(dir_row)
        lay.addWidget(fmt_box)

        job_box = QGroupBox("Jobs")
        jl = QVBoxLayout(job_box)

        self.job_list = QListWidget()
        jl.addWidget(self.job_list, 1)

        btns = QWidget()
        bl = QHBoxLayout(btns)
        bl.setContentsMargins(0, 0, 0, 0)

        self.save_job_button = QPushButton("Save job")
        self.save_job_button.setToolTip("Capture Tabs 1-5 as a job and start a new one")
        self.save_job_button.clicked.connect(self._save_job)
        bl.addWidget(self.save_job_button)

        load = QPushButton("Load")
        load.clicked.connect(self._load_job)
        bl.addWidget(load)

        dup = QPushButton("Duplicate")
        dup.clicked.connect(self._duplicate_job)
        bl.addWidget(dup)

        delete = QPushButton("Delete")
        delete.clicked.connect(self._delete_job)
        bl.addWidget(delete)

        bl.addStretch(1)
        jl.addWidget(btns)
        lay.addWidget(job_box, 1)

        self.export_button = QPushButton("Export data")
        self.export_button.setMinimumHeight(34)
        self.export_button.clicked.connect(self._export)
        lay.addWidget(self.export_button)

        self.export_hint = QLabel("")
        self.export_hint.setObjectName("hint")
        self.export_hint.setWordWrap(True)
        lay.addWidget(self.export_hint)
        return w

    def _build_right(self) -> QWidget:
        box = QGroupBox("Dataset classes (union over all jobs)")
        lay = QVBoxLayout(box)

        self.class_list = QListWidget()
        lay.addWidget(self.class_list, 1)

        note = QLabel(
            "The index shown here is the YOLO class id written to data.yaml. "
            "The order is fixed by (kind, name) so adding a job does not renumber "
            "the existing classes."
        )
        note.setObjectName("hint")
        note.setWordWrap(True)
        lay.addWidget(note)
        return box

    # ==================================================================

    def _set_format(self, value: str) -> None:
        self.state.set_export(fmt=value)
        self.refresh()  # the hint below the button names the format

    def _push_split(self) -> None:
        self.state.set_export(split=tuple(s.value() for s in self.split_spins))

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Output directory", self.dir_edit.text())

        if path:
            self.dir_edit.setText(path)

    # ==================================================================
    # jobs
    # ==================================================================

    def _save_job(self) -> None:
        errors = validate_classes(self.state.classes) if self.state.classes else ["no classes defined"]

        text = (
            "Save the current definition as a job?\n\n"
            f"characters: {', '.join(self.state.job_characters()) or '-'}\n"
            f"lines: {len(self.state.lines)}\n"
            f"backgrounds: {len(self.state.backgrounds)}\n"
            f"character formats: {len(self.state.char_formats)}\n"
            f"classes: {len(self.state.classes)}\n"
            f"dot samples: {len(self.state.dot_samples)}\n\n"
            "Tabs 1-5 are then cleared so you can define the next job.\n"
            "(Loaded sample images stay.)"
        )

        if errors:
            text += "\n\nWarning - the class list is not valid yet:\n" + "\n".join(
                "  - " + e for e in errors
            )

        if QMessageBox.question(self, "Save job", text) != QMessageBox.Yes:
            return

        job = self.state.save_job()
        self.state.reset_job_definition()
        self.statusMessage.emit(f"Saved {job.name}. Tabs 1-5 are ready for the next job.")

    def _selected_job(self) -> int:
        return self.job_list.currentRow()

    def _load_job(self) -> None:
        index = self._selected_job()

        if index < 0:
            return

        if QMessageBox.question(
            self,
            "Load job",
            "Loading replaces the current contents of Tabs 1-5. Continue?",
        ) != QMessageBox.Yes:
            return

        self.state.load_job(index)
        self.statusMessage.emit(f"Loaded {self.state.jobs[index].name} into Tabs 1-5.")

    def _duplicate_job(self) -> None:
        index = self._selected_job()

        if index >= 0:
            self.state.duplicate_job(index)

    def _delete_job(self) -> None:
        index = self._selected_job()

        if index >= 0:
            self.state.delete_job(index)

    # ==================================================================
    # export
    # ==================================================================

    def _export(self) -> None:
        """Run the export on a worker thread behind a modal progress dialog.

        A local :class:`QEventLoop` -- not ``dialog.exec()`` -- is what keeps
        this method synchronous for its caller while the GUI keeps repainting.
        Two reasons, in order of severity: closing a ``QProgressDialog`` from
        inside a queued slot while its own ``exec`` is running crashes PySide6
        6.9.2, and a plain loop makes the finishing order explicit (quit the
        loop, then close the dialog, then read the result).

        The worker's signals are received by *methods of this widget*, never by
        local closures.  A closure is not a ``QObject``, so Qt has no receiver
        thread to queue the call to and invokes it directly in the thread that
        emitted -- which put every ``dialog.setValue`` on the worker thread and
        froze the window.  A bound method of a widget living on the GUI thread
        is what makes the connection queued, which is the whole point of moving
        the work off this thread in the first place.
        """
        jobs = list(self.state.jobs)

        if not jobs or not self.state.export.out_dir:
            return

        total = max(len(jobs) * self.state.export.images_per_job, 1)

        dialog = QProgressDialog("Exporting dataset...", "Cancel", 0, total, self)
        dialog.setWindowTitle("Export")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setMinimumDuration(0)
        dialog.setAutoClose(False)  # closing is ours to do, once the worker is done
        dialog.setAutoReset(False)

        thread = QThread(self)
        worker = _ExportWorker(jobs, self.state.export)
        worker.moveToThread(thread)

        loop = QEventLoop()
        result: dict = {}

        self._dialog = dialog
        self._loop = loop
        self._result = result

        worker.progressed.connect(self._on_export_progress)
        worker.finished.connect(self._on_export_finished)
        worker.failed.connect(self._on_export_failed)
        thread.started.connect(worker.run)

        # Direct, deliberately.  The worker sits inside ``run`` for the whole
        # export and never returns to its own event loop, so a queued call would
        # be delivered after the export it was meant to stop.  Setting the flag
        # from the GUI thread is what makes Cancel arrive.
        dialog.canceled.connect(worker.cancel, Qt.DirectConnection)

        self.export_button.setEnabled(False)
        thread.start()
        dialog.show()

        # A short export can finish before the loop is entered.  ``quit`` on a
        # loop that is not running is forgotten, so entering it then would wait
        # for a signal that has already been delivered -- and hang.
        if not result:
            loop.exec()

        dialog.close()

        thread.quit()
        thread.wait()
        worker.deleteLater()

        self._dialog = None
        self._loop = None
        self._result = None

        self.refresh()

        if "error" in result:
            QMessageBox.critical(self, "Export failed", result["error"])
            self.statusMessage.emit("Export failed.")
            return

        report = result.get("report")

        if report is None:  # the dialog was closed some other way
            self.statusMessage.emit("Export stopped.")
            return

        self._show_report(report)

    # -- worker signals, received on the GUI thread ---------------------
    #
    # Guarded because a queued signal can still be in flight after the loop has
    # quit; by then the dialog is closed and there is nothing left to update.

    def _on_export_progress(self, done: int, total: int) -> None:
        if self._dialog is None:
            return

        self._dialog.setMaximum(max(total, 1))
        self._dialog.setValue(done)
        self._dialog.setLabelText(f"Exporting dataset...  {done} / {total}")

    def _on_export_finished(self, report) -> None:
        if self._result is None:
            return

        self._result["report"] = report
        self._loop.quit()

    def _on_export_failed(self, message: str) -> None:
        if self._result is None:
            return

        self._result["error"] = message
        self._loop.quit()

    def _show_report(self, report) -> None:
        box = QMessageBox(self)
        box.setWindowTitle("Export cancelled" if report.cancelled else "Export finished")
        box.setIcon(QMessageBox.Warning if report.empty_classes else QMessageBox.Information)
        box.setText(report_text(report))
        box.addButton(QMessageBox.Ok)
        open_button = box.addButton("Open folder", QMessageBox.ActionRole)
        box.exec()

        if box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(report.out_dir)))

        self.statusMessage.emit(
            f"Exported {report.images} image(s) and {report.boxes} box(es) to {report.out_dir}"
        )

    # ==================================================================

    def refresh(self) -> None:
        self.job_list.clear()

        for job in self.state.jobs:
            self.job_list.addItem(
                f"{job.name}   {len(job.lines)} line(s), {len(job.backgrounds)} background(s), "
                f"{len(job.classes)} class(es)"
            )

        names = dataset_classes(self.state.jobs)
        self.class_list.clear()

        for i, name in enumerate(names):
            self.class_list.addItem(f"{i:>3}  {name}")

        ready = bool(self.state.jobs) and bool(self.state.export.out_dir)
        self.export_button.setEnabled(ready)

        if not self.state.jobs:
            self.export_hint.setText("Save at least one job before exporting.")
        elif not self.state.export.out_dir:
            self.export_hint.setText("Choose an output directory.")
        else:
            fmt = self.state.export.fmt
            shape = (
                "<class> x1 y1 x2 y2 x3 y3 x4 y4"
                if fmt == "yolo-obb"
                else "<class> cx cy w h"
            )
            text = (
                f"{len(self.state.jobs)} job(s) x {self.state.export.images_per_job} images "
                f"= {len(self.state.jobs) * self.state.export.images_per_job} images, "
                f"{len(names)} classes.\n"
                f"Label lines: {shape}"
            )

            # The pre-flight runs here as well as in the exporter so the problem
            # is visible before a long export is started, not after it refuses.
            problems = preflight(self.state.jobs, self.state.export)

            if problems:
                text += "\n\nExport will refuse:\n" + "\n".join("  - " + p for p in problems)

            self.export_hint.setText(text)


