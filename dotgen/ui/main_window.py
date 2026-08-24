"""MainWindow -- the seven tabs, the status bar and the progressive gating.

A tab that is not ready is disabled and the reason is shown in the status bar
when the user hovers it, rather than letting them walk into a half-defined job.

The window is also where :mod:`~dotgen.core.session` is given a clock and a
voice.  The module underneath is deliberately Qt-free: it knows how to write an
autosave atomically and how to format a crash, but nothing about timers or
dialogs.  This class supplies the 60-second :class:`QTimer`, the save on a clean
exit, and the one prompt on the next start that offers the recovered work back.

Two rules govern that wiring.  *An autosave is silent.*  It runs once a minute
whether or not the user is looking, so a failure gets a line in the status bar
and nothing more -- a modal every minute on a read-only network drive would
train the user to dismiss dialogs without reading them, which is the one habit
this program cannot afford.  *Recovery happens before any work exists.*
:func:`~dotgen.core.session.load_session` overwrites ``AppState`` field by
field, so it is only safe on a window that has not been touched yet; hence
:meth:`offer_recovery` is a separate method called from
:func:`dotgen.app.main` after ``show()``, and never from ``__init__`` -- a
constructor that opens a modal cannot be built in a test.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QGuiApplication, QKeySequence, QCloseEvent
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QTabWidget,
)

from ..core import session
from ..core.io_config import CONFIG_FILTER, load_config, save_config
from ..core.state import AppState
from .tabs.tab1_sample import Tab1Sample
from .tabs.tab2_matrix import Tab2Matrix
from .tabs.tab3_summary import Tab3Summary
from .tabs.tab4_job import Tab4Job
from .tabs.tab5_defect import Tab5Defect
from .tabs.tab6_class import Tab6Class
from .tabs.tab7_export import Tab7Export

# The size the seven tabs were laid out for, before the screen gets a say.
WANTED_SIZE = (1600, 1000)

TAB_TITLES = [
    "1 - Sample collection",
    "2 - Number matrix",
    "3 - Summary",
    "4 - Create job",
    "5 - Defect generation",
    "6 - Class definition",
    "7 - Save job and export",
]


class MainWindow(QMainWindow):
    def __init__(self, state: AppState | None = None, parent=None) -> None:
        super().__init__(parent)
        self.state = state or AppState()

        self.setWindowTitle("DOT-CODE-DATA-TOOL")
        self.resize(self._preferred_size())

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)

        self.tab1 = Tab1Sample(self.state)
        self.tab2 = Tab2Matrix(self.state)
        self.tab3 = Tab3Summary(self.state)
        self.tab4 = Tab4Job(self.state)
        self.tab5 = Tab5Defect(self.state)
        self.tab6 = Tab6Class(self.state)
        self.tab7 = Tab7Export(self.state)

        ordered = (
            self.tab1,
            self.tab2,
            self.tab3,
            self.tab4,
            self.tab5,
            self.tab6,
            self.tab7,
        )

        for title, widget in zip(TAB_TITLES, ordered):
            self.tabs.addTab(widget, title)

        for tab in ordered:
            tab.statusMessage.connect(self.show_message)

        self.state.statusMessage.connect(self.show_message)

        self.status_label = QLabel("")
        self.gate_label = QLabel("")
        self.gate_label.setObjectName("hint")
        self.statusBar().addWidget(self.status_label, 1)
        self.statusBar().addPermanentWidget(self.gate_label)

        self._build_menu()

        for signal in (
            self.state.samplesChanged,
            self.state.paramsChanged,
            self.state.charFormatsChanged,
            self.state.backgroundsChanged,
            self.state.linesChanged,
            self.state.lineDefectsChanged,
            self.state.classesChanged,
            self.state.jobsChanged,
        ):
            signal.connect(lambda *_: self._refresh_gating())

        self.tabs.currentChanged.connect(lambda _i: self._refresh_gating())
        self._refresh_gating()

        # The timer is the only thing started here; the recovery prompt that
        # goes with it is left for main() to call, so constructing a window
        # stays free of dialogs.
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(session.AUTOSAVE_SECONDS * 1000)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()

    @staticmethod
    def _preferred_size() -> QSize:
        """:data:`WANTED_SIZE`, or as much of it as the screen actually has.

        The tabs are wide -- Tab 1 carries three columns and Tab 3 puts the
        parameter bars beside two preview frames.  Asking for 1600x1000 on a
        1536x864 laptop puts the rightmost column off the edge of the display
        with no horizontal scrollbar to reach it, so the bars the user is
        looking for are not merely cramped, they are invisible.
        """
        screen = QGuiApplication.primaryScreen()

        if screen is None:
            return QSize(*WANTED_SIZE)

        available = screen.availableGeometry()

        return QSize(
            min(WANTED_SIZE[0], available.width()),
            min(WANTED_SIZE[1], available.height()),
        )

    # ==================================================================
    # crash survival
    # ==================================================================

    def has_recoverable_work(self) -> bool:
        """Whether the state holds anything an autosave would be worth keeping.

        The check is deliberately generous: any sample image, dot, character
        format, background, line or saved job counts, and so does a measured
        distance unit, because each of those took mouse work to produce.  What
        it excludes is the state a window has one second after it opens -- and
        that exclusion is the point.  Writing a blank session once a minute
        would quietly overwrite the recoverable file left by the run that
        crashed, so the empty case must not reach the disk at all.
        """
        return bool(
            self.state.sample_images
            or self.state.dot_samples
            or self.state.char_formats
            or self.state.backgrounds
            or self.state.lines
            or self.state.jobs
            or self.state.has_distance_units()
        )

    def autosave(self) -> None:
        """Write the session, reporting a failure to the status bar only.

        :func:`~dotgen.core.session.save_session` swallows everything and
        returns ``None``, so there is no exception to handle here -- only a
        result to mention quietly.  A dialog would be wrong: this fires on a
        timer the user did not ask for, and a disk that refuses one write will
        refuse the next sixty as well.
        """
        if not self.has_recoverable_work():
            return

        if session.save_session(self.state) is None:
            self.show_message("Autosave failed -- see the log directory for the reason.")

    def closeEvent(self, event: QCloseEvent) -> None:
        """Autosave on the way out, then close normally.

        A clean exit is exactly when the last minute of work is most likely to
        be unsaved, and the recovery prompt on the next start is harmless if it
        turns out the user did not want it.
        """
        self.autosave_timer.stop()
        self.autosave()

        super().closeEvent(event)

    def offer_recovery(self) -> bool:
        """Offer the autosaved session back, and return whether it was restored.

        Call this once, on a fresh window, before the user has touched
        anything: :func:`~dotgen.core.session.load_session` replaces the state
        field by field and a damaged file leaves it half-replaced, so there
        must be nothing there worth half-replacing.

        Declining deletes the file.  That looks harsh, but the alternative is
        the same stale prompt on every start until the user gives in, and a
        session they have already refused once is not work they are protecting.
        """
        if not session.has_session():
            return False

        answer = QMessageBox.question(
            self,
            "Recover session",
            f"DotGen has an autosaved session from {self._describe_age()}.\n\n"
            "Restore it? Choosing No deletes it and starts empty.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes,
        )

        if answer != QMessageBox.Yes:
            session.clear_session()
            self.show_message("Autosaved session discarded.")
            return False

        if not session.load_session(self.state):
            # has_session() screened the obvious rubbish, so getting here means
            # the archive parsed as a zip and still would not load.  Say so
            # plainly and remove it rather than offering it again tomorrow.
            QMessageBox.warning(
                self,
                "Recovery failed",
                "The autosaved session could not be read and has been discarded.\n"
                "Details are in the log directory.",
            )
            session.clear_session()
            self.show_message("Autosaved session could not be restored.")
            return False

        # The recovery has served its purpose; keeping the file would offer the
        # same work again next time on top of the work just restored.
        session.clear_session()
        self.show_message("Autosaved session restored.")
        self._refresh_gating()

        return True

    @staticmethod
    def _describe_age() -> str:
        """The session's age in the units a person would use to say it."""
        seconds = session.session_age()

        if seconds is None:
            return "an earlier run"

        if seconds < 90:
            return "less than a minute ago" if seconds < 60 else "about a minute ago"

        minutes = int(seconds // 60)

        if minutes < 60:
            return f"{minutes} minutes ago"

        hours = int(seconds // 3600)

        if hours < 24:
            return f"{hours} hour{'s' if hours != 1 else ''} ago"

        days = int(seconds // 86400)

        return f"{days} day{'s' if days != 1 else ''} ago"

    # ==================================================================

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")

        save = QAction("&Save configuration...", self)
        save.setShortcut(QKeySequence.Save)
        save.triggered.connect(self._save_config)
        file_menu.addAction(save)

        load = QAction("&Load configuration...", self)
        load.setShortcut(QKeySequence.Open)
        load.triggered.connect(self._load_config)
        file_menu.addAction(load)

        file_menu.addSeparator()

        quit_action = QAction("&Quit", self)
        quit_action.setShortcut(QKeySequence.Quit)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        help_menu = self.menuBar().addMenu("&Help")

        about = QAction("&About", self)
        about.triggered.connect(self._about)
        help_menu.addAction(about)

    # ==================================================================

    def show_message(self, text: str) -> None:
        self.status_label.setText(text)

    def gating_reasons(self) -> dict[int, str]:
        """index -> blocking reason ('' when the tab is open)."""
        reasons = {i: "" for i in range(7)}

        if not self.state.has_distance_units():
            reasons[1] = (
                "Tab 2 needs the horizontal and vertical distance units: "
                "measure dot pairs with the distance tool in Tab 1."
            )

        if not self.state.char_formats:
            reasons[2] = "Tab 3 needs at least one saved character format from Tab 2."
        else:
            # A format saved before a rule tightened, or edited through a loaded
            # config, must not reach the renderer: Phase 5 solves its metrics
            # from the links, and an invalid link set has no pitch.
            broken = [c for c, f in sorted(self.state.char_formats.items()) if f.validate()]

            if broken:
                reasons[2] = (
                    "Tab 3 needs every saved format to be valid; fix "
                    + ", ".join(f"'{c}'" for c in broken)
                    + " in Tab 2."
                )

        # Tabs 5 and 6 stand or fall together: the defect tab composes the job
        # to preview it and the class tab reads its classes off the same lines
        # and backgrounds, so a state that blocks one blocks the other.  The
        # reason is written once into a local and assigned to both, because two
        # copies of it would drift apart at the first edit.
        content: str = ""

        if not self.state.backgrounds:
            content = "Tabs 5 and 6 need at least one background in Tab 4."
        elif self.state.backgrounds_missing_quad():
            missing = ", ".join(f"#{i + 1}" for i in self.state.backgrounds_missing_quad())
            content = f"Tabs 5 and 6 are locked: backgrounds {missing} have no base quadrilateral."
        elif not self.state.has_content():
            content = "Tabs 5 and 6 need at least one line with one character in Tab 4."

        reasons[4] = content
        reasons[5] = content

        # ... or a saved job.  "Save job" lives *in* Tab 7 and clears Tabs 1-6
        # behind it, so gating on the class list alone locks the tab the moment
        # it is used -- and the Export button, which is a child of the tab and
        # therefore disabled with it, becomes unreachable for the jobs already
        # saved.  Classes get you in the first time; a saved job keeps you in.
        if not self.state.classes_ready() and not self.state.jobs:
            reasons[6] = "Tab 7 needs a valid class list in Tab 6, or a saved job."

        return reasons

    def _refresh_gating(self) -> None:
        reasons = self.gating_reasons()

        for i, reason in reasons.items():
            self.tabs.setTabEnabled(i, not reason)
            self.tabs.setTabToolTip(i, reason)

        current = self.tabs.currentIndex()

        if reasons.get(current):
            self.gate_label.setText(reasons[current])
        else:
            blocked = [f"Tab {i + 1}" for i, r in reasons.items() if r]
            self.gate_label.setText(
                "locked: " + ", ".join(blocked) if blocked else "all tabs unlocked"
            )

    # ==================================================================

    def _save_config(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save configuration", "config.dotcfg", CONFIG_FILTER
        )

        if not path:
            return

        try:
            save_config(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Save failed", str(exc))
            return

        self.show_message(f"Configuration saved to {path}")

    def _load_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load configuration", "", CONFIG_FILTER)

        if not path:
            return

        try:
            load_config(self.state, path)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            QMessageBox.critical(self, "Load failed", str(exc))
            return

        self.show_message(f"Configuration loaded from {path}")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "DOT-CODE-DATA-TOOL",
            "Synthetic dot-matrix character dataset generator.\n\n"
            "Seven tabs: sample the dots, draw the characters, check the\n"
            "ranges, build the job, generate the defects, define the classes,\n"
            "export the dataset.\n\n"
            "PLAN.md phases 1-10 built the program.  plan2.md (phase 11) added\n"
            "Tab 5's line-level defect generation and the classes that go with\n"
            "it, and renumbered the two tabs after it.",
        )
