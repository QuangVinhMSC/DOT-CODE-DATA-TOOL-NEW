"""Plan 10.4: the autosave timer, the recovery prompt and the crash dialog.

Everything here runs against a throwaway ``DOTGEN_HOME``.  That override is not
a convenience -- without it the suite would autosave into the developer's real
profile directory and greet them with somebody else's recovery prompt on the
next launch -- so it is applied by an autouse fixture rather than left to each
test to remember.

The second autouse fixture makes every modal unreachable.  A test that opens a
real ``QMessageBox`` does not fail, it hangs, and it hangs on a machine nobody
is watching; so the dialogs are replaced by default and each test that wants
one re-patches it to answer the way that test needs.  The third saves
``sys.excepthook``, which :func:`~dotgen.core.session.install_excepthook`
replaces process-wide with no way to uninstall.
"""

from __future__ import annotations

import os
import sys

import pytest
from PySide6.QtWidgets import QApplication, QMessageBox

from dotgen import app as app_module
from dotgen.core import session
from dotgen.core.state import AppState
from dotgen.ui.main_window import MainWindow


# ======================================================================
# fixtures
# ======================================================================


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    """Point the session module at a directory this test owns."""
    monkeypatch.setenv(session.HOME_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def no_modals(monkeypatch):
    """Make any unexpected modal a failure instead of a hang."""

    def forbidden(*args, **kwargs):
        raise AssertionError("a modal dialog was opened by a test")

    for name in ("question", "warning", "critical", "information", "about", "exec"):
        monkeypatch.setattr(QMessageBox, name, forbidden)


@pytest.fixture(autouse=True)
def keep_excepthook():
    """Restore ``sys.excepthook``; nothing in the module can uninstall it."""
    original = sys.excepthook

    yield

    sys.excepthook = original


@pytest.fixture
def worked_state(state):
    """A state with enough in it to count as work worth autosaving."""
    state.add_line()
    state.add_char(0, "1")

    return state


def answer(monkeypatch, button):
    """Make the next ``QMessageBox.question`` return *button*."""
    asked: list[str] = []

    def fake_question(parent, title, text, *args, **kwargs):
        asked.append(text)
        return button

    monkeypatch.setattr(QMessageBox, "question", fake_question)

    return asked


def allow(monkeypatch, name):
    """Let one static QMessageBox call through, recording its message."""
    seen: list[str] = []

    monkeypatch.setattr(
        QMessageBox, name, lambda parent, title, text, *a, **k: seen.append(text)
    )

    return seen


# ======================================================================
# autosave
# ======================================================================


def test_the_window_starts_a_timer_at_the_module_interval(qtbot, state):
    w = MainWindow(state)
    qtbot.addWidget(w)

    assert w.autosave_timer.isActive()
    assert w.autosave_timer.interval() == session.AUTOSAVE_SECONDS * 1000


def test_firing_the_timer_writes_the_session_file(qtbot, worked_state):
    w = MainWindow(worked_state)
    qtbot.addWidget(w)

    assert not os.path.exists(session.session_path())

    # Emitted rather than called: this asserts the slot is connected, which is
    # the half of the wiring a direct call would skip.
    w.autosave_timer.timeout.emit()

    assert os.path.isfile(session.session_path())
    assert session.has_session() is True


def test_a_failed_autosave_is_reported_in_the_status_bar_only(qtbot, worked_state, monkeypatch):
    """A read-only disk must not produce a dialog once a minute."""
    monkeypatch.setattr(session, "save_session", lambda state: None)

    w = MainWindow(worked_state)
    qtbot.addWidget(w)

    w.autosave()  # no_modals turns any dialog into an AssertionError

    assert "Autosave failed" in w.status_label.text()


def test_an_untouched_window_does_not_autosave_over_a_recoverable_file(qtbot, state, monkeypatch):
    """The empty-state skip: a blank save would destroy the thing worth having."""
    calls: list[AppState] = []
    monkeypatch.setattr(session, "save_session", lambda s: calls.append(s))

    w = MainWindow(state)
    qtbot.addWidget(w)

    assert w.has_recoverable_work() is False

    w.autosave()

    assert calls == []
    assert not os.path.exists(session.session_path())

    # ...and the moment there is anything to lose, it saves again.
    state.add_line()
    state.add_char(0, "1")

    assert w.has_recoverable_work() is True

    w.autosave()

    assert calls == [state]


@pytest.mark.parametrize(
    "fill",
    [
        lambda s: s.add_line(),
        lambda s: s.set_param("dist.h", "mean", 12.0) or s.set_param("dist.v", "mean", 16.0),
    ],
)
def test_any_kind_of_work_counts_as_recoverable(qtbot, state, fill):
    w = MainWindow(state)
    qtbot.addWidget(w)

    fill(state)

    assert w.has_recoverable_work() is True


def test_closing_the_window_saves_and_still_closes(qtbot, worked_state):
    w = MainWindow(worked_state)
    qtbot.addWidget(w)

    assert w.close() is True
    assert os.path.isfile(session.session_path())
    assert w.autosave_timer.isActive() is False


# ======================================================================
# recovery
# ======================================================================


def write_session(state) -> None:
    """Put a genuine autosave on disk, the way the timer would."""
    assert session.save_session(state) is not None
    assert session.has_session() is True


def test_offer_recovery_does_nothing_without_a_session(qtbot, state):
    w = MainWindow(state)
    qtbot.addWidget(w)

    # no_modals would raise if this reached QMessageBox.question.
    assert w.offer_recovery() is False


def test_accepting_the_offer_restores_the_saved_work(qtbot, worked_state, monkeypatch):
    write_session(worked_state)

    fresh = AppState()
    w = MainWindow(fresh)
    qtbot.addWidget(w)

    asked = answer(monkeypatch, QMessageBox.Yes)

    assert w.offer_recovery() is True
    assert asked and "autosaved session" in asked[0]
    assert len(fresh.lines) == 1
    assert fresh.lines[0].chars[0].char == "1"

    # The file has done its job; it must not be offered again next start.
    assert session.has_session() is False


def test_the_prompt_says_how_old_the_session_is(qtbot, worked_state, monkeypatch):
    write_session(worked_state)

    monkeypatch.setattr(session, "session_age", lambda: 4 * 60.0)

    w = MainWindow(AppState())
    qtbot.addWidget(w)

    asked = answer(monkeypatch, QMessageBox.No)
    w.offer_recovery()

    assert "4 minutes ago" in asked[0]


def test_declining_the_offer_deletes_the_file(qtbot, worked_state, monkeypatch):
    write_session(worked_state)

    loaded: list[int] = []
    monkeypatch.setattr(session, "load_session", lambda s: loaded.append(1))

    fresh = AppState()
    w = MainWindow(fresh)
    qtbot.addWidget(w)

    answer(monkeypatch, QMessageBox.No)

    assert w.offer_recovery() is False
    assert loaded == []
    assert session.has_session() is False
    assert fresh.lines == []


def test_an_unreadable_session_is_reported_and_discarded(qtbot, worked_state, monkeypatch):
    """has_session passed, load_session still refused: say so and clear it."""
    write_session(worked_state)

    monkeypatch.setattr(session, "load_session", lambda s: False)

    w = MainWindow(AppState())
    qtbot.addWidget(w)

    answer(monkeypatch, QMessageBox.Yes)
    warned = allow(monkeypatch, "warning")

    assert w.offer_recovery() is False
    assert warned and "could not be read" in warned[0]
    assert session.has_session() is False


# ======================================================================
# the crash dialog
# ======================================================================

DETAILS = "DotGen crash report\ntime      : now\n\nValueError: exploded on purpose\n"


def click_copy(monkeypatch):
    """Run the crash dialog without showing it, pressing ``Copy details``."""
    boxes: list[QMessageBox] = []

    def fake_exec(self):
        boxes.append(self)

        for button in self.buttons():
            if button.text().replace("&", "") == app_module.COPY_BUTTON_TEXT:
                button.click()

        return QMessageBox.Close

    monkeypatch.setattr(QMessageBox, "exec", fake_exec)

    return boxes


def test_the_crash_dialog_puts_the_details_on_the_clipboard(app, monkeypatch):
    boxes = click_copy(monkeypatch)

    QApplication.clipboard().setText("")
    app_module.show_crash_dialog(r"C:\logs\crash-1.log", DETAILS)

    assert len(boxes) == 1

    box = boxes[0]
    assert "crash-1.log" in box.informativeText()
    assert "exploded on purpose" in box.informativeText()

    # The whole report is in the collapsible detail area, not in a label that
    # would stretch the dialog past the edge of the screen.
    assert box.detailedText() == DETAILS
    assert DETAILS not in box.text()

    assert QApplication.clipboard().text() == DETAILS


def test_the_crash_dialog_still_copies_when_the_log_could_not_be_written(app, monkeypatch):
    """log_path is None exactly when the details matter most."""
    boxes = click_copy(monkeypatch)

    QApplication.clipboard().setText("")
    app_module.show_crash_dialog(None, DETAILS)

    assert "could not be written" in boxes[0].informativeText()
    assert QApplication.clipboard().text() == DETAILS


def test_the_crash_dialog_is_silent_without_a_live_application(monkeypatch):
    """From an excepthook the GUI may already be gone; that is not an error."""
    monkeypatch.setattr(QApplication, "instance", staticmethod(lambda: None))

    # no_modals is still armed, so constructing any dialog here would raise.
    app_module.show_crash_dialog("/tmp/crash.log", DETAILS)


def test_the_crash_dialog_never_raises_out_of_the_excepthook(app, monkeypatch):
    """A handler that raises replaces the report it was meant to deliver."""

    def exploding_exec(self):
        raise RuntimeError("the GUI is tearing down")

    monkeypatch.setattr(QMessageBox, "exec", exploding_exec)

    app_module.show_crash_dialog(None, DETAILS)


def test_the_excepthook_is_installed_with_the_dialog_attached(app, monkeypatch):
    """``main`` wires the hook and the recovery prompt without running exec."""
    original = sys.excepthook
    windows: list[MainWindow] = []

    monkeypatch.setattr(MainWindow, "show", lambda self: windows.append(self))
    monkeypatch.setattr(QApplication, "exec", lambda self: 0)

    offered: list[int] = []
    monkeypatch.setattr(MainWindow, "offer_recovery", lambda self: offered.append(1))

    try:
        assert app_module.main([]) == 0
        assert sys.excepthook is not original
        assert offered == [1]
    finally:
        for window in windows:
            window.autosave_timer.stop()
            window.deleteLater()

        sys.excepthook = original

    assert sys.excepthook is original
