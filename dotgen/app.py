"""QApplication bootstrap.

Besides building the application object this module owns the two things that
can only be done once per process: installing the crash hook, and asking about
a recovered session before the user has started work in this run.

Both are deliberately here rather than in :class:`~dotgen.ui.main_window.MainWindow`.
``sys.excepthook`` is process-global state, so a window constructor that
installed it would fight itself in a test suite that builds windows freely; and
the recovery prompt is a modal, which a constructor must never open.
"""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QMessageBox

from .core import session
from .ui.main_window import MainWindow
from .ui.theme import STYLESHEET

APP_NAME = "DOT-CODE-DATA-TOOL"

# Named so a test can find the button without matching on prose.
COPY_BUTTON_TEXT = "Copy details"


def build_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance()

    if app is None:
        app = QApplication(argv if argv is not None else sys.argv)

    app.setApplicationName(APP_NAME)
    app.setOrganizationName("dotgen")
    app.setStyleSheet(STYLESHEET)
    return app


def show_crash_dialog(log_path: str | None, details: str) -> None:
    """The dialog :func:`~dotgen.core.session.install_excepthook` calls.

    ``details`` is the report as text, handed over by the hook rather than read
    back from *log_path*, and it goes straight onto the ``Copy details``
    button.  That is what makes the button work in the case that matters most:
    a crash caused by a disk that is full or read-only, where the log write
    failed too and ``log_path`` is ``None``.  The user can still paste the
    traceback into a bug report.

    Every step is guarded.  This runs from an excepthook, which means the
    program is already failing and the GUI may be half torn down -- there may
    be no live :class:`QApplication` left at all.  A crash handler that raises
    replaces the report with its own, and the original is lost; so the worst
    outcome allowed here is that nothing is shown.
    """
    try:
        app = QApplication.instance()

        if app is None:
            return

        summary = details.strip().splitlines()[-1] if details.strip() else "Unknown error."

        box = QMessageBox()
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("DotGen has crashed")
        box.setText("DotGen hit an unexpected error and may no longer be in a usable state.")
        box.setInformativeText(
            f"{summary}\n\n"
            + (
                f"A crash log was written to:\n{log_path}"
                if log_path
                else "The crash log could not be written -- use Copy details instead."
            )
        )

        # setDetailedText, not a label: a traceback is arbitrarily long and a
        # QLabel would stretch the dialog off the screen rather than scroll.
        box.setDetailedText(details)

        copy_button = box.addButton(COPY_BUTTON_TEXT, QMessageBox.ActionRole)
        box.addButton(QMessageBox.Close)
        box.setDefaultButton(QMessageBox.Close)

        # The copy is wired to the button's own signal rather than tested for
        # after exec() returns.  QMessageBox dismisses itself on any button
        # including an ActionRole one, so both spellings behave the same for
        # the user, but this one does not depend on clickedButton() surviving
        # the teardown that is already under way.
        copy_button.clicked.connect(lambda: _copy_to_clipboard(details))

        box.exec()

    except Exception:  # noqa: BLE001 - a failing crash handler must stay silent
        pass


def _copy_to_clipboard(text: str) -> None:
    """Put *text* on the system clipboard, or do nothing at all."""
    try:
        clipboard = QApplication.clipboard()

        if clipboard is not None:
            clipboard.setText(text)
    except Exception:  # noqa: BLE001 - see show_crash_dialog
        pass


def main(argv: list[str] | None = None) -> int:
    app = build_app(argv)

    session.install_excepthook(on_crash=show_crash_dialog)

    window = MainWindow()
    window.show()

    # After show(), so the prompt has a parent window to sit over, and before
    # the user can type anything into the fresh state it would overwrite.
    window.offer_recovery()

    return app.exec()
