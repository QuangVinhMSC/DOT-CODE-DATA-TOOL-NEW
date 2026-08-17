"""Small Qt helpers shared by the widgets.

The one thing here exists because getting it wrong is visible: a widget removed
from a layout with ``setParent(None)`` is not destroyed, it becomes a *top-level
window*.  Anything still holding a reference to it -- a lambda that captured
``self``, a pending signal -- keeps it alive, and it can surface as a stray
floating panel over the canvas.  Rebuilding a list is the common case, so the
teardown gets one correct implementation instead of five copies.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtWidgets import QLayout, QWidget


def clear_layout(layout: QLayout, keep: Iterable[QWidget] = ()) -> None:
    """Empty ``layout``, destroying the widgets it held.

    Widgets in ``keep`` are only detached -- they are owned by the caller and
    get re-added, so they must survive.

    Everything is hidden *before* being unparented.  That order is the fix: an
    unparented widget Qt still considers visible is shown as a window, and
    ``deleteLater`` does not run until control returns to the event loop, which
    leaves a real gap for it to appear in.
    """
    kept = list(keep)

    while layout.count():
        item = layout.takeAt(0)
        w = item.widget()

        if w is None:
            continue

        w.hide()
        w.setParent(None)

        if w in kept:
            continue

        w.deleteLater()
