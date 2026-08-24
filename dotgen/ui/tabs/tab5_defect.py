"""Tab 5 -- Defect generation.

Left: a live preview of the current job, composed through the very engine the
exporter runs, so what is on screen is what gets written.  A defect fires
probabilistically, so one draw is not a preview of a distribution -- hence the
seed spinner and the **Reroll** button beside it.

Right: one :class:`~dotgen.ui.widgets.defect_card.DefectCard` per kind, and the
class summary underneath.

The one rule of this feature a user cannot verify from the picture alone is that
a character a defect touches loses its own bounding box and hands the label to
its line.  "Show defect boxes" draws the composed boxes over the preview so that
rule is visible *before* twenty thousand images are exported rather than after.

Wiring follows the house rule that tabs never call each other: every card's
``changed`` goes into :meth:`~dotgen.core.state.AppState.set_line_defect`, and
this tab redraws from ``lineDefectsChanged`` like any other listener would.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPointF, QTimer, Qt, Signal
from PySide6.QtGui import QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGraphicsPolygonItem,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ...core import registry
from ...core.models import DEFECT_CLASS_PREFIX, DEFECT_KINDS, DEFECT_LABELS
from ...core.state import AppState
from .. import theme
from ..widgets.defect_card import DefectCard
from ..widgets.image_canvas import ImageCanvas
from ..widgets.overlay_items import cosmetic_pen

PREVIEW_SEED = 7

# The same debounce Tab 4 uses, for the same reason: a spinbox emits on every
# step of a drag and compose renders a full photograph.
PREVIEW_DELAY_MS = 200

BOX_NOTE = (
    "A character touched by a defect gets no bounding box.  Its line's box "
    "carries the defect class instead of its own."
)


class Tab5Defect(QWidget):
    statusMessage = Signal(str)

    def __init__(self, state: AppState, parent=None) -> None:
        super().__init__(parent)
        self.state = state
        self.cards: dict[str, DefectCard] = {}
        self._shown_bg = -2

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._build_preview_column())
        splitter.addWidget(self._build_settings_column())
        splitter.setSizes([900, 660])

        lay = QVBoxLayout(self)
        lay.setContentsMargins(theme.PAD, theme.PAD, theme.PAD, theme.PAD)
        lay.addWidget(splitter)

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PREVIEW_DELAY_MS)
        self._preview_timer.timeout.connect(self.refresh)

        for signal in (
            state.lineDefectsChanged,
            state.linesChanged,
            state.backgroundsChanged,
            state.paramsChanged,
        ):
            signal.connect(lambda *_: self._preview_timer.start())

        # The cards and the summary are cheap and must never lag behind the
        # state they are editing, so only the composed picture is debounced.
        state.lineDefectsChanged.connect(self._sync_cards)
        state.classesChanged.connect(lambda *_: self._refresh_summary())

        self.refresh()

    # ==================================================================
    # construction
    # ==================================================================

    def _build_preview_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)

        rl.addWidget(QLabel("Background"))

        self.bg_combo = QComboBox()
        self.bg_combo.setMinimumWidth(90)
        self.bg_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        rl.addWidget(self.bg_combo)

        rl.addWidget(QLabel("seed"))

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(PREVIEW_SEED)
        self.seed_spin.valueChanged.connect(lambda _v: self._preview_timer.start())
        rl.addWidget(self.seed_spin)

        reroll = QPushButton("Reroll")
        reroll.setToolTip(
            "Draw the defects again.  A defect fires by chance, so one picture\n"
            "is not a preview of what twenty thousand images will look like."
        )
        reroll.clicked.connect(self._reroll)
        rl.addWidget(reroll)

        self.show_boxes = QCheckBox("Show defect boxes")
        self.show_boxes.setToolTip(BOX_NOTE)
        self.show_boxes.toggled.connect(lambda _on: self.refresh())
        rl.addWidget(self.show_boxes)

        rl.addStretch(1)
        lay.addWidget(row)

        self.banner = QLabel("")
        self.banner.setObjectName("banner")
        self.banner.setWordWrap(True)
        self.banner.setVisible(False)
        lay.addWidget(self.banner)

        self.canvas = ImageCanvas()
        lay.addWidget(self.canvas, 1)

        note = QLabel(BOX_NOTE)
        note.setObjectName("hint")
        note.setWordWrap(True)
        lay.addWidget(note)

        return w

    def _build_settings_column(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(theme.GAP)

        inner = QWidget()
        il = QVBoxLayout(inner)
        il.setContentsMargins(0, 0, 0, 0)
        il.setSpacing(theme.GAP)

        for kind in DEFECT_KINDS:
            card = DefectCard(kind, self.state.line_defects.get(kind))
            card.changed.connect(self._on_card_changed)
            self.cards[kind] = card
            il.addWidget(card)

        il.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        lay.addWidget(scroll, 1)

        self.summary = QLabel("")
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-weight: 600;")
        lay.addWidget(self.summary)

        return w

    # ==================================================================
    # cards
    # ==================================================================

    def _on_card_changed(self, kind: str, fields: dict) -> None:
        self.state.set_line_defect(kind, **fields)

    def _sync_cards(self) -> None:
        """Push the state back into every card, and redo the summary.

        A card that raised the change re-reads its own value here, which is
        harmless because :meth:`DefectCard.set_defect` does not re-emit.  What
        it buys is a Load configuration or a job restore showing up without the
        tab knowing where the change came from.
        """
        for kind, card in self.cards.items():
            card.set_defect(self.state.line_defects.get(kind))

        self._refresh_summary()

    def _refresh_summary(self) -> None:
        classes = self.state.line_defect_classes()

        if classes:
            self.summary.setText(
                f"{len(classes)} defect kind{'s' if len(classes) != 1 else ''} enabled "
                f"-> {len(classes)} extra line "
                f"class{'es' if len(classes) != 1 else ''}: " + ", ".join(classes)
            )
        else:
            self.summary.setText("No defect kinds enabled -- no extra line classes.")

        self._refresh_banner()

    def _refresh_banner(self) -> None:
        """The two ways this tab can be lying about what export will produce."""
        warnings: list[str] = []

        armed = [
            k
            for k in DEFECT_KINDS
            if self.state.line_defects.get(k).enabled
            and k not in self.state.line_defects.enabled_kinds()
        ]

        if armed:
            warnings.append(
                "Ticked but can never fire (chance per line or max lines is 0): "
                + ", ".join(DEFECT_LABELS[k] for k in armed)
                + "."
            )

        if self.state.classes:
            have = {c.name for c in self.state.classes}
            want = set(self.state.line_defect_classes())
            stale = {
                c.name
                for c in self.state.classes
                if c.name.startswith(DEFECT_CLASS_PREFIX) and c.name not in want
            }
            missing = want - have

            if missing or stale:
                warnings.append(
                    "The class list no longer matches the enabled kinds -- press "
                    '"Load class" in Tab 6 before exporting.'
                )

        self.banner.setText("  ".join(warnings))
        self.banner.setVisible(bool(warnings))

    # ==================================================================
    # preview
    # ==================================================================

    def _reroll(self) -> None:
        self.seed_spin.setValue((self.seed_spin.value() + 1) % (self.seed_spin.maximum() + 1))

    def _sync_bg_combo(self) -> None:
        count = len(self.state.backgrounds)

        if self.bg_combo.count() == count:
            return

        keep = self.bg_combo.currentIndex()

        self.bg_combo.blockSignals(True)
        self.bg_combo.clear()

        for i in range(count):
            self.bg_combo.addItem(f"#{i + 1}")

        self.bg_combo.setCurrentIndex(min(max(keep, 0), count - 1))
        self.bg_combo.blockSignals(False)

    def active_background(self) -> int:
        index = self.bg_combo.currentIndex()

        return index if 0 <= index < len(self.state.backgrounds) else -1

    def refresh(self) -> None:
        self._sync_cards()
        self._sync_bg_combo()

        index = self.active_background()

        if index < 0:
            self.canvas.set_image(None)
            self.canvas.clear_overlays()
            self._shown_bg = -2
            return

        spec = self.state.backgrounds[index]
        image = spec.array
        quads: list[tuple] = []

        if image is not None and self.state.has_content():
            job = self.state.snapshot_job("preview")

            try:
                composed = registry.get_engines().compose(
                    job, index, np.random.default_rng(self.seed_spin.value())
                )
                image = composed.image
                quads = list(composed.quads)
            except Exception as exc:  # noqa: BLE001 - a preview must never crash the tab
                self.statusMessage.emit(f"Preview unavailable: {exc}")

        first = self._shown_bg != index
        self.canvas.set_image(image, keep_view=not first)
        self._shown_bg = index

        self.canvas.clear_overlays()

        if self.show_boxes.isChecked() and image is not None:
            self._draw_boxes(quads, (image.shape[1], image.shape[0]))

    def _draw_boxes(self, quads: list[tuple], size: tuple[int, int]) -> None:
        """The composed labels, drawn where they landed.

        Defect classes are drawn in the warning colour and everything else in
        the plain outline one, because the thing being checked here is which
        boxes survived -- a smeared character with no box of its own, inside a
        line box that turned amber.

        Drawn as polygons, which is what the labels are: a band cut or a blob
        keeps the angle of the line it damaged, and a rectangle here would not
        show that.
        """
        width, height = float(size[0]), float(size[1])

        for name, *coords in quads:
            item = QGraphicsPolygonItem(
                QPolygonF(
                    [
                        QPointF(coords[i] * width, coords[i + 1] * height)
                        for i in range(0, 8, 2)
                    ]
                )
            )
            defect = name.startswith(DEFECT_CLASS_PREFIX)
            item.setPen(cosmetic_pen(theme.WARN_AMBER if defect else theme.BOUND_BLUE))
            item.setZValue(20.0)
            self.canvas.add_overlay(item)
