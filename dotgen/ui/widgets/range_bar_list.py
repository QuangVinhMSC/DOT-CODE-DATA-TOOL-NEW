"""RangeBarList -- a scrollable, filterable, collapsible list of RangeBars.

Tab 1 uses it with group-level enable checkboxes for the optional parameter
groups; Tab 3 uses it in *draft* mode with per-bar compare checkboxes.

Draft mode is what puts Tab 3's Load button in charge.  The rows then edit a
private copy of the parameters instead of the live ones, so the two preview
frames can follow a typed number while nothing downstream has moved yet; the copy
reaches :class:`AppState` only when the tab asks it to.  A bar the user has not
touched keeps tracking its measurement, exactly as ``ParamSet.merge`` does with
``user_set`` -- the draft is that same rule, one step earlier.

Only the three *values* are staged.  ``enabled`` and ``compare`` go straight to
the state: whether a group is used at all is not a number being tuned, and the
compare ticks are a property of the preview itself.
"""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ...core.params import GROUP_ORDER, ParamSet, group_of
from ...core.state import AppState
from .. import theme
from ..qtutil import clear_layout
from .range_bar import RangeBar
from .value_row import FIELD_WIDTH, ValueRow

# How far a section indents its rows, and the width the numeric header leaves
# for an enable checkbox -- both only so the column captions land over their
# columns.  ``_Section`` below is what actually applies the indent.
_SECTION_INDENT = 14
_ENABLE_BOX_WIDTH = 18

# Groups the draft marks as "can be used or unused".
OPTIONAL_GROUPS: dict[str, tuple[str, ...]] = {
    "Perspective / Tilt": ("persp", "tilt"),
    "Curve": ("curve",),
}


class _Section(QFrame):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.title = title

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(0, 2, 0, 2)
        hl.setSpacing(4)

        self.toggle = QToolButton()
        self.toggle.setText(title)
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setArrowType(Qt.DownArrow)
        self.toggle.setStyleSheet("QToolButton { font-weight: 600; border: none; }")
        self.toggle.toggled.connect(self._on_toggled)
        hl.addWidget(self.toggle)

        self.group_box: QCheckBox | None = None
        hl.addStretch(1)

        self.header = header
        self.header_layout = hl
        outer.addWidget(header)

        self.body = QWidget()
        self.body_layout = QVBoxLayout(self.body)
        self.body_layout.setContentsMargins(_SECTION_INDENT, 0, 0, 4)
        self.body_layout.setSpacing(1)
        outer.addWidget(self.body)

    def add_group_checkbox(self, text: str = "use") -> QCheckBox:
        self.group_box = QCheckBox(text)
        self.group_box.setToolTip("Use this parameter group when rendering")
        self.header_layout.insertWidget(1, self.group_box)
        return self.group_box

    def add(self, w: QWidget) -> None:
        self.body_layout.addWidget(w)

    def _on_toggled(self, checked: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.body.setVisible(checked)


class RangeBarList(QWidget):
    draftEdited = Signal(str)  # key -- draft mode only
    editRejected = Signal(str)  # numeric mode only: what was typed is not a number

    def __init__(
        self,
        state: AppState,
        keys: Iterable[str] | None = None,
        show_enable: bool = False,
        show_compare: bool = False,
        editable: bool = True,
        group_enable: bool = False,
        show_filter: bool = True,
        label_width: int = 128,
        draft: bool = False,
        numeric: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.state = state
        self.keys = list(keys) if keys is not None else None
        self.show_enable = show_enable
        self.show_compare = show_compare
        self.editable = editable
        self.group_enable = group_enable
        self.label_width = label_width
        # Two ways to show the same parameter: a bar with three handles, or
        # three fields with the numbers in them.  Tab 1 measures and displays,
        # so it keeps the bar; Tab 3 is where values are stated, and there a
        # typed number beats a dragged one.
        self.numeric = numeric

        self.draft: ParamSet | None = state.params.deep_copy() if draft else None
        self._edited: set[str] = set()

        self.bars: dict[str, RangeBar] = {}
        self.sections: dict[str, _Section] = {}
        self._extra: list[tuple[str, QWidget]] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(theme.GAP)

        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter parameters...")
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.textChanged.connect(self._apply_filter)
        self.filter_edit.setVisible(show_filter)
        outer.addWidget(self.filter_edit)

        if self.numeric:
            outer.addWidget(self._build_header())

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.NoFrame)
        outer.addWidget(self.scroll, 1)

        self.container = QWidget()
        self.container_layout = QVBoxLayout(self.container)
        self.container_layout.setContentsMargins(0, 0, 0, 0)
        self.container_layout.setSpacing(2)
        self.scroll.setWidget(self.container)

        self.empty_label = QLabel("No parameters yet.")
        self.empty_label.setObjectName("hint")
        self.container_layout.addWidget(self.empty_label)

        self.rebuild()

        state.paramsChanged.connect(self.refresh)

    # ------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        """Name the three columns, once, above the scroll area.

        Three anonymous boxes are three chances to type a mean into the max.
        The margins and widths here are the ones a row uses -- the section
        indent, the row's own margin, then the label -- so the captions sit
        over their fields instead of near them, and it stays put while the
        list underneath scrolls.
        """
        header = QWidget()
        hl = QHBoxLayout(header)
        hl.setContentsMargins(_SECTION_INDENT + 2, 0, 2, 0)
        hl.setSpacing(theme.GAP)

        if self.show_enable:
            hl.addSpacing(_ENABLE_BOX_WIDTH + theme.GAP)

        hl.addSpacing(self.label_width)

        for name in ("Min", "Mean", "Max"):
            cap = QLabel(name)
            cap.setObjectName("hint")
            cap.setFixedWidth(FIELD_WIDTH)
            cap.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
            hl.addWidget(cap)

        hl.addStretch(1)

        return header

    # ------------------------------------------------------------------
    def add_section(self, title: str, widget: QWidget) -> None:
        """Attach an extra panel (Tab 1's test panel, geometry debug, ...)."""
        self._extra.append((title, widget))
        self.rebuild()

    def source_params(self) -> ParamSet:
        """What the bars are drawn from -- the draft when there is one."""
        return self.state.params if self.draft is None else self.draft

    def visible_params(self) -> ParamSet:
        params = self.source_params()

        if self.keys is None:
            return params

        return ParamSet([params[k] for k in self.keys if k in params])

    # ------------------------------------------------------------------
    # draft mode
    # ------------------------------------------------------------------

    def edited_keys(self) -> list[str]:
        """The bars the user has changed since the last :meth:`reseed`."""
        return sorted(self._edited)

    def reseed(self) -> None:
        """Forget the edits and take the live parameters back.

        Called after a commit, and by whoever wants a Revert: with nothing
        marked edited, the next state change flows through untouched.
        """
        if self.draft is None:
            return

        self._edited.clear()
        self.refresh()

    def set_compare_all(self, compare: bool) -> None:
        for key in list(self.state.params):
            self.state.set_param_compare(key, compare)

    def _reseed_draft(self) -> None:
        """Pull the live parameters into the draft, holding the edited values back.

        Done in place, because every bar holds a reference to the draft's
        :class:`RangeParam`: replacing the object would leave each bar that
        :meth:`refresh` was not told about pointing at the previous copy.

        Everything except min/mean/max is taken from the state even on an edited
        bar -- the label, the bounds and the enabled flag are not what the user
        typed, and a bar frozen against its own group's checkbox would be a lie.
        """
        if self.draft is None:
            return

        live_params = self.state.params

        for key, live in live_params.items():
            old = self.draft.get(key)

            if old is None:
                self.draft[key] = live.copy()
                continue

            staged = (old.mean, old.min, old.max) if key in self._edited else None

            for name, value in live.to_dict().items():
                setattr(old, name, value)

            if staged is not None:
                old.mean, old.min, old.max = staged

            old.clamp()

        for key in [k for k in self.draft if k not in live_params]:
            del self.draft[key]
            self._edited.discard(key)

    def _edit_value(self, key: str, field: str, value: float) -> None:
        p = self.draft.get(key) if self.draft is not None else None

        if p is None:
            return

        p.set_field(field, value)
        self._edited.add(key)

        bar = self.bars.get(key)

        if bar is not None:
            bar.refresh()

        self.draftEdited.emit(key)

    # ------------------------------------------------------------------
    def rebuild(self) -> None:
        # The attached panels and the empty label belong to whoever supplied
        # them and are re-added below; everything else is a section this method
        # built last time and must be destroyed, not merely unparented.
        for _title, widget in self._extra:
            widget.hide()
            widget.setParent(None)

        clear_layout(self.container_layout, keep=[self.empty_label])

        self.bars.clear()
        self.sections.clear()

        params = self.visible_params()
        by_group: dict[str, list] = {}

        for key, p in params.items():
            by_group.setdefault(group_of(key), []).append(p)

        order = [g for g in GROUP_ORDER if g in by_group]
        order += [g for g in by_group if g not in order]

        for gname in order:
            section = _Section(gname)
            self.sections[gname] = section

            if self.group_enable and gname in OPTIONAL_GROUPS:
                box = section.add_group_checkbox()
                prefixes = OPTIONAL_GROUPS[gname]
                box.setChecked(any(p.enabled for p in by_group[gname]))
                box.toggled.connect(
                    lambda v, pref=prefixes: self.state.set_group_enabled(pref, v)
                )

            row_class = ValueRow if self.numeric else RangeBar

            for p in by_group[gname]:
                bar = row_class(
                    p,
                    show_enable=self.show_enable,
                    show_compare=self.show_compare,
                    label_width=self.label_width,
                )
                bar.setEditable(self.editable)
                bar.valueChanged.connect(
                    self.state.set_param if self.draft is None else self._edit_value
                )
                bar.enabledToggled.connect(self.state.set_param_enabled)
                bar.compareToggled.connect(self.state.set_param_compare)

                if self.numeric:
                    bar.editRejected.connect(self.editRejected)
                self.bars[p.key] = bar
                section.add(bar)

            self.container_layout.addWidget(section)

        for title, widget in self._extra:
            section = _Section(title)
            section.add(widget)
            self.sections[title] = section
            self.container_layout.addWidget(section)

        self.empty_label.setVisible(not params)
        self.container_layout.addWidget(self.empty_label)
        self.container_layout.addStretch(1)
        self._apply_filter(self.filter_edit.text())

    # ------------------------------------------------------------------
    def refresh(self, keys: list[str] | None = None) -> None:
        self._reseed_draft()
        params = self.source_params()

        # A key we have never shown means the visible set changed.
        wanted = set(self.visible_params())

        if wanted - set(self.bars):
            self.rebuild()
            return

        for key in keys or list(self.bars):
            bar = self.bars.get(key)
            p = params.get(key)

            if bar is not None and p is not None:
                bar.setParam(p)

        for gname, section in self.sections.items():
            if section.group_box is not None:
                prefixes = OPTIONAL_GROUPS.get(gname, ())
                on = any(
                    p.enabled for k, p in params.items() if k.split(".", 1)[0] in prefixes
                )
                section.group_box.blockSignals(True)
                section.group_box.setChecked(on)
                section.group_box.blockSignals(False)

    # ------------------------------------------------------------------
    def _apply_filter(self, text: str) -> None:
        needle = text.strip().lower()

        for key, bar in self.bars.items():
            match = (
                not needle
                or needle in key.lower()
                or needle in bar.param.label.lower()
            )
            bar.setVisible(match)

        for gname, section in self.sections.items():
            if gname in [t for t, _ in self._extra]:
                continue

            any_visible = any(
                b.isVisible() for k, b in self.bars.items() if group_of(k) == gname
            )
            section.setVisible(any_visible or not needle)
