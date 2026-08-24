"""DefectCard -- the one widget seven defect kinds share.

The card is generated from the kind, so what is worth testing is not any one
kind's layout but the rule that generates it: the rows a kind ignores are absent
rather than dead, every widget follows the checkbox, and showing a value never
looks like the user typing it.
"""

import pytest

from dotgen.core.models import (
    DEFECT_FULL_SPAN,
    DEFECT_KINDS,
    DEFECT_LABELS,
    DEFECT_NO_SIDE,
    LineDefect,
    defect_range,
)
from dotgen.ui.widgets.defect_card import DEFECT_AMOUNT_LABELS, DefectCard


@pytest.fixture
def card(app):
    def make(kind: str, defect: LineDefect | None = None) -> DefectCard:
        return DefectCard(kind, defect or LineDefect(kind=kind))

    return make


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_every_kind_builds_a_card(card, kind):
    c = card(kind)

    assert c.enable.text() == DEFECT_LABELS[kind]
    assert c.kind == kind


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_the_ignored_rows_are_absent(card, kind):
    c = card(kind)

    assert (c.span_lo is None) is (kind in DEFECT_FULL_SPAN)
    assert (c.side is None) is (kind in DEFECT_NO_SIDE)
    assert (c.amount_lo is None) is (DEFECT_AMOUNT_LABELS[kind] is None)


@pytest.mark.parametrize("kind", DEFECT_KINDS)
def test_the_checkbox_governs_every_child(card, kind):
    c = card(kind)

    assert all(not w.isEnabled() for w in c._children())

    c.enable.setChecked(True)

    assert all(w.isEnabled() for w in c._children())

    c.enable.setChecked(False)

    assert all(not w.isEnabled() for w in c._children())


def test_set_defect_shows_the_values(card):
    c = card("ink_cover")
    c.set_defect(
        LineDefect(
            kind="ink_cover",
            enabled=True,
            p_line=0.3,
            max_lines=2,
            amount=(0.85, 1.0),
            span=(0.1, 0.4),
            side="right",
        )
    )

    assert c.enable.isChecked() is True
    assert c.p_line.value() == pytest.approx(0.3)
    assert c.max_lines.value() == 2
    assert (c.amount_lo.value(), c.amount_hi.value()) == pytest.approx((0.85, 1.0))
    assert (c.span_lo.value(), c.span_hi.value()) == pytest.approx((0.1, 0.4))
    assert c.side.currentData() == "right"


def test_set_defect_does_not_re_emit(card):
    """Otherwise the tab's refresh on lineDefectsChanged is an infinite loop."""
    c = card("collapse_side")
    seen = []
    c.changed.connect(lambda kind, fields: seen.append((kind, fields)))

    c.set_defect(
        LineDefect(
            kind="collapse_side",
            enabled=True,
            p_line=0.5,
            max_lines=4,
            amount=(0.05, 0.3),
            span=(0.2, 0.8),
            side="left",
        )
    )

    assert seen == []


def test_changed_carries_only_the_field_that_moved(card):
    c = card("top_loss")
    seen = []
    c.changed.connect(lambda kind, fields: seen.append((kind, fields)))

    c.enable.setChecked(True)

    assert seen == [("top_loss", {"enabled": True})]

    seen.clear()
    c.p_line.setValue(0.25)

    assert seen == [("top_loss", {"p_line": 0.25})]

    seen.clear()
    c.amount_lo.setValue(0.2)

    assert seen == [("top_loss", {"amount": (0.2, defect_range("top_loss")[0][1])})]

    seen.clear()
    c.amount_hi.setValue(0.6)

    assert seen == [("top_loss", {"amount": (0.2, 0.6)})]


def test_a_kind_with_no_side_never_emits_one(card):
    c = card("squeeze")
    seen = []
    c.changed.connect(lambda kind, fields: seen.append(fields))

    c.enable.setChecked(True)
    c.amount_lo.setValue(0.35)

    assert all("side" not in f and "span" not in f for f in seen)
