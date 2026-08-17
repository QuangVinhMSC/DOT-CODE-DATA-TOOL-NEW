import pytest

from dotgen.core.matrix import CharMetrics, solve_metrics, validate
from dotgen.core.models import CharFormat, DotLink

DIST_H = 12.0
DIST_V = 20.0


def fmt(
    dots: list[tuple[int, int]],
    links: list[DotLink] | None = None,
    char: str = "1",
) -> CharFormat:
    return CharFormat(char, 5, 7, list(dots), list(links or []))


def valid_format() -> CharFormat:
    """Two columns, two rows: one legal link per axis."""
    return fmt(
        [(1, 1), (1, 3), (3, 1)],
        [DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0)],
    )


# ----------------------------------------------------------------------
# pitch from a link
# ----------------------------------------------------------------------


def test_headline_case_two_cells_one_unit():
    """The plan's example: '1' with two dots 2 cells apart, coeff 1."""
    f = fmt([(2, 1), (2, 3)], [DotLink(0, 1, "v", 1.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_v == pytest.approx(DIST_V / 2)
    assert m.positions[1][1] - m.positions[0][1] == pytest.approx(DIST_V)


def test_a_third_dot_one_cell_below_sits_one_pitch_away():
    f = fmt([(2, 1), (2, 3), (2, 4)], [DotLink(0, 1, "v", 1.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.positions[2][1] - m.positions[1][1] == pytest.approx(m.pitch_v)
    assert m.pitch_v == pytest.approx(DIST_V / 2)


def test_horizontal_mirror_of_the_headline_case():
    f = fmt([(1, 2), (3, 2), (4, 2)], [DotLink(0, 1, "h", 1.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_h == pytest.approx(DIST_H / 2)
    assert m.positions[1][0] - m.positions[0][0] == pytest.approx(DIST_H)
    assert m.positions[2][0] - m.positions[1][0] == pytest.approx(m.pitch_h)


def test_coeff_two_over_two_cells_gives_one_unit_per_cell():
    f = fmt([(2, 0), (2, 2)], [DotLink(0, 1, "v", 2.0)])

    assert solve_metrics(f, DIST_H, DIST_V).pitch_v == pytest.approx(DIST_V)


def test_coeff_one_over_one_cell_gives_one_unit_per_cell():
    f = fmt([(2, 0), (2, 1)], [DotLink(0, 1, "v", 1.0)])

    assert solve_metrics(f, DIST_H, DIST_V).pitch_v == pytest.approx(DIST_V)


def test_fractional_coefficient():
    f = fmt([(0, 0), (3, 0)], [DotLink(0, 1, "h", 1.5)])

    assert solve_metrics(f, DIST_H, DIST_V).pitch_h == pytest.approx(1.5 * DIST_H / 3)


def test_link_direction_does_not_matter():
    a = fmt([(2, 1), (2, 4)], [DotLink(0, 1, "v", 1.0)])
    b = fmt([(2, 1), (2, 4)], [DotLink(1, 0, "v", 1.0)])

    assert solve_metrics(a, DIST_H, DIST_V).pitch_v == pytest.approx(
        solve_metrics(b, DIST_H, DIST_V).pitch_v
    )


def test_axes_are_solved_independently():
    f = fmt(
        [(0, 0), (0, 2), (4, 0)],
        [DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 2.0)],
    )

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_v == pytest.approx(DIST_V / 2)
    assert m.pitch_h == pytest.approx(2 * DIST_H / 4)


# ----------------------------------------------------------------------
# positions, size, scaling
# ----------------------------------------------------------------------


def test_origin_is_the_characters_own_top_left():
    """A character painted at (2, 3) still starts at (0, 0)."""
    f = fmt([(2, 3), (2, 5), (4, 3)], [DotLink(0, 1, "v", 1.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.positions[0] == (0.0, 0.0)
    assert m.positions[1][0] == 0.0
    assert m.positions[2][1] == 0.0
    assert min(x for x, _ in m.positions.values()) == 0.0
    assert min(y for _, y in m.positions.values()) == 0.0


def test_positions_are_keyed_by_dot_index():
    f = valid_format()

    m = solve_metrics(f, DIST_H, DIST_V)

    assert set(m.positions) == set(range(len(f.dots)))


def test_size_spans_the_painted_cells():
    f = fmt(
        [(1, 1), (1, 4), (3, 1)],
        [DotLink(0, 1, "v", 3.0), DotLink(0, 2, "h", 2.0)],
    )

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_v == pytest.approx(DIST_V)
    assert m.pitch_h == pytest.approx(DIST_H)
    assert m.height == pytest.approx(3 * DIST_V)
    assert m.width == pytest.approx(2 * DIST_H)


def test_single_column_character_has_zero_width():
    """Not faked to a minimum -- the renderer adds the dot radius itself."""
    f = fmt([(2, 0), (2, 1), (2, 2)], [DotLink(0, 2, "v", 2.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.width == 0.0
    assert m.height > 0.0


def test_doubling_the_vertical_unit_doubles_height_and_every_y():
    """This is what makes Tab 3's Min and Max frames differ."""
    f = valid_format()

    small = solve_metrics(f, DIST_H, DIST_V)
    big = solve_metrics(f, DIST_H, 2 * DIST_V)

    assert big.height == pytest.approx(2 * small.height)
    assert big.pitch_v == pytest.approx(2 * small.pitch_v)

    for i, (x, y) in small.positions.items():
        assert big.positions[i][0] == pytest.approx(x)
        assert big.positions[i][1] == pytest.approx(2 * y)


def test_doubling_the_horizontal_unit_only_touches_x():
    f = valid_format()

    small = solve_metrics(f, DIST_H, DIST_V)
    big = solve_metrics(f, 2 * DIST_H, DIST_V)

    assert big.width == pytest.approx(2 * small.width)
    assert big.height == pytest.approx(small.height)

    for i, (x, y) in small.positions.items():
        assert big.positions[i][0] == pytest.approx(2 * x)
        assert big.positions[i][1] == pytest.approx(y)


def test_characters_are_independent():
    """Same units, different links -> different pitches."""
    one = fmt([(2, 0), (2, 2)], [DotLink(0, 1, "v", 1.0)], char="1")
    seven = fmt([(2, 0), (2, 2)], [DotLink(0, 1, "v", 2.0)], char="7")

    a = solve_metrics(one, DIST_H, DIST_V)
    b = solve_metrics(seven, DIST_H, DIST_V)

    assert a.pitch_v == pytest.approx(DIST_V / 2)
    assert b.pitch_v == pytest.approx(DIST_V)
    assert a.pitch_v != b.pitch_v


# ----------------------------------------------------------------------
# fallbacks -- solve_metrics is called live on invalid formats
# ----------------------------------------------------------------------


def test_no_links_falls_back_to_one_unit_per_cell():
    f = fmt([(0, 0), (1, 2)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_h == pytest.approx(DIST_H)
    assert m.pitch_v == pytest.approx(DIST_V)
    assert m.width == pytest.approx(DIST_H)
    assert m.height == pytest.approx(2 * DIST_V)


def test_missing_axis_falls_back_only_on_that_axis():
    f = fmt([(0, 0), (0, 2)], [DotLink(0, 1, "v", 1.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_v == pytest.approx(DIST_V / 2)
    assert m.pitch_h == pytest.approx(DIST_H)


def test_out_of_range_link_index_falls_back():
    f = fmt([(0, 0), (0, 2)], [DotLink(0, 9, "v", 1.0), DotLink(-1, 0, "h", 1.0)])

    m = solve_metrics(f, DIST_H, DIST_V)

    assert m.pitch_v == pytest.approx(DIST_V)
    assert m.pitch_h == pytest.approx(DIST_H)


def test_zero_cell_link_falls_back_instead_of_dividing_by_zero():
    f = fmt([(1, 2), (3, 2)], [DotLink(0, 1, "v", 4.0)])

    assert solve_metrics(f, DIST_H, DIST_V).pitch_v == pytest.approx(DIST_V)


def test_self_link_falls_back():
    f = fmt([(1, 2), (3, 2)], [DotLink(0, 0, "h", 4.0)])

    assert solve_metrics(f, DIST_H, DIST_V).pitch_h == pytest.approx(DIST_H)


def test_non_positive_coefficient_falls_back():
    for coeff in (0.0, -2.0):
        f = fmt([(0, 0), (0, 2)], [DotLink(0, 1, "v", coeff)])

        assert solve_metrics(f, DIST_H, DIST_V).pitch_v == pytest.approx(DIST_V)


def test_no_dots_gives_an_empty_metric():
    m = solve_metrics(fmt([]), DIST_H, DIST_V)

    assert isinstance(m, CharMetrics)
    assert m.positions == {}
    assert (m.width, m.height) == (0.0, 0.0)
    assert (m.pitch_h, m.pitch_v) == (0.0, 0.0)


def test_single_dot_is_a_point_at_the_origin():
    m = solve_metrics(fmt([(3, 4)]), DIST_H, DIST_V)

    assert m.positions == {0: (0.0, 0.0)}
    assert (m.width, m.height) == (0.0, 0.0)


def test_solve_metrics_never_raises_on_broken_formats():
    broken = [
        fmt([]),
        fmt([], [DotLink(0, 1, "v", 1.0)]),
        fmt([(0, 0)], [DotLink(0, 1, "h", 1.0)]),
        fmt([(0, 0), (0, 1)], [DotLink(0, 1, "h", 0.0)]),
        fmt([(0, 0), (0, 1)], [DotLink(5, 6, "v", -1.0)]),
    ]

    for f in broken:
        m = solve_metrics(f, DIST_H, DIST_V)

        assert m.pitch_h >= 0.0 and m.pitch_v >= 0.0


def test_zero_units_are_tolerated():
    """Tab 1 may not have measured anything yet."""
    m = solve_metrics(valid_format(), 0.0, 0.0)

    assert (m.width, m.height) == (0.0, 0.0)
    assert all(p == (0.0, 0.0) for p in m.positions.values())


# ----------------------------------------------------------------------
# validate -- delegates to CharFormat.validate
# ----------------------------------------------------------------------


def test_a_valid_format_reports_no_errors():
    assert validate(valid_format()) == []


def test_validate_delegates_to_the_format():
    f = valid_format()

    assert validate(f) == f.validate()


def test_missing_vertical_link_is_an_error():
    f = fmt([(1, 1), (1, 3), (3, 1)], [DotLink(0, 2, "h", 1.0)])

    assert validate(f)


def test_missing_horizontal_link_is_an_error():
    f = fmt([(1, 1), (1, 3), (3, 1)], [DotLink(0, 1, "v", 1.0)])

    assert validate(f)


def test_two_horizontal_links_are_an_error():
    f = fmt(
        [(1, 1), (1, 3), (3, 1), (4, 1)],
        [DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0), DotLink(0, 3, "h", 2.0)],
    )

    assert validate(f)


def test_two_vertical_links_are_an_error():
    f = fmt(
        [(1, 1), (1, 3), (1, 5), (3, 1)],
        [DotLink(0, 1, "v", 1.0), DotLink(1, 2, "v", 1.0), DotLink(0, 3, "h", 1.0)],
    )

    assert validate(f)


def test_link_to_a_removed_dot_is_an_error():
    f = valid_format()
    f.links.append(DotLink(0, 7, "v", 1.0))

    assert any("no longer exists" in e for e in validate(f))


def test_vertical_link_across_columns_is_an_error():
    f = fmt(
        [(1, 1), (2, 3), (3, 1)],
        [DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0)],
    )

    assert any("same column" in e for e in validate(f))


def test_horizontal_link_across_rows_is_an_error():
    f = fmt(
        [(1, 1), (1, 3), (3, 2)],
        [DotLink(0, 1, "v", 1.0), DotLink(0, 2, "h", 1.0)],
    )

    assert any("same row" in e for e in validate(f))


def test_zero_and_negative_coefficients_are_errors():
    for coeff in (0.0, -1.0):
        f = valid_format()
        f.links[0].coeff = coeff

        assert any("Coefficient" in e for e in validate(f))


def test_a_single_dot_is_an_error():
    f = fmt([(2, 2)])

    assert any("at least 2 dots" in e for e in validate(f))


def test_errors_are_human_readable_strings():
    errors = validate(fmt([(2, 2)]))

    assert errors
    assert all(isinstance(e, str) and e for e in errors)
