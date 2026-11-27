# test_plotting.py

from itertools import pairwise

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from catella._plotting import (
    _BP_UNITS,
    _MCS_UNITS,
    _panel_grid,
    _ShiftedLocator,
    _unit_axis,
    _UnitFormatter,
)


def _labels(locs, start=0, units=_BP_UNITS):
    fmt = _UnitFormatter(units, start)
    fmt.set_locs(np.asarray(locs, dtype=float))
    return [fmt(x) for x in locs]


class TestTimeUnits:
    def test_thousands_of_mcs(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 20000)
        _unit_axis(ax.xaxis, _MCS_UNITS, label="Time [{unit}]")
        fig.canvas.draw()
        assert ax.get_xlabel() == "Time [$10^3$ MCS]"
        plt.close(fig)

    def test_labels(self):
        assert _labels([0, 5000, 10000], units=_MCS_UNITS) == [
            "0",
            "5",
            "10",
        ]
        assert _labels([0, 50, 100], units=_MCS_UNITS) == ["0", "50", "100"]


class TestUnitFormatter:
    @pytest.mark.parametrize(
        ("locs", "start", "expected"),
        [
            ([0, 2000, 4000], 3170000, ["3,170", "3,172", "3,174"]),
            ([0, 500, 1000], 3170000, ["3,170.0", "3,170.5", "3,171.0"]),
            ([0, 250, 500], 3170000, ["3,170,000", "3,170,250", "3,170,500"]),
            ([50, 100], 3170000, ["3,170,050", "3,170,100"]),
            ([0, 500000, 1000000], 0, ["0.0", "0.5", "1.0"]),
            ([0, 4, 8], 0, ["0", "4", "8"]),
            ([0, 0.5, 1], 0, ["0.0", "0.5", "1.0"]),
            ([1000], 0, ["1"]),
        ],
    )
    def test_largest_unit_with_one_decimal(self, locs, start, expected):
        assert _labels(locs, start) == expected

    def test_label_unit_set_when_drawn(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 10000)
        _unit_axis(ax.xaxis, _BP_UNITS, label="Position [{unit}]")
        assert ax.get_xlabel() == "Position [bp]"
        fig.canvas.draw()
        assert ax.get_xlabel() == "Position [kb]"
        plt.close(fig)

    def test_label_unit_kept_for_short_range(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 30)
        _unit_axis(ax.xaxis, _BP_UNITS, label="Position [{unit}]")
        fig.canvas.draw()
        assert ax.get_xlabel() == "Position [bp]"
        plt.close(fig)

    def test_no_scientific_notation(self):
        fig, ax = plt.subplots()
        ax.set_xlim(0, 300)
        _unit_axis(ax.xaxis, _BP_UNITS, start=30000000)
        fig.canvas.draw()
        texts = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
        assert texts
        assert all("e" not in t for t in texts)
        plt.close(fig)


class TestShiftedLocator:
    def test_ticks_round_in_shifted_values(self):
        start = 3171234
        ticks = _ShiftedLocator(start).tick_values(0, 10000)
        assert np.allclose(np.mod(ticks + start, 500), 0)

    def test_no_start_matches_plain_ticks(self):
        ticks = _ShiftedLocator().tick_values(0, 100)
        assert np.allclose(np.mod(ticks, 5), 0)

    @pytest.mark.parametrize("vmax", [3, 20, 25])
    def test_whole_number_ticks(self, vmax):
        ticks = _ShiftedLocator(1000000).tick_values(0, vmax)
        assert np.allclose(ticks, np.round(ticks))


def _drawn_ticks(width=6.4, start=0, vmax=10000, axis="x"):
    with plt.rc_context({"font.size": 14}):
        fig, ax = plt.subplots(figsize=(width, 4.8))
        if axis == "x":
            ax.set_xlim(0, vmax)
            _unit_axis(ax.xaxis, _BP_UNITS, start=start, label="[{unit}]")
            ticklabels = ax.get_xticklabels
        else:
            ax.set_ylim(0, vmax)
            _unit_axis(ax.yaxis, _BP_UNITS, start=start, label="[{unit}]")
            ticklabels = ax.get_yticklabels
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        shown = [t for t in ticklabels() if t.get_text()]
        texts = [t.get_text() for t in shown]
        boxes = [t.get_window_extent(renderer) for t in shown]
    plt.close(fig)
    return texts, boxes


class TestTickSpacing:
    def test_genomic_labels_spaced_by_width(self):
        texts, boxes = _drawn_ticks(start=3170000)
        assert texts == ["3,170", "3,175", "3,180"]
        gaps = [b.x0 - a.x1 for a, b in pairwise(boxes)]
        assert min(gaps) >= max(b.width for b in boxes)

    def test_short_labels_keep_default_ticks(self):
        texts, _ = _drawn_ticks()
        assert texts == ["0", "2", "4", "6", "8", "10"]

    def test_narrower_axis_fewer_ticks(self):
        wide, _ = _drawn_ticks(width=12, start=3170000)
        narrow, _ = _drawn_ticks(width=3, start=3170000)
        assert len(narrow) < len(wide)
        assert len(narrow) >= 2

    def test_y_axis_not_thinned(self):
        texts, _ = _drawn_ticks(start=3170000, axis="y")
        assert len(texts) > 3


class TestPanelGridSize:
    @pytest.mark.parametrize(
        ("nregion", "factor"), [(1, 1.5), (2, 4 / 3), (3, 7 / 6), (4, 1.0)]
    )
    def test_default_width_tapers(self, nregion, factor):
        w, h = plt.rcParams["figure.figsize"]
        fig, _ = _panel_grid(nregion, [20, 1], dendrogram=False)
        width, height = fig.get_size_inches()
        assert width == pytest.approx(w * factor * nregion)
        assert height == pytest.approx(h)
        plt.close(fig)

    def test_more_rows_keep_four_column_width(self):
        w, h = plt.rcParams["figure.figsize"]
        fig, _ = _panel_grid(6, [20, 1], dendrogram=False)
        assert tuple(fig.get_size_inches()) == pytest.approx((4 * w, 2 * h))
        plt.close(fig)

    def test_scale_w_multiplies_width(self):
        w, _ = plt.rcParams["figure.figsize"]
        fig, _ = _panel_grid(1, [20, 1], dendrogram=False, scale_w=1.2)
        assert fig.get_size_inches()[0] == pytest.approx(w * 1.5 * 1.2)
        plt.close(fig)

    def test_figsize_overrides_default(self):
        fig, _ = _panel_grid(
            1, [20, 1], dendrogram=False, figsize=(7, 3)
        )
        assert tuple(fig.get_size_inches()) == pytest.approx((7, 3))
        plt.close(fig)
