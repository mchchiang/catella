# test_plotting.py

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from catella._plotting import (
    _BP_UNITS,
    _ShiftedLocator,
    _unit_axis,
    _UnitFormatter,
)


def _labels(locs, start=0):
    fmt = _UnitFormatter(_BP_UNITS, start)
    fmt.set_locs(np.asarray(locs, dtype=float))
    return [fmt(x) for x in locs]


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
