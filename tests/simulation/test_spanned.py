# test_spanned.py

import numpy as np
import pytest
from catella.simulation import _inputs
from catella.simulation._inputs import _spanned_intervals


def _windows_from_read(read_mask, lnuc, seed=0):
    rng = np.random.default_rng(seed)
    nmol, nbp = read_mask.shape
    block = rng.uniform(0.1, 0.9, size=(nmol, nbp))
    for row in range(nmol):
        for i in range(nbp):
            if not read_mask[row, i : min(i + lnuc, nbp)].any():
                block[row, i] = np.nan
    return block


def _reference(block, lnuc):
    nmol, nbp = block.shape
    npos = nbp - lnuc + 1
    finite = ~np.isnan(block[:, :npos])
    out = []
    for row in range(nmol):
        if not finite[row].any():
            out.append([])
            continue
        mask = [
            finite[row, max(0, b - lnuc + 1) : min(b, npos - 1) + 1].all()
            for b in range(nbp)
        ]
        runs = []
        for b, value in enumerate(mask):
            if value and (b == 0 or not mask[b - 1]):
                runs.append([b, b + 1])
            elif value:
                runs[-1][1] = b + 1
        out.append(runs)
    return out


def _as_lists(result, nmol):
    intervals, offset = result
    return [
        intervals[offset[i] : offset[i + 1]].tolist() for i in range(nmol)
    ]


def _read_mask(nbp, reads):
    mask = np.zeros((len(reads), nbp), dtype=bool)
    for row, pieces in enumerate(reads):
        for a, b in pieces:
            mask[row, a:b] = True
    return mask


class TestSpannedIntervals:
    def test_none_when_every_start_is_finite(self):
        block = np.full((3, 20), 0.5)
        assert _spanned_intervals(block, 5) is None

    def test_none_when_nan_only_in_no_data_rows(self):
        block = np.full((3, 20), 0.5)
        block[1] = np.nan
        assert _spanned_intervals(block, 5) is None

    def test_none_when_region_shorter_than_lnuc(self):
        block = np.full((2, 4), np.nan)
        block[0, 0] = 0.5
        assert _spanned_intervals(block, 5) is None

    def test_nan_after_last_start_is_ignored(self):
        block = np.full((2, 20), 0.5)
        block[:, 16:] = np.nan
        assert _spanned_intervals(block, 5) is None

    def test_read_is_recovered(self):
        mask = _read_mask(50, [[(12, 30)], [(0, 50)]])
        block = _windows_from_read(mask, 5)
        got = _as_lists(_spanned_intervals(block, 5), 2)
        assert got == [[[12, 30]], [[0, 50]]]

    def test_read_reaching_region_ends(self):
        mask = _read_mask(50, [[(0, 20)], [(30, 50)]])
        block = _windows_from_read(mask, 5)
        block[:, 46:] = np.nan
        got = _as_lists(_spanned_intervals(block, 5), 2)
        assert got == [[[0, 20]], [[30, 50]]]

    def test_long_gap_is_left_out(self):
        mask = _read_mask(50, [[(0, 20), (30, 50)]])
        block = _windows_from_read(mask, 5)
        got = _as_lists(_spanned_intervals(block, 5), 1)
        assert got == [[[0, 20], [30, 50]]]

    def test_short_gap_is_spanned(self):
        mask = _read_mask(50, [[(5, 20), (23, 40)], [(0, 50)]])
        block = _windows_from_read(mask, 5)
        got = _as_lists(_spanned_intervals(block, 5), 2)
        assert got == [[[5, 40]], [[0, 50]]]

    def test_no_data_rows_have_no_intervals(self):
        mask = _read_mask(30, [[(5, 20)], [], [(0, 30)]])
        block = _windows_from_read(mask, 5)
        intervals, offset = _spanned_intervals(block, 5)
        assert offset.tolist() == [0, 1, 1, 2]
        assert intervals.dtype == np.int64
        assert offset.dtype == np.int64

    def test_isolated_finite_start_spans_nothing(self):
        block = np.full((1, 20), np.nan)
        block[0, 7] = 0.5
        got = _as_lists(_spanned_intervals(block, 5), 1)
        assert got == [[]]

    def test_one_read_bp_is_spanned(self):
        mask = _read_mask(20, [[(9, 10)]])
        block = _windows_from_read(mask, 5)
        got = _as_lists(_spanned_intervals(block, 5), 1)
        assert got == [[[9, 10]]]

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_reference(self, seed):
        rng = np.random.default_rng(seed)
        block = rng.uniform(size=(12, 60))
        for row in range(12):
            for _ in range(rng.integers(0, 4)):
                a = rng.integers(0, 60)
                block[row, a : a + rng.integers(1, 25)] = np.nan
        block[3] = np.nan
        got = _spanned_intervals(block, 7)
        assert got is not None
        assert _as_lists(got, 12) == _reference(block, 7)

    def test_batched_matches_single_batch(self, monkeypatch):
        rng = np.random.default_rng(1)
        block = rng.uniform(size=(9, 40))
        block[rng.uniform(size=block.shape) < 0.2] = np.nan
        single = _spanned_intervals(block, 4)
        monkeypatch.setattr(_inputs, "_SPANNED_BATCH_CELLS", 80)
        batched = _spanned_intervals(block, 4)
        np.testing.assert_array_equal(batched[0], single[0])
        np.testing.assert_array_equal(batched[1], single[1])
