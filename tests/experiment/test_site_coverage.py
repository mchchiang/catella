# test_site_coverage.py

import numpy as np
import pandas as pd
import pytest
from catella.experiment._dropout import _dropout_fracs, _site_coverage

_L = 10


def _mask(idx):
    m = np.zeros(_L, dtype=bool)
    m[idx] = True
    return m


_SITES = {"X": (_mask([1, 3, 5, 7]), _mask([2, 4, 6, 8]))}


def _rows(rows):
    return pd.DataFrame(rows, columns=["mol_index", "pos", "strand"])


def _df():
    return _rows([(0, 3, "+"), (0, 5, "+"), (1, 4, "-")])


def _gaps(per_mol):
    flat = [g for gaps in per_mol for g in gaps]
    offset = np.concatenate([[0], np.cumsum([len(g) for g in per_mol])])
    return np.array(flat, dtype=np.int64).reshape(-1, 2), offset


def _extent(starts, ends):
    return np.array(starts, np.int64), np.array(ends, np.int64)


def _cov(df=None, nmol=2, **kwargs):
    df = _df() if df is None else df
    covered, total = _site_coverage(df, nmol, _SITES, "union", **kwargs)["X"]
    return covered.tolist(), total.tolist()


class TestSpan:
    def test_full_counts_every_site(self):
        assert _cov() == ([2, 1], [4, 4])

    def test_read_counts_sites_in_extent(self):
        got = _cov(span="read", extent=_extent([2, 4], [6, 5]))
        assert got == ([2, 1], [2, 1])

    def test_read_ignores_rows_outside_extent(self):
        got = _cov(span="read", extent=_extent([4, 0], [10, 10]))
        assert got == ([1, 1], [2, 4])

    def test_calls_from_first_to_last_call(self):
        assert _cov(span="calls") == ([2, 1], [2, 1])

    def test_calls_without_rows_measures_nothing(self):
        assert _cov(nmol=3, span="calls") == ([2, 1, 0], [2, 1, 0])


class TestGaps:
    def test_gap_sites_left_out(self):
        got = _cov(gaps=_gaps([[(6, 9)], []]))
        assert got == ([2, 1], [3, 4])

    def test_gap_clipped_to_span(self):
        got = _cov(
            span="read",
            extent=_extent([2, 0], [8, 10]),
            gaps=_gaps([[(6, 9)], []]),
        )
        assert got == ([2, 1], [2, 4])

    def test_call_inside_gap_still_counts(self):
        got = _cov(gaps=_gaps([[(4, 8)], []]))
        assert got == ([2, 1], [3, 4])

    def test_gaps_of_other_molecules_ignored(self):
        got = _cov(gaps=_gaps([[], [(0, 3)]]))
        assert got == ([2, 1], [4, 3])

    def test_no_gaps_is_unchanged(self):
        assert _cov(gaps=_gaps([[], []])) == _cov()


class TestStrandHandling:
    def test_ignore_strand_uses_both_strands(self):
        covered, total = _site_coverage(
            _df(), 2, _SITES, "union", ignore_strand=True
        )["X"]
        assert covered.tolist() == [2, 1]
        assert total.tolist() == [8, 8]

    @pytest.mark.parametrize(
        ("unmapped", "total"), [("+", 4), ("-", 4), ("union", 8), ("drop", 8)]
    )
    def test_unmapped_strand_total(self, unmapped, total):
        df = _rows([(0, 3, ".")])
        got = _site_coverage(df, 1, _SITES, unmapped)["X"][1]
        assert got.tolist() == [total]


class TestDropoutFracs:
    def test_fraction_of_sites_without_call(self):
        cov = {"X": (np.array([1, 2]), np.array([4, 2]))}
        frac, pooled = _dropout_fracs(cov, ["X"])
        assert frac.tolist() == [[0.75, 0.0]]
        assert pooled.tolist() == [0.75, 0.0]

    def test_nan_without_sites(self):
        cov = {
            "X": (np.array([0, 1]), np.array([0, 2])),
            "Y": (np.array([0, 0]), np.array([0, 0])),
        }
        frac, pooled = _dropout_fracs(cov, ["X", "Y"])
        assert np.isnan(frac[:, 0]).all()
        assert frac[0, 1] == 0.5
        assert np.isnan(frac[1, 1])
        assert np.isnan(pooled[0])
        assert pooled[1] == 0.5
