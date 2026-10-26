import numpy as np
import pandas as pd
import pytest
from catella.experiment._empirical import (
    _clip_bounds,
    _fill_smoothed_nans,
    _normalize_to_controls,
)


def _frame():
    return pd.DataFrame(
        {
            0: [np.nan, 1.0, np.nan, 3.0, np.nan],
            1: [2.0, np.nan, np.nan, 6.0, 8.0],
        }
    )


class TestFillSmoothedNans:
    def test_none_fills_only_edges(self):
        out = _fill_smoothed_nans(_frame(), "none", -1.0)
        np.testing.assert_array_equal(
            out[0].to_numpy(), [-1.0, 1.0, -1.0, 3.0, -1.0]
        )
        np.testing.assert_array_equal(
            out[1].to_numpy(), [2.0, -1.0, -1.0, 6.0, 8.0]
        )

    def test_interpolate_fills_interior_linearly(self):
        out = _fill_smoothed_nans(_frame(), "interpolate", np.nan)
        np.testing.assert_array_equal(
            out[0].to_numpy(), [np.nan, 1.0, 2.0, 3.0, np.nan]
        )
        np.testing.assert_allclose(
            out[1].to_numpy(), [2.0, 10 / 3, 14 / 3, 6.0, 8.0]
        )

    def test_mean_fills_interior_with_molecule_mean(self):
        out = _fill_smoothed_nans(_frame(), "mean", np.nan)
        np.testing.assert_array_equal(
            out[0].to_numpy(), [np.nan, 1.0, 2.0, 3.0, np.nan]
        )
        np.testing.assert_array_equal(
            out[1].to_numpy(), [2.0, 16 / 3, 16 / 3, 6.0, 8.0]
        )

    @pytest.mark.parametrize("nan_method", ["mean", "interpolate", "none"])
    def test_edge_mean_uses_molecule_mean(self, nan_method):
        out = _fill_smoothed_nans(_frame(), nan_method, "mean")
        assert out[0].iloc[0] == pytest.approx(out[0].iloc[1:4].mean())
        assert not out.isna().to_numpy().any()

    def test_does_not_modify_input(self):
        res = _frame()
        _fill_smoothed_nans(res, "mean", 0.0)
        assert res.equals(_frame())


class TestNormalizeToControls:
    def test_none_returns_batch_unchanged(self):
        batch = np.array([[0.2, 0.4]])
        assert _normalize_to_controls(batch, None, None, 1e-12) is batch

    def test_pooled_maps_unmeth_to_0_and_meth_to_1(self):
        means = {None: (np.array([0.9, 0.8]), np.array([0.1, 0.2]))}
        batch = np.array([[0.1, 0.8], [0.5, 0.5]])
        out = _normalize_to_controls(batch, None, means, 1e-12)
        np.testing.assert_allclose(out, [[0.0, 1.0], [0.5, 0.5]])

    def test_by_strand_uses_each_strand_means(self):
        means = {
            "+": (np.array([1.0]), np.array([0.0])),
            "-": (np.array([0.5]), np.array([0.25])),
        }
        batch = np.array([[0.5], [0.5]])
        labels = np.array(["+", "-"])
        out = _normalize_to_controls(batch, labels, means, 1e-12)
        np.testing.assert_allclose(out, [[0.5], [1.0]])

    def test_eps_floors_equal_means(self):
        means = {None: (np.array([0.5]), np.array([0.5]))}
        out = _normalize_to_controls(np.array([[0.75]]), None, means, 0.5)
        np.testing.assert_allclose(out, [[0.5]])


class TestClipBounds:
    def _kw(self):
        return {
            "sample_size": 1000,
            "batch_size": 2,
            "end_idx": None,
            "rng": np.random.default_rng(0),
            "eps": 1e-12,
        }

    def test_without_controls_uses_test_percentiles(self):
        test = np.arange(20, dtype=float).reshape(5, 4)
        vmin, vmax = _clip_bounds(
            {"test": test},
            {"test": None},
            None,
            clip_low=0,
            clip_high=100,
            **self._kw(),
        )
        assert (vmin, vmax) == (0.0, 19.0)

    def test_with_controls_uses_unmeth_low_and_meth_high(self):
        means = {None: (np.ones(2), np.zeros(2))}
        arrs = {
            "test": np.full((3, 2), 100.0),
            "meth": np.array([[0.5, 0.9], [0.7, 0.8]]),
            "unmeth": np.array([[0.1, 0.3], [0.2, 0.4]]),
        }
        vmin, vmax = _clip_bounds(
            arrs,
            dict.fromkeys(arrs),
            means,
            clip_low=0,
            clip_high=100,
            **self._kw(),
        )
        assert (vmin, vmax) == (0.1, 0.9)

    def test_end_idx_excludes_trailing_positions(self):
        test = np.array([[0.0, 1.0, 50.0], [2.0, 3.0, -50.0]])
        kw = self._kw()
        kw["end_idx"] = -1
        vmin, vmax = _clip_bounds(
            {"test": test},
            {"test": None},
            None,
            clip_low=0,
            clip_high=100,
            **kw,
        )
        assert (vmin, vmax) == (0.0, 3.0)
