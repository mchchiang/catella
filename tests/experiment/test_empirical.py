import numpy as np
import pandas as pd
import pytest
from catella.experiment._empirical import _fill_smoothed_nans


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
