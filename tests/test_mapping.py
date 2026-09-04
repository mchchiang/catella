# test_mapping.py

import numpy as np
import pandas as pd
import pytest

from catella.h5_array import H5Array
from catella.mapping import CoordsTransform


def _h5_from(values, index=None, columns=None):
    arr = H5Array.create(values.shape, dtype=values.dtype, index=index,
                         columns=columns)
    arr.write_batch(0, values.shape[0], values)
    return arr


@pytest.mark.parametrize("batch_size", [2, 20000])
def test_left_to_center_aligned_h5array_axis1_matches_ndarray(batch_size):
    values = np.arange(30, dtype=np.float64).reshape(6, 5)
    trans = CoordsTransform(binsize=4)

    expected = trans.left_to_center_aligned(values.copy(), axis=1)
    h5 = _h5_from(values)
    got = trans.left_to_center_aligned(h5, axis=1, batch_size=batch_size)

    assert isinstance(got, H5Array)
    np.testing.assert_array_equal(got.to_numpy(), expected)


@pytest.mark.parametrize("batch_size", [2, 20000])
def test_center_to_left_aligned_h5array_axis1_matches_ndarray(batch_size):
    values = np.arange(30, dtype=np.float64).reshape(6, 5)
    trans = CoordsTransform(binsize=4)

    expected = trans.center_to_left_aligned(values.copy(), axis=1)
    h5 = _h5_from(values)
    got = trans.center_to_left_aligned(h5, axis=1, batch_size=batch_size)

    assert isinstance(got, H5Array)
    np.testing.assert_array_equal(got.to_numpy(), expected)


@pytest.mark.parametrize("batch_size", [2, 20000])
def test_left_to_center_aligned_h5array_axis0_matches_ndarray(batch_size):
    values = np.arange(30, dtype=np.float64).reshape(6, 5)
    trans = CoordsTransform(binsize=4)

    expected = trans.left_to_center_aligned(values.copy(), axis=0)
    h5 = _h5_from(values)
    got = trans.left_to_center_aligned(h5, axis=0, batch_size=batch_size)

    assert isinstance(got, H5Array)
    np.testing.assert_array_equal(got.to_numpy(), expected)


@pytest.mark.parametrize("batch_size", [2, 20000])
def test_center_to_left_aligned_h5array_axis0_matches_ndarray(batch_size):
    values = np.arange(30, dtype=np.float64).reshape(6, 5)
    trans = CoordsTransform(binsize=4)

    expected = trans.center_to_left_aligned(values.copy(), axis=0)
    h5 = _h5_from(values)
    got = trans.center_to_left_aligned(h5, axis=0, batch_size=batch_size)

    assert isinstance(got, H5Array)
    np.testing.assert_array_equal(got.to_numpy(), expected)


def test_left_to_center_aligned_h5array_trim_matches_ndarray():
    values = np.arange(30, dtype=np.float64).reshape(6, 5)
    trans = CoordsTransform(binsize=4)

    expected = trans.left_to_center_aligned(values.copy(), axis=1,
                                            trim=True)
    h5 = _h5_from(values)
    got = trans.left_to_center_aligned(h5, axis=1, trim=True)

    np.testing.assert_array_equal(np.asarray(got), expected)


def test_h5array_preserves_index_and_columns():
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    index = pd.Index(["a", "b", "c"])
    columns = pd.Index([10, 20, 30, 40])
    h5 = _h5_from(values, index=index, columns=columns)
    trans = CoordsTransform(binsize=2)

    got = trans.left_to_center_aligned(h5, axis=1)

    pd.testing.assert_index_equal(got.index, index)
    pd.testing.assert_index_equal(got.columns, columns)


def test_h5array_uses_fill_edge_value():
    values = np.ones((4, 4), dtype=np.float64)
    h5 = _h5_from(values)
    trans = CoordsTransform(binsize=4, fill_edge=-1.0)

    got = trans.left_to_center_aligned(h5, axis=1)

    np.testing.assert_array_equal(got.to_numpy()[:, :2],
                                  np.full((4, 2), -1.0))
