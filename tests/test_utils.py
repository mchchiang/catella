# test_utils.py

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch

from nucmc import utils
from nucmc.h5_array import H5Array


def _reference_linkage(data, *, metric="euclidean", method="ward"):
    dist_vec = pdist(data, metric=metric)
    link_mat = sch.linkage(dist_vec, method=method)
    order = sch.leaves_list(link_mat)
    return order, link_mat


def _nan_masked_reference(data, *, method="ward"):
    n = data.shape[0]
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            u, v = data[i], data[j]
            mask = ~np.isnan(u) & ~np.isnan(v)
            n_present = mask.sum()
            if n_present == 0:
                dist[i, j] = np.nan
                continue
            sq = np.sum((u[mask] - v[mask]) ** 2)
            dist[i, j] = np.sqrt(len(u) / n_present * sq)
    dist_vec = squareform(dist, checks=False)
    link_mat = sch.linkage(dist_vec, method=method)
    order = sch.leaves_list(link_mat)
    return order, link_mat


def test_plain_ndarray_matches_reference():
    data = np.random.default_rng(0).random((15, 6))
    order, link_mat = utils.compute_linkage(data)
    ref_order, ref_link = _reference_linkage(data)
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)


@pytest.mark.parametrize("batch_size", [2, 5, 1000])
def test_h5array_matches_reference_across_batch_sizes(batch_size):
    data = np.random.default_rng(1).random((13, 4))
    arr = H5Array.create(data.shape)
    arr.write_batch(0, data.shape[0], data)

    order, link_mat = utils.compute_linkage(arr, batch_size=batch_size)
    ref_order, ref_link = _reference_linkage(data)
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)


def test_h5array_scratch_dir_is_cleaned_up():
    data = np.random.default_rng(2).random((10, 3))
    arr = H5Array.create(data.shape)
    arr.write_batch(0, data.shape[0], data)

    before = set(os.listdir(tempfile.gettempdir()))
    utils.compute_linkage(arr, batch_size=3)
    after = set(os.listdir(tempfile.gettempdir()))
    leaked = [d for d in after - before if d.startswith("nucmc_")]
    assert not leaked


def test_h5array_respects_explicit_dir(tmp_path):
    data = np.random.default_rng(3).random((6, 3))
    arr = H5Array.create(data.shape)
    arr.write_batch(0, data.shape[0], data)

    utils.compute_linkage(arr, batch_size=2, dir=tmp_path)
    # Only files created by the H5Array source/reorder machinery
    # should remain; the scratch distance matrix must be gone.
    assert all(not f.endswith(".h5") for f in os.listdir(tmp_path))


def test_warns_above_cluster_warn_rows(monkeypatch):
    monkeypatch.setattr(utils, "_CLUSTER_WARN_ROWS", 5)
    data = np.random.default_rng(4).random((8, 3))
    with pytest.warns(UserWarning):
        utils.compute_linkage(data)


def test_no_warning_below_cluster_warn_rows(monkeypatch):
    monkeypatch.setattr(utils, "_CLUSTER_WARN_ROWS", 100)
    data = np.random.default_rng(5).random((8, 3))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        utils.compute_linkage(data)
    assert not any(issubclass(w.category, UserWarning) for w in caught)


@pytest.mark.parametrize("as_h5array", [False, True])
def test_nan_masked_distance_matches_reference(as_h5array):
    data = np.random.default_rng(6).random((9, 5))
    data[1, 3:] = np.nan
    data[4, 0] = np.nan
    data[6, :] = np.nan
    data[6, 2] = 0.5

    source = data
    if as_h5array:
        source = H5Array.create(data.shape)
        source.write_batch(0, data.shape[0], data)

    order, link_mat = utils.compute_linkage(source, batch_size=3)
    ref_order, ref_link = _nan_masked_reference(data)
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)


@pytest.mark.parametrize("as_h5array", [False, True])
def test_non_euclidean_metric_with_nan_raises(as_h5array):
    data = np.random.default_rng(7).random((6, 3))
    data[0, 0] = np.nan

    source = data
    if as_h5array:
        source = H5Array.create(data.shape)
        source.write_batch(0, data.shape[0], data)

    with pytest.raises(ValueError):
        utils.compute_linkage(source, metric="cityblock", batch_size=2)


def test_non_euclidean_metric_without_nan_still_works():
    data = np.random.default_rng(8).random((7, 3))
    order, link_mat = utils.compute_linkage(data, metric="cityblock")
    ref_order, ref_link = _reference_linkage(data, metric="cityblock")
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)


def _filled(data, fill_nan):
    if fill_nan == "mean":
        colmeans = np.nanmean(data, axis=0)
        colmeans = np.where(np.isnan(colmeans), 0.0, colmeans)
        return np.where(np.isnan(data), colmeans, data)
    return np.where(np.isnan(data), fill_nan, data)


@pytest.mark.parametrize("as_h5array", [False, True])
@pytest.mark.parametrize("fill_nan", ["mean", 0.0])
def test_fill_nan_matches_reference(as_h5array, fill_nan):
    data = np.random.default_rng(9).random((9, 5))
    data[1, 3:] = np.nan
    data[4, 0] = np.nan
    data[6, :] = np.nan
    data[6, 2] = 0.5

    source = data
    if as_h5array:
        source = H5Array.create(data.shape)
        source.write_batch(0, data.shape[0], data)

    order, link_mat = utils.compute_linkage(source, batch_size=3,
                                            fill_nan=fill_nan)
    ref_order, ref_link = _reference_linkage(_filled(data, fill_nan))
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)


@pytest.mark.parametrize("as_h5array", [False, True])
@pytest.mark.parametrize("fill_nan", ["mean", 0.0])
def test_fill_nan_handles_zero_overlap_row(as_h5array, fill_nan):
    data = np.array([
        [1.0, 2.0, np.nan],
        [np.nan, np.nan, np.nan],
        [3.0, 4.0, 5.0],
    ])

    source = data
    if as_h5array:
        source = H5Array.create(data.shape)
        source.write_batch(0, data.shape[0], data)

    with pytest.raises(ValueError):
        utils.compute_linkage(source, batch_size=1)

    order, link_mat = utils.compute_linkage(source, batch_size=1,
                                            fill_nan=fill_nan)
    ref_order, ref_link = _reference_linkage(_filled(data, fill_nan))
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)
    assert np.isfinite(link_mat).all()


def test_fill_nan_all_nan_column_falls_back_to_zero():
    data = np.random.default_rng(10).random((6, 3))
    data[:, 1] = np.nan

    order, link_mat = utils.compute_linkage(data, fill_nan="mean")
    ref_order, ref_link = _reference_linkage(_filled(data, "mean"))
    np.testing.assert_array_equal(order, ref_order)
    np.testing.assert_allclose(link_mat, ref_link)


def test_invalid_fill_nan_value_raises():
    data = np.random.default_rng(11).random((5, 3))
    with pytest.raises(ValueError):
        utils.compute_linkage(data, fill_nan="bogus")


@pytest.mark.parametrize("how", ["mean", "sum", "min", "max", "stride"])
def test_downsample_h5array_and_array_paths_agree(how):
    values = np.random.default_rng(9).random((23, 3))
    max_rows = 4

    arr = H5Array.create(values.shape)
    arr.write_batch(0, values.shape[0], values)

    h5_matrix = utils.downsample(arr, max_rows, how=how)
    arr_matrix = utils.downsample(values, max_rows, how=how)
    df_matrix = utils.downsample(pd.DataFrame(values), max_rows, how=how)

    np.testing.assert_allclose(h5_matrix, arr_matrix)
    np.testing.assert_allclose(h5_matrix, df_matrix)


def test_downsample_dispatches_h5array_to_its_own_method():
    values = np.random.default_rng(10).random((10, 3))
    arr = H5Array.create(values.shape)
    arr.write_batch(0, values.shape[0], values)

    expected = arr.downsample(3, how="mean")
    result = utils.downsample(arr, 3, how="mean")
    np.testing.assert_allclose(result, expected)


def test_downsample_noop_when_nrow_within_max_rows():
    values = np.random.default_rng(11).random((5, 2))
    np.testing.assert_array_equal(utils.downsample(values, 10), values)


@pytest.mark.parametrize("kind", ["h5array", "dataframe", "ndarray"])
def test_downsample_invalid_how_raises(kind):
    values = np.random.default_rng(12).random((5, 3))
    data = {"h5array": lambda: _filled_h5array(values),
           "dataframe": lambda: pd.DataFrame(values),
           "ndarray": lambda: values}[kind]()
    with pytest.raises(ValueError):
        utils.downsample(data, 2, how="bogus")


def _filled_h5array(values):
    arr = H5Array.create(values.shape)
    arr.write_batch(0, values.shape[0], values)
    return arr
