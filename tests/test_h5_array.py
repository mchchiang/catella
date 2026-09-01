# test_h5_array.py

from pathlib import Path

import numpy as np
import pytest

from catella.h5_array import H5Array


def test_write_batch_and_read_back():
    arr = H5Array.create((10, 4))
    values = np.arange(40, dtype=np.float64).reshape(10, 4)
    arr.write_batch(0, 5, values[0:5])
    arr.write_batch(5, 10, values[5:10])
    np.testing.assert_array_equal(arr.to_numpy(), values)
    assert arr.shape == (10, 4)
    assert arr.dtype == np.float64


def test_getitem_slicing():
    arr = H5Array.create((6, 3))
    values = np.arange(18, dtype=np.float64).reshape(6, 3)
    arr.write_batch(0, 6, values)
    np.testing.assert_array_equal(arr[2:4, :], values[2:4, :])


def test_scratch_file_cleanup_on_close():
    arr = H5Array.create((3, 3))
    path = Path(arr.path)
    assert path.exists()
    arr.close()
    assert not path.exists()


def test_permanent_path_survives_close(tmp_path):
    path = tmp_path / "permanent.h5"
    arr = H5Array.create((3, 3), path=path)
    arr.write_batch(0, 3, np.ones((3, 3)))
    arr.close()
    assert path.exists()


def test_save_to_and_load_from_round_trip(tmp_path):
    import h5py

    src = H5Array.create((4, 2))
    values = np.arange(8, dtype=np.float64).reshape(4, 2)
    src.write_batch(0, 4, values)

    dest_path = tmp_path / "dest.h5"
    with h5py.File(dest_path, "w") as f:
        group = f.create_group("analysis")
        src.save_to(group, "test_smoothed")

    loaded = H5Array.load_from(dest_path, "analysis/test_smoothed")
    np.testing.assert_array_equal(loaded.to_numpy(), values)


def test_create_with_dir_uses_given_directory(tmp_path):
    arr = H5Array.create((2, 2), dir=tmp_path)
    assert Path(arr.path).parent == tmp_path
    arr.close()


def _filled(shape, seed=0):
    rng = np.random.default_rng(seed)
    values = rng.random(shape)
    arr = H5Array.create(shape)
    arr.write_batch(0, shape[0], values)
    return arr, values


def test_repr_mentions_shape_and_dtype():
    arr, _ = _filled((5, 4))
    text = repr(arr)
    assert "(5, 4)" in text
    assert "float64" in text


def test_repr_truncates_like_numpy_for_large_arrays():
    nrow, ncol = 50, 8
    values = np.arange(nrow * ncol, dtype=np.float64).reshape(nrow, ncol)
    arr = H5Array.create((nrow, ncol))
    arr.write_batch(0, nrow, values)

    text = repr(arr)
    assert "..." in text
    expected_body = np.array2string(values, threshold=0, edgeitems=3)
    assert text.endswith(expected_body)


def test_repr_no_truncation_for_small_arrays():
    values = np.arange(12, dtype=np.float64).reshape(4, 3)
    arr = H5Array.create((4, 3))
    arr.write_batch(0, 4, values)

    text = repr(arr)
    assert "..." not in text
    expected_body = np.array2string(values, threshold=0, edgeitems=3)
    assert text.endswith(expected_body)


def test_iloc_matches_getitem():
    arr, values = _filled((6, 3))
    np.testing.assert_array_equal(arr.iloc[1:4, :], values[1:4, :])
    np.testing.assert_array_equal(arr.iloc[:, 0], values[:, 0])


def test_array_protocol_round_trips():
    arr, values = _filled((4, 4))
    np.testing.assert_array_equal(np.asarray(arr), values)
    assert np.asarray(arr, dtype=np.float32).dtype == np.float32


def test_len_returns_row_count():
    arr, _ = _filled((7, 2))
    assert len(arr) == 7


def test_to_pandas_default_range_index():
    arr, values = _filled((3, 2))
    df = arr.to_pandas()
    np.testing.assert_array_equal(df.to_numpy(), values)
    assert list(df.index) == [0, 1, 2]
    assert list(df.columns) == [0, 1]


def test_to_pandas_uses_explicit_labels():
    values = np.arange(6, dtype=np.float64).reshape(3, 2)
    arr = H5Array.create((3, 2), index=["m0", "m1", "m2"],
                         columns=["x", "y"])
    arr.write_batch(0, 3, values)
    df = arr.to_pandas()
    assert list(df.index) == ["m0", "m1", "m2"]
    assert list(df.columns) == ["x", "y"]
    np.testing.assert_array_equal(df.to_numpy(), values)


def test_labels_survive_save_to_and_load_from(tmp_path):
    import h5py

    values = np.arange(6, dtype=np.float64).reshape(3, 2)
    src = H5Array.create((3, 2), index=["a", "b", "c"], columns=["x", "y"])
    src.write_batch(0, 3, values)

    dest_path = tmp_path / "dest.h5"
    with h5py.File(dest_path, "w") as f:
        group = f.create_group("analysis")
        src.save_to(group, "meth_prob")

    loaded = H5Array.load_from(dest_path, "analysis/meth_prob")
    assert list(loaded.index) == ["a", "b", "c"]
    assert list(loaded.columns) == ["x", "y"]
    np.testing.assert_array_equal(loaded.to_numpy(), values)


@pytest.mark.parametrize("batch_size", [2, 1000])
@pytest.mark.parametrize("op", ["mean", "sum", "min", "max"])
def test_streaming_reductions_match_numpy(op, batch_size):
    arr, values = _filled((9, 5), seed=1)
    numpy_op = getattr(np, op)
    for axis in (0, 1):
        result = getattr(arr, op)(axis=axis, batch_size=batch_size)
        expected = numpy_op(values, axis=axis)
        np.testing.assert_allclose(result, expected)


def test_downsample_noop_when_nrow_within_max_rows():
    arr, values = _filled((5, 3), seed=2)
    np.testing.assert_array_equal(arr.downsample(10), values)
    np.testing.assert_array_equal(arr.downsample(5), values)


@pytest.mark.parametrize("batch_size", [1, 3, 1000])
@pytest.mark.parametrize("op", ["mean", "sum", "min", "max"])
def test_downsample_aggregate_matches_manual_bins(op, batch_size):
    nrow, ncol, max_rows = 10, 4, 3
    values = np.arange(nrow * ncol, dtype=np.float64).reshape(nrow, ncol)
    arr = H5Array.create((nrow, ncol))
    arr.write_batch(0, nrow, values)

    result = arr.downsample(max_rows, how=op, batch_size=batch_size)
    assert result.shape == (max_rows, ncol)

    edges = np.linspace(0, nrow, max_rows + 1).astype(int)
    numpy_op = getattr(np, op)
    expected = np.stack([numpy_op(values[edges[i]:edges[i+1]], axis=0)
                         for i in range(max_rows)])
    np.testing.assert_allclose(result, expected)


def test_downsample_stride_selects_expected_rows():
    nrow, ncol, max_rows = 10, 3, 3
    values = np.arange(nrow * ncol, dtype=np.float64).reshape(nrow, ncol)
    arr = H5Array.create((nrow, ncol))
    arr.write_batch(0, nrow, values)

    result = arr.downsample(max_rows, how="stride")
    step = -(-nrow // max_rows)
    np.testing.assert_array_equal(result, values[0:nrow:step, :])


def test_downsample_invalid_how_raises():
    arr, _ = _filled((5, 2), seed=3)
    with pytest.raises(ValueError):
        arr.downsample(2, how="bogus")


def test_downsample_nan_aware_within_a_bin():
    nrow, ncol, max_rows = 4, 2, 2
    values = np.array([[1.0, np.nan],
                       [3.0, 4.0],
                       [5.0, 6.0],
                       [7.0, 8.0]])
    arr = H5Array.create((nrow, ncol))
    arr.write_batch(0, nrow, values)

    result = arr.downsample(max_rows, how="mean")
    expected = np.stack([np.nanmean(values[0:2], axis=0),
                         np.nanmean(values[2:4], axis=0)])
    np.testing.assert_allclose(result, expected)


def test_downsample_all_nan_bin_produces_nan_without_raising():
    nrow, ncol, max_rows = 4, 2, 2
    values = np.array([[np.nan, np.nan],
                       [np.nan, np.nan],
                       [7.0, 8.0],
                       [9.0, 10.0]])
    arr = H5Array.create((nrow, ncol))
    arr.write_batch(0, nrow, values)

    result = arr.downsample(max_rows, how="mean")
    assert np.all(np.isnan(result[0]))
    np.testing.assert_allclose(result[1], [8.0, 9.0])


def _sparse_array(nrow, ncol, frac_observed, seed):
    rng = np.random.default_rng(seed)
    values = np.full((nrow, ncol), np.nan)
    n_observed = int(frac_observed * nrow * ncol)
    idx = rng.choice(nrow * ncol, size=n_observed, replace=False)
    values.flat[idx] = rng.random(n_observed)
    return values


def _dense_correlated_array(nrow, ncol):
    x = np.linspace(0, 4 * np.pi, ncol)
    row = np.sin(x)
    return np.tile(row, (nrow, 1))


def test_default_compression_shrinks_sparse_array(tmp_path):
    values = _sparse_array(500, 200, frac_observed=0.02, seed=0)

    gz_path = tmp_path / "gz.h5"
    gz = H5Array.create(values.shape, path=gz_path)
    gz.write_batch(0, values.shape[0], values)
    gz.close()

    none_path = tmp_path / "none.h5"
    none = H5Array.create(values.shape, path=none_path, compression=None)
    none.write_batch(0, values.shape[0], values)
    none.close()

    gz_size = gz_path.stat().st_size
    none_size = none_path.stat().st_size
    assert gz_size * 5 < none_size


def test_default_compression_round_trips_values_exactly():
    values = _sparse_array(50, 20, frac_observed=0.3, seed=1)
    arr = H5Array.create(values.shape)
    arr.write_batch(0, values.shape[0], values)
    np.testing.assert_array_equal(arr.to_numpy(), values)


def test_compression_none_disables_compression_and_chunking():
    arr = H5Array.create((10, 5), compression=None)
    assert arr._dataset.compression is None
    assert arr._dataset.chunks is None


def test_compression_and_chunks_can_be_overridden():
    arr = H5Array.create((10, 5), compression="lzf", chunks=(3, 5))
    assert arr._dataset.compression == "lzf"
    assert arr._dataset.chunks == (3, 5)


def test_shuffle_false_disables_shuffle_filter():
    arr = H5Array.create((10, 5), shuffle=False)
    assert arr._dataset.shuffle is False


def test_shuffle_reduces_size_for_dense_correlated_array(tmp_path):
    values = _dense_correlated_array(2000, 300)

    shuffled_path = tmp_path / "shuffled.h5"
    shuffled = H5Array.create(values.shape, path=shuffled_path,
                              shuffle=True)
    shuffled.write_batch(0, values.shape[0], values)
    shuffled.close()

    unshuffled_path = tmp_path / "unshuffled.h5"
    unshuffled = H5Array.create(values.shape, path=unshuffled_path,
                                shuffle=False)
    unshuffled.write_batch(0, values.shape[0], values)
    unshuffled.close()

    assert shuffled_path.stat().st_size < unshuffled_path.stat().st_size


def test_save_to_and_load_from_preserves_compression(tmp_path):
    import h5py

    values = _sparse_array(50, 20, frac_observed=0.3, seed=2)
    src = H5Array.create(values.shape)
    src.write_batch(0, values.shape[0], values)

    dest_path = tmp_path / "dest.h5"
    with h5py.File(dest_path, "w") as f:
        group = f.create_group("analysis")
        src.save_to(group, "test_raw")

    loaded = H5Array.load_from(dest_path, "analysis/test_raw")
    assert loaded._dataset.compression is not None
    np.testing.assert_array_equal(loaded.to_numpy(), values)


@pytest.mark.parametrize("shape", [(0, 5), (5, 0)])
def test_create_with_degenerate_shape_does_not_raise(shape):
    arr = H5Array.create(shape)
    assert arr.shape == shape


def test_h5py_dataset_rejects_decreasing_order_indices():
    # Regression/documentation test: this is the exact limitation
    # reorder_rows works around -- a plain HDF5 dataset only accepts
    # strictly increasing fancy-index arrays.
    arr, _ = _filled((5, 3))
    with pytest.raises(TypeError):
        arr._dataset[np.array([3, 0, 4, 1])]


def test_reorder_rows_matches_fancy_indexing_for_increasing_order():
    arr, values = _filled((6, 3))
    order = np.array([1, 3, 4])
    result = arr.reorder_rows(order)
    np.testing.assert_array_equal(result.to_numpy(), values[order])


def test_reorder_rows_handles_arbitrary_permutation():
    arr, values = _filled((6, 3))
    order = np.array([3, 0, 4, 1, 2, 5])
    result = arr.reorder_rows(order)
    np.testing.assert_array_equal(result.to_numpy(), values[order])


def test_reorder_rows_supports_repeated_indices():
    arr, values = _filled((5, 2))
    order = np.array([0, 0, 2])
    result = arr.reorder_rows(order)
    np.testing.assert_array_equal(result.to_numpy(), values[order])


@pytest.mark.parametrize("batch_size", [1, 3, 1000])
def test_reorder_rows_batch_size_does_not_affect_result(batch_size):
    arr, values = _filled((7, 4), seed=5)
    order = np.array([5, 0, 6, 2, 1, 4, 3])
    result = arr.reorder_rows(order, batch_size=batch_size)
    np.testing.assert_array_equal(result.to_numpy(), values[order])


def test_reorder_rows_result_is_new_independent_array():
    arr, _ = _filled((4, 2))
    result = arr.reorder_rows([2, 0, 1, 3])
    assert isinstance(result, H5Array)
    assert result.path != arr.path


def test_reorder_rows_preserves_column_labels_and_permutes_index_labels():
    values = np.arange(8, dtype=np.float64).reshape(4, 2)
    arr = H5Array.create((4, 2), index=["a", "b", "c", "d"],
                         columns=["x", "y"])
    arr.write_batch(0, 4, values)
    result = arr.reorder_rows([3, 1, 0])
    assert list(result.index) == ["d", "b", "a"]
    assert list(result.columns) == ["x", "y"]
    np.testing.assert_array_equal(result.to_numpy(), values[[3, 1, 0]])


def test_reorder_rows_empty_order_produces_empty_array():
    arr, _ = _filled((4, 3))
    result = arr.reorder_rows([])
    assert result.shape == (0, 3)


def test_reorder_rows_respects_dir_kwarg(tmp_path):
    arr, _ = _filled((4, 2))
    result = arr.reorder_rows([1, 0, 2, 3], dir=tmp_path)
    assert Path(result.path).parent == tmp_path
