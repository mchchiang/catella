# utils.py

import shutil
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import cdist, pdist, squareform

from catella import h5_utils
from catella._warnings import _warn
from catella.h5_array import _DOWNSAMPLE_HOW, H5Array

IndexType = int | slice | Sequence[int]

# Row-count threshold above which compute_linkage() warns, since
# pairwise-distance clustering needs an nrow x nrow matrix regardless
# of how it is built.
_CLUSTER_WARN_ROWS = 5000


def downsample(arr, max_rows, *, how="mean", batch_size=20000):
    """
    Collapse rows of a dense array to at most `max_rows`.

    Parameters
    ----------
    arr : H5Array, pd.DataFrame, or np.ndarray
        2D data to collapse (rows=molecules, columns=bp position).
        `H5Array` input is dispatched to its own `.downsample`
        (streamed from disk); other input is binned in memory.
    max_rows : int
        Target number of rows.
    how : {"mean", "sum", "min", "max", "stride"}, default "mean"
        How to collapse groups of consecutive rows into one row.
        "stride" reads every k-th row instead of aggregating bins.
    batch_size : int, default 20000
        Number of rows read per streamed chunk. Only used when `arr`
        is an `H5Array`.

    Returns
    -------
    np.ndarray
        Array of shape (min(nrow, max_rows), ncol).

    Raises
    ------
    ValueError
        If `how` is not a recognized option.
    """
    if isinstance(arr, H5Array):
        return arr.downsample(max_rows, how=how, batch_size=batch_size)
    if how not in _DOWNSAMPLE_HOW:
        raise ValueError(f"'how' must be one of {_DOWNSAMPLE_HOW}")

    dense = (
        arr.to_numpy() if isinstance(arr, pd.DataFrame) else np.asarray(arr)
    )
    nrow = dense.shape[0]
    if nrow <= max_rows:
        return dense
    if how == "stride":
        step = -(-nrow // max_rows)  # ceil div
        return dense[0:nrow:step]
    edges = _bin_edges(nrow, max_rows)
    return np.stack(
        [
            _reduce_rows(dense[edges[i] : edges[i + 1]], how)
            for i in range(len(edges) - 1)
        ]
    )


def _bin_edges(n, max_rows):
    n_bins = min(n, max_rows)
    return np.linspace(0, n, n_bins + 1).astype(int)


def _reduce_rows(block, how):
    if block.shape[0] == 0:
        return np.full(block.shape[1], np.nan)
    fn = {
        "mean": np.nanmean,
        "sum": np.nansum,
        "min": np.nanmin,
        "max": np.nanmax,
    }[how]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return fn(block, axis=0)


def compute_linkage(
    matrix,
    *,
    metric="euclidean",
    method="ward",
    batch_size=20000,
    dir=None,
    fill_nan=None,
):
    """
    Compute a hierarchical-clustering leaf order for a matrix's rows.

    Parameters
    ----------
    matrix : H5Array, pd.DataFrame, or np.ndarray
        2D data whose rows are to be clustered. If an `H5Array`, the
        pairwise-distance matrix is built from disk in row batches
        rather than materializing `matrix` up front.
    metric : str, default "euclidean"
        Distance metric, forwarded to `scipy.spatial.distance.pdist`.
    method : str, default "ward"
        Linkage method, forwarded to `scipy.cluster.hierarchy.linkage`.
    batch_size : int, default 20000
        Number of rows read (and held in memory) per batch. Only used
        when `matrix` is an `H5Array`.
    dir : str or pathlib.Path, optional
        Directory for the scratch file backing the intermediate
        distance matrix. Only used when `matrix` is an `H5Array`. If
        None, a fresh directory is created and removed again once
        this call returns.
    fill_nan : {"mean"}, float, or None, default None
        How to handle `nan` values before computing distances. If
        None (default), `nan` is instead handled by masking (see
        Notes below), which can leave some pairwise distances
        undefined. If "mean", each `nan` is replaced by its column's
        mean (ignoring `nan`); columns that are entirely `nan` fall
        back to 0.0. If a number, every `nan` is replaced by that
        constant. Either mode guarantees finite pairwise distances.

    Returns
    -------
    order : np.ndarray
        Row indices of `matrix`, in leaf order.
    link_mat : np.ndarray
        The linkage matrix, as returned by
        `scipy.cluster.hierarchy.linkage`.

    Raises
    ------
    ValueError
        If `fill_nan` is invalid, or if `fill_nan` is None and some row is
        nan in every column (its distances are undefined).

    Warns
    -----
    UserWarning
        If `matrix` has more than `_CLUSTER_WARN_ROWS` rows: clustering
        needs the full `nrow x nrow` pairwise-distance matrix no
        matter how it is built or stored, so this may be slow and
        memory-intensive. Downsample first (e.g. via `H5Array.downsample`)
        to avoid it.

    Notes
    -----
    By default (`fill_nan=None`), rows with `nan` (e.g. missing
    coverage, or unfilled `smooth` edges) are handled by masking:
    each distance uses only the jointly non-`nan` columns of that
    pair, scaled to the full column count. Only implemented for
    `metric="euclidean"`; other metrics raise `ValueError` if any
    row-block being compared contains `nan`. Rows that are entirely
    `nan` (e.g. molecules with no data) are rejected up front. A pair
    of partly-`nan` rows that shares no non-`nan` column still has an
    undefined distance, which `scipy.cluster.hierarchy.linkage` rejects
    with `ValueError`; use `fill_nan` to avoid both.
    """
    if (
        fill_nan is not None
        and fill_nan != "mean"
        and not isinstance(fill_nan, (int, float))
    ):
        raise ValueError(
            f"fill_nan must be None, 'mean', or a number, got {fill_nan!r}"
        )

    nrow = matrix.shape[0]
    if fill_nan is None:
        n_empty = _count_nan_rows(matrix, batch_size)
        if n_empty:
            raise ValueError(
                f"{n_empty} of {nrow} rows have no data (all nan), so "
                "their distances are undefined; pass fill_nan"
            )
    if nrow > _CLUSTER_WARN_ROWS:
        _warn(
            f"clustering {nrow} rows requires an {nrow}x{nrow} pairwise "
            "distance matrix; this may be slow and memory-intensive; "
            "consider downsampling first (e.g. H5Array.downsample)"
        )

    if isinstance(matrix, H5Array):
        dist_vec = _streamed_pdist(
            matrix,
            metric=metric,
            batch_size=batch_size,
            dir=dir,
            fill_nan=fill_nan,
        )
    else:
        dense = np.asarray(matrix)
        if fill_nan is not None:
            dense = _apply_fill_nan(dense, fill_nan)
            has_nan = False
        else:
            has_nan = np.isnan(dense).any()
        dist_vec = pdist(dense, metric=_resolve_metric(metric, has_nan))

    link_mat = sch.linkage(dist_vec, method=method)
    order = sch.leaves_list(link_mat)
    return order, link_mat


def _count_nan_rows(matrix, batch_size):
    """
    Count the rows of `matrix` that are nan in every column.

    Parameters
    ----------
    matrix : H5Array, pd.DataFrame, or np.ndarray
        2D data.
    batch_size : int
        Number of rows read per batch when `matrix` is an `H5Array`.

    Returns
    -------
    int
        The number of all-nan rows.
    """
    if not isinstance(matrix, H5Array):
        dense = np.asarray(matrix, dtype=np.float64)
        return int(np.isnan(dense).all(axis=1).sum())
    count = 0
    for start in range(0, matrix.shape[0], batch_size):
        stop = min(start + batch_size, matrix.shape[0])
        block = np.asarray(matrix[start:stop, :], dtype=np.float64)
        count += int(np.isnan(block).all(axis=1).sum())
    return count


def _nan_euclidean(u, v):
    # Euclidean distance using only jointly-observed columns, scaled
    # up to the full column count -- matches
    # sklearn.metrics.pairwise.nan_euclidean_distances.
    mask = ~np.isnan(u) & ~np.isnan(v)
    n_present = mask.sum()
    if n_present == 0:
        return np.nan
    sq = np.sum((u[mask] - v[mask]) ** 2)
    return np.sqrt(len(u) / n_present * sq)


def _resolve_metric(metric, has_nan):
    if not has_nan:
        return metric
    if metric != "euclidean":
        raise ValueError(
            "NaN-masked distances are only supported for "
            f"metric='euclidean', got {metric!r} with nan present"
        )
    return _nan_euclidean


def _apply_fill_nan(dense, fill_nan):
    if fill_nan == "mean":
        # All-nan columns intentionally fall back to 0.0 below; silence
        # numpy's "Mean of empty slice" warning for that expected case.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            colmeans = np.nanmean(dense, axis=0)
        fill_values = np.where(np.isnan(colmeans), 0.0, colmeans)
    else:
        fill_values = fill_nan
    return np.where(np.isnan(dense), fill_values, dense)


def _streamed_pdist(h5arr, *, metric, batch_size, dir, fill_nan=None):
    # Builds the nrow x nrow distance matrix on disk in row-blocks (so
    # peak memory is O(batch_size * nrow), not O(nrow^2)) and only
    # materializes it once, transiently, to hand to squareform/linkage
    # (which have no disk-backed mode of their own).
    made_own_dir = dir is None
    if made_own_dir:
        dir = h5_utils.fresh_tmp_dir()

    dist = None
    try:
        fill_values = None
        if fill_nan == "mean":
            # Means over all rows (as the dense path), 0 for all-nan columns
            means = h5arr.mean(axis=0, batch_size=batch_size)
            fill_values = np.where(np.isnan(means), 0.0, means)
        elif fill_nan is not None:
            fill_values = fill_nan

        nrow = h5arr.shape[0]
        dist = H5Array.create((nrow, nrow), dtype=np.float64, dir=dir)
        for a0 in range(0, nrow, batch_size):
            a1 = min(a0 + batch_size, nrow)
            block_a = h5arr[a0:a1, :]
            if fill_values is not None:
                block_a = np.where(np.isnan(block_a), fill_values, block_a)
                has_nan_a = False
            else:
                has_nan_a = np.isnan(block_a).any()
            row_block = np.empty((a1 - a0, nrow), dtype=np.float64)
            for b0 in range(0, nrow, batch_size):
                b1 = min(b0 + batch_size, nrow)
                if b0 == a0:
                    if a1 - a0 == 1:
                        row_block[:, b0:b1] = 0.0
                    else:
                        m = _resolve_metric(metric, has_nan_a)
                        row_block[:, b0:b1] = squareform(
                            pdist(block_a, metric=m)
                        )
                else:
                    block_b = h5arr[b0:b1, :]
                    if fill_values is not None:
                        block_b = np.where(
                            np.isnan(block_b), fill_values, block_b
                        )
                        has_nan_b = False
                    else:
                        has_nan_b = np.isnan(block_b).any()
                    m = _resolve_metric(metric, has_nan_a or has_nan_b)
                    row_block[:, b0:b1] = cdist(block_a, block_b, metric=m)
            dist.write_batch(a0, a1, row_block)
        dense = dist.to_numpy()
        return squareform(dense, checks=False)
    finally:
        if dist is not None:
            dist.close()
        if made_own_dir:
            shutil.rmtree(dir, ignore_errors=True)
