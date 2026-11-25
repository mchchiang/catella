# utils.py

import shutil
import warnings
from collections.abc import Sequence

import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import cdist, pdist, squareform

from catella import h5_utils
from catella._batches import _row_batches
from catella._warnings import _warn
from catella.h5_array import H5Array, _downsample_rows

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
        (streamed from disk); other input is reduced in memory.
    max_rows : int
        Target number of rows.
    how : {"mean", "sum", "min", "max", "stride"}, default "mean"
        How to collapse groups of consecutive rows into one row.
        "stride" reads every k-th row instead of aggregating bins.
    batch_size : int, default 20000
        Number of rows reduced at a time within a bin, when `how`
        aggregates bins. Ignored for "stride".

    Returns
    -------
    np.ndarray
        Array of shape (min(nrow, max_rows), ncol). If nrow <=
        `max_rows`, the data are returned as given. Otherwise, "mean"
        keeps a float dtype and gives float64 for other input; "sum"
        follows `np.nansum` (e.g., int8 gives int64); "min", "max", and
        "stride" keep the input dtype.

    Raises
    ------
    ValueError
        If `how` is not a recognized option, or if `max_rows` or
        `batch_size` is not positive.
    """
    if isinstance(arr, H5Array):
        return arr.downsample(max_rows, how=how, batch_size=batch_size)
    dense = (
        arr.to_numpy() if isinstance(arr, pd.DataFrame) else np.asarray(arr)
    )
    return _downsample_rows(dense, max_rows, how, batch_size)


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
        Notes below). If "mean", each `nan` is replaced by its column's
        mean (ignoring `nan`); columns that are entirely `nan` fall
        back to 0.0. If a number, every `nan` is replaced by that
        constant.

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
        If `fill_nan` is invalid, or if `nan` is present with
        `fill_nan` None and a metric other than "euclidean".

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
    By default (`fill_nan=None`), rows with `nan` (e.g. unspanned
    positions, or unfilled `smooth` edges) are handled by masking:
    each distance uses only the jointly non-`nan` columns of that
    pair, scaled to the full column count. Only implemented for
    `metric="euclidean"`; other metrics raise `ValueError` if any
    row-block being compared contains `nan`. Pairs that share no
    non-`nan` column, including rows that are entirely `nan` (e.g.
    molecules with no data), get the largest defined distance, so
    unknown similarity counts as dissimilar.
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
        dense = np.asarray(matrix, dtype=np.float64)
        if fill_nan is not None:
            dense = _apply_fill_nan(dense, fill_nan)
            has_nan = False
        else:
            has_nan = np.isnan(dense).any()
        _check_nan_metric(metric, has_nan)
        if has_nan:
            full = _nan_euclidean_block(dense, dense)
            np.fill_diagonal(full, 0.0)
            dist_vec = squareform(full, checks=False)
        else:
            dist_vec = pdist(dense, metric=metric)
    dist_vec = _fill_undefined(dist_vec)

    link_mat = sch.linkage(dist_vec, method=method)
    order = sch.leaves_list(link_mat)
    return order, link_mat


def _nan_euclidean_block(a, b):
    """
    Euclidean distances between the rows of `a` and `b`, using only
    jointly observed columns, scaled up to the full column count (as
    sklearn's `nan_euclidean_distances`); nan for pairs that share no
    observed column.

    Parameters
    ----------
    a, b : np.ndarray
        float, (m, p) and (n, p).

    Returns
    -------
    np.ndarray
        float64, (m, n).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    mask_a = (~np.isnan(a)).astype(np.float64)
    mask_b = (~np.isnan(b)).astype(np.float64)
    a0, b0 = np.nan_to_num(a), np.nan_to_num(b)
    sq = (a0 * a0) @ mask_b.T + mask_a @ (b0 * b0).T - 2.0 * (a0 @ b0.T)
    count = mask_a @ mask_b.T
    with np.errstate(invalid="ignore", divide="ignore"):
        d2 = a.shape[1] / count * np.maximum(sq, 0.0)
    return np.where(count > 0, np.sqrt(d2), np.nan)


def _check_nan_metric(metric, has_nan):
    """Raise if nan-masked distances are needed for another metric."""
    if has_nan and metric != "euclidean":
        raise ValueError(
            "NaN-masked distances are only supported for "
            f"metric='euclidean', got {metric!r} with nan present"
        )


def _block_distances(a, b, metric, has_nan):
    """Distances between the rows of `a` and `b` (nan-masked if needed)."""
    _check_nan_metric(metric, has_nan)
    if has_nan:
        return _nan_euclidean_block(a, b)
    return cdist(a, b, metric=metric)


def _fill_undefined(dist_vec):
    """
    Replace undefined (nan) distances by the largest defined one, in
    place: pairs that share no observed column count as dissimilar.
    """
    undefined = np.isnan(dist_vec)
    if undefined.any():
        finite = dist_vec[~undefined]
        dist_vec[undefined] = finite.max() if finite.size else 0.0
    return dist_vec


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
        for a0, a1 in _row_batches(0, nrow, batch_size):
            block_a = h5arr[a0:a1, :]
            if fill_values is not None:
                block_a = np.where(np.isnan(block_a), fill_values, block_a)
                has_nan_a = False
            else:
                has_nan_a = np.isnan(block_a).any()
            row_block = np.empty((a1 - a0, nrow), dtype=np.float64)
            for b0, b1 in _row_batches(0, nrow, batch_size):
                if b0 == a0:
                    block_b, has_nan_b = block_a, has_nan_a
                else:
                    block_b = h5arr[b0:b1, :]
                    if fill_values is not None:
                        block_b = np.where(
                            np.isnan(block_b), fill_values, block_b
                        )
                        has_nan_b = False
                    else:
                        has_nan_b = np.isnan(block_b).any()
                row_block[:, b0:b1] = _block_distances(
                    block_a, block_b, metric, has_nan_a or has_nan_b
                )
                if b0 == a0:
                    np.fill_diagonal(row_block[:, b0:b1], 0.0)
            dist.write_batch(a0, a1, row_block)
        dense = dist.to_numpy()
        return squareform(dense, checks=False)
    finally:
        if dist is not None:
            dist.close()
        if made_own_dir:
            shutil.rmtree(dir, ignore_errors=True)
