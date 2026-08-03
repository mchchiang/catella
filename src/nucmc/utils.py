# utils.py

import shutil
import warnings
from dataclasses import fields
from typing import List, Iterable, Sequence
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist, cdist, squareform
from . import h5_utils
from .h5_array import H5Array

IndexType = int | slice | Sequence[int]

# Row-count threshold above which compute_linkage() warns, since
# pairwise-distance clustering needs an nrow x nrow matrix regardless
# of how it is built.
_CLUSTER_WARN_ROWS = 5000

def add_frozen_properties(cls):
    keyword = "_frozen_"
    for f in fields(cls):
        if not f.name.startswith(keyword):
            continue
        
        public_name = f.name.removeprefix(keyword)
        private_name = f.name
        doc_str = f.metadata.get("doc", "")
        
        def make_getter(p_name):
            def getter(self):
                val = getattr(self, p_name)
                # NumPy: return read-only view
                if isinstance(val, np.ndarray):
                    view = val.view()
                    view.flags.writeable = False
                    return view
                # Pandas 3.0: standard copy is O(1) and safe
                if isinstance(val, (pd.DataFrame, pd.Series)):
                    return val.copy()
                # Nested lists/tuples: convert to tuple of views
                if isinstance(val, (tuple, list)):
                    return tuple(v.view() if isinstance(v, np.ndarray)
                                 else v for v in val)
                return val
            return getter
        # Attach property + docstring. Properties don't need a slot.
        prop = property(make_getter(private_name))
        prop.__doc__ = doc_str
        setattr(cls, public_name, prop)
    return cls

# Standard utility to normalize chromosome inputs into a list
def normalize_chroms(chroms: str | Iterable[str] | None = None, 
                     default_chroms: Iterable[str] | None = None) -> List[str]:
    if chroms is None:
        return list(default_chroms) if default_chroms is not None else []
    if isinstance(chroms, str):
        if chroms == "":
            raise ValueError("chroms must not be an empty string.")
        return [chroms]
    if not isinstance(chroms, Iterable):
        raise TypeError("chroms must be a str or an iterable.")
    return list(chroms)

def compute_linkage(matrix, *, metric="euclidean", method="ward",
                    batch_size=20000, dir=None):
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

    Returns
    -------
    order : np.ndarray
        Row indices of `matrix`, in leaf order.
    link_mat : np.ndarray
        The linkage matrix, as returned by
        `scipy.cluster.hierarchy.linkage`.

    Warns
    -----
    UserWarning
        If `matrix` has more than `_CLUSTER_WARN_ROWS` rows: clustering
        needs the full `nrow x nrow` pairwise-distance matrix no
        matter how it is built or stored, so this may be slow and
        memory-intensive. Downsample first (e.g. via `H5Array.downsample`)
        to avoid it.
    """
    nrow = matrix.shape[0]
    if nrow > _CLUSTER_WARN_ROWS:
        warnings.warn(
            f"Clustering {nrow} rows requires an {nrow}x{nrow} pairwise "
            "distance matrix; this may be slow and memory-intensive. "
            "Consider downsampling first (e.g. H5Array.downsample).",
            stacklevel=2)

    if isinstance(matrix, H5Array):
        dist_vec = _streamed_pdist(matrix, metric=metric,
                                   batch_size=batch_size, dir=dir)
    else:
        dist_vec = pdist(np.asarray(matrix), metric=metric)

    link_mat = sch.linkage(dist_vec, method=method)
    order = sch.leaves_list(link_mat)
    return order, link_mat

def _streamed_pdist(h5arr, *, metric, batch_size, dir):
    # Builds the nrow x nrow distance matrix on disk in row-blocks (so
    # peak memory is O(batch_size * nrow), not O(nrow^2)) and only
    # materializes it once, transiently, to hand to squareform/linkage
    # (which have no disk-backed mode of their own).
    made_own_dir = dir is None
    if made_own_dir:
        dir = h5_utils.fresh_tmp_dir()

    nrow = h5arr.shape[0]
    dist = H5Array.create((nrow, nrow), dtype=np.float64, dir=dir)
    for a0 in range(0, nrow, batch_size):
        a1 = min(a0 + batch_size, nrow)
        block_a = h5arr[a0:a1, :]
        row_block = np.empty((a1 - a0, nrow), dtype=np.float64)
        for b0 in range(0, nrow, batch_size):
            b1 = min(b0 + batch_size, nrow)
            if b0 == a0:
                row_block[:, b0:b1] = 0.0 if a1 - a0 == 1 else \
                    squareform(pdist(block_a, metric=metric))
            else:
                block_b = h5arr[b0:b1, :]
                row_block[:, b0:b1] = cdist(block_a, block_b, metric=metric)
        dist.write_batch(a0, a1, row_block)
    dense = dist.to_numpy()
    dist.close()
    if made_own_dir:
        shutil.rmtree(dir, ignore_errors=True)
    return squareform(dense, checks=False)
