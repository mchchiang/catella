# _jagged.py

import numpy as np


def _pack_jagged(arrays, dtype=np.int32):
    """
    Pack a sequence of 1D arrays into flat/offset form.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays to pack.
    dtype : data-type, default np.int32
        The data type of the flat array.

    Returns
    -------
    flat : np.ndarray
        All arrays concatenated, of type `dtype`.
    offset : np.ndarray
        A uint64 array of length `len(arrays) + 1`; array `i` is
        `flat[offset[i]:offset[i + 1]]`.
    """
    offset = np.zeros(len(arrays) + 1, dtype=np.uint64)
    offset[1:] = np.cumsum([len(a) for a in arrays])
    if len(arrays) == 0:
        return np.array([], dtype=dtype), offset
    return np.concatenate(arrays).astype(dtype), offset


def _unpack_jagged(flat, offset):
    """
    Split a flat/offset pair back into its arrays.

    Parameters
    ----------
    flat : np.ndarray
        The concatenated values.
    offset : np.ndarray
        The start of each array in `flat`, followed by the total length.

    Returns
    -------
    tuple of np.ndarray
        The `len(offset) - 1` arrays, with the data type of `flat`.
    """
    return tuple(
        np.asarray(flat[offset[i] : offset[i + 1]], dtype=flat.dtype)
        for i in range(len(offset) - 1)
    )


def _interval_mask(intervals, offset, start, stop, nbp):
    """
    Positions inside the intervals of molecules `start:stop`.

    Parameters
    ----------
    intervals : np.ndarray
        int64, (n_interval, 2), half-open intervals grouped by molecule.
    offset : np.ndarray
        int64, length nmol + 1; the intervals of molecule `i` are
        `intervals[offset[i]:offset[i + 1]]`.
    start, stop : int
        Molecule range.
    nbp : int
        Number of positions.

    Returns
    -------
    np.ndarray
        bool, (stop - start, nbp).
    """
    lo, hi = offset[start], offset[stop]
    mol = np.repeat(np.arange(stop - start), np.diff(offset[start : stop + 1]))
    edges = np.zeros((stop - start, nbp + 1), dtype=np.int64)
    np.add.at(edges, (mol, intervals[lo:hi, 0]), 1)
    np.add.at(edges, (mol, intervals[lo:hi, 1]), -1)
    return np.cumsum(edges, axis=1)[:, :nbp] > 0
