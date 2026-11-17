# _plotting.py

import numpy as np
import scipy.cluster.hierarchy as sch

from catella._warnings import _warn

# Estimated peak memory of rendering and saving one heatmap cell
# (measured with matplotlib 3.10 Agg: ~70 B PNG, ~30 B PDF, whatever
# the input dtype)
_RENDER_BYTES_PER_CELL = 80

# Warn when rendering a heatmap is estimated to need more than this
_RENDER_WARN_BYTES = 10**9


def _warn_if_large(nrow, ncol, hint):
    """
    Warn if rendering an `nrow` x `ncol` heatmap may need a lot of memory.

    Parameters
    ----------
    nrow, ncol : int
        The shape of the heatmap as drawn.
    hint : str
        How to make the plot smaller, appended to the warning.
    """
    nbytes = nrow * ncol * _RENDER_BYTES_PER_CELL
    if nbytes > _RENDER_WARN_BYTES:
        _warn(
            f"plotting {nrow} rows x {ncol} bp needs about "
            f"{nbytes / 1e9:.1f} GB of memory to render; {hint}"
        )


def _leaf_order(link_mat, nrow, mols=None):
    """
    Return the row order of a linkage matrix, checked against the data.

    Parameters
    ----------
    link_mat : np.ndarray or pd.DataFrame
        A linkage matrix, as returned (or stored) by `sort_by_linkage`.
    nrow : int
        The number of rows (molecules) of the plotted data.
    mols : iterable of int, optional
        The plot's molecule subset; must be None with a linkage, which
        covers every molecule.

    Returns
    -------
    link_mat : np.ndarray
        The linkage matrix as a float array.
    order : np.ndarray
        The row indices in leaf order.

    Raises
    ------
    ValueError
        If `mols` is given, or if the linkage has a different number of
        leaves than `nrow`.
    """
    if mols is not None:
        raise ValueError(
            "mols cannot be combined with link_mat, which covers every "
            "molecule; use max_rows to show fewer rows"
        )
    link_mat = np.asarray(link_mat, dtype=np.float64)
    order = sch.leaves_list(link_mat)
    if len(order) != nrow:
        raise ValueError(
            f"link_mat has {len(order)} leaves but the data has {nrow} rows"
        )
    return link_mat, order
