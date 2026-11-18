# _plotting.py

from pathlib import Path

import matplotlib.pyplot as plt
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


def _panel_grid(
    nchrom, width_ratios, *, dendrogram, eseq, scale_w=1.0, scale_h=1.0
):
    """
    Create a figure with one heatmap panel per chromosome.

    The panels are stacked vertically in one grid whose columns
    (heatmap, optional dendrogram, colorbar or legend) are shared, so
    the panels line up.

    Parameters
    ----------
    nchrom : int
        The number of panels.
    width_ratios : list of float
        The relative widths of the columns.
    dendrogram : bool
        Whether the second column holds a dendrogram.
    eseq : bool
        Whether each panel has an energy row below its heatmap.
    scale_w, scale_h : float, default 1.0
        Factors applied to the default figure width and to the height
        of each panel.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure; it uses constrained layout if `nchrom` > 1.
    panels : list of dict
        Per panel, the axes "hm_ax", "dend_ax" (None without a
        dendrogram), "side_ax" (last column), and "seq_ax" (None
        without an energy row).
    """
    nsub = 2 if eseq else 1
    gridspec_kw = {"width_ratios": width_ratios}
    if eseq:
        gridspec_kw["height_ratios"] = [4, 1] * nchrom
    w, h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(
        nrows=nchrom * nsub,
        ncols=len(width_ratios),
        gridspec_kw=gridspec_kw,
        squeeze=False,
        figsize=(w * scale_w, h * scale_h * nchrom),
        layout="constrained" if nchrom > 1 else None,
    )
    panels = []
    for i in range(nchrom):
        row = ax[i * nsub]
        panel = {
            "hm_ax": row[0],
            "dend_ax": row[1] if dendrogram else None,
            "side_ax": row[-1],
            "seq_ax": None,
        }
        if eseq:
            seq_row = ax[i * nsub + 1]
            panel["seq_ax"] = seq_row[0]
            for extra in seq_row[1:]:
                extra.axis("off")
        panels.append(panel)
    return fig, panels


def _finish(fig, show, out_file):
    """
    Lay out `fig`, then show it and/or save it to `out_file`.

    Figures without a layout engine get `tight_layout`. Directories of
    `out_file` are created if they do not exist.
    """
    if fig.get_layout_engine() is None:
        fig.tight_layout()
    if show:
        plt.show()
    if out_file is not None:
        out_file = Path(out_file)
        out_file.parents[0].mkdir(exist_ok=True, parents=True)
        fig.savefig(out_file)
