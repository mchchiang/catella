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

# Warn when stacking more chromosome panels than this in one figure
_MAX_PANELS = 10


def _chrom_list(chroms):
    """
    Return `chroms` (a str or an iterable of str) as a list.

    Raises
    ------
    ValueError
        If `chroms` is empty.
    """
    chroms = [chroms] if isinstance(chroms, str) else list(chroms)
    if not chroms:
        raise ValueError("chroms must not be empty")
    return chroms


def _per_chrom(value, chroms, name):
    """
    Map a per-chromosome argument onto each chromosome of `chroms`.

    Parameters
    ----------
    value : object, dict, or None
        None (None for every chromosome), a dict keyed by chromosome
        (missing chromosomes get None; extra keys are ignored), or a
        single value (only allowed for one chromosome).
    chroms : list of str
        The chromosomes plotted.
    name : str
        The argument name, for the error message.

    Returns
    -------
    dict
        The value for each chromosome of `chroms`.

    Raises
    ------
    ValueError
        If `value` is a single value and `chroms` has several entries.
    """
    if value is None:
        return dict.fromkeys(chroms)
    if isinstance(value, dict):
        return {c: value.get(c) for c in chroms}
    if len(chroms) > 1:
        raise ValueError(
            f"{name} must be a dict keyed by chrom when plotting several "
            "chroms"
        )
    return {chroms[0]: value}


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

    Columns: heatmap, optional dendrogram, and a last column for the
    colorbar or legend. With several chromosomes, the panels are
    stacked vertically in one grid with aligned columns, and the last
    column is left out: one colorbar (`_colorbar`) or legend is then
    shared by all panels, placed by the layout engine to the right.

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
        dendrogram), "side_ax" (last column; None if `nchrom` > 1),
        and "seq_ax" (None without an energy row).
    """
    if nchrom > _MAX_PANELS:
        _warn(
            f"plotting {nchrom} chromosomes in one figure; consider fewer "
            "chroms, or one figure per chrom with out_file"
        )
    nsub = 2 if eseq else 1
    height_ratios = [4, 1] if eseq else None
    w, h = plt.rcParams["figure.figsize"]
    figsize = (w * scale_w, h * scale_h * nchrom)
    # A shared colorbar or legend is attached to all axes instead of
    # spanning a grid column, which would equalize all row gaps.
    shared = nchrom > 1
    cols = width_ratios[:-1] if shared else width_ratios
    fig, ax = plt.subplots(
        nrows=nchrom * nsub,
        ncols=len(cols),
        gridspec_kw={
            "width_ratios": cols,
            "height_ratios": None if not eseq else height_ratios * nchrom,
        },
        squeeze=False,
        figsize=figsize,
        layout="constrained" if shared else None,
    )
    panels = []
    for i in range(nchrom):
        row = ax[i * nsub]
        panel = {
            "hm_ax": row[0],
            "dend_ax": row[1] if dendrogram else None,
            "side_ax": None if shared else row[-1],
            "seq_ax": None,
        }
        if eseq:
            seq_row = ax[i * nsub + 1]
            panel["seq_ax"] = seq_row[0]
            for extra in seq_row[1:]:
                extra.axis("off")
        panels.append(panel)
    return fig, panels


def _share_xlim(panels, xmax):
    """Set the x-limits of every heatmap and energy axis to (0, xmax)."""
    for panel in panels:
        for ax in (panel["hm_ax"], panel["seq_ax"]):
            if ax is not None:
                ax.set_xlim(0, xmax)


def _colorbar(fig, panels, mappable, label):
    """
    Draw the colorbar of `mappable` in the last column of a single
    panel, or as one colorbar shared by several panels.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar.
    """
    side_ax = panels[0]["side_ax"]
    if side_ax is not None:
        cbar = fig.colorbar(mappable, cax=side_ax)
    else:
        cbar = fig.colorbar(
            mappable, ax=fig.axes, fraction=0.05, aspect=20 * len(panels)
        )
    cbar.set_label(label, rotation=270, labelpad=15)
    return cbar


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
