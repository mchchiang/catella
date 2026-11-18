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

# Warn when plotting more chromosome panels than this in one figure
_MAX_PANELS = 10

# Chromosome panels per row (as in MethPlot.plot_dropout_ecdf)
_PANELS_PER_ROW = 4


def _power_of_10(x, name=None):
    """
    Return the base-10 logarithm of `x`, checked to be a power of 10.

    Parameters
    ----------
    x : float
        The value to check.
    name : str, optional
        The variable name to include in the error message.

    Returns
    -------
    float
        The log10 of `x`.

    Raises
    ------
    ValueError
        If `x` is not a non-negative power of 10.
    """
    log10x = np.log10(x)
    if log10x < 0.0 or not np.isclose(np.mod(log10x, 1), 0, atol=1e-9):
        xstr = name if name is not None else "x"
        raise ValueError(f"'{xstr}' must be a non-negative power of 10")
    return log10x


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

    Panels are laid out left to right, at most 4 per row (as in
    `MethPlot.plot_dropout_ecdf`), in one grid so that panels in a row
    share their height and panels in a column their edges. Each panel
    has a heatmap column and an optional dendrogram column; the last
    grid column holds the colorbar or legend, next to the top-right
    heatmap only. Empty slots of the last row are hidden.

    Parameters
    ----------
    nchrom : int
        The number of panels.
    width_ratios : list of float
        The relative widths of the columns of one panel (heatmap,
        optional dendrogram) followed by that of the colorbar or legend.
    dendrogram : bool
        Whether the second column of a panel holds a dendrogram.
    eseq : bool
        Whether each panel has an energy row below its heatmap.
    scale_w, scale_h : float, default 1.0
        Factors applied to the default figure width and height, per
        panel column and per panel row.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure; it uses constrained layout if `nchrom` > 1.
    panels : list of dict
        Per panel, the axes "hm_ax", "dend_ax" (None without a
        dendrogram), "side_ax" (the top-right cell of the last column,
        the same for all panels), "seq_ax" (None without an energy
        row), and its grid column "col".
    """
    if nchrom > _MAX_PANELS:
        _warn(
            f"plotting {nchrom} chromosomes in one figure; consider fewer "
            "chroms, or one figure per chrom with out_file"
        )
    nsub = 2 if eseq else 1
    ncols = min(nchrom, _PANELS_PER_ROW)
    nrows = -(-nchrom // ncols)
    panel_ratios = list(width_ratios[:-1])
    width = len(panel_ratios)
    w, h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(
        nrows=nrows * nsub,
        ncols=width * ncols + 1,
        gridspec_kw={
            "width_ratios": panel_ratios * ncols + [width_ratios[-1]],
            "height_ratios": [4, 1] * nrows if eseq else None,
        },
        squeeze=False,
        figsize=(w * scale_w * ncols, h * scale_h * nrows),
        layout="constrained" if nchrom > 1 else None,
    )
    side_ax = ax[0, -1]
    for cell in ax[1:, -1]:
        cell.axis("off")
    panels = []
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        cells = ax[r * nsub, c * width : (c + 1) * width]
        seq_cells = (
            ax[r * nsub + 1, c * width : (c + 1) * width] if eseq else []
        )
        if i >= nchrom:
            for cell in [*cells, *seq_cells]:
                cell.set_visible(False)
            continue
        panel = {
            "hm_ax": cells[0],
            "dend_ax": cells[1] if dendrogram else None,
            "side_ax": side_ax,
            "seq_ax": seq_cells[0] if eseq else None,
            "col": c,
        }
        for extra in seq_cells[1:]:
            extra.axis("off")
        panels.append(panel)
    return fig, panels


def _left_ylabels(panels):
    """Keep the y-axis labels of the leftmost panels of each row only."""
    for panel in panels:
        if panel["col"] > 0:
            for ax in (panel["hm_ax"], panel["seq_ax"]):
                if ax is not None:
                    ax.set_ylabel("")


def _share_xlim(panels, xmax):
    """Set the x-limits of every heatmap and energy axis to (0, xmax)."""
    for panel in panels:
        for ax in (panel["hm_ax"], panel["seq_ax"]):
            if ax is not None:
                ax.set_xlim(0, xmax)


def _colorbar(fig, panels, mappable, label):
    """
    Draw the colorbar of `mappable`, shared by all panels, next to the
    top-right heatmap.

    Returns
    -------
    matplotlib.colorbar.Colorbar
        The colorbar.
    """
    cbar = fig.colorbar(mappable, cax=panels[0]["side_ax"])
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
