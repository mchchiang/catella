# _plotting.py

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from matplotlib.axis import XAxis
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import Formatter, Locator, MaxNLocator
from matplotlib.transforms import Bbox

from catella._warnings import _warn

# Units (factor, name) of position and time axes, smallest first
_BP_UNITS = ((1, "bp"), (1000, "kb"), (1000000, "Mb"))
_MCS_UNITS = (
    (1, "MCS"),
    (1000, "$10^3$ MCS"),
    (1000000, "$10^6$ MCS"),
    (1000000000, "$10^9$ MCS"),
)

# Most decimals in tick labels of an automatic unit
_MAX_UNIT_DECIMALS = 1

# Estimated width of a tick label character, as a share of its font size
_CHAR_WIDTH = 0.6

# Most passes lowering the tick count to fit x tick labels
_MAX_TICK_PASSES = 5

# Estimated peak memory of rendering and saving one heatmap cell
# (measured with matplotlib 3.10 Agg: ~70 B PNG, ~30 B PDF, whatever
# the input dtype)
_RENDER_BYTES_PER_CELL = 80

# Warn when rendering a heatmap is estimated to need more than this
_RENDER_WARN_BYTES = 10**9

# Warn when plotting more region panels than this in one figure
_MAX_PANELS = 10

# Region panels per row (as in MethPlot.plot_dropout_ecdf)
_PANELS_PER_ROW = 4


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
    nregion,
    width_ratios,
    *,
    dendrogram,
    extra_rows=(),
    scale_w=1.0,
    scale_h=1.0,
):
    """
    Create a figure with one heatmap panel per region.

    Panels are laid out left to right, at most 4 per row (as in
    `MethPlot.plot_dropout_ecdf`), in one grid so that panels in a row
    share their height and panels in a column their edges. Each panel
    has a heatmap column and an optional dendrogram column; the last
    grid column holds the colorbar or legend, next to the top-right
    heatmap only. Empty slots of the last row are hidden.

    Parameters
    ----------
    nregion : int
        The number of panels.
    width_ratios : list of float
        The relative widths of the columns of one panel (heatmap,
        optional dendrogram) followed by that of the colorbar or legend.
    dendrogram : bool
        Whether the second column of a panel holds a dendrogram.
    extra_rows : sequence of {"mean", "edata"}, default ()
        Rows below each heatmap, top to bottom: "mean" for the mean
        across molecules, "edata" for the data energy.
    scale_w, scale_h : float, default 1.0
        Factors applied to the default figure width and height, per
        panel column and per panel row.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure; it uses constrained layout if `nregion` > 1.
    panels : list of dict
        Per panel, the axes "hm_ax", "dend_ax" (None without a
        dendrogram), "side_ax" (the top-right cell of the last column,
        the same for all panels), "mean_ax" and "edata_ax" (None without
        that row), and its grid column "col".
    """
    if nregion > _MAX_PANELS:
        _warn(
            f"plotting {nregion} regions in one figure; consider fewer "
            "regions, or one figure per region with out_file"
        )
    extra_rows = list(extra_rows)
    nsub = 1 + len(extra_rows)
    ncols = min(nregion, _PANELS_PER_ROW)
    nrows = -(-nregion // ncols)
    panel_ratios = list(width_ratios[:-1])
    width = len(panel_ratios)
    w, h = plt.rcParams["figure.figsize"]
    fig, ax = plt.subplots(
        nrows=nrows * nsub,
        ncols=width * ncols + 1,
        gridspec_kw={
            "width_ratios": panel_ratios * ncols + [width_ratios[-1]],
            "height_ratios": (
                ([4] + [1] * len(extra_rows)) * nrows if extra_rows else None
            ),
        },
        squeeze=False,
        figsize=(w * scale_w * ncols, h * scale_h * nrows),
        layout="constrained" if nregion > 1 else None,
    )
    side_ax = ax[0, -1]
    for cell in ax[1:, -1]:
        cell.axis("off")
    panels = []
    for i in range(nrows * ncols):
        r, c = divmod(i, ncols)
        cols = slice(c * width, (c + 1) * width)
        cells = ax[r * nsub, cols]
        extra = {
            name: ax[r * nsub + 1 + k, cols]
            for k, name in enumerate(extra_rows)
        }
        if i >= nregion:
            for cell in [*cells, *(x for row in extra.values() for x in row)]:
                cell.set_visible(False)
            continue
        panel = {
            "hm_ax": cells[0],
            "dend_ax": cells[1] if dendrogram else None,
            "side_ax": side_ax,
            "mean_ax": extra["mean"][0] if "mean" in extra else None,
            "edata_ax": extra["edata"][0] if "edata" in extra else None,
            "col": c,
        }
        for row in extra.values():
            for cell in row[1:]:
                cell.axis("off")
        panels.append(panel)
    return fig, panels


def _row_axes(panel):
    """The heatmap, mean and energy axes of `panel` that exist."""
    keys = ("hm_ax", "mean_ax", "edata_ax")
    return [panel[k] for k in keys if panel[k] is not None]


def _left_ylabels(panels):
    """Keep the y-axis labels of the leftmost panels of each row only."""
    for panel in panels:
        if panel["col"] > 0:
            for ax in _row_axes(panel):
                ax.set_ylabel("")


def _share_xlim(panels, xmax):
    """Set the x-limits of every heatmap, mean and energy axis to
    (0, xmax), in bp."""
    for panel in panels:
        for ax in _row_axes(panel):
            ax.set_xlim(0, xmax)


def _draw_mean(ax, mean):
    """
    Plot the mean across molecules (one value per bp) into `ax`, on a
    fixed (0, 1) y-range.
    """
    nbp = len(mean)
    ax.plot(np.arange(nbp), mean)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, nbp)
    ax.set_ylabel("Mean")


def _decimals(values, most=6):
    """Fewest decimals (at most `most`) that write `values` exactly."""
    values = np.asarray(values, dtype=float)
    steps = np.diff(np.unique(values))
    ref = steps.min() if steps.size else np.abs(values).max(initial=0)
    if ref == 0:
        return 0
    # Tolerance relative to the tick spacing, not absolute
    atol = 1e-3 * ref
    for ndec in range(most + 1):
        if np.allclose(values, np.round(values, ndec), rtol=0, atol=atol):
            return ndec
    return most


class _ShiftedLocator(Locator):
    """
    Ticks at round whole values of `start + x`, for an axis drawn in
    `x` (positions in bp and times in MCS are whole numbers). On an
    x-axis with a `_UnitFormatter`, fewer ticks are placed until each
    label (width estimated from its characters) has a slot of at least
    twice its width, keeping at least two ticks.

    Parameters
    ----------
    start : float, default 0
        The value of `x = 0` (e.g., the genomic start of a region).
    """

    def __init__(self, start=0):
        self.start = start

    def __call__(self):
        vmin, vmax = self.axis.get_view_interval()
        return self.tick_values(vmin, vmax)

    def _ticks(self, vmin, vmax, nbins):
        base = MaxNLocator(
            nbins=nbins, steps=[1, 2, 2.5, 5, 10], integer=True
        )
        if nbins == "auto":
            base.set_axis(self.axis)
        return base.tick_values(vmin + self.start, vmax + self.start) - (
            self.start
        )

    def tick_values(self, vmin, vmax):
        ticks = self._ticks(vmin, vmax, "auto")
        fmt = None if self.axis is None else self.axis.get_major_formatter()
        if not isinstance(self.axis, XAxis) or not isinstance(
            fmt, _UnitFormatter
        ):
            return ticks
        # Fewer ticks until each label has a slot of twice its width
        length, size = _axis_points(self.axis)
        lo, hi = sorted((vmin, vmax))
        for _ in range(_MAX_TICK_PASSES):
            shown = ticks[(ticks >= lo) & (ticks <= hi)]
            if len(shown) <= 2 or hi <= lo:
                break
            labels = fmt.labels(shown)
            width = max(len(s) for s in labels) * _CHAR_WIDTH * size
            gap = length * (shown[1] - shown[0]) / (hi - lo)
            if gap >= 2 * width:
                break
            ticks = self._ticks(vmin, vmax, len(shown) - 2)
        return ticks


def _axis_points(axis):
    """Length (points) of an x-axis and the font size of its labels."""
    ax = axis.axes
    ends = Bbox.unit().transformed(
        ax.transAxes - ax.get_figure(root=False).dpi_scale_trans
    )
    size = (
        axis.majorTicks[0].label1.get_size()
        if axis.majorTicks
        else FontProperties(size=plt.rcParams["xtick.labelsize"]).get_size()
    )
    return ends.width * 72, size


def _choose_unit(values, units):
    """
    The largest unit whose labels of `values` need at most one decimal.

    Returns
    -------
    factor : float
    name : str
    ndec : int
        Decimals the labels need in that unit.
    """
    factor, name = units[0]
    ndec = _decimals(values / factor)
    for unit_factor, unit_name in units[1:]:
        unit_ndec = _decimals(values / unit_factor)
        if unit_ndec <= _MAX_UNIT_DECIMALS:
            factor, name, ndec = unit_factor, unit_name, unit_ndec
    return factor, name, ndec


class _UnitFormatter(Formatter):
    """
    Tick labels of `start + x` in the largest unit that needs at most
    one decimal, chosen when the ticks are drawn.

    Parameters
    ----------
    units : sequence of (float, str)
        Unit factors and names, smallest first (e.g., `_BP_UNITS`).
    start : float, default 0
        The value of `x = 0`.
    label : str, optional
        Axis label with a "{unit}" placeholder, set to the chosen unit;
        None leaves the label alone.
    """

    def __init__(self, units, start=0, label=None):
        self.units = units
        self.start = start
        self.label = label
        self.factor, self.ndec = units[0][0], 0

    def set_locs(self, locs):
        super().set_locs(locs)
        values = np.asarray(locs, dtype=float) + self.start
        self.factor, name, self.ndec = _choose_unit(values, self.units)
        if self.label is not None and self.axis is not None:
            self.axis.set_label_text(self.label.replace("{unit}", name))

    def labels(self, locs):
        """Labels of `locs` in the unit they would get, without state."""
        values = np.asarray(locs, dtype=float) + self.start
        factor, _, ndec = _choose_unit(values, self.units)
        return [f"{v / factor:,.{ndec}f}" for v in values]

    def __call__(self, x, pos=None):
        return f"{(x + self.start) / self.factor:,.{self.ndec}f}"


def _unit_axis(axis, units, *, start=0, label=None):
    """
    Give a matplotlib axis round ticks and automatic-unit labels.

    Parameters
    ----------
    axis : matplotlib.axis.Axis
        E.g., `ax.xaxis`.
    units : sequence of (float, str)
        As in `_UnitFormatter`.
    start : float, default 0
        The value of `x = 0`.
    label : str, optional
        Axis label with a "{unit}" placeholder; set at once to the
        smallest unit, then updated when drawn. None sets no label.
    """
    axis.set_major_locator(_ShiftedLocator(start))
    axis.set_major_formatter(_UnitFormatter(units, start, label))
    if label is not None:
        axis.set_label_text(label.replace("{unit}", units[0][1]))


def _region_starts(coords, nbp):
    """
    Reference chromosome and start of each plotted region.

    Parameters
    ----------
    coords : pd.DataFrame
        Indexed by region, with columns `ref_chrom`, `ref_start`,
        `ref_end` and `nbp` (as `MethPrintExperiment.region_coords`).
    nbp : dict of str to int
        Plotted region -> its length (bp).

    Returns
    -------
    dict of str to (str, int)
        Region -> `(ref_chrom, ref_start)`.

    Raises
    ------
    ValueError
        If a region is not in `coords`, if its `nbp` differs, if it has
        no coordinates, or if it is folded (`wrap`).
    """
    starts = {}
    for region, length in nbp.items():
        if region not in coords.index:
            raise ValueError(f"coords has no region '{region}'")
        row = coords.loc[region]
        if row["nbp"] != length:
            raise ValueError(
                f"coords has nbp {row['nbp']} for region '{region}', but "
                f"the region has {length}"
            )
        if pd.isna(row["ref_start"]):
            raise ValueError(f"region '{region}' has no genomic coordinates")
        if row["ref_end"] - row["ref_start"] != length:
            raise ValueError(
                f"region '{region}' is folded (wrap); it has no genomic "
                "positions"
            )
        starts[region] = (str(row["ref_chrom"]), int(row["ref_start"]))
    return starts


def _position_xaxis(panel, label_ax, *, start=0, chrom=None):
    """
    Give the heatmap, mean and energy axes of `panel` position ticks in
    bp, kb or Mb, and label `label_ax`.

    Parameters
    ----------
    panel : dict
        As returned by `_panel_grid`.
    label_ax : matplotlib.axes.Axes
        The axis that carries the x label (the bottom one).
    start : float, default 0
        The position of `x = 0` (e.g., `ref_start` of the region).
    chrom : str, optional
        If given, the label reads "Position on `chrom`".
    """
    where = "$x$" if chrom is None else f"on {chrom}"
    for ax in _row_axes(panel):
        label = f"Position {where} [{{unit}}]" if ax is label_ax else None
        _unit_axis(ax.xaxis, _BP_UNITS, start=start, label=label)


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
