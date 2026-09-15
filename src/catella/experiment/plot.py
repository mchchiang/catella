# plot.py

import warnings
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib.colors import Normalize
from scipy.special import logit

from catella import utils

# Row-count threshold above which plot_meth_prob() warns, since it plots
# data at full resolution -- downsample first (utils.downsample) to
# avoid materializing/plotting more than this many rows.
_PLOT_WARN_ROWS = 5000


@dataclass(slots=True, kw_only=True)
class MethPlot:
    """
    A utility class for generating methylation heatmaps from
    experimental data.

    This class manages global matplotlib settings and provides methods
    to visualize dense or raw-derived methylation signal matrices
    without requiring the full array to be materialized in memory.
    """

    # Global plot settings
    linewidth: int = 1.0
    """The border and axis line thickness for all plots."""

    fontsize: int = 14
    """The base font size for labels, ticks, and titles."""

    cmap: str = "OrRd"
    """The Matplotlib colormap name used for heatmaps."""

    nan_color: str = "lightgray"
    """The color used to render NaN cells in heatmaps."""

    _rc: dict = field(init=None)

    def __post_init__(self):
        """Initializes the runtime configuration dictionary for
        matplotlib styling."""
        self._rc = {
            "font.size": self.fontsize,
            "axes.linewidth": self.linewidth,
            "axes.axisbelow": False,
            "axes.unicode_minus": False,
        }

    def _apply_style(func):
        """Decorator to apply the class-defined matplotlib RC context to
        a method."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with plt.rc_context(rc=self._rc):
                return func(self, *args, **kwargs)

        return wrapper

    @_apply_style
    def plot_meth_prob(
        self,
        data=None,
        *,
        exp=None,
        chrom: str | None = None,
        raw_which: str | None = None,
        mols=None,
        mask_name: str | None = None,
        max_rows: int | None = None,
        downsample_how: str = "mean",
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
        cbar_label: str = "Methylation prob.",
        out_file: str | Path | None = None,
        link_mat: np.ndarray | None = None,
        show: bool = True,
    ):
        """
        Plot a methylation heatmap.

        Plots `data` at full resolution -- for large data, pass
        `max_rows` to downsample first, or downsample it yourself
        (`utils.downsample`, works uniformly for `H5Array`,
        `pd.DataFrame`, or `np.ndarray`) and pass the reduced result.

        Parameters
        ----------
        data : H5Array, pd.DataFrame, or np.ndarray, optional
            Dense signal matrix to plot (rows=molecules, columns=bp
            position), e.g. `exp.analysis[chrom]["meth_prob"]`,
            `exp.analysis[chrom]["test_smoothed"]`, or the result of
            `MethPrintExperiment.to_dense()`. Required unless `exp`,
            `chrom`, and `raw_which` are given instead.
        exp : MethPrintExperiment, optional
            Experiment to pull raw data from. Must be given together
            with `chrom` and `raw_which`, and not combined with `data`.
        chrom : str, optional
            Chromosome to plot, when using `exp`/`raw_which`.
        raw_which : {"test", "unmeth", "meth"}, optional
            Which raw table to plot, when using `exp`/`chrom`. Calls
            `exp.to_dense(chrom, which=raw_which, mols=mols,
            mask_name=mask_name)` internally.
        mols : int or sequence of int, optional
            Molecule index/indices to include, when using `exp`/
            `chrom`/`raw_which`. See `MethPrintExperiment.to_dense`.
        mask_name : str, optional
            Dropout mask name to apply, when using `exp`/`chrom`/
            `raw_which`. See `MethPrintExperiment.to_dense`.
        max_rows : int, optional
            If given, `data` is downsampled to at most this many rows
            (via `utils.downsample`) before plotting. Applied after
            `data`/`raw_which` resolution and before the
            `_PLOT_WARN_ROWS` check.
        downsample_how : {"mean", "sum", "min", "max", "stride"},
            default "mean"
            How to collapse rows when `max_rows` is given. See
            `utils.downsample`.
        vmin : float, optional
            Lower bound for the color scale. If None, inferred from
            `data`.
        vmax : float, optional
            Upper bound for the color scale. If None, inferred from
            `data`.
        cmap : str, optional
            Matplotlib colormap name to use for this plot. If None,
            `self.cmap` is used.
        cbar_label : str, default "Methylation prob."
            Label drawn next to the colorbar.
        out_file : str or pathlib.Path, optional
            Path to save the generated figure. Directories are created
            if they do not exist.
        link_mat : np.ndarray, optional
            Linkage matrix to draw as a dendrogram alongside the
            heatmap (as returned by `MethPrintAnalysis.sort_by_linkage`).
            If given, `data` should already be the correspondingly-sorted
            array, not the original unsorted one.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If neither `data` nor all of `exp`/`chrom`/`raw_which` are
            given, or if `data` is combined with any of `exp`/`chrom`/
            `raw_which`/`mols`/`mask_name`.

        Warns
        -----
        UserWarning
            If `data` has more than `_PLOT_WARN_ROWS` rows, since it is
            plotted at full resolution. Pass `max_rows`, or downsample
            first (`utils.downsample`), to avoid it.
        """
        raw_data = None
        if data is None:
            if exp is None or chrom is None or raw_which is None:
                raise ValueError(
                    "Provide either 'data', or all of 'exp', 'chrom', "
                    "and 'raw_which'."
                )
            data = exp.to_dense(
                chrom, which=raw_which, mols=mols, mask_name=mask_name
            )
            raw_data = data
        elif (
            exp is not None
            or chrom is not None
            or raw_which is not None
            or mols is not None
            or mask_name is not None
        ):
            raise ValueError(
                "'exp'/'chrom'/'raw_which'/'mols'/'mask_name' cannot "
                "be combined with 'data'."
            )

        try:
            if max_rows is not None:
                data = utils.downsample(data, max_rows, how=downsample_how)

            nrow = data.shape[0]
            if nrow > _PLOT_WARN_ROWS:
                warnings.warn(
                    f"Plotting {nrow} rows at full resolution; this may "
                    "be slow and memory-intensive. Consider passing "
                    "'max_rows', or downsampling first "
                    "(utils.downsample).",
                    stacklevel=2,
                )

            matrix = np.asarray(data)
            nrow, ncol = matrix.shape
            norm = Normalize(vmin=vmin, vmax=vmax)
            cmap_obj = plt.get_cmap(cmap if cmap is not None else self.cmap)
            cmap_obj = cmap_obj.with_extremes(bad=self.nan_color)

            if link_mat is not None:
                fig, ax = plt.subplots(
                    ncols=3, gridspec_kw={"width_ratios": [5, 1, 0.25]}
                )
                hm_ax, dend_ax, cbar_ax = ax
            else:
                fig, ax = plt.subplots(
                    ncols=2, gridspec_kw={"width_ratios": [20, 1]}
                )
                hm_ax, cbar_ax = ax

            im = hm_ax.imshow(
                matrix,
                cmap=cmap_obj,
                norm=norm,
                aspect="auto",
                origin="lower",
                interpolation="none",
                extent=[0, ncol, 0, nrow],
            )
            hm_ax.set_xlabel("Position [bp]")
            hm_ax.set_ylabel("Molecule index")

            if link_mat is not None:
                sch.dendrogram(
                    link_mat,
                    orientation="right",
                    ax=dend_ax,
                    no_labels=True,
                    link_color_func=lambda x: "black",
                )
                dend_ax.axis("off")

            # cbar_ax is always the rightmost column, so the colorbar
            # appears to the right of the dendrogram when one is shown.
            cbar = fig.colorbar(im, cax=cbar_ax)
            cbar.set_label(cbar_label, rotation=270, labelpad=15)

            fig.tight_layout()

            if show:
                plt.show()

            if out_file is not None:
                out_file = Path(out_file)
                out_dir = out_file.parents[0]
                out_dir.mkdir(exist_ok=True, parents=True)
                fig.savefig(out_file)
        finally:
            if raw_data is not None:
                raw_data.close()

    @_apply_style
    def plot_meth_energy(
        self,
        data,
        *,
        emax: float | None = None,
        max_rows: int | None = None,
        downsample_how: str = "mean",
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
        cbar_label: str = r"Energy [$k_BT$]",
        out_file: str | Path | None = None,
        link_mat: np.ndarray | None = None,
        show: bool = True,
    ):
        """
        Plot a methylation heatmap on an energy scale.

        Thin wrapper around `plot_meth_prob` showing `logit(data)`
        instead of the raw probability, the same quantity
        `catella.simulation.engine` treats as physical energy.

        Parameters
        ----------
        data : H5Array, pd.DataFrame, or np.ndarray
            Dense probability matrix in [0, 1], e.g.
            `exp.analysis[chrom]["meth_prob"]`.
        emax : float, optional
            If given, clamp energy to [-emax, emax] (matching
            `SimSettings.emax`'s clamping in `NucPosModel.setEnergy`)
            and use it as the default `vmin`/`vmax`. If None, energy
            is unclamped and the default color-scale range is
            inferred from the finite values in `data`.
        max_rows : int, optional
            If given, downsample to at most this many rows (via
            `utils.downsample`) before plotting.
        downsample_how : {"mean", "sum", "min", "max", "stride"},
            default "mean"
            How to collapse rows when `max_rows` is given.
        vmin : float, optional
            Combined with `vmax` into a symmetric half-range so the
            colormap stays centered at zero.
        vmax : float, optional
            See `vmin`.
        cmap : str, optional
            Matplotlib colormap name. Defaults to "RdBu_r".
        cbar_label : str, default "Energy [$k_BT$]"
            Label drawn next to the colorbar.
        out_file : str or pathlib.Path, optional
            Path to save the generated figure.
        link_mat : np.ndarray, optional
            Linkage matrix to draw as a dendrogram alongside the
            heatmap. `data` should already be sorted to match.
        show : bool, default True
            Whether to display the plot using `plt.show()`.
        """
        energy = logit(np.asarray(data))

        if emax is not None:
            energy = np.clip(energy, -emax, emax)

        if vmin is not None or vmax is not None:
            halfrange = max(abs(v) for v in (vmin, vmax) if v is not None)
        elif emax is not None:
            halfrange = emax
        else:
            finite = energy[np.isfinite(energy)]
            halfrange = np.nanmax(np.abs(finite)) if finite.size > 0 else None

        if halfrange is not None:
            vmin, vmax = -halfrange, halfrange

        self.plot_meth_prob(
            energy,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap if cmap is not None else "RdBu_r",
            cbar_label=cbar_label,
            out_file=out_file,
            max_rows=max_rows,
            downsample_how=downsample_how,
            link_mat=link_mat,
            show=show,
        )

    @_apply_style
    def plot_dropout_ecdf(
        self,
        dropout_fractions,
        chrom,
        *,
        source: str | None = None,
        out_file: str | Path | None = None,
        show: bool = True,
    ):
        """
        Plot the empirical CDF of dropout rate per group.

        For each matching group, at threshold `t` the plotted value
        is `100 * mean(dropout_frac <= t)` over that group's
        molecules.

        Parameters
        ----------
        dropout_fractions : dict of (str, str, str) to np.ndarray
            Output of `MethPrintExperiment.dropout_fractions`.
        chrom : str
            Chromosome to plot.
        source : {"test", "meth", "unmeth"}, optional
            Restrict the plot to this source. None plots every
            source present for `chrom`.
        out_file : str or pathlib.Path, optional
            Path to save the generated figure. Directories are
            created if they do not exist.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `chrom` (or `source`, when given) has no matching
            entries in `dropout_fractions`.
        """
        keys = [
            key
            for key in dropout_fractions
            if key[0] == chrom and (source is None or key[1] == source)
        ]
        if not keys:
            raise ValueError(
                f"No entries for chrom={chrom!r}, source={source!r} "
                "in dropout_fractions."
            )

        fig, ax = plt.subplots()
        for src, label in sorted((k[1], k[2]) for k in keys):
            frac = np.sort(dropout_fractions[(chrom, src, label)])
            n = len(frac)
            ecdf_pct = 100 * np.arange(1, n + 1) / n
            ax.step(frac, ecdf_pct, where="post", label=f"{src}/{label}")

        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("ECDF [%]")
        ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False)
        fig.tight_layout()

        if show:
            plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file, bbox_inches="tight")
