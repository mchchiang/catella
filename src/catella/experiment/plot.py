# plot.py

import warnings
from functools import wraps
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib.colors import Normalize

# Row-count threshold above which plot_methmap() warns, since it plots
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
    linewidth : int = 1.0
    """The border and axis line thickness for all plots."""

    fontsize : int = 14
    """The base font size for labels, ticks, and titles."""

    cmap : str = "viridis"
    """The Matplotlib colormap name used for heatmaps."""

    _rc : dict = field(init=None)

    def __post_init__(self):
        """Initializes the runtime configuration dictionary for
        matplotlib styling."""
        self._rc = {"font.size" : self.fontsize,
                    "axes.linewidth" : self.linewidth,
                    "axes.axisbelow" : False,
                    "axes.unicode_minus" : False
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
    def plot_methmap(self, data, *,
                     vmin : float | None = None,
                     vmax : float | None = None,
                     out_file : str | Path | None = None,
                     link_mat : np.ndarray | None = None,
                     show : bool = True):
        """
        Plot a methylation heatmap.

        Plots `data` at full resolution -- for large data, downsample
        it yourself first (`utils.downsample`, works uniformly for
        `H5Array`, `pd.DataFrame`, or `np.ndarray`) and pass the
        reduced result.

        Parameters
        ----------
        data : H5Array, pd.DataFrame, or np.ndarray
            Dense signal matrix to plot (rows=molecules, columns=bp
            position), e.g. `exp.analysis[chrom]["meth_prob"]`,
            `exp.analysis[chrom]["test_smoothed"]`, or the result of
            `MethPrintExperiment.to_dense()`.
        vmin : float, optional
            Lower bound for the color scale. If None, inferred from
            `data`.
        vmax : float, optional
            Upper bound for the color scale. If None, inferred from
            `data`.
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

        Warns
        -----
        UserWarning
            If `data` has more than `_PLOT_WARN_ROWS` rows, since it is
            plotted at full resolution. Downsample first (`utils.downsample`)
            to avoid it.
        """
        nrow = data.shape[0]
        if nrow > _PLOT_WARN_ROWS:
            warnings.warn(
                f"Plotting {nrow} rows at full resolution; this may be "
                "slow and memory-intensive. Consider downsampling first "
                "(utils.downsample).", stacklevel=2)

        matrix = np.asarray(data)
        nrow, ncol = matrix.shape
        norm = Normalize(vmin=vmin, vmax=vmax)

        if link_mat is not None:
            fig, ax = plt.subplots(ncols=2,
                                   gridspec_kw={"width_ratios": [5, 1]})
            hm_ax, dend_ax = ax
        else:
            fig, hm_ax = plt.subplots()

        hm_ax.imshow(matrix, cmap=self.cmap, norm=norm, aspect="auto",
                     origin="lower", interpolation="none",
                     extent=[0, ncol, 0, nrow])
        hm_ax.set_xlabel("Position [bp]")
        hm_ax.set_ylabel("Molecule index")

        if link_mat is not None:
            sch.dendrogram(link_mat, orientation="right", ax=dend_ax,
                           no_labels=True, link_color_func=lambda x: "black")
            dend_ax.axis("off")

        fig.tight_layout()

        if show: plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)

    @_apply_style
    def plot_dropout_filter(self, dropout_fractions, chrom, *,
                            source : str | None = None,
                            out_file : str | Path | None = None,
                            show : bool = True):
        """
        Plot percent of molecules filtered vs. dropout-rate
        threshold.

        For each matching group, at threshold `t` the plotted value
        is `100 * mean(dropout_frac > t)` over that group's
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
        keys = [key for key in dropout_fractions if key[0] == chrom
               and (source is None or key[1] == source)]
        if not keys:
            raise ValueError(
                f"No entries for chrom={chrom!r}, source={source!r} "
                "in dropout_fractions.")

        fig, ax = plt.subplots()
        for src, label in sorted((k[1], k[2]) for k in keys):
            frac = np.sort(dropout_fractions[(chrom, src, label)])
            n = len(frac)
            filtered_pct = 100 - 100 * np.arange(1, n + 1) / n
            ax.step(frac, filtered_pct, where="post",
                   label=f"{src}/{label}")

        ax.set_xlabel("Dropout rate")
        ax.set_ylabel("Molecules filtered [%]")
        ax.legend()
        fig.tight_layout()

        if show: plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
