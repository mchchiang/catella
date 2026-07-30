# plot.py

import warnings
from functools import wraps
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from ..h5_array import H5Array, _DOWNSAMPLE_HOW


def _bin_edges(n, max_rows):
    n_bins = min(n, max_rows)
    return np.linspace(0, n, n_bins + 1).astype(int)


def _reduce_rows(block, how):
    if block.shape[0] == 0:
        return np.full(block.shape[1], np.nan)
    fn = {"mean": np.nanmean, "sum": np.nansum,
         "min": np.nanmin, "max": np.nanmax}[how]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return fn(block, axis=0)


def _downsample_array(arr, max_rows, how):
    nrow = arr.shape[0]
    if nrow <= max_rows:
        return arr
    if how == "stride":
        step = -(-nrow // max_rows)  # ceil div
        return arr[0:nrow:step]
    edges = _bin_edges(nrow, max_rows)
    return np.stack([_reduce_rows(arr[edges[i]:edges[i+1]], how)
                     for i in range(len(edges) - 1)])


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
                     max_rows : int = 2000,
                     how : str = "mean",
                     batch_size : int = 20000,
                     vmin : float | None = None,
                     vmax : float | None = None,
                     out_file : str | Path | None = None,
                     show : bool = True):
        """
        Plot a methylation heatmap, downsampled to bounded memory.

        Rows are collapsed to at most `max_rows` using memory-bounded
        streaming (for `H5Array` data) or in-memory binning (for
        `pd.DataFrame`/`np.ndarray` data), so the full array is never
        materialized in memory regardless of its original row count.

        Parameters
        ----------
        data : H5Array, pd.DataFrame, or np.ndarray
            Dense signal matrix to plot (rows=molecules, columns=bp
            position), e.g. `exp.analysis[chrom]["meth_prob"]`,
            `exp.analysis[chrom]["test_smoothed"]`, or the result of
            `MethPrintExperiment.to_dense()`.
        max_rows : int, default 2000
            Target number of rows to plot. See `H5Array.downsample`
            for how rows are collapsed.
        how : {"mean", "sum", "min", "max", "stride"}, default "mean"
            How to collapse groups of consecutive rows. See
            `H5Array.downsample` for details.
        batch_size : int, default 20000
            Number of rows read (and held in memory) per streamed
            chunk. Only used when `data` is an `H5Array`.
        vmin : float, optional
            Lower bound for the color scale. If None, inferred from
            the downsampled data.
        vmax : float, optional
            Upper bound for the color scale. If None, inferred from
            the downsampled data.
        out_file : str or pathlib.Path, optional
            Path to save the generated figure. Directories are created
            if they do not exist.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `how` is not a recognized option.
        """
        if how not in _DOWNSAMPLE_HOW:
            raise ValueError(f"'how' must be one of {_DOWNSAMPLE_HOW}")

        if isinstance(data, H5Array):
            nrow, ncol = data.shape
            matrix = data.downsample(max_rows, how=how,
                                     batch_size=batch_size)
        else:
            arr = data.to_numpy() if isinstance(data, pd.DataFrame) \
                else np.asarray(data)
            nrow, ncol = arr.shape
            matrix = _downsample_array(arr, max_rows, how)

        norm = Normalize(vmin=vmin, vmax=vmax)
        fig, ax = plt.subplots()
        ax.imshow(matrix, cmap=self.cmap, norm=norm, aspect="auto",
                 origin="lower", interpolation="none",
                 extent=[0, ncol, 0, nrow])
        ax.set_xlabel("Position [bp]")
        ax.set_ylabel("Molecule index")

        fig.tight_layout()

        if show: plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
