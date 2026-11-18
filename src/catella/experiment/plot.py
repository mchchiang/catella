# plot.py

from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib.colors import Normalize
from scipy.special import logit

from catella import utils
from catella._plotting import _leaf_order, _warn_if_large
from catella._warnings import _warn

# Position of each plot_dropout_ecdf split_by field in a dropout key.
_ECDF_FIELDS = {"chrom": 0, "source": 1, "channel": 2}


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
            `data`/`raw_which` resolution and before the size check
            (see Warns).
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
        link_mat : np.ndarray or pd.DataFrame, optional
            Linkage matrix of these molecules (as returned or stored by
            `MethPrintAnalysis.sort_by_linkage`). If given, the rows are
            put in its leaf order before `max_rows` downsampling, and its
            dendrogram is drawn alongside. Cannot be combined with
            `mols`. With `data`, the whole array is loaded to reorder it.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If neither `data` nor all of `exp`/`chrom`/`raw_which` are
            given, if `data` is combined with any of `exp`/`chrom`/
            `raw_which`/`mols`/`mask_name`, or if `link_mat` is combined
            with `mols` or does not match the number of molecules.

        Warns
        -----
        UserWarning
            If rendering `data` at full resolution is estimated to need
            more than 1 GB of memory (about 80 bytes per row and base
            pair). Pass `max_rows`, or downsample first
            (`utils.downsample`), to avoid it.
        """
        raw_data = None
        if data is None:
            if exp is None or chrom is None or raw_which is None:
                raise ValueError(
                    "provide either 'data', or all of 'exp', 'chrom', "
                    "and 'raw_which'"
                )
            if link_mat is not None:
                mol_id = getattr(exp.raw[chrom], f"{raw_which}_mol_id")
                nmol = 0 if mol_id is None else len(mol_id)
                link_mat, order = _leaf_order(link_mat, nmol, mols)
                mols = order
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
                "be combined with 'data'"
            )
        elif link_mat is not None:
            link_mat, order = _leaf_order(link_mat, data.shape[0])
            data = np.asarray(data)[order]

        try:
            if max_rows is not None:
                data = utils.downsample(data, max_rows, how=downsample_how)

            _warn_if_large(
                data.shape[0],
                data.shape[1],
                "consider passing 'max_rows', or downsampling first "
                "(utils.downsample)",
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
            sorted_rows = link_mat is not None
            hm_ax.set_ylabel(
                "Row (leaf order)" if sorted_rows else "Molecule index"
            )

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
            `SimSettings.emax`'s clamping in `NucPosMCMC.setEnergy`)
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
        link_mat : np.ndarray or pd.DataFrame, optional
            Linkage matrix of these molecules; the rows are put in its
            leaf order and its dendrogram is drawn alongside. See
            `plot_meth_prob`.
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
        chroms,
        *,
        which: str | None = None,
        pooled: bool = True,
        split_by: str | None = None,
        out_file: str | Path | None = None,
        show: bool = True,
    ):
        """
        Plot the empirical CDF of dropout rate per group.

        For each matching `(chrom, source, channel)` group, at
        threshold `t` the plotted value is
        `100 * mean(dropout_frac <= t)` over that group's molecules.
        The legend shows only the key fields that differ between
        lines (other than `split_by`); fields shared by all lines go in
        the title.

        Parameters
        ----------
        dropout_fractions : dict of (str, str, str) to np.ndarray
            Output of `MethPrintExperiment.dropout_fractions`.
        chroms : str or list of str
            Chromosome(s) to plot.
        which : {"test", "meth", "unmeth"}, optional
            Restrict the plot to this source. None plots every
            source present.
        pooled : bool, default True
            Whether to plot the pooled `channel="all"` groups.
        split_by : {"chrom", "source", "channel"} or None, default None
            Key field to split into panels, one per value, on shared
            axes (at most 4 per row). None draws a single panel.
        out_file : str or pathlib.Path, optional
            Path to save the generated figure. Directories are
            created if they do not exist.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `chroms` is empty, `split_by` is invalid, or some
            chromosome in `chroms` has no matching entries in
            `dropout_fractions`.
        """
        chroms = [chroms] if isinstance(chroms, str) else list(chroms)
        if not chroms:
            raise ValueError("chroms must not be empty")
        if split_by is not None and split_by not in _ECDF_FIELDS:
            raise ValueError(
                f"split_by must be one of {list(_ECDF_FIELDS)} or None, "
                f"got {split_by!r}"
            )
        keys = sorted(
            (
                key
                for key in dropout_fractions
                if key[0] in chroms
                and (which is None or key[1] == which)
                and (pooled or key[2] != "all")
            ),
            key=lambda k: chroms.index(k[0]),
        )
        missing = [c for c in chroms if not any(k[0] == c for k in keys)]
        if missing:
            raise ValueError(
                f"no entries for chroms {missing} with which={which!r} "
                f"and pooled={pooled} in dropout_fractions"
            )
        f = None if split_by is None else _ECDF_FIELDS[split_by]
        varying = [i for i in range(3) if len({k[i] for k in keys}) > 1]
        legend_idx = [i for i in varying if i != f]
        title = "/".join(
            keys[0][i] for i in range(3) if i not in varying and i != f
        )
        if f is None:
            panels = [(title, keys)]
        else:
            values = list(dict.fromkeys(k[f] for k in keys))
            panels = [(v, [k for k in keys if k[f] == v]) for v in values]

        ncols = min(len(panels), 4)
        nrows = -(-len(panels) // ncols)
        width, height = plt.rcParams["figure.figsize"]
        scale = 1.0 if f is None else 0.6
        # Room for suptitle and shared x label, so one row is not squashed.
        extra = 0.0 if f is None else 1.2
        fig, axes = plt.subplots(
            nrows,
            ncols,
            sharex=True,
            sharey=True,
            squeeze=False,
            figsize=(
                width * scale * ncols,
                height * scale * nrows + extra,
            ),
        )
        labels = list(
            dict.fromkeys("/".join(k[i] for i in legend_idx) for k in keys)
        )
        if len(labels) > 10:
            _warn(
                f"plotting {len(labels)} lines per panel; colors repeat "
                "after 10, so use split_by or fewer chroms to keep lines "
                "distinguishable"
            )
        for ax, (panel_title, panel_keys) in zip(axes.flat, panels):
            for key in panel_keys:
                frac = np.sort(dropout_fractions[key])
                n = len(frac)
                ecdf_pct = 100 * np.arange(1, n + 1) / n
                label = "/".join(key[i] for i in legend_idx)
                color = f"C{labels.index(label) % 10}"
                ax.step(
                    frac, ecdf_pct, where="post", label=label, color=color
                )
            ax.set_title(panel_title)
        for idx in range(len(panels), nrows * ncols):
            axes.flat[idx].set_visible(False)
            axes.flat[idx - ncols].tick_params(labelbottom=True)

        if f is None:
            axes[0, 0].set_xlabel("Dropout rate")
            axes[0, 0].set_ylabel("ECDF [%]")
        else:
            fig.supxlabel("Dropout rate")
            fig.supylabel("ECDF [%]")
            if title:
                fig.suptitle(title)
        if legend_idx:
            handles = {}
            for ax in axes.flat[: len(panels)]:
                for h, lab in zip(*ax.get_legend_handles_labels()):
                    handles.setdefault(lab, h)
            axes[0, -1].legend(
                [handles[lab] for lab in labels],
                labels,
                loc="center left",
                bbox_to_anchor=(1, 0.5),
                frameon=False,
            )
        fig.tight_layout()

        if show:
            plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file, bbox_inches="tight")
