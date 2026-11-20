# plot.py

from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib.colors import Normalize
from scipy.special import logit

from catella import utils
from catella._chroms import _chrom_list, _per_chrom
from catella._plotting import (
    _colorbar,
    _finish,
    _leaf_order,
    _left_ylabels,
    _panel_grid,
    _power_of_10,
    _share_xlim,
    _warn_if_large,
)
from catella._warnings import _warn

# Position of each plot_dropout_ecdf split_by field in a dropout key.
_ECDF_FIELDS = {"chrom": 0, "source": 1, "channel": 2}


def _close_raw(sources):
    """Close the `to_dense` results among the panel sources."""
    for source in sources:
        if source["raw"]:
            source["data"].close()


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
        chroms: str | Iterable[str] | None = None,
        raw_which: str | None = None,
        data_name: str | None = None,
        mols=None,
        mask_name: str | None = None,
        max_rows: int | None = None,
        downsample_how: str = "mean",
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
        cbar_label: str = "Methylation prob.",
        out_file: str | Path | None = None,
        link_mat: np.ndarray | dict | None = None,
        xscale: int = 1000,
        sharex: bool = False,
        show: bool = True,
    ):
        """
        Plot a methylation heatmap of one matrix, or of one or more
        chromosomes of an experiment.

        Plots at full resolution -- for large data, pass `max_rows` to
        downsample first, or downsample it yourself (`utils.downsample`,
        works uniformly for `H5Array`, `pd.DataFrame`, or `np.ndarray`)
        and pass the reduced result. Several chromosomes get one panel
        each, laid out side by side (at most 4 per row) in one figure,
        aligned and sharing one colorbar on a common color scale.

        Parameters
        ----------
        data : H5Array, pd.DataFrame, or np.ndarray, optional
            Dense signal matrix to plot (rows=molecules, columns=bp
            position), e.g. `exp.analysis[chrom]["meth_prob"]` or the
            result of `MethPrintExperiment.to_dense()`. Required unless
            `exp` and `chroms` are given instead.
        exp : MethPrintExperiment, optional
            Experiment to read from, together with `chroms` and exactly
            one of `raw_which`/`data_name`. Not combined with `data`.
        chroms : str or iterable of str, optional
            Chromosome(s) to plot from `exp`, one panel each, in
            reading order.
        raw_which : {"test", "unmeth", "meth"}, optional
            Which raw table to plot per chromosome. Calls
            `exp.to_dense(chrom, which=raw_which, mols=mols,
            mask_name=mask_name)` internally.
        data_name : str, optional
            Key of a dense array to plot per chromosome, read from
            `exp.analysis[chrom][data_name]` (e.g., "meth_prob").
        mols : int, sequence of int, or dict, optional
            Molecule index/indices to include; only with `raw_which`.
            See `MethPrintExperiment.to_dense`. With several chroms, a
            dict keyed by chrom (chroms not in it show all molecules).
        mask_name : str, optional
            Dropout mask name to apply; only with `raw_which`. See
            `MethPrintExperiment.to_dense`.
        max_rows : int, optional
            If given, each panel is downsampled to at most this many
            rows (via `utils.downsample`) before plotting, and before
            the size check (see Warns).
        downsample_how : {"mean", "sum", "min", "max", "stride"},
            default "mean"
            How to collapse rows when `max_rows` is given. See
            `utils.downsample`.
        vmin : float, optional
            Lower bound for the color scale. If None, inferred from the
            data (over all panels).
        vmax : float, optional
            Upper bound for the color scale. If None, inferred from the
            data (over all panels).
        cmap : str, optional
            Matplotlib colormap name to use for this plot. If None,
            `self.cmap` is used.
        cbar_label : str, default "Methylation prob."
            Label drawn next to the colorbar.
        out_file : str or pathlib.Path, optional
            Path to save the generated figure. Directories are created
            if they do not exist.
        link_mat : np.ndarray, pd.DataFrame, or dict, optional
            Linkage matrix of these molecules (as returned or stored by
            `MethPrintAnalysis.sort_by_linkage`). If given, the rows are
            put in its leaf order before `max_rows` downsampling, and its
            dendrogram is drawn alongside. Cannot be combined with
            `mols`. With `data` or `data_name`, the whole array is
            loaded to reorder it. With several chroms, a dict keyed by
            chrom, e.g. the result of `sort_by_linkage` (chroms not in
            it are not sorted; extra keys are ignored).
        xscale : int, default 1000
            Spatial scaling factor (a power of 10) for the x-axis labels.
        sharex : bool, default False
            If True, all panels span the widest chromosome, so they
            share one bp scale. If False, each panel spans its own.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `xscale` is not a power of 10, if neither `data` nor
            `exp` and `chroms` with exactly one of
            `raw_which`/`data_name` are given, if `data` is combined
            with any of `exp`/`chroms`/`raw_which`/`data_name`/`mols`/
            `mask_name`, if `mols` or `mask_name` is given without
            `raw_which`, if `chroms` is empty, if `mols` or `link_mat`
            is not a dict with several chroms, or if `link_mat` is
            combined with `mols` or does not match the number of
            molecules.

        Warns
        -----
        UserWarning
            If rendering is estimated to need more than 1 GB of memory
            (about 80 bytes per row and base pair, over all panels).
            Pass `max_rows`, or downsample first (`utils.downsample`),
            to avoid it. Also if more than 10 chroms are plotted.
        """
        _power_of_10(xscale, "xscale")
        sources = self._meth_sources(
            data,
            exp=exp,
            chroms=chroms,
            raw_which=raw_which,
            data_name=data_name,
            mols=mols,
            mask_name=mask_name,
            link_mat=link_mat,
        )
        try:
            self._plot_panels(
                sources,
                max_rows=max_rows,
                downsample_how=downsample_how,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                cbar_label=cbar_label,
                xscale=xscale,
                sharex=sharex,
                show=show,
                out_file=out_file,
            )
        finally:
            _close_raw(sources)

    def _meth_sources(
        self,
        data,
        *,
        exp,
        chroms,
        raw_which,
        data_name,
        mols,
        mask_name,
        link_mat,
    ):
        """
        Check the inputs of `plot_meth_prob`/`plot_meth_energy` and read
        the matrix of each panel, in the leaf order of its `link_mat`.

        Returns
        -------
        list of dict
            Per panel: "title" (the chrom with several chroms, else
            None), "data", "link_mat" (float array or None), and "raw"
            (True if "data" is a `to_dense` result to close).
        """
        if data is not None:
            given = (exp, chroms, raw_which, data_name, mols, mask_name)
            if any(v is not None for v in given):
                raise ValueError(
                    "'exp'/'chroms'/'raw_which'/'data_name'/'mols'/"
                    "'mask_name' cannot be combined with 'data'"
                )
            if link_mat is not None:
                link_mat, order = _leaf_order(link_mat, data.shape[0])
                data = np.asarray(data)[order]
            return [
                {"title": None, "data": data, "link_mat": link_mat,
                 "raw": False}
            ]
        if exp is None or chroms is None or (
            (raw_which is None) == (data_name is None)
        ):
            raise ValueError(
                "provide either 'data', or 'exp' and 'chroms' with "
                "exactly one of 'raw_which' and 'data_name'"
            )
        if raw_which is None and (mols is not None or mask_name is not None):
            raise ValueError("'mols' and 'mask_name' require 'raw_which'")
        chroms = _chrom_list(chroms)
        mols = _per_chrom(mols, chroms, "mols")
        link_mat = _per_chrom(link_mat, chroms, "link_mat")
        sources = []
        try:
            for chrom in chroms:
                link, sel = link_mat[chrom], mols[chrom]
                if raw_which is not None:
                    if link is not None:
                        mol_id = getattr(
                            exp.raw[chrom], f"{raw_which}_mol_id"
                        )
                        nmol = 0 if mol_id is None else len(mol_id)
                        link, sel = _leaf_order(link, nmol, sel)
                    matrix = exp.to_dense(
                        chrom, which=raw_which, mols=sel, mask_name=mask_name
                    )
                else:
                    matrix = exp.analysis[chrom][data_name]
                    if link is not None:
                        link, order = _leaf_order(link, matrix.shape[0])
                        matrix = np.asarray(matrix)[order]
                sources.append(
                    {
                        "title": chrom if len(chroms) > 1 else None,
                        "data": matrix,
                        "link_mat": link,
                        "raw": raw_which is not None,
                    }
                )
        except BaseException:
            _close_raw(sources)
            raise
        return sources

    def _plot_panels(
        self,
        sources,
        *,
        max_rows,
        downsample_how,
        vmin,
        vmax,
        cmap,
        cbar_label,
        xscale,
        sharex,
        show,
        out_file,
    ):
        """
        Downsample, draw and lay out the panels from `_meth_sources`,
        with one colorbar on a common color scale.
        """
        xpow = int(_power_of_10(xscale, "xscale"))
        matrices = []
        for source in sources:
            matrix = source["data"]
            if max_rows is not None:
                matrix = utils.downsample(matrix, max_rows, how=downsample_how)
            matrices.append(matrix)
        _warn_if_large(
            sum(m.shape[0] for m in matrices),
            max(m.shape[1] for m in matrices),
            "consider passing 'max_rows', or downsampling first "
            "(utils.downsample)",
        )
        matrices = [np.asarray(m) for m in matrices]
        if len(matrices) > 1 and (vmin is None or vmax is None):
            finite = np.concatenate(
                [m[np.isfinite(m)].ravel() for m in matrices]
            )
            if finite.size > 0:
                vmin = finite.min() if vmin is None else vmin
                vmax = finite.max() if vmax is None else vmax

        sorted_rows = any(s["link_mat"] is not None for s in sources)
        fig, panels = _panel_grid(
            len(sources),
            [5, 1, 0.25] if sorted_rows else [20, 1],
            dendrogram=sorted_rows,
        )
        for source, matrix, panel in zip(sources, matrices, panels):
            im = self._draw_meth(
                panel,
                matrix,
                vmin=vmin,
                vmax=vmax,
                cmap=cmap,
                link_mat=source["link_mat"],
                xscale=xscale,
                xpow=xpow,
            )
            if source["title"] is not None:
                panel["hm_ax"].set_title(source["title"], pad=12)
        _colorbar(fig, panels, im, cbar_label)
        _left_ylabels(panels)
        if sharex:
            _share_xlim(panels, max(m.shape[1] for m in matrices) / xscale)
        _finish(fig, show, out_file)

    def _draw_meth(
        self, panel, matrix, *, vmin, vmax, cmap, link_mat, xscale, xpow
    ):
        """
        Draw a methylation heatmap into the axes of `panel` (from
        `_panel_grid`), with a dendrogram if `link_mat` is given (rows
        already in its leaf order).

        Returns
        -------
        matplotlib.image.AxesImage
            The heatmap, for the colorbar drawn by the caller.
        """
        hm_ax = panel["hm_ax"]
        nrow, ncol = matrix.shape
        norm = Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap if cmap is not None else self.cmap)
        cmap_obj = cmap_obj.with_extremes(bad=self.nan_color)

        im = hm_ax.imshow(
            matrix,
            cmap=cmap_obj,
            norm=norm,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=[0, ncol / xscale, 0, nrow],
        )
        hm_ax.set_xlabel(rf"Position $x$ [$10^{{{xpow}}}$ bp]")
        sorted_rows = link_mat is not None
        hm_ax.set_ylabel(
            "Row (leaf order)" if sorted_rows else "Molecule index"
        )

        dend_ax = panel["dend_ax"]
        if sorted_rows:
            sch.dendrogram(
                link_mat,
                orientation="right",
                ax=dend_ax,
                no_labels=True,
                link_color_func=lambda x: "black",
            )
        if dend_ax is not None:
            dend_ax.axis("off")
        return im

    @_apply_style
    def plot_meth_energy(
        self,
        data=None,
        *,
        exp=None,
        chroms: str | Iterable[str] | None = None,
        data_name: str | None = None,
        emax: float | None = None,
        max_rows: int | None = None,
        downsample_how: str = "mean",
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str | None = None,
        cbar_label: str = r"Energy [$k_BT$]",
        out_file: str | Path | None = None,
        link_mat: np.ndarray | dict | None = None,
        xscale: int = 1000,
        sharex: bool = False,
        show: bool = True,
    ):
        """
        Plot a methylation heatmap on an energy scale.

        Like `plot_meth_prob`, but showing `logit(data)` instead of the
        raw probability, the same quantity `catella.simulation.engine`
        treats as physical energy, on a color scale centered at zero
        (common to all panels).

        Parameters
        ----------
        data : H5Array, pd.DataFrame, or np.ndarray, optional
            Dense probability matrix in [0, 1], e.g.
            `exp.analysis[chrom]["meth_prob"]`. Required unless `exp`,
            `chroms`, and `data_name` are given instead.
        exp : MethPrintExperiment, optional
            Experiment to read `data_name` from. Not combined with
            `data`.
        chroms : str or iterable of str, optional
            Chromosome(s) to plot from `exp`, one panel each.
        data_name : str, optional
            Key of the probability array per chromosome, read from
            `exp.analysis[chrom][data_name]`.
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
        link_mat : np.ndarray, pd.DataFrame, or dict, optional
            Linkage matrix of these molecules; the rows are put in its
            leaf order and its dendrogram is drawn alongside. With
            several chroms, a dict keyed by chrom. See `plot_meth_prob`.
        xscale : int, default 1000
            Spatial scaling factor (a power of 10) for the x-axis labels.
        sharex : bool, default False
            If True, all panels span the widest chromosome.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `xscale` is not a power of 10, if neither `data` nor
            `exp`, `chroms`, and `data_name` are given, if `data` is
            combined with any of them, or for an invalid `chroms` or
            `link_mat` (see `plot_meth_prob`).
        """
        _power_of_10(xscale, "xscale")
        if data is None and exp is not None and data_name is None:
            raise ValueError(
                "provide either 'data', or 'exp', 'chroms' and 'data_name'"
            )
        sources = self._meth_sources(
            data,
            exp=exp,
            chroms=chroms,
            raw_which=None,
            data_name=data_name,
            mols=None,
            mask_name=None,
            link_mat=link_mat,
        )
        for source in sources:
            energy = logit(np.asarray(source["data"]))
            if emax is not None:
                energy = np.clip(energy, -emax, emax)
            source["data"] = energy

        if vmin is not None or vmax is not None:
            halfrange = max(abs(v) for v in (vmin, vmax) if v is not None)
        elif emax is not None:
            halfrange = emax
        else:
            finite = np.concatenate(
                [s["data"][np.isfinite(s["data"])] for s in sources]
            )
            halfrange = np.nanmax(np.abs(finite)) if finite.size > 0 else None

        if halfrange is not None:
            vmin, vmax = -halfrange, halfrange

        self._plot_panels(
            sources,
            max_rows=max_rows,
            downsample_how=downsample_how,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap if cmap is not None else "RdBu_r",
            cbar_label=cbar_label,
            xscale=xscale,
            sharex=sharex,
            show=show,
            out_file=out_file,
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
