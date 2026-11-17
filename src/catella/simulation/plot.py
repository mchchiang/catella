# plot.py

import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import wraps
from numbers import Integral
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as sch
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from catella._plotting import _warn_if_large
from catella.mapping import CoordsTransform
from catella.simulation.analysis import NucFiberMap, SimAnalysis
from catella.simulation.results import SimDataset

_DOWNSAMPLE_HOW = ("stride", "mean", "max")


def _check_downsample(max_rows, how):
    """Check `max_rows` and `downsample_how`."""
    if how not in _DOWNSAMPLE_HOW:
        raise ValueError(
            f"downsample_how must be one of {_DOWNSAMPLE_HOW}, got {how!r}"
        )
    if max_rows is not None and max_rows < 1:
        raise ValueError(f"max_rows must be positive, got {max_rows}")


def _check_eseq(dataset, plot_eseq):
    """Check that the energies to plot are stored."""
    if plot_eseq and not dataset.has_eseq:
        raise ValueError(
            "plot_eseq requires stored energies; run with store_eseq=True"
        )


def _stride_step(nrow, max_rows):
    """Step that keeps at most `max_rows` of `nrow` rows (1 if all fit)."""
    if max_rows is None or nrow <= max_rows:
        return 1
    return -(-nrow // max_rows)


def _shown_rows(nrow, max_rows):
    """Number of rows drawn out of `nrow`, at most `max_rows`."""
    return nrow if max_rows is None else min(nrow, max_rows)


def _rows_label(unit, ntotal, nshown):
    """
    Y-axis label of a heatmap whose rows count `unit`s.

    Rows are the `unit`s themselves ("Molecule"), or, after
    downsampling, rows drawn from `ntotal` of them ("Row (100
    molecules)").
    """
    if nshown == ntotal:
        return unit.capitalize()
    return f"Row ({ntotal} {unit}s)"


def _one_run(runs):
    """Return the single run of `runs` (0 if None)."""
    if runs is None:
        return 0
    if isinstance(runs, Integral):
        return int(runs)
    values = list(runs)
    if len(values) != 1:
        raise ValueError(
            f"over='time' shows one run, got {len(values)} runs; use "
            "over='runs' to compare runs"
        )
    return int(values[0])


def _run_list(runs, nsim):
    """Return `runs` as a list of ints (all of `range(nsim)` if None)."""
    if runs is None:
        values = list(range(nsim))
    elif isinstance(runs, Integral):
        values = [int(runs)]
    else:
        values = [int(r) for r in runs]
    if not values:
        raise ValueError("runs must not be empty")
    return values


def _footprints(configs, nbp, nucbp):
    """
    Stack configurations into an int8 footprint matrix.

    Parameters
    ----------
    configs : list of np.ndarray or None
        Nucleosome start positions per row; None for no data.
    nbp : int
        The number of base pairs.
    nucbp : int
        The nucleosome footprint size.

    Returns
    -------
    np.ndarray
        int8, (len(configs), nbp): 1 covered, 0 linker, -1 no data.
    """
    nuc_map = NucFiberMap(nbp=nbp, nucbp=nucbp)
    out = np.full((len(configs), nbp), -1, dtype=np.int8)
    for i, starts in enumerate(configs):
        if starts is not None:
            out[i] = nuc_map.project(np.asarray(starts))
    return out


def _merge_rows(matrix, max_rows, how):
    """
    Merge groups of rows of a footprint matrix into at most `max_rows`.

    Parameters
    ----------
    matrix : np.ndarray
        int8 footprints (1 covered, 0 linker, -1 no data).
    max_rows : int or None
        The maximum number of rows; `matrix` is returned as is if None
        or if it already fits.
    how : {"stride", "mean", "max"}
        "stride" returns `matrix` as is (rows are picked before
        reading). "mean" gives the covered share of each group among
        rows with data (nan if none); "max" gives 1 if any row of the
        group is covered (-1 if no row has data).

    Returns
    -------
    np.ndarray
        The merged matrix: int8 for "max", float64 for "mean".
    """
    nrow = matrix.shape[0]
    if how == "stride" or max_rows is None or nrow <= max_rows:
        return matrix
    edges = np.linspace(0, nrow, max_rows + 1).astype(int)
    out = np.empty(
        (max_rows, matrix.shape[1]),
        dtype=np.float64 if how == "mean" else np.int8,
    )
    for i in range(max_rows):
        block = matrix[edges[i] : edges[i + 1]]
        if how == "max":
            out[i] = block.max(axis=0)
            continue
        count = (block >= 0).sum(axis=0)
        covered = (block > 0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[i] = np.where(count > 0, covered / count, np.nan)
    return out


@dataclass(slots=True, kw_only=True)
class SimPlot:
    """
    A utility class for generating heatmaps and occupancy plots from
    simulation data.

    This class manages global matplotlib settings and provides methods to
    visualize nucleosome positions over time or across different molecules
    within a dataset.
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
        """Initializes the runtime configuration dictionary for matplotlib
        styling."""
        self._rc = {
            "font.size": self.fontsize,
            "axes.linewidth": self.linewidth,
            "axes.axisbelow": False,
            "axes.unicode_minus": False,
        }

    def _apply_style(func):
        """Decorator to apply the class-defined matplotlib RC context to a
        method."""

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            with plt.rc_context(rc=self._rc):
                return func(self, *args, **kwargs)

        return wrapper

    def _log10(self, x, name=None):
        """
        Validates and returns the base-10 logarithm of a value.

        Parameters
        ----------
        x : float
            The value to check.
        name : str, optional
            The variable name to include in the error message.

        Returns
        -------
        float
            The log10 of x.

        Raises
        ------
        ValueError
            If x is not a non-negative power of 10.
        """
        log10x = np.log10(x)
        if log10x < 0.0 or not np.isclose(np.mod(log10x, 1), 0, atol=1e-9):
            xstr = name if name is not None else "x"
            raise ValueError(f"'{xstr}' must be a non-negative power of 10")
        return log10x

    @_apply_style
    def plot_energy(
        self,
        dataset: SimDataset,
        *,
        chrom: str,
        mol: int,
        run: int,
        tstart: int | None = None,
        tend: int | None = None,
        tscale: int = 1000,
        out_file: str | Path | None = None,
        show: bool = True,
    ):
        """
        Plot the total energy of the system for a specific simulation run
        of a molecule over time.

        Requires 'energy' (or 'all') in the `out_types` of `run`.

        Parameters
        ----------
        dataset : SimDataset
            The dataset object containing raw simulation results.
        chrom : str
            Chromosome identifier.
        mol : int
            Molecule index within the dataset.
        run : int
            Simulation run index.
        tstart : int, optional
            Starting time step. If None, defaults to the beginning of the
            simulation.
        tend : int, optional
            Ending time step. If None, defaults to the end of the simulation.
        tscale : int, default 1000
            Time scaling factor (must be a power of 10) for the y-axis labels.
        out_file : str or Path, optional
            Path to save the generated figure. Directories are created if
            they do not exist.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `tend` < `tstart` or if scaling factors are not powers of 10.
        """
        # Check the time values are valid
        if tstart is not None and tend is not None and tend < tstart:
            raise ValueError("'tend' must be greater than 'tstart'")

        tpow = int(self._log10(tscale, "tscale"))

        # Retrieve the data
        data = dataset.raw[chrom, mol, run]

        # Normalize time indices and validate time values
        nframes = len(data.time)
        start_idx = 0 if tstart is None else data.time_index(tstart)
        end_idx = (
            nframes
            if tend is None
            else min(data.time_index(tend) + 1, nframes)
        )
        time = data.time[start_idx:end_idx] / tscale
        energy = data.energy[start_idx:end_idx]

        # Plot the nucleosome position as a heat map
        fig, ax = plt.subplots()
        ax.plot(time, energy)

        tpow_str = rf"$10^{{{tpow}}}$"
        ax.set_xlabel(rf"Time $t$ [{tpow_str} MCS]")
        ax.set_ylabel(r"Energy [$k_BT$]")

        fig.tight_layout()

        if show:
            plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)

    @_apply_style
    def plot_samples(
        self,
        dataset: SimDataset,
        *,
        chrom: str,
        mol: int,
        over: str | None = None,
        runs: int | Iterable[int] | None = None,
        time: int | None = None,
        tstart: int | None = None,
        tend: int | None = None,
        tscale: int = 1000,
        xscale: int = 1000,
        max_rows: int | None = None,
        downsample_how: str = "stride",
        cmap: str | None = None,
        out_file: str | Path | None = None,
        plot_eseq: bool = False,
        show: bool = True,
    ):
        """
        Plot many nucleosome configurations of one molecule.

        Each row is one configuration of molecule `mol`, showing the
        bases covered by nucleosomes (1) and the linkers (0):

        - `over="time"` (MCMC only): the recorded frames of one run
          between `tstart` and `tend`; with `plot_energy`, a check of the
          convergence of the run.
        - `over="runs"`: one configuration per MCMC run at time `time`,
          or one per exact TM sample; shows how variable the molecule's
          positioning is.

        Missing runs or samples, and molecules without data, are drawn
        in `nan_color`. Use `plot_config` to compare molecules.

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the raw configurations.
        chrom : str
            Chromosome identifier.
        mol : int
            Molecule index within `chrom`.
        over : {"time", "runs"}, optional
            What the rows run over. If None (default), "time" for MCMC
            datasets and "runs" for TM datasets.
        runs : int or iterable of int, optional
            With `over="time"`, the one run to show (0 if None). With
            `over="runs"`, the runs (MCMC) or samples (TM) to show, in
            this order (all if None). The y-axis counts rows, not run
            indices.
        time : int, optional
            MCMC time point of the configurations with `over="runs"`; the
            final frame if None.
        tstart : int, optional
            First time step shown with `over="time"`; the beginning if
            None.
        tend : int, optional
            Last time step shown with `over="time"`; the end if None.
        tscale : int, default 1000
            Time scaling factor (a power of 10) for the y-axis labels with
            `over="time"`.
        xscale : int, default 1000
            Spatial scaling factor (a power of 10) for the x-axis labels.
        max_rows : int, optional
            If given, show at most this many rows, merged as
            `downsample_how` says.
        downsample_how : {"stride", "mean", "max"}, default "stride"
            How to reduce the rows to `max_rows`: "stride" keeps every
            k-th row (real configurations), "mean" shows the covered share
            of each group of rows, and "max" whether any of them is
            covered.
        cmap : str, optional
            Matplotlib colormap name. If None, `self.cmap` is used.
        out_file : str or Path, optional
            Path to save the figure. Directories are created if they do
            not exist.
        plot_eseq : bool, default False
            Whether to plot the sequence-specific binding energy of `mol`
            below the heatmap.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `over` is invalid; if `over="time"` is used for a TM
            dataset, with several `runs`, with `time`, or for a molecule
            with no data; if `tstart` or `tend` is used with
            `over="runs"`; if a TM dataset stores no samples; if `tend` <
            `tstart`; if `max_rows` or `downsample_how` is invalid; if a
            scaling factor is not a power of 10; or if `plot_eseq` is True
            and the dataset stores no energies.

        Warns
        -----
        UserWarning
            If rendering the plot is estimated to need more than 1 GB of
            memory; pass `max_rows` to reduce it.

        Notes
        -----
        Requires 'position' (or 'all') in the `out_types` of an MCMC
        `run`.
        """
        xpow = int(self._log10(xscale, "xscale"))
        _check_downsample(max_rows, downsample_how)
        _check_eseq(dataset, plot_eseq)
        if over is None:
            over = "time" if dataset.method == "mcmc" else "runs"
        if over not in ("time", "runs"):
            raise ValueError(f"over must be 'time' or 'runs', got {over!r}")
        nbp = dataset.nbp[chrom]
        hint = "consider passing 'max_rows'"
        if over == "time":
            if dataset.method != "mcmc":
                raise ValueError(
                    "over='time' requires a method='mcmc' dataset; TM "
                    "samples have no time axis"
                )
            if time is not None:
                raise ValueError(
                    "time is only used with over='runs'; use tstart and "
                    "tend"
                )
            run = _one_run(runs)
            if tstart is not None and tend is not None and tend < tstart:
                raise ValueError("'tend' must be greater than 'tstart'")
            if dataset.no_data[chrom][mol]:
                raise ValueError(
                    f"molecule {mol} of chrom '{chrom}' has no data"
                )
            tpow = int(self._log10(tscale, "tscale"))
            data = dataset.raw[chrom, mol, run]
            nframes = len(data.time)
            start_idx = 0 if tstart is None else data.time_index(tstart)
            end_idx = (
                nframes
                if tend is None
                else min(data.time_index(tend) + 1, nframes)
            )
            idx = np.arange(start_idx, end_idx)
            if downsample_how == "stride":
                idx = idx[:: _stride_step(len(idx), max_rows)]
            _warn_if_large(_shown_rows(len(idx), max_rows), nbp, hint)
            matrix = NucFiberMap(nbp=nbp, nucbp=data.nucbp).stack(
                [data.position[i] for i in idx]
            )
            matrix = _merge_rows(matrix, max_rows, downsample_how)
            extent_y = (
                data.time[idx[0]] / tscale,
                data.time[idx[-1]] / tscale,
            )
            ylabel = rf"Time $t$ [$10^{{{tpow}}}$ MCS]"
            invert_y = True
        else:
            if tstart is not None or tend is not None:
                raise ValueError(
                    "tstart and tend are only used with over='time'"
                )
            if dataset.method == "tm" and dataset.nsim == 0:
                raise ValueError("no samples stored (nsim is 0)")
            run_ids = _run_list(runs, dataset.nsim)
            ntotal = len(run_ids)
            if downsample_how == "stride":
                run_ids = run_ids[:: _stride_step(len(run_ids), max_rows)]
            _warn_if_large(_shown_rows(len(run_ids), max_rows), nbp, hint)
            ana = SimAnalysis()
            configs = [
                ana.extract_config(
                    dataset, chrom=chrom, run=r, time=time, mols=[mol]
                )[0]
                for r in run_ids
            ]
            matrix = _footprints(configs, nbp, dataset.settings["nucbp"])
            matrix = _merge_rows(matrix, max_rows, downsample_how)
            extent_y = None
            unit = "run" if dataset.method == "mcmc" else "sample"
            ylabel = _rows_label(unit, ntotal, matrix.shape[0])
            invert_y = False

        eseq = dataset.eseq[chrom][mol, :] if plot_eseq else None
        fig, hm_ax = self._draw_footprints(
            dataset,
            matrix,
            extent_y=extent_y,
            ylabel=ylabel,
            invert_y=invert_y,
            xpow=xpow,
            xscale=xscale,
            cmap=cmap,
            eseq=eseq,
            eseq_label=r"$E_{\text{seq}}$ [$k_BT$]",
        )
        hm_ax.set_title(f"Molecule {mol} of {chrom}", pad=12)
        fig.tight_layout()

        if show:
            plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_file.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)

    @_apply_style
    def plot_config(
        self,
        dataset: SimDataset,
        *,
        chrom: str,
        name: str | None = None,
        mols: Iterable[int] | None = None,
        run: int | None = None,
        time: int | None = None,
        max_rows: int | None = None,
        downsample_how: str = "stride",
        xscale: int = 1000,
        cmap: str | None = None,
        out_file: str | Path | None = None,
        plot_eseq: bool = False,
        show: bool = True,
    ):
        """
        Plot one nucleosome configuration per molecule.

        Each row is a molecule, showing the bases covered by the
        nucleosomes (1) and the linkers (0) of one configuration read with
        `SimAnalysis.extract_config`: the calls stored under `name` (e.g.,
        by `compute_config`), or the raw configuration of run `run` at
        time `time` (MCMC) or of exact sample `run` (TM). Molecules
        without data are drawn in `nan_color`. Use `plot_samples` for
        many configurations of one molecule.

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the configurations.
        chrom : str
            Chromosome identifier.
        name : str, optional
            The prefix the calls were stored under. If None (default),
            raw configurations are shown.
        mols : iterable of int, optional
            Molecules to show, in this order (all if None). The y-axis
            counts rows, not molecule indices.
        run : int, optional
            MCMC run or TM sample to show; 0 if None. Only used if
            `name` is None.
        time : int, optional
            MCMC time point to show; the final frame if None. Only used
            for MCMC datasets with `name` None.
        max_rows : int, optional
            If given, show at most this many rows, merged as
            `downsample_how` says.
        downsample_how : {"stride", "mean", "max"}, default "stride"
            How to reduce the rows to `max_rows`: "stride" keeps every
            k-th molecule (real configurations; only those are read),
            "mean" shows the covered share of each group of molecules,
            and "max" whether any of them is covered.
        xscale : int, default 1000
            Spatial scaling factor (a power of 10) for the x-axis labels.
        cmap : str, optional
            Matplotlib colormap name. If None, `self.cmap` is used.
        out_file : str or Path, optional
            Path to save the figure. Directories are created if they do
            not exist.
        plot_eseq : bool, default False
            Whether to plot the sequence-specific binding energy, averaged
            over the shown molecules, below the heatmap.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `max_rows` or `downsample_how` is invalid, if `xscale` is
            not a power of 10, if `plot_eseq` is True and the dataset
            stores no energies, or for invalid arguments of
            `SimAnalysis.extract_config`.
        KeyError
            If no calls are stored under `name` for `chrom`.
        IndexError
            If a molecule or `run` is out of range.

        Warns
        -----
        UserWarning
            If rendering the plot is estimated to need more than 1 GB of
            memory; pass `max_rows` or `mols` to reduce it.

        Notes
        -----
        The footprints are held as an int8 matrix (1 byte per molecule
        and base pair) before downsampling.
        """
        xpow = int(self._log10(xscale, "xscale"))
        _check_downsample(max_rows, downsample_how)
        _check_eseq(dataset, plot_eseq)
        mol_ids = (
            list(range(dataset.nmol[chrom]))
            if mols is None
            else [int(m) for m in mols]
        )
        ntotal = len(mol_ids)
        if downsample_how == "stride":
            mol_ids = mol_ids[:: _stride_step(len(mol_ids), max_rows)]
        nbp = dataset.nbp[chrom]
        _warn_if_large(
            _shown_rows(len(mol_ids), max_rows),
            nbp,
            "consider passing 'max_rows' or 'mols'",
        )
        configs = SimAnalysis().extract_config(
            dataset, chrom=chrom, name=name, run=run, time=time, mols=mol_ids
        )
        matrix = _footprints(configs, nbp, dataset.settings["nucbp"])
        matrix = _merge_rows(matrix, max_rows, downsample_how)

        eseq = None
        if plot_eseq:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                eseq = np.nanmean(np.asarray(dataset.eseq[chrom])[mol_ids], 0)
        fig, hm_ax = self._draw_footprints(
            dataset,
            matrix,
            extent_y=None,
            ylabel=_rows_label("molecule", ntotal, matrix.shape[0]),
            invert_y=False,
            xpow=xpow,
            xscale=xscale,
            cmap=cmap,
            eseq=eseq,
            eseq_label=r"$\langle E_{\text{seq}} \rangle$ [$k_BT$]",
        )
        hm_ax.set_title(f"Configurations for {chrom}", pad=12)
        fig.tight_layout()

        if show:
            plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_file.parents[0].mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)

    def _draw_footprints(
        self,
        dataset,
        matrix,
        *,
        extent_y,
        ylabel,
        invert_y,
        xpow,
        xscale,
        cmap,
        eseq,
        eseq_label,
    ):
        """
        Draw a footprint matrix (1 covered, 0 linker, -1 or nan no data),
        with an optional energy panel below.

        An int8 matrix (0/1 calls) gets a legend; a float matrix (merged
        with "mean") gets a colorbar of the covered share. If `extent_y`
        is None, the y-axis counts rows: row i is centered on y = i, with
        integer ticks.

        Returns
        -------
        tuple
            The figure and the heatmap axis.
        """
        nbp = matrix.shape[1]
        binary = matrix.dtype == np.int8
        # The second column is a slot for the legend or the colorbar
        gridspec_kw = {"width_ratios": [4, 1] if binary else [20, 1]}
        nrows = 1 if eseq is None else 2
        if eseq is not None:
            gridspec_kw["height_ratios"] = [4, 1]
        fig, ax = plt.subplots(
            nrows=nrows, ncols=2, gridspec_kw=gridspec_kw, squeeze=False
        )
        w, h = fig.get_size_inches()
        fig.set_size_inches(
            w * (1.2 if binary else 1.0), h * (1.0 if eseq is None else 1.25)
        )
        hm_ax, side_ax = ax[0, 0], ax[0, 1]
        seq_ax = None
        if eseq is not None:
            seq_ax = ax[1, 0]
            ax[1, 1].axis("off")

        # -1 (no data) is below vmin; nan comes from "mean" merging
        cmap_obj = plt.get_cmap(cmap if cmap is not None else self.cmap)
        cmap_obj = cmap_obj.with_extremes(
            under=self.nan_color, bad=self.nan_color
        )
        rows_axis = extent_y is None
        if rows_axis:
            extent_y = (-0.5, matrix.shape[0] - 0.5)
        im = hm_ax.imshow(
            matrix,
            cmap=cmap_obj,
            norm=Normalize(vmin=0, vmax=1),
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=[0, nbp / xscale, *extent_y],
        )
        if rows_axis:
            hm_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if invert_y:
            hm_ax.invert_yaxis()
        hm_ax.set_xlim(0, nbp / xscale)
        hm_ax.set_ylabel(ylabel)
        if binary:
            style = {"edgecolor": "black", "linewidth": 0.5}
            handles = [
                Patch(facecolor=cmap_obj(1.0), label="Nucleosome", **style),
                Patch(facecolor=cmap_obj(0.0), label="Linker", **style),
            ]
            if (matrix < 0).any():
                handles.append(
                    Patch(facecolor=self.nan_color, label="No data", **style)
                )
            side_ax.axis("off")
            side_ax.legend(
                handles=handles, loc="center left", frameon=False
            )
        else:
            cbar = fig.colorbar(im, cax=side_ax)
            cbar.set_label("Covered share", rotation=270, labelpad=15)

        if eseq is not None:
            hm_ax.get_xaxis().set_visible(False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                emin = max(np.nanmin(eseq), -dataset.settings["emax"])
                emax = min(np.nanmax(eseq), dataset.settings["emax"])
            if np.isfinite(emin) and np.isfinite(emax):
                seq_ax.set_ylim(emin, emax)
            lnuc = dataset.settings["nucbp"]
            eseq = CoordsTransform(lnuc=lnuc).left_to_center_aligned(eseq)
            eseq[: lnuc // 2] = np.nan
            eseq[len(eseq) - lnuc // 2 :] = np.nan
            seq_ax.plot(np.arange(0, nbp) / xscale, eseq)
            seq_ax.set_ylabel(eseq_label)
            seq_ax.set_xlim(0, nbp / xscale)

        btm_ax = seq_ax if eseq is not None else hm_ax
        btm_ax.set_xlabel(rf"Position $x$ [$10^{{{xpow}}}$ bp]")
        return fig, hm_ax

    @_apply_style
    def plot_occup(
        self,
        dataset: SimDataset,
        *,
        chrom: str,
        time: int | None = None,
        occup_name: str = "occup",
        mols: Iterable[int] | None = None,
        xscale: int = 1000,
        cmap: str | None = None,
        out_file: str | Path | None = None,
        plot_eseq: bool = False,
        link_mat: np.ndarray | None = None,
        show: bool = True,
    ):
        """
        Plot the nucleosome occupancy across all molecules for a specific
        chromosome.

        If the occupancy data is not found in the dataset analysis, it will be
        computed on-the-fly using `SimAnalysis`.

        Parameters
        ----------
        dataset : SimDataset
            The dataset object containing analysis results or raw data.
        chrom : str
            Chromosome identifier.
        time : int, optional
            The time step of interest if occupancy needs to be computed.
        occup_name : str, default "occup"
            The key used to look up or store the occupancy analysis in the
            dataset.
        mols : iterable of int, optional
            Molecule indices to display, restricting the heatmap (and,
            if `plot_eseq` is True, the sequence-energy average) to
            this subset. If None (default), all molecules are shown.
            The y-axis reflects positions within this subset (0..n),
            not the original molecule indices.
        xscale : int, default 1000
            Spatial scaling factor (must be a power of 10) for the x-axis
            labels.
        cmap : str, optional
            Matplotlib colormap name to use for this plot. If None,
            `self.cmap` is used.
        out_file : str or Path, optional
            Path to save the generated figure. Directories are created if
            they do not exist.
        plot_eseq : bool, default False
            Whether to plot the underlying sequence-specific nucleosome binding
            energy, averaged across all molecules.
        link_mat : np.ndarray, optional
            Linkage matrix to draw as a dendrogram alongside the heatmap
            (as returned by `SimAnalysis.sort_by_linkage`). If given,
            `occup_name` should point at the correspondingly-sorted
            array (e.g. `SimAnalysis.sort_by_linkage`'s output) rather
            than the unsorted data. If None (default), no dendrogram is
            drawn and `occup_name` is plotted as-is.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If the occupancy is missing and no `time` is provided for
            computation.

        Warns
        -----
        UserWarning
            If rendering the plot is estimated to need more than 1 GB of
            memory; pass `mols` to plot a subset.
        """
        xpow = int(self._log10(xscale, "xscale"))

        # Retrieve the occupancy data
        if occup_name not in dataset.analysis[chrom]:
            # Compute occupancy on-the-fly if the analysis cannot be found
            ana = SimAnalysis()
            ana.compute_occup(
                dataset=dataset,
                time=time,
                chroms=chrom,
                name=occup_name,
                mean_nnuc_name=None,
            )
        occup = dataset.analysis[chrom][occup_name]
        if mols is not None:
            mols = list(mols)
        _warn_if_large(
            occup.shape[0] if mols is None else len(mols),
            occup.shape[1],
            "consider passing 'mols' to plot a subset",
        )
        if mols is not None:
            occup = np.asarray(occup)[mols]
        sort_data = link_mat is not None

        # Set up the figure. The rightmost column is a dedicated,
        # narrow slot for the colorbar, so it appears to the right of
        # the dendrogram (when shown) instead of shrinking hm_ax.
        ncols = 3 if sort_data else 2
        width_ratios = [5, 1, 0.25] if sort_data else [20, 1]
        gridspec_kw = {"width_ratios": width_ratios}
        if plot_eseq:
            nrows = 2
            gridspec_kw["height_ratios"] = [4, 1]
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, gridspec_kw=gridspec_kw
            )
            w, h = fig.get_size_inches()
            fig.set_size_inches(w * 1.25, h * 1.25)
            hm_ax = ax[0, 0]
            dend_ax = ax[0, 1] if sort_data else None
            cbar_ax = ax[0, 2] if sort_data else ax[0, 1]
            seq_ax = ax[1, 0]
            ax[1, 1].axis("off")
            if sort_data:
                ax[1, 2].axis("off")
        else:
            nrows = 1
            fig, ax = plt.subplots(
                nrows=nrows, ncols=ncols, gridspec_kw=gridspec_kw
            )
            hm_ax = ax[0]
            dend_ax = ax[1] if sort_data else None
            cbar_ax = ax[2] if sort_data else ax[1]
            seq_ax = None

        # Plot the nucleosome position as a heat map
        nbp = dataset.nbp[chrom]
        norm = Normalize(vmin=0, vmax=1)
        cmap_obj = plt.get_cmap(cmap if cmap is not None else self.cmap)
        cmap_obj = cmap_obj.with_extremes(bad=self.nan_color)
        im = hm_ax.imshow(
            occup,
            cmap=cmap_obj,
            norm=norm,
            aspect="auto",
            origin="lower",
            interpolation="none",
            extent=[0, occup.shape[1] / xscale, 0, occup.shape[0]],
        )
        xpow_str = rf"$10^{{{xpow}}}$"
        hm_ax.set_xlim(0, nbp / xscale)
        if plot_eseq:
            hm_ax.get_xaxis().set_visible(False)
        hm_ax.set_ylabel(r"Molecule index")
        cbar = fig.colorbar(im, cax=cbar_ax)
        cbar.set_label("Occupancy", rotation=270, labelpad=15)

        # Plot dendrogram on the right panel
        if sort_data:
            sch.dendrogram(
                link_mat,
                orientation="right",
                ax=dend_ax,
                no_labels=True,
                link_color_func=lambda x: "black",
            )
            dend_ax.axis("off")

        # Plot the sequence energy if needed
        if plot_eseq:
            eseq_src = dataset.eseq[chrom]
            if mols is not None:
                eseq_src = np.asarray(eseq_src)[list(mols)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                eseq = np.nanmean(eseq_src, axis=0)
                emin = max(np.nanmin(eseq), -dataset.settings["emax"])
                emax = min(np.nanmax(eseq), dataset.settings["emax"])
            if np.isfinite(emin) and np.isfinite(emax):
                seq_ax.set_ylim(emin, emax)
            lnuc = dataset.settings["nucbp"]
            trans = CoordsTransform(lnuc=lnuc)
            eseq = trans.left_to_center_aligned(eseq)
            eseq[: lnuc // 2] = np.nan
            eseq[len(eseq) - lnuc // 2 :] = np.nan
            seq_ax.plot(np.arange(0, nbp) / xscale, eseq)
            seq_ax.set_ylabel(r"$\langle E_{\text{seq}} \rangle$ [$k_BT$]")
            seq_ax.set_xlim(0, nbp / xscale)

        # Common x-axis label
        btm_ax = seq_ax if plot_eseq else hm_ax
        btm_ax.set_xlabel(rf"Position $x$ [{xpow_str} bp]")

        hm_ax.set_title(f"Occupancy Profile for {chrom}", pad=12)

        if show:
            plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
