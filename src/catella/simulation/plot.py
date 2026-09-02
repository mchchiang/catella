# plot.py

import warnings
import matplotlib.pyplot as plt
from functools import wraps
from dataclasses import dataclass, field
from collections.abc import Iterable
from catella.simulation.results import SimDataset
from catella.simulation.analysis import NucFiberMap, SimAnalysis
from pathlib import Path
from matplotlib.colors import Normalize
import numpy as np
import scipy.cluster.hierarchy as sch
from catella.mapping import CoordsTransform

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
    linewidth : int = 1.0
    """The border and axis line thickness for all plots."""
    
    fontsize : int = 14
    """The base font size for labels, ticks, and titles."""
    
    cmap : str = "viridis"
    """The Matplotlib colormap name used for heatmaps."""
    
    _rc : dict = field(init=None)

    def __post_init__(self):
        """Initializes the runtime configuration dictionary for matplotlib
        styling."""
        self._rc = {"font.size" : self.fontsize,
                    "axes.linewidth" : self.linewidth,
                    "axes.axisbelow" : False,
                    "axes.unicode_minus" : False
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
            raise ValueError(f"'{name}' must be a non-negative power of 10.")
        return log10x

    @_apply_style
    def plot_energy(self, *,
                    chrom : str,
                    mol : int,
                    run : int,
                    dataset : SimDataset,
                    tstart : int | None = None,
                    tend : int | None = None,
                    tscale : int = 1000000,
                    out_file : str | Path | None = None,
                    show : bool = True):
        """
        Plot the total energy of the system for a specific simulation run
        of a molecule over time.

        Parameters
        ----------
        chrom : str
            Chromosome identifier.
        mol : int
            Molecule index within the dataset.
        run : int
            Simulation run index.
        dataset : SimDataset
            The dataset object containing raw simulation results.
        tstart : int, optional
            Starting time step. If None, defaults to the beginning of the
            simulation.
        tend : int, optional
            Ending time step. If None, defaults to the end of the simulation.
        tscale : int, default 1000000
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
            raise ValueError("'tend' must be greater than 'tstart'.")

        tpow = int(self._log10(tscale, "tscale"))
        
        # Retrieve the data
        data = dataset.raw[chrom,mol,run]
        
        # Normalize time indices and validate time values
        nframes = len(data.time)
        start_idx = 0 if tstart is None else data.time_index(tstart)
        end_idx = nframes if tend is None else \
            min(data.time_index(tend)+1, nframes)
        time = data.time[start_idx:end_idx] / tscale
        energy = data.energy[start_idx:end_idx]
        
        # Plot the nucleosome position as a heat map        
        fig, ax = plt.subplots()
        ax.plot(time, energy)
        
        tpow_str = rf"$10^{{{tpow}}}$"
        ax.set_xlabel(rf"Time $t$ [{tpow_str} MCS]")
        ax.set_ylabel(rf"Energy [$k_BT$]")        

        fig.tight_layout()
        
        if show: plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
    
    
    @_apply_style
    def plot_nuc_pos(self, *,
                     chrom : str,
                     mol : int,
                     run : int,
                     dataset : SimDataset,
                     tstart : int | None = None,
                     tend : int | None = None,
                     tscale : int = 1000000,
                     xscale : int = 1000,
                     out_file : str | Path | None = None,
                     plot_eseq : bool = False,
                     show : bool = True):
        """
        Plot the nucleosome position heatmap for a specific simulation run of
        a molecule over time.

        Parameters
        ----------
        chrom : str
            Chromosome identifier.
        mol : int
            Molecule index within the dataset.
        run : int
            Simulation run index.
        dataset : SimDataset
            The dataset object containing raw simulation results.
        tstart : int, optional
            Starting time step. If None, defaults to the beginning of the
            simulation.
        tend : int, optional
            Ending time step. If None, defaults to the end of the simulation.
        tscale : int, default 1000000
            Time scaling factor (must be a power of 10) for the y-axis labels.
        xscale : int, default 1000
            Spatial scaling factor (must be a power of 10) for the x-axis
            labels.
        out_file : str or Path, optional
            Path to save the generated figure. Directories are created if
            they do not exist.
        plot_eseq : bool, default False
            Whether to plot the underlying sequence-specific nucleosome binding
            energy.
        show : bool, default True
            Whether to display the plot using `plt.show()`.

        Raises
        ------
        ValueError
            If `tend` < `tstart` or if scaling factors are not powers of 10.
        """
        # Check the time values are valid
        if tstart is not None and tend is not None and tend < tstart:
            raise ValueError("'tend' must be greater than 'tstart'.")

        tpow = int(self._log10(tscale, "tscale"))
        xpow = int(self._log10(xscale, "xscale"))
        
        # Retrieve the data
        data = dataset.raw[chrom,mol,run]
        nuc_map = NucFiberMap(nbp=data.nbp, nucbp=data.nucbp)

        # Normalize time indices and validate time values
        nframes = len(data.time)
        start_idx = 0 if tstart is None else data.time_index(tstart)
        end_idx = nframes if tend is None else \
            min(data.time_index(tend)+1, nframes)

        # Compute the nucleosome occupancy map
        occup = nuc_map.stack(data.position[start_idx:end_idx])
        tstart = data.time[start_idx]
        tend = data.time[end_idx-1]
        
        # Set up the figure
        if plot_eseq:
            nplots = 2
            fig, ax = plt.subplots(nrows=nplots, ncols=1,
                                   gridspec_kw={"height_ratios": [4, 1]})
            w, h = fig.get_size_inches()
            fig.set_size_inches(w, h*1.25) 
        else:
            nplots = 1
            fig, ax = plt.subplots(nrows=nplots, ncols=1)
            ax = [ax]

        # Plot the nucleosome position as a heat map            
        norm = Normalize(vmin=0, vmax=1)
        ax[0].imshow(occup, cmap=self.cmap, norm=norm, aspect="auto",
                     origin="lower", interpolation="none",
                     extent=[0, data.nbp/xscale, tstart/tscale, tend/tscale])
        xpow_str = rf"$10^{{{xpow}}}$"
        tpow_str = rf"$10^{{{tpow}}}$"
        ax[0].invert_yaxis()
        ax[0].set_xlim(0, data.nbp/xscale)
        if plot_eseq:
            ax[0].get_xaxis().set_visible(False)
        ax[0].set_ylabel(rf"Time $t$ [{tpow_str} MCS]")

        # Plot the sequence energy if needed
        if plot_eseq:
            eseq = dataset.eseq[chrom][mol,:]
            med = np.median(eseq)
            sigma = np.median(np.abs(eseq-med)) * 1.4826 # MAD to SD
            nsig = 3 # Plot up to how many sigma
            emin = -nsig*sigma+med
            emax = min(med+nsig*sigma,dataset.settings["emax"])
            ax[1].set_ylim(emin,emax)            
            binsize = dataset.settings["nucbp"]            
            trans = CoordsTransform(binsize=binsize)
            eseq = trans.left_to_center_aligned(eseq)
            eseq[:binsize//2] = np.nan
            eseq[len(eseq)-binsize//2:] = np.nan            
            ax[1].plot(np.arange(0,data.nbp)/xscale, eseq)
            ax[1].set_ylabel(r"$E_{\text{seq}}$ [$k_BT$]")            
            ax[1].set_xlim(0, data.nbp/xscale)

        # Common x-axis label
        ax[nplots-1].set_xlabel(rf"Position $x$ [{xpow_str} bp]")
        
        fig.tight_layout()
        
        if show: plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
        
    @_apply_style
    def plot_occup(self, *,
                   chrom : str,
                   dataset : SimDataset,
                   time : int | None = None,
                   occup_name : str = "occup",
                   mols : Iterable[int] | None = None,
                   xscale : int = 1000,
                   out_file : str | Path | None = None,
                   plot_eseq : bool = False,
                   link_mat : np.ndarray | None = None,
                   show : bool = True):
        """
        Plot the nucleosome occupancy across all molecules for a specific
        chromosome.

        If the occupancy data is not found in the dataset analysis, it will be
        computed on-the-fly using `SimAnalysis`.

        Parameters
        ----------
        chrom : str
            Chromosome identifier.
        dataset : SimDataset
            The dataset object containing analysis results or raw data.
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
        """
        xpow = int(self._log10(xscale, "xscale"))

        # Retrieve the occupancy data
        if occup_name not in dataset.analysis[chrom]:
            # Compute occupancy on-the-fly if the analysis cannot be found
            ana = SimAnalysis()
            ana.compute_occup(dataset=dataset, time=time, chroms=chrom,
                              name=occup_name)
        occup = dataset.analysis[chrom][occup_name]
        if mols is not None:
            occup = np.asarray(occup)[list(mols)]
        sort_data = link_mat is not None
        
        # Set up the figure
        ncols = 2 if sort_data else 1
        gridspec_kw = {"width_ratios": [5, 1]} if sort_data else {}
        if plot_eseq:
            nrows = 2
            gridspec_kw["height_ratios"] = [4, 1]
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                                   gridspec_kw=gridspec_kw)
            w, h = fig.get_size_inches()
            fig.set_size_inches(w*1.25, h*1.25)
            hm_ax = ax[0,0] if sort_data else ax[0]
            dend_ax = ax[0,1] if sort_data else None
            seq_ax = ax[1,0] if sort_data else ax[1]
            if sort_data: ax[1,1].axis("off")
        else:
            nrows = 1
            fig, ax = plt.subplots(nrows=nrows, ncols=ncols,
                                   gridspec_kw=gridspec_kw)
            hm_ax = ax[0] if sort_data else ax
            dend_ax = ax[1] if sort_data else None
            seq_ax = None
        
        # Plot the nucleosome position as a heat map
        nbp = dataset.nbp[chrom]
        norm = Normalize(vmin=0, vmax=1)
        hm_ax.imshow(occup, cmap=self.cmap, norm=norm, aspect="auto",
                     origin="lower", interpolation="none",
                     extent=[0,occup.shape[1]/xscale,0,occup.shape[0]])
        xpow_str = rf"$10^{{{xpow}}}$"
        hm_ax.set_xlim(0, nbp/xscale)
        if plot_eseq:
            hm_ax.get_xaxis().set_visible(False)
        hm_ax.set_ylabel(r"Molecule index")

        # Plot dendrogram on the right panel
        if sort_data:
            sch.dendrogram(link_mat, orientation="right", ax=dend_ax,
                           no_labels=True, link_color_func=lambda x : "black")
            dend_ax.axis("off")

        # Plot the sequence energy if needed
        if plot_eseq:
            eseq_src = dataset.eseq[chrom]
            if mols is not None:
                eseq_src = np.asarray(eseq_src)[list(mols)]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                eseq = np.nanmean(eseq_src, axis=0)
                med = np.nanmedian(eseq)
                sigma = np.nanmedian(np.abs(eseq-med)) * 1.4826 # MAD to SD
            nsig = 3 # Plot up to how many sigma
            emin = -nsig*sigma+med
            emax = min(med+nsig*sigma,dataset.settings["emax"])
            seq_ax.set_ylim(emin,emax)            
            binsize = dataset.settings["nucbp"]            
            trans = CoordsTransform(binsize=binsize)
            eseq = trans.left_to_center_aligned(eseq)
            eseq[:binsize//2] = np.nan
            eseq[len(eseq)-binsize//2:] = np.nan
            seq_ax.plot(np.arange(0,nbp)/xscale, eseq)
            seq_ax.set_ylabel(r"$\langle E_{\text{seq}} \rangle$ [$k_BT$]")
            seq_ax.set_xlim(0, nbp/xscale)

        # Common x-axis label
        btm_ax = seq_ax if plot_eseq else hm_ax
        btm_ax.set_xlabel(rf"Position $x$ [{xpow_str} bp]")

        hm_ax.set_title(f"Occupancy Profile for {chrom}", pad=12)
        
        if show: plt.show()

        if out_file is not None:
            out_file = Path(out_file)
            out_dir = out_file.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
        
