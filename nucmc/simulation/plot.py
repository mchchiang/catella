# plot.py

import matplotlib.pyplot as plt
from functools import wraps
from dataclasses import dataclass, field
from .results import SimDataset
from .analysis import NucFiberMap, SimAnalysis
from pathlib import Path
from matplotlib.colors import Normalize
import numpy as np

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
    
    fontsize : int = 20
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
                     show : bool = True):
        """
        Plot the nucleosome position heatmap for a specific molecule over time.

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
        tscale : int, default=1000000
            Time scaling factor (must be a power of 10) for the y-axis labels.
        xscale : int, default=1000
            Spatial scaling factor (must be a power of 10) for the x-axis
            labels.
        out_file : str or Path, optional
            Path to save the generated figure. Directories are created if
            they do not exist.
        show : bool, default=True
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
        
        # Plot the nucleosome position as a heat map        
        fig, ax = plt.subplots()
        norm = Normalize(vmin=0, vmax=1)
        ax.imshow(occup, cmap=self.cmap, norm=norm, aspect="auto",
                  origin="lower", interpolation="none",
                  extent=[0,occup.shape[1]/xscale,tstart/tscale,tend/tscale])
        xpow_str = rf"$10^{{{xpow}}}$"
        tpow_str = rf"$10^{{{tpow}}}$"
        ax.invert_yaxis()
        ax.set_xlabel(rf"Position $x$ [{xpow_str} bp]")
        ax.set_ylabel(rf"Time $t$ [{tpow_str} MCS]")

        fig.tight_layout()
        
        if show: plt.show()

        if out_file is not None:
            out_path = Path(out_file)
            out_dir = out_path.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
        
    @_apply_style
    def plot_occup(self, *,
                   chrom : str,
                   dataset : SimDataset,
                   time : int | None = None,
                   occup_name : str = "occup",                       
                   xscale : int = 1000,
                   out_file : str | Path | None = None,
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
        xscale : int, default 1000
            Spatial scaling factor (must be a power of 10) for the x-axis
            labels.
        out_file : str or Path, optional
            Path to save the generated figure. Directories are created if
            they do not exist.
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

        # Plot the nucleosome position as a heat map        
        fig, ax = plt.subplots()
        norm = Normalize(vmin=0, vmax=1)
        ax.imshow(occup, cmap=self.cmap, norm=norm, aspect="auto",
                  origin="lower", interpolation="none",
                  extent=[0,occup.shape[1]/xscale,0,occup.shape[0]])
        xpow_str = rf"$10^{{{xpow}}}$"
        ax.set_xlabel(rf"Position $x$ [{xpow_str} bp]")
        ax.set_ylabel(r"Molecule index")

        fig.tight_layout()
        
        if show: plt.show()

        if out_file is not None:
            out_path = Path(out_file)
            out_dir = out_path.parents[0]
            out_dir.mkdir(exist_ok=True, parents=True)
            fig.savefig(out_file)
        
