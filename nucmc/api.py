# api.py

# A high-level interface for processing methylation footprinting data and
# running simulations

from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment.preprocessing import MethPrintAnalysis
from .experiment.methdata import MethPrintExperiment
from .simulation.config import SimSettings
from .simulation.engine import SimManager
from .simulation.results import SimDataset
from .simulation.analysis import SimAnalysis
from .simulation.plot import SimPlot
from .utils import IndexType

def preprocess(*, chromsize : str | Path,               
               test_file : str | Path,
               out_file : str | Path | None = None,
               unmeth_file : str | Path | None = None,
               meth_file : str | Path | None = None,
               binsize : int = 147,               
               wrap : bool = False,
               colidx : Iterable | None = None) -> MethPrintExperiment:
    """
    Preprocess raw methylation data to create a MethPrintExperiment.

    This high-level API loads raw modkit data, optionally performs
    normalization if both methylated and unmethylated control files are
    provided, and can persist the resulting experiment object to disk.

    Parameters
    ----------
    chromsize : str | Path
        Path to the chromosome sizes file or a string identifier for the
        genome.
    test_file : str | Path
        Path to the primary experimental methylation data file.
    out_file : str | Path, optional
        Path where the processed `MethPrintExperiment` will be saved. 
        If None, the result is only returned in-memory.
    unmeth_file : str | Path, optional
        Path to the unmethylated control file. Required for normalization.
    meth_file : str | Path, optional
        Path to the methylated control file. Required for normalization.
    binsize : int, default 147
        The genomic window size (in base pairs) used for data aggregation. The
        default value corresponds to the typical DNA footprint of a nucleosome.
    wrap : bool, default False
        If True, calculates positions relative to the fiber center
        (useful for circular or symmetrical fibers). 
    colidx : Iterable, optional
        Specific column indices to use if the input file does not follow    
        the standard modkit format.
    
    Returns
    -------
    MethPrintExperiment
        An initialized (and potentially normalized) experiment object
        containing the processed methylation data.

    .. note::
       Normalization is only triggered if **both** `meth_file` and
       `unmeth_file` are provided. This process utilizes
       `MethPrintAnalysis.normalize` to adjust the experimental signal
       against the control baselines.
    """

    # Load the raw data (generated from modkit)
    exp_data = MethPrintExperiment.load_raw(
        chromsize=chromsize, test_file=test_file, unmeth_file=unmeth_file,
        meth_file=meth_file, wrap=wrap, colidx=colidx)

    # Normalize the data as required
    if meth_file is not None and unmeth_file is not None:
        ana = MethPrintAnalysis()
        ana.normalize(binsize, exp_data)
    
    # Save the results
    if out_file is not None:
        exp_data.save(out_file)

    return exp_data


def run(*, chroms : str | Iterable[str],
        nsim : int,
        settings : str | Path | SimSettings,
        meth : np.ndarray | Mapping[str,np.ndarray|pd.DataFrame],
        out_path : str | Path,
        out_types : str | Iterable[str] = "all",        
        seed : int | None = None,        
        mols : IndexType | Mapping[str,IndexType] = slice(None),
        store_emeth : bool = True,
        use_zero_point_mu : bool = False,
        nworker : int = 1,
        mp_context : str | None = None,
        verbose : bool = True) -> SimDataset:
    """
    Execute a parallelized methylation simulation.

    Orchestrates the simulation process using a `SimManager` to handle data 
    distribution across multiple workers. It generates synthetic datasets 
    based on provided methylation profiles and simulation settings.

    Parameters
    ----------
    chroms : str | Iterable[str]
        The identifier(s) of chromosome(s) to include in the simulation.
    nsim : int
        Number of independent simulation runs per molecule.
    settings : str | Path | SimSettings
        Simulation parameters. Can be a path to a configuration file or a 
        `SimSettings` object.
    meth : np.ndarray | Mapping[str, np.ndarray | pd.DataFrame]
        Methylation data. If multiple chromosomes are provided, this
        must be a mapping of {chrom_name: data}. Data can be NumPy arrays or
        Pandas DataFrames. 
    out_path : str | Path
        Directory or file prefix where simulation results will be stored.
    out_types : str | Iterable[str], default 'all'
        Types of data to record. Options include 'energy', 'position',      
        'temp', or 'all'. 
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    mols : IndexType | Mapping[str, IndexType], default slice(None)
        Specific molecule indices to subset for the simulation.
    store_emeth : bool, default True
        Whether to store the energy landscape derived from the methylation
        data to output dataset file or not.
    use_zero_point_mu : bool : default False
        Whether to modify the chemical potential parameter so that it is
        equal to the mean of the methlyation energy.    
    nworker : int, default 1
        Number of parallel processes to spawn.
    mp_context : {'spawn', 'forkserver'}, optional
        The multiprocessing start method. Default to 'spawn' on macOS
        and 'forkserver' on other platforms. 
    verbose : bool, default True
        If True, print progress updates to the console.

    Returns
    -------
    SimDataset
        A container object providing access to the generated simulation
        results.
    """
    manager = SimManager(nworker=nworker, mp_context=mp_context,
                         verbose=verbose)
    dataset = manager.run(chroms=chroms, nsim=nsim, settings=settings,
                          meth=meth, out_types=out_types, out_path=out_path,
                          seed=seed, mols=mols, store_emeth=store_emeth,
                          use_zero_point_mu=use_zero_point_mu)    
    return dataset

def analyze(*, dataset : SimDataset,
            time : int | None = None,
            occup_name : str = "occup",
            mean_nnuc_name : str = "mean_nnuc"):
    """
    Perform post-simulation statistical analysis on a dataset.

    This function calculates key nucleosome metrics, such as site occupancy 
    and average nucleosome counts, across the simulated trajectories. Results 
    are stored directly back into the provided SimDataset.

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset to analyze.
    time : int, optional
        A specific time point (snapshot) from the simulation to analyze. 
        If None, the final state of the simulation is typically used.
    occup_name : str, default "occup"
        The key/name under which to store the computed occupancy data 
        within the dataset.
    mean_nnuc_name : str, default "mean_nnuc"
        The key/name under which to store the computed mean number of 
        nucleosomes within the dataset.

    Returns
    -------
    SimDataset
        The input dataset object, updated with the newly computed 
        analysis metrics.
    """
    ana = SimAnalysis()
    ana.compute_occup(dataset=dataset, time=time, name=occup_name)
    ana.compute_mean_nnuc(dataset=dataset, time=time, name=mean_nnuc_name)

def plot_occup(*, chrom : str,
               dataset : SimDataset,
               out_file : str | Path | None = None,
               occup_name : str = "occup",
               show : bool = True):
    """
    Visualize nucleosome occupancy profiles for a specific chromosome.

    Generate a plot showing the probability of nucleosome occupancy across 
    the genomic coordinates based on previously computed analysis within 
    the SimDataset.

    Parameters
    ----------
    chrom : str
        The identifier of the chromosome to plot.
    dataset : SimDataset
        The simulation dataset containing the computed occupancy data.
    out_file : str | Path, optional
        Path where the generated plot will be saved. If None, the plot 
        is not saved to disk.
    occup_name : str, default "occup"
        The key/name of the occupancy data to retrieve from the dataset. 
        This should match the name used during the `analyze` step.
    show : bool, default True
        If True, invokes the active plotting backend to display the 
        figure immediately.

    Notes
    -----
    This function requires that `analyze()` (specifically `compute_occupancy`) 
    has been called on the dataset prior to plotting.
    """
    simplot = SimPlot()
    simplot.plot_occup(chrom=chrom, dataset=dataset, occup_name=occup_name,
                       out_file=out_file, show=show)

def plot_nuc_pos(*, chrom : str,
                 mol : int,
                 run : int,
                 dataset : SimDataset,
                 out_file : str | Path | None = None,
                 show : bool = True):
    """
    Plot the time-course positions of nucleosomes for a single simulation run.

    Generate a trajectory plot (often a kymograph or "spaghetti plot") 
    visualizing how nucleosomes move or remain stable over simulation time 
    steps for a specific molecule.

    Parameters
    ----------
    chrom : str
        The identifier of the chromosome to plot.
    mol : int
        The index of the specific molecule (fiber) to visualize.
    run : int
        The specific simulation run index for the chosen molecule.
    dataset : SimDataset
        The simulation dataset containing the positional coordinates.
    out_file : str | Path, optional
        Path where the generated plot will be saved. If None, the plot 
        is not saved to disk.
    show : bool, default True
        If True, displays the figure using the active plotting backend.

    Notes
    -----
    This visualization requires that 'position' (or 'all') was included 
    in the `out_types` during the `run` execution.
    """
    simplot = SimPlot()
    simplot.plot_nuc_pos(chrom=chrom, mol=mol, run=run, dataset=dataset,
                         out_file=out_file, show=show)
