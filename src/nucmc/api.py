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
               colidx : Iterable | None = None,
               max_nmol : int | None = None,
               seed : int | None = None,
               clip_low : float = 0.1,
               clip_high : float = 99.9,
               norm_by_strand : bool = False,
               batch_size : int = 20_000,
               percentile_sample_size : int = 100_000,
               tmp_dir : str | Path | None = None) -> MethPrintExperiment:
    """
    Preprocess raw methylation data to create a MethPrintExperiment.

    Load raw ModKit data and perform normalization to convert the data into a
    methylation probability score, and can persist the resulting experiment
    object to disk.

    Parameters
    ----------
    chromsize : str or Path
        Path to the chromosome sizes file or a string identifier for the
        genome.
    test_file : str or Path
        Path to the primary experimental methylation data file.
    out_file : str or Path, optional
        Path where the processed `MethPrintExperiment` will be saved. 
        If None, the result is only returned in-memory.
    unmeth_file : str or Path, optional
        Path to the unmethylated control file.
    meth_file : str or Path, optional
        Path to the methylated control file.
    binsize : int, default 147
        The genomic window size (in base pairs) used for data aggregation. The
        default value corresponds to the typical DNA footprint of a nucleosome.
    wrap : bool, default False
        If True, calculates positions relative to the fiber center
        (useful for circular or symmetrical fibers).
    colidx : Iterable, optional
        Specific column indices to use if the input file does not follow    
        the standard ModKit format.
    max_nmol : int, optional
        Maximum number of molecules to extract for each chromosome.
    seed : int, optional
        The seed for the random number generator selecting the molecules if
        `max_nmol` is specified.        
    clip_low : float, default 0.1
        Lower percentile bound for signal clipping. Values below this
        percentile are set to 0. 
    clip_high : float, default 99.9
        Upper percentile bound for signal clipping. Values above this
        percentile are set to 1.    
    norm_by_strand : bool, default False
        Whether to perform normalization separately based on strandedness.
    batch_size : int, default 20_000
        Number of molecules processed (and held in memory) per batch
        during smoothing and probability calculation.
    percentile_sample_size : int, default 100_000
        Approximate number of molecules used to estimate percentile clip
        bounds during probability calculation.
    tmp_dir : str or Path, optional
        Directory used for scratch files backing intermediate analysis
        results. If None, the system default temporary directory is used.

    Returns
    -------
    MethPrintExperiment
        An initialized experiment object containing the processed methylation
        data. Normalization is done to convert the signal into a methylation
        probability score (accounting for the control datasets if provided).
    """

    # Load the raw data (generated from ModKit)
    exp_data = MethPrintExperiment.load_raw(
        chromsize=chromsize, test_file=test_file, unmeth_file=unmeth_file,
        meth_file=meth_file, wrap=wrap, colidx=colidx, max_nmol=max_nmol,
        seed=seed)

    # Smooth and normalize the data - compute methylation probability
    ana = MethPrintAnalysis()
    ana.smooth(binsize=binsize, exp=exp_data, batch_size=batch_size,
              tmp_dir=tmp_dir)
    ana.meth_prob(exp=exp_data, clip_low=clip_low, clip_high=clip_high,
                  norm_by_strand=norm_by_strand, batch_size=batch_size,
                  percentile_sample_size=percentile_sample_size,
                  tmp_dir=tmp_dir)
    
    # Save the results
    if out_file is not None:
        exp_data.save(out_file)

    return exp_data


def run(*, chroms : str | Iterable[str],
        nsim : int,
        settings : str | Path | SimSettings,
        meth_prob : np.ndarray | Mapping[str,np.ndarray|pd.DataFrame],
        out_dir : str | Path,
        dataset_name : str = "results",
        out_types : str | Iterable[str] = "all",        
        seed : int | None = None,        
        mols : IndexType | Mapping[str,IndexType] = slice(None),
        store_eseq : bool = True,
        use_zero_point_mu : bool = False,
        nworker : int = 1,
        mp_context : str | None = None,
        verbose : bool = True) -> SimDataset:
    """
    Execute a parallelized methylation simulation.

    Orchestrate the simulation process using a `SimManager` to handle data 
    distribution across multiple workers. Generate synthetic datasets based
    on provided methylation profiles and simulation settings.

    Parameters
    ----------
    chroms : str or iterable of str
        The identifier(s) of chromosome(s) to include in the simulation.
    nsim : int
        Number of independent simulation runs per molecule.
    settings : str or Path or SimSettings
        Simulation parameters. Can be a path to a configuration file or a 
        `SimSettings` object.
    meth_prob : np.ndarray or Mapping[str, np.ndarray | pd.DataFrame]
        Probability of methylation. If multiple chromosomes are provided, this
        must be a mapping of {chrom_name: data}. Data can be NumPy arrays or
        Pandas DataFrames. 
    out_dir : str or Path
        Directory or file prefix where simulation results will be stored.
    out_types : str | Iterable[str], default 'all'
        Types of data to record. Options include 'energy', 'position',      
        'temp', or 'all'. 
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    mols : IndexType | Mapping[str, IndexType], default slice(None
        Specific molecule indices to subset for the simulation.
    store_eseq : bool, default True
        Whether to store the sequence-specific nucleosome binding energy
        landscape derived from the methylation data to output dataset file.
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
                          meth_prob=meth_prob, out_types=out_types,
                          out_dir=out_dir, dataset_name=dataset_name,
                          seed=seed, mols=mols, store_eseq=store_eseq,
                          use_zero_point_mu=use_zero_point_mu)    
    return dataset

def analyze(*, dataset : SimDataset,
            time : int | None = None,
            occup_name : str = "occup",
            mean_nnuc_name : str = "mean_nnuc"):
    """
    Perform post-simulation statistical analysis on a dataset.

    Calculate key nucleosome metrics, such as site occupancy and average
    nucleosome counts, across the simulated trajectories. Results are stored
    directly back into the provided SimDataset.

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
    """
    ana = SimAnalysis()
    ana.compute_occup(dataset=dataset, time=time, name=occup_name)
    ana.compute_mean_nnuc(dataset=dataset, time=time, name=mean_nnuc_name)

def plot_occup(*, chrom : str,
               dataset : SimDataset,
               out_file : str | Path | None = None,
               occup_name : str = "occup",
               plot_eseq : bool = False,               
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
        The key or name of the occupancy data to retrieve from the dataset. 
        This should match the name used during the `analyze` step.
    plot_eseq : bool, default True
        Whether to plot the underlying sequence-specific nucleosome binding 
        energy, averaged across all molecules.
    show : bool, default True
        If True, invokes the active plotting backend to display the 
        figure immediately.

    Notes
    -----
    This function requires that `analyze()` (specifically `compute_occup`) 
    has been called on the dataset prior to plotting.
    """
    simplot = SimPlot()
    simplot.plot_occup(chrom=chrom, dataset=dataset, occup_name=occup_name,
                       out_file=out_file, plot_eseq=plot_eseq, show=show)

def plot_nuc_pos(*, chrom : str,
                 mol : int,
                 run : int,
                 dataset : SimDataset,
                 out_file : str | Path | None = None,
                 plot_eseq : bool = False,
                 show : bool = True):
    """
    Plot the time-course positions of nucleosomes for a specific simulation
    run of a molecule.

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
    plot_eseq : bool, default False
        Whether to plot the underlying sequence-specific nucleosome binding 
        energy.
    show : bool, default True
        Whether to display the figure using the active plotting backend.

    Notes
    -----
    This visualization requires that 'position' (or 'all') was included 
    in the `out_types` during the `run` execution. If `plot_seq` is True,
    it also requires that `store_eseq` was set to True during `run`.
    """
    simplot = SimPlot()
    simplot.plot_nuc_pos(chrom=chrom, mol=mol, run=run, dataset=dataset,
                         out_file=out_file, plot_eseq=plot_eseq, show=show)


def plot_energy(*, chrom : str,
                mol : int,
                run : int,
                dataset : SimDataset,
                out_file : str | Path | None = None,
                show : bool = True):
    """
    Plot the total energy of the system for a specific simulation run of a
    molecule over time.

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
        Whether to display the figure using the active plotting backend.

    Notes
    -----
    This visualization requires that 'energy' (or 'all') was included 
    in the `out_types` during the `run` execution.
    """
    simplot = SimPlot()
    simplot.plot_energy(chrom=chrom, mol=mol, run=run, dataset=dataset,
                        out_file=out_file, show=show)
    
