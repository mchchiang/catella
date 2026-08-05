# api.py

# A high-level interface for processing methylation footprinting data and
# running simulations

from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np
import pandas as pd
from .experiment.preprocessing import MethPrintAnalysis
from .experiment.methdata import MethPrintExperiment
from .experiment.plot import MethPlot
from .h5_array import H5Array
from .simulation.config import SimSettings
from .simulation.engine import SimManager
from .simulation.results import SimDataset
from .simulation.analysis import SimAnalysis
from .simulation.plot import SimPlot
from . import utils
from .utils import IndexType

def preprocess(*, chromsize : str | Path,               
               test_file : str | Path,
               out_file : str | Path | None = None,
               unmeth_file : str | Path | None = None,
               meth_file : str | Path | None = None,
               fasta_file : str | Path | None = None,
               mtase : str | Iterable[str] | None = None,
               binsize : int = 147,
               wrap : bool = False,
               colidx : Iterable | None = None,
               max_nmol : int | None = None,
               seed : int | None = None,
               clip_low : float = 0.1,
               clip_high : float = 99.9,
               norm_by_strand : bool = False,
               batch_size : int = 20000,
               percentile_sample_size : int = 100000,
               tmp_dir : str | Path | None = None,
               chunk_size : int = 1000000,
               max_cached_chroms : int = 1) -> MethPrintExperiment:
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
    fasta_file : str or Path, optional
        Multi-FASTA file of per-chromosome reference sequences (record
        id matching `chromsize`); stored on `MethPrintData.refseq`.
    mtase : str or iterable of str, optional
        Methyltransferase(s) used to generate the test data: 'A'
        (any-context adenine, e.g., EcoGII), 'CG' (CpG, e.g., M.SssI),
        'GC' (GpC, e.g., M.CviPI). One label, a list of labels, or
        None (default) if unspecified.
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
        percentile are set to 0. If `unmeth_file`/`meth_file` controls
        are provided, this percentile is estimated from the normalized
        unmeth control channel (making the bound independent of which
        test data is processed); otherwise it is estimated from the
        test signal itself.
    clip_high : float, default 99.9
        Upper percentile bound for signal clipping. Values above this
        percentile are set to 1. If `unmeth_file`/`meth_file` controls
        are provided, this percentile is estimated from the normalized
        meth control channel (making the bound independent of which
        test data is processed); otherwise it is estimated from the
        test signal itself.
    norm_by_strand : bool, default False
        Whether to perform normalization separately based on strandedness.
    batch_size : int, default 20000
        Number of molecules processed (and held in memory) per batch
        during smoothing and probability calculation.
    percentile_sample_size : int, default 100000
        Approximate number of molecules used to estimate percentile clip
        bounds during probability calculation.
    tmp_dir : str or Path, optional
        Directory used for the scratch file backing the experiment's
        raw data staging file. A fresh `nucmc_<timestamp>_<hex>`
        subfolder is created for it — under this directory if given,
        otherwise under the system default temporary directory — and
        reused for the scratch files backing the smoothing and
        probability calculation steps that follow. The staging file is
        removed once the returned experiment is closed or
        garbage-collected.
    chunk_size : int, default 1000000
        Approximate number of rows read (and held in memory) per
        streamed chunk while ingesting raw data files.
    max_cached_chroms : int, default 1
        Maximum number of chromosomes' raw data kept in memory at once.

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
        meth_file=meth_file, fasta_file=fasta_file, mtase=mtase, wrap=wrap,
        colidx=colidx, max_nmol=max_nmol, seed=seed, chunk_size=chunk_size,
        tmp_dir=tmp_dir, max_cached_chroms=max_cached_chroms)

    # Smooth and normalize the data - compute methylation probability
    ana = MethPrintAnalysis()
    ana.smooth(binsize=binsize, exp=exp_data, batch_size=batch_size)
    ana.meth_prob(exp=exp_data, clip_low=clip_low, clip_high=clip_high,
                  norm_by_strand=norm_by_strand, batch_size=batch_size,
                  percentile_sample_size=percentile_sample_size)
    
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
    verbose : bool, default True
        If True, print progress updates to the console.

    Returns
    -------
    SimDataset
        A container object providing access to the generated simulation
        results.
    """
    manager = SimManager(nworker=nworker, verbose=verbose)
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
               link_mat : np.ndarray | None = None,
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
    plot_eseq : bool, default False
        Whether to plot the underlying sequence-specific nucleosome binding
        energy, averaged across all molecules.
    link_mat : np.ndarray, optional
        Linkage matrix to draw as a dendrogram alongside the heatmap (as
        returned by `SimAnalysis.sort_by_linkage`). If given, `occup_name`
        should point at the correspondingly-sorted array rather than the
        unsorted data. If None (default), no dendrogram is drawn.
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
                       out_file=out_file, plot_eseq=plot_eseq,
                       link_mat=link_mat, show=show)

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


def plot_methmap(*, data : H5Array | pd.DataFrame | np.ndarray,
                 vmin : float | None = None,
                 vmax : float | None = None,
                 out_file : str | Path | None = None,
                 link_mat : np.ndarray | None = None,
                 show : bool = True):
    """
    Plot a methylation heatmap.

    Plots `data` at full resolution -- for large data, downsample it
    yourself first (`utils.downsample`, works uniformly for `H5Array`,
    `pd.DataFrame`, or `np.ndarray`) and pass the reduced result.

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
    out_file : str | Path, optional
        Path where the generated plot will be saved. If None, the plot
        is not saved to disk.
    link_mat : np.ndarray, optional
        Linkage matrix to draw as a dendrogram alongside the heatmap
        (as returned by `MethPrintAnalysis.sort_by_linkage`). If given,
        `data` should already be the correspondingly-sorted array
        rather than the original unsorted one.
    show : bool, default True
        Whether to display the figure using the active plotting
        backend.
    """
    methplot = MethPlot()
    methplot.plot_methmap(data, vmin=vmin, vmax=vmax, out_file=out_file,
                          link_mat=link_mat, show=show)


def downsample(*, data : H5Array | pd.DataFrame | np.ndarray,
               max_rows : int,
               how : str = "mean",
               batch_size : int = 20000) -> np.ndarray:
    """
    Collapse rows of a dense array to at most `max_rows`.

    Public wrapper around `utils.downsample`, for reducing plot data
    ahead of `plot_occup`/`plot_methmap`.

    Parameters
    ----------
    data : H5Array, pd.DataFrame, or np.ndarray
        2D data to collapse (rows=molecules, columns=bp position).
    max_rows : int
        Target number of rows.
    how : {"mean", "sum", "min", "max", "stride"}, default "mean"
        How to collapse groups of consecutive rows into one row.
    batch_size : int, default 20000
        Rows read per streamed chunk. Only used when `data` is an
        `H5Array`.

    Returns
    -------
    np.ndarray
        Array of shape (min(nrow, max_rows), ncol).

    Raises
    ------
    ValueError
        If `how` is not a recognized option.
    """
    return utils.downsample(data, max_rows, how=how, batch_size=batch_size)


def sort_by_linkage(*, dataset : SimDataset | None = None,
                    exp : MethPrintExperiment | None = None,
                    chroms : str | Iterable[str] | None = None,
                    data_name : str | None = None,
                    raw_which : str | None = None,
                    sorted_name : str | None = None,
                    store_link_mat : bool = True,
                    link_mat_name : str | None = None,
                    metric : str = "euclidean",
                    method : str = "ward",
                    batch_size : int = 20000,
                    fill_nan : str | float | None = None
                    ) -> dict[str, np.ndarray]:
    """
    Sort molecules by hierarchical-clustering similarity.

    Dispatches to `SimAnalysis.sort_by_linkage` (if `dataset` is given)
    or `MethPrintAnalysis.sort_by_linkage` (if `exp` is given) -- exactly
    one of the two must be provided.

    Parameters
    ----------
    dataset : SimDataset, optional
        Simulation dataset to sort. Mutually exclusive with `exp`.
    exp : MethPrintExperiment, optional
        Methylation experiment to sort. Mutually exclusive with
        `dataset`.
    chroms : str or iterable of str, optional
        Chromosome(s) to process. Only used with `dataset`; `exp` is
        always sorted across all of `exp.chroms`.
    data_name : str, optional
        Key of the analysis array to sort. If None, the underlying
        method's own default is used ("occup" for `dataset`,
        "test_smoothed" for `exp`).
    raw_which : {"test", "meth", "unmeth"}, optional
        Only used with `exp`. If given, sorts raw long-form data
        instead of an `exp.analysis` entry.
    sorted_name : str, optional
        Key used to store the sorted result. Defaults to
        `f"{data_name}_sorted"` (or `f"{raw_which}_sorted"`).
    store_link_mat : bool, default True
        Whether to also persist each chromosome's linkage matrix into
        `.analysis[chrom][link_mat_name]`.
    link_mat_name : str, optional
        Key used to store the linkage matrix if `store_link_mat` is
        True. Defaults to `f"{data_name}_linkage"` (or
        `f"{raw_which}_linkage"`).
    metric : str, default "euclidean"
        Distance metric, forwarded to `utils.compute_linkage`.
    method : str, default "ward"
        Linkage method, forwarded to `utils.compute_linkage`.
    batch_size : int, default 20000
        Rows processed (and held in memory) per batch.
    fill_nan : {"mean"}, float, or None, default None
        How to handle `nan` values, forwarded to
        `utils.compute_linkage`.

    Returns
    -------
    dict of str to np.ndarray
        A mapping from chromosome name to that chromosome's linkage
        matrix.

    Raises
    ------
    ValueError
        If neither or both of `dataset`/`exp` are given.
    """
    if (dataset is None) == (exp is None):
        raise ValueError("Exactly one of 'dataset' or 'exp' must be given.")
    kwargs = dict(sorted_name=sorted_name, store_link_mat=store_link_mat,
                 link_mat_name=link_mat_name, metric=metric, method=method,
                 batch_size=batch_size, fill_nan=fill_nan)
    if data_name is not None:
        kwargs["data_name"] = data_name
    if dataset is not None:
        if chroms is not None:
            kwargs["chroms"] = chroms
        return SimAnalysis().sort_by_linkage(dataset=dataset, **kwargs)
    if raw_which is not None:
        kwargs["raw_which"] = raw_which
    return MethPrintAnalysis().sort_by_linkage(exp=exp, **kwargs)

