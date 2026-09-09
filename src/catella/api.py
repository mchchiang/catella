# api.py

# A high-level interface for processing methylation footprinting data and
# running simulations

from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np
import pandas as pd
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.experiment.methdata import MethPrintExperiment
from catella.experiment.plot import MethPlot
from catella.h5_array import H5Array
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset
from catella.simulation.analysis import SimAnalysis
from catella.simulation.plot import SimPlot
from catella import utils
from catella.utils import IndexType

def load_raw(*, chromsize : str | Path,
             test_file : str | Path,
             unmeth_file : str | Path | None = None,
             meth_file : str | Path | None = None,
             fasta_file : str | Path | None = None,
             mtase : str | Iterable[str] | None = None,
             chroms : Iterable[str] | None = None,
             wrap : bool = False,
             colidx : Iterable | None = None,
             max_nmol : int | None = None,
             seed : int | None = None,
             chunk_size : int = 1000000,
             tmp_dir : str | Path | None = None,
             max_cached_chroms : int = 1) -> MethPrintExperiment:
    """
    Load raw methylation footprinting data into a MethPrintExperiment.

    First stage of catella's preprocessing pipeline: `load_raw` (raw
    data ingestion) -> `filter_dropout` (optional QC) ->
    `compute_empirical_prob`/`compute_model_prob` (probability
    calculation). Thin wrapper around `MethPrintExperiment.load_raw`.

    Parameters
    ----------
    chromsize : str or Path
        Path to the chromosome sizes file or a string identifier for the
        genome.
    test_file : str or Path
        Path to the primary experimental methylation data file.
    unmeth_file : str or Path, optional
        Path to the unmethylated control file.
    meth_file : str or Path, optional
        Path to the methylated control file.
    fasta_file : str or Path, optional
        Multi-FASTA file of per-chromosome reference sequences (record
        id matching `chromsize`); stored on `MethPrintData.refseq`.
        Required by `compute_model_prob` but not by `load_raw` itself.
    mtase : str or iterable of str, optional
        Methyltransferase(s) used to generate the test data: 'A'
        (any-context adenine, e.g., EcoGII), 'CG' (CpG, e.g., M.SssI),
        'GC' (GpC, e.g., M.CviPI). One label, a list of labels, or
        None (default) if unspecified.
    chroms : iterable of str, optional
        Restrict processing to these chromosomes. If None (default),
        all chromosomes in `chromsize` are processed.
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
    chunk_size : int, default 1000000
        Approximate number of rows read (and held in memory) per
        streamed chunk while ingesting raw data files.
    tmp_dir : str or Path, optional
        Directory used for the scratch file backing the experiment's
        raw data staging file. A fresh `catella_<timestamp>_<hex>`
        subfolder is created for it -- under this directory if given,
        otherwise under the system default temporary directory -- and
        reused for the scratch files backing later smoothing/
        probability calculation steps. The staging file is removed
        once the returned experiment is closed or garbage-collected.
    max_cached_chroms : int, default 1
        Maximum number of chromosomes' raw data kept in memory at once.

    Returns
    -------
    MethPrintExperiment
        A newly created experiment containing the raw, unprocessed
        data. Pass it to `filter_dropout` (optional QC) and then to
        `compute_empirical_prob`/`compute_model_prob`.
    """
    return MethPrintExperiment.load_raw(
        chromsize=chromsize, test_file=test_file,
        unmeth_file=unmeth_file, meth_file=meth_file,
        fasta_file=fasta_file, mtase=mtase, chroms=chroms, wrap=wrap,
        colidx=colidx, max_nmol=max_nmol, seed=seed, chunk_size=chunk_size,
        tmp_dir=tmp_dir, max_cached_chroms=max_cached_chroms)


def compute_empirical_prob(exp : MethPrintExperiment, *,
               out_file : str | Path | None = None,
               binsize : int = 147,
               prob_name : str = "meth_prob",
               clip_low : float = 0.1,
               clip_high : float = 99.9,
               norm_by_strand : bool = False,
               fill_edge : float = np.nan,
               batch_size : int = 20000,
               percentile_sample_size : int = 100000,
               seed : int | None = None,
               mask_name : str | None = None) -> MethPrintExperiment:
    """
    Compute methylation probabilities via empirical normalization.

    Smooth `exp`'s raw signal and perform empirical (percentile/
    control-based) normalization to convert it into a methylation
    probability score, mutating `exp` in place and optionally
    persisting it to disk. See `compute_model_prob` for a model-based
    alternative. `exp` should already be loaded (`load_raw`) and, if
    desired, QC'd (`filter_dropout`).

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to process (as returned by `load_raw`).
    out_file : str or Path, optional
        Path where the processed `exp` will be saved. If None, the
        result is only mutated in-memory.
    binsize : int, default 147
        The genomic window size (in base pairs) used for data aggregation. The
        default value corresponds to the typical DNA footprint of a nucleosome.
    prob_name : str, default "meth_prob"
        The key used to store the resulting methylation probabilities
        in `exp.analysis`.
    clip_low : float, default 0.1
        Lower percentile bound for signal clipping. Values below this
        percentile are set to 0. If `exp` has meth/unmeth controls,
        this percentile is estimated from the normalized unmeth
        control source; otherwise it is estimated from the test
        signal itself.
    clip_high : float, default 99.9
        Upper percentile bound for signal clipping. Values above this
        percentile are set to 1. If `exp` has meth/unmeth controls,
        this percentile is estimated from the normalized meth
        control source; otherwise it is estimated from the test
        signal itself.
    norm_by_strand : bool, default False
        Whether to perform normalization separately based on strandedness.
    fill_edge : float, default np.nan
        Probability used to fill the trailing positions with no full
        window to average. Interior gaps are always filled with 0.5,
        the neutral, no-evidence probability.
    batch_size : int, default 20000
        Number of molecules processed (and held in memory) per batch
        during smoothing and probability calculation.
    percentile_sample_size : int, default 100000
        Approximate number of molecules used to estimate percentile clip
        bounds during probability calculation.
    seed : int, optional
        Seed for the random number generator used for percentile
        subsampling.
    mask_name : str, optional
        If given, molecules flagged as dropout by a prior
        `filter_dropout(mask_name=mask_name)` call are excluded from
        both the smoothing step and the probability calculation, per
        source: set to all-NaN in the smoothed signal, and excluded
        (set to NaN) from the test signal, control-based normalization
        statistics, and the resulting probabilities.

    Returns
    -------
    MethPrintExperiment
        `exp`, mutated in place with the computed probabilities.
    """

    # Smooth and normalize the data - compute methylation probability
    ana = MethPrintAnalysis()
    ana.empirical_prob(exp=exp, binsize=binsize, prob_name=prob_name,
                       clip_low=clip_low, clip_high=clip_high,
                       norm_by_strand=norm_by_strand, fill_edge=fill_edge,
                       batch_size=batch_size,
                       percentile_sample_size=percentile_sample_size,
                       seed=seed, mask_name=mask_name)

    # Save the results
    if out_file is not None:
        exp.save(out_file)

    return exp


def compute_model_prob(exp : MethPrintExperiment, *,
               out_file : str | Path | None = None,
               prob_name : str = "meth_prob",
               pi0 : float = 0.5,
               eta : float | dict[str, float] | None = None,
               eta_max_lag : int = 10,
               store_rho : bool = False,
               nu : float = 10.0,
               rho_leak : float = 0.1,
               min_gap : float = 0.05,
               l_nuc : int = 147,
               n_min : int = 10,
               max_iters : int = 200,
               init_prot : float = 0.05,
               init_acc : float = 0.95,
               tol : float = 1e-8,
               fill_edge : float = np.nan,
               norm_by_strand : bool = False,
               batch_size : int = 20000,
               mask_name : str | None = None) -> MethPrintExperiment:
    """
    Compute methylation probabilities via a calibrated log-odds model.

    Convert `exp`'s per-read calls into a methylation probability
    score using a calibrated Bayesian log-odds model, mutating `exp`
    in place and optionally persisting it to disk. See
    `compute_empirical_prob` for a percentile/control-based
    alternative. `exp` should already be loaded (`load_raw`, with
    `fasta_file` given) and, if desired, QC'd (`filter_dropout`).

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to process (as returned by `load_raw`).
    out_file : str or Path, optional
        Path where the processed `exp` will be saved. If None, the
        result is only mutated in-memory.
    prob_name : str, default "meth_prob"
        The key used to store the resulting methylation probabilities
        in `exp.analysis`.
    pi0 : float, default 0.5
        Prior probability that an assayable site is methylated, used
        to convert each site's `mod_qual` confidence score into a
        likelihood ratio. 0.5 is the uninformative choice used when
        the base caller's training prior is unknown.
    eta : float, dict of str to float, or None, default None
        Per-channel multiplicative correction for inflated log-
        likelihood ratios from correlated nearby sites (e.g.
        palindromic CpG/GpC positions). If None (default), each
        channel's ("M6A"/"GCH"/"HCG"/"GCG") eta is auto-estimated
        from lag-k autocorrelation in its log-odds -- from the
        methylated control when available, or from the test data
        otherwise. A float pins every channel to that value; a dict
        pins only the named channels, leaving any not mentioned to
        be auto-estimated. The value(s) actually applied are stored
        in `exp.global_analysis[f"{prob_name}_eta"]` afterward.
    eta_max_lag : int, default 10
        Maximum lag (bp) summed over when auto-estimating eta.
    store_rho : bool, default False
        If True, store the lag-k autocorrelation `rho(k)` used in
        each channel's eta estimation to
        `exp.global_analysis[f"{prob_name}_rho"]`, one column per
        channel that was actually auto-estimated (channels pinned
        via `eta`, or with too little data to estimate, are omitted).
    nu : float, default 10.0
        Pseudo-count strength for shrinking each position's call
        rate toward its context group's mean; larger values shrink
        harder at low depth.
    rho_leak : float, default 0.1
        Leak fraction in [0, 1] interpolating the protected-state
        call rate between the unmethylated control's false-positive
        rate (0 for perfect protection) and the accessible-state rate.
        Used only when meth/unmeth controls are available.
    min_gap : float, default 0.05
        Minimum required gap between the accessible and protected
        call rates for a position to be treated as informative;
        positions below this gap contribute no evidence.
    l_nuc : int, default 147
        Nucleosome footprint size (bp): the expectation-maximization
        window size (no-controls path) and the output window-sum
        size.
    n_min : int, default 10
        Minimum number of context-eligible sites a window must have
        to be used in the no-controls expectation-maximization fit.
        Used only when no meth/unmeth controls are available.
    max_iters : int, default 200
        Maximum number of expectation-maximization iterations for
        the no-controls rate fit.
    init_prot : float, default 0.05
        Initial guess for the protected-state call rate in the
        no-controls expectation-maximization fit.
    init_acc : float, default 0.95
        Initial guess for the accessible-state call rate in the
        no-controls expectation-maximization fit.
    tol : float, default 1e-8
        Relative log-likelihood convergence tolerance for the
        no-controls expectation-maximization fit.
    fill_edge : float, default nan
        Probability used to fill the trailing `l_nuc - 1` positions
        of the result, which have no full window to summarize. The
        nan default leaves those positions unfilled.
    norm_by_strand : bool, default False
        Whether to calibrate theta_prot/theta_acc separately per
        strand before scoring. If True, the source(s) feeding
        calibration (meth/unmeth controls when available, otherwise
        the test data via expectation-maximization) are split by
        strand, and each test read is scored against its own strand's
        calibration. `eta` is still auto-estimated pooled across
        strands, since the crosstalk it corrects for is an
        assay-chemistry property rather than a strand-specific one.
    batch_size : int, default 20000
        Number of molecules processed (and held in memory) per
        batch.
    mask_name : str, optional
        If given, molecules flagged as dropout by a prior
        `filter_dropout(mask_name=mask_name)` call are excluded from
        rate calibration, eta estimation, and the output, per source.

    Returns
    -------
    MethPrintExperiment
        `exp`, mutated in place with the computed probabilities.

    Raises
    ------
    ValueError
        If a chromosome has no reference sequence, if the
        no-controls path cannot find enough windows with `n_min`
        context-eligible sites, if `eta` is a dict with an
        unrecognized channel name, or if `norm_by_strand` is True but
        molecules with unmapped strands ('.') exist. See
        `MethPrintAnalysis.model_prob`.

    Notes
    -----
    Per-chromosome (and, if `norm_by_strand`, per-strand) calibration
    diagnostics -- whether controls were used, fraction of informative
    positions per context, and (no-controls path only) actual EM
    iterations run, final log-likelihood, fitted mixing fraction, and
    fitted theta_prot/theta_acc per context -- are stored in
    `exp.global_analysis[f"{prob_name}_calib"]`. The full per-position
    theta_prot/theta_acc/informative actually used to score each
    chromosome are stored in `exp.analysis[chrom][f"{prob_name}_theta"]`.
    """

    # Compute methylation probability via the calibrated log-odds model
    ana = MethPrintAnalysis()
    ana.model_prob(exp=exp, prob_name=prob_name, pi0=pi0, eta=eta,
                   eta_max_lag=eta_max_lag, store_rho=store_rho, nu=nu,
                   rho_leak=rho_leak, min_gap=min_gap, l_nuc=l_nuc,
                   n_min=n_min, max_iters=max_iters, init_prot=init_prot,
                   init_acc=init_acc, tol=tol, fill_edge=fill_edge,
                   norm_by_strand=norm_by_strand, batch_size=batch_size,
                   mask_name=mask_name)

    # Save the results
    if out_file is not None:
        exp.save(out_file)

    return exp


def filter_dropout(exp : MethPrintExperiment, *,
           which : str | None = None,
           mtase : list | None = None,
           chroms : list | None = None,
           thres_min : float = 0.0,
           thres_max : float = 1.0,
           unmapped_strand : str = "union",
           method : str = "separate",
           mask_name : str = "dropout_mask") -> None:
    """
    Flag molecules with poor coverage at methylatable positions.

    Thin wrapper around `MethPrintExperiment.filter_dropout`.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to evaluate. Mutated in place.
    which : {"test", "meth", "unmeth"} or None, default None
        Source(s) to evaluate. None evaluates every source present
        for each chromosome.
    mtase : list of str, optional
        Subset of `exp.mtase` labels to evaluate. None uses all of
        them.
    chroms : list of str, optional
        Subset of `exp.chroms` to evaluate. None (default) evaluates
        every chromosome.
    thres_min : float, default 0.0
        Min allowed no-signal fraction (per label, or of the pooled
        total under `method="aggregate"`) to be kept.
    thres_max : float, default 1.0
        Max allowed no-signal fraction (per label, or of the pooled
        total under `method="aggregate"`) to be kept.
    unmapped_strand : {"union", "drop", "+", "-"}, default "union"
        How to evaluate unmapped ('.') strand molecules: union of
        '+'/'-' position sets, always dropout, or treat as that
        strand.
    method : {"separate", "aggregate"}, default "separate"
        How multiple `mtase` labels combine into `keep`. "separate":
        must clear `thres_min`/`thres_max` per label. "aggregate":
        site counts pooled across labels into one fraction first.
        Irrelevant for a single label.
    mask_name : str, default "dropout_mask"
        Key for the mask in `exp.analysis[chrom]`, as
        `f"{source}_{mask_name}"`.

    Raises
    ------
    ValueError
        If `mtase` is unset on `exp`, `mtase` contains a label not in
        `exp.mtase`, `chroms` contains a chromosome not in
        `exp.chroms`, `thres_min` or `thres_max` is not in [0, 1],
        `thres_min` is greater than `thres_max`,
        `unmapped_strand`/`method` is invalid, a requested source is
        missing for some chromosome, or `refseq` is missing.
    """
    exp.filter_dropout(which=which, mtase=mtase, chroms=chroms,
                       thres_min=thres_min, thres_max=thres_max,
                       unmapped_strand=unmapped_strand,
                       method=method, mask_name=mask_name)


def summarize_dropout(exp : MethPrintExperiment, *,
              which : str | None = None,
              mtase : list | None = None,
              chroms : list | None = None,
              unmapped_strand : str = "union") -> pd.DataFrame:
    """
    Return a per-label dropout fraction summary (QC check).

    Thin wrapper around `MethPrintExperiment.summarize_dropout`.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to summarize.
    which : {"test", "meth", "unmeth"} or None, default None
        Source(s) to summarize. None summarizes every source present
        for each chromosome.
    mtase : list of str, optional
        Subset of `exp.mtase` labels to summarize. None uses all.
    chroms : list of str, optional
        Subset of `exp.chroms` to summarize. None (default)
        summarizes every chromosome.
    unmapped_strand : {"union", "drop", "+", "-"}, default "union"
        How to evaluate unmapped ('.') strand molecules.

    Returns
    -------
    pd.DataFrame
        Columns `chrom`, `source`, `label`, `n_sites_plus`,
        `n_sites_minus`, `p0`/`p25`/`p50`/`p75`/`p100` (dropout
        fraction percentiles) -- one row per (chrom, source, label),
        plus a `label="aggregate"` row per (chrom, source) pooling
        all labels, if more than one label is evaluated.

    Raises
    ------
    ValueError
        If `mtase` is unset, contains an unknown label, `chroms`
        contains a chromosome not in `exp.chroms`, `unmapped_strand`
        is invalid, a source is missing for some chromosome, or
        `refseq` is missing.
    """
    return exp.summarize_dropout(which=which, mtase=mtase, chroms=chroms,
                                 unmapped_strand=unmapped_strand)


def plot_dropout_ecdf(exp : MethPrintExperiment, *,
                chrom : str,
                which : str | None = None,
                mtase : list | None = None,
                unmapped_strand : str = "union",
                source : str | None = None,
                out_file : str | Path | None = None,
                show : bool = True):
    """
    Plot the empirical CDF of dropout rate per group.

    Computes `exp.dropout_fractions(...)`, restricted to `chrom`, and
    plots the result via `MethPlot.plot_dropout_ecdf`. For each
    matching group, at threshold `t` the plotted value is
    `100 * mean(dropout_frac <= t)` over that group's molecules.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to evaluate.
    chrom : str
        Chromosome to plot.
    which : {"test", "meth", "unmeth"} or None, default None
        Source(s) to evaluate. None evaluates every source present
        for each chromosome.
    mtase : list of str, optional
        Subset of `exp.mtase` labels to evaluate. None uses all of
        them.
    unmapped_strand : {"union", "drop", "+", "-"}, default "union"
        How to evaluate unmapped ('.') strand molecules.
    source : {"test", "meth", "unmeth"}, optional
        Restrict the plot to this source. None plots every source
        present for `chrom`.
    out_file : str or pathlib.Path, optional
        Path to save the generated figure. Directories are created if
        they do not exist.
    show : bool, default True
        Whether to display the plot using `plt.show()`.

    Raises
    ------
    ValueError
        If `mtase` is unset, contains an unknown label,
        `unmapped_strand` is invalid, a source is missing for some
        chromosome, or `refseq` is missing (from
        `exp.dropout_fractions`); or if `chrom` (or `source`, when
        given) has no matching entries in the result (from
        `MethPlot.plot_dropout_ecdf`).
    """
    fractions = exp.dropout_fractions(which=which, mtase=mtase,
                                      chroms=[chrom],
                                      unmapped_strand=unmapped_strand)
    methplot = MethPlot()
    methplot.plot_dropout_ecdf(fractions, chrom, source=source,
                               out_file=out_file, show=show)


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

def analyze(dataset : SimDataset, *,
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

def plot_occup(dataset : SimDataset, *,
               chrom : str,
               out_file : str | Path | None = None,
               occup_name : str = "occup",
               mols : Iterable[int] | None = None,
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
    dataset : SimDataset
        The simulation dataset containing the computed occupancy data.
    chrom : str
        The identifier of the chromosome to plot.
    out_file : str | Path, optional
        Path where the generated plot will be saved. If None, the plot
        is not saved to disk.
    occup_name : str, default "occup"
        The key or name of the occupancy data to retrieve from the dataset.
        This should match the name used during the `analyze` step.
    mols : iterable of int, optional
        Molecule indices to display, restricting the heatmap (and, if
        `plot_eseq` is True, the sequence-energy average) to this
        subset. If None (default), all molecules are shown.
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
                       mols=mols, out_file=out_file, plot_eseq=plot_eseq,
                       link_mat=link_mat, show=show)

def plot_nuc_pos(dataset : SimDataset, *,
                 chrom : str,
                 mol : int,
                 run : int,
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
    dataset : SimDataset
        The simulation dataset containing the positional coordinates.
    chrom : str
        The identifier of the chromosome to plot.
    mol : int
        The index of the specific molecule (fiber) to visualize.
    run : int
        The specific simulation run index for the chosen molecule.
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


def plot_energy(dataset : SimDataset, *,
                chrom : str,
                mol : int,
                run : int,
                out_file : str | Path | None = None,
                show : bool = True):
    """
    Plot the total energy of the system for a specific simulation run of a
    molecule over time.

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset containing the positional coordinates.
    chrom : str
        The identifier of the chromosome to plot.
    mol : int
        The index of the specific molecule (fiber) to visualize.
    run : int
        The specific simulation run index for the chosen molecule.
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


def plot_methmap(data : H5Array | pd.DataFrame | np.ndarray, *,
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
                    mask_name : str | None = None,
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
        Chromosome(s) to process. If None (default), all chromosomes
        in `dataset.chroms` (or `exp.chroms`) are processed.
    data_name : str, optional
        Key of the analysis array to sort. If None, the underlying
        method's own default is used ("occup" for `dataset`,
        "test_smoothed" for `exp`).
    raw_which : {"test", "meth", "unmeth"}, optional
        Only used with `exp`. If given, sorts raw long-form data
        instead of an `exp.analysis` entry.
    mask_name : str, optional
        Only used with `exp` and `raw_which`. If given, molecules
        flagged as dropout by a prior `filter_dropout(mask_name=
        mask_name)` call are excluded before sorting.
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
        If neither or both of `dataset`/`exp` are given, or if
        `mask_name` is given with `dataset`.
    """
    if (dataset is None) == (exp is None):
        raise ValueError("Exactly one of 'dataset' or 'exp' must be given.")
    kwargs = dict(sorted_name=sorted_name, store_link_mat=store_link_mat,
                 link_mat_name=link_mat_name, metric=metric, method=method,
                 batch_size=batch_size, fill_nan=fill_nan)
    if data_name is not None:
        kwargs["data_name"] = data_name
    if chroms is not None:
        kwargs["chroms"] = chroms
    if dataset is not None:
        if mask_name is not None:
            raise ValueError("'mask_name' is only used with 'exp'.")
        return SimAnalysis().sort_by_linkage(dataset=dataset, **kwargs)
    if raw_which is not None:
        kwargs["raw_which"] = raw_which
    if mask_name is not None:
        kwargs["mask_name"] = mask_name
    return MethPrintAnalysis().sort_by_linkage(exp=exp, **kwargs)

