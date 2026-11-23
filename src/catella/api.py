# api.py

# A high-level interface for processing methylation footprinting data and
# running simulations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from catella import utils
from catella.experiment.methdata import MethPrintExperiment
from catella.experiment.plot import MethPlot
from catella.experiment.preprocessing import MethPrintAnalysis
from catella.h5_array import H5Array
from catella.simulation.analysis import SimAnalysis
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.plot import SimPlot
from catella.simulation.results import SimDataset
from catella.utils import IndexType


def load_raw(
    *,
    chromsize: str | Path,
    test_file: str | Path,
    unmeth_file: str | Path | None = None,
    meth_file: str | Path | None = None,
    fasta_file: str | Path | None = None,
    mtase: str | Iterable[str] | None = None,
    chroms: Iterable[str] | None = None,
    wrap: bool = False,
    ignore_strand: bool = False,
    colidx: Iterable | None = None,
    max_nmol: int | None = None,
    balance_strand: bool = False,
    seed: int | None = None,
    chunk_size: int = 1000000,
    tmp_dir: str | Path | None = None,
    max_cached_chroms: int = 1,
    nworker: int = 1,
    min_gap: int | None = 20,
) -> MethPrintExperiment:
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
    ignore_strand : bool, default False
        If True, ignore each row's recorded `strand` for `mtase`
        context filtering, and in `filter_dropout`,
        `dropout_fractions` and `qc`. Independent of
        `norm_by_strand`.
    colidx : Iterable, optional
        Specific column indices to use if the input file does not follow
        the standard ModKit format; see
        `MethPrintExperiment.load_raw`.
    max_nmol : int, optional
        Maximum number of molecules to extract for each chromosome.
    balance_strand : bool, default False
        If True, keep an equal number of '+' and '-' strand molecules
        per chromosome: at most `max_nmol`; fewer if either strand has
        fewer than `max_nmol // 2` molecules (warns). Molecules with
        unmapped strand ('.') are dropped. Requires `max_nmol` of at
        least 2.
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
    nworker : int, default 1
        Maximum number of threads used to read test/unmeth/meth files
        concurrently. Only relevant when more than one is given.
    min_gap : int or None, default 20
        Minimum deletion (bp) inside a read recorded as an internal
        gap; None records none. See `MethPrintExperiment.load_raw`.

    Returns
    -------
    MethPrintExperiment
        A newly created experiment containing the raw, unprocessed
        data. Pass it to `filter_dropout` (optional QC) and then to
        `compute_empirical_prob`/`compute_model_prob`.
    """
    return MethPrintExperiment.load_raw(
        chromsize=chromsize,
        test_file=test_file,
        unmeth_file=unmeth_file,
        meth_file=meth_file,
        fasta_file=fasta_file,
        mtase=mtase,
        chroms=chroms,
        wrap=wrap,
        ignore_strand=ignore_strand,
        colidx=colidx,
        max_nmol=max_nmol,
        balance_strand=balance_strand,
        seed=seed,
        chunk_size=chunk_size,
        tmp_dir=tmp_dir,
        max_cached_chroms=max_cached_chroms,
        nworker=nworker,
        min_gap=min_gap,
    )


def compute_empirical_prob(
    exp: MethPrintExperiment,
    *,
    out_file: str | Path | None = None,
    lnuc: int = 147,
    prob_name: str = "meth_prob",
    clip_low: float = 0.1,
    clip_high: float = 99.9,
    norm_by_strand: bool = False,
    fill_edge: float = np.nan,
    batch_size: int = 20000,
    percentile_sample_size: int = 100000,
    seed: int | None = None,
    mask_name: str | None = None,
) -> MethPrintExperiment:
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
    lnuc : int, default 147
        Nucleosome footprint size (bp): the rolling-average smoothing
        window size.
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
    ana.empirical_prob(
        exp=exp,
        lnuc=lnuc,
        prob_name=prob_name,
        clip_low=clip_low,
        clip_high=clip_high,
        norm_by_strand=norm_by_strand,
        fill_edge=fill_edge,
        batch_size=batch_size,
        percentile_sample_size=percentile_sample_size,
        seed=seed,
        mask_name=mask_name,
    )

    # Save the results
    if out_file is not None:
        exp.save(out_file)

    return exp


def compute_model_prob(
    exp: MethPrintExperiment,
    *,
    out_file: str | Path | None = None,
    prob_name: str = "meth_prob",
    q_thres: float | None = None,
    channels: str | Iterable[str] | None = None,
    eta: float | dict[str, float] | None = None,
    eta_max_lag: int = 10,
    store_rho: bool = False,
    nu: float | None = None,
    rho_leak: float | None = None,
    min_rel_contrast: float = 0.2,
    lnuc: int = 147,
    lbin: int = 30,
    max_iters: int = 200,
    tol: float = 1e-8,
    fill_edge: float = np.nan,
    norm_by_strand: bool = False,
    span: str = "full",
    batch_size: int = 20000,
    mask_name: str | None = None,
) -> MethPrintExperiment:
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
    q_thres : float, optional
        If given, binarize each `mod_qual` confidence score: 1 if it is
        at least `q_thres` and 0 otherwise. This applies to all sources,
        so to rate calibration, eta estimation, and the output alike.
        None (default) uses the scores as they are. Must be in (0, 1].
    channels : str or iterable of str, optional
        Context channel(s) to use as evidence, among "A", "GCH", "HCG",
        "GCG", "GC" and "CG"; each must be assayed by `exp.mtase`. "GC"
        merges GCH and GCG into one channel, and "CG" merges HCG and
        GCG, for in vitro data with a single C mtase (`exp.mtase`
        holding "GC" or "CG" but not both); this assumes GCG sites are
        methylated as efficiently as the rest, and a merged channel
        cannot be combined with its parts. If None (default), all
        assayed channels are used, unmerged. A selected channel is
        still dropped if it fails the tests of `min_rel_contrast`.
    eta : float, dict of str to float, or None, default None
        Per-channel multiplicative correction for inflated log-
        likelihood ratios from correlated nearby sites (e.g.
        palindromic CpG/GpC positions). If None (default), each
        selected channel's eta is auto-estimated from lag-k
        autocorrelation in its log-odds -- from the methylated control
        when available, or from the test data otherwise. A float pins
        every selected channel to that value; a dict pins only the
        named channels, leaving any not mentioned to be
        auto-estimated. The value(s) actually applied are stored
        in `exp.global_analysis[f"{prob_name}_eta"]` afterward.
    eta_max_lag : int, default 10
        Maximum lag (bp) summed over when auto-estimating eta.
    store_rho : bool, default False
        If True, store the lag-k autocorrelation `rho(k)` used in
        each channel's eta estimation to
        `exp.global_analysis[f"{prob_name}_rho"]`, one column per
        channel that was actually auto-estimated (channels pinned
        via `eta`, or with too little data to estimate, are omitted).
    nu : float, optional
        Pseudo-count strength for shrinking each position's control
        call rate toward its context group's mean; larger values
        shrink harder at low depth. If None (default), it is estimated
        by empirical Bayes per control and per channel; a float pins
        it. Used only when meth/unmeth controls are available.
    rho_leak : float, optional
        Leak fraction in [0, 1] interpolating the protected-state
        call rate between the unmethylated control's false-positive
        rate (0 for perfect protection) and the accessible-state rate.
        If None (default), it is fitted per channel by
        expectation-maximization on the test sample; a float pins it.
        Used only when meth/unmeth controls are available.
    min_rel_contrast : float, default 0.2
        Minimum relative contrast `(kappa - tau) / kappa` of a context
        after an expectation-maximization fit (which must also pass a
        BIC test) for its positions to be treated as informative.
    lnuc : int, default 147
        Nucleosome footprint size (bp), used as the output window-sum
        size.
    lbin : int, default 30
        Bin width (bp) of the non-overlapping windows in the
        no-controls expectation-maximization fit. Used only when no
        meth/unmeth controls are available.
    max_iters : int, default 200
        Maximum number of expectation-maximization iterations for
        the no-controls rate fit.
    tol : float, default 1e-8
        Relative log-likelihood convergence tolerance for the
        no-controls expectation-maximization fit.
    fill_edge : float, default nan
        Probability used to fill the trailing `lnuc - 1` positions
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
    span : {"full", "calls", "read"}, default "full"
        Extent of each molecule; see `MethPrintAnalysis.model_prob`.
        "full" assumes every molecule covers the whole fiber, "read"
        uses each molecule's read extent (for partial reads, e.g.
        in-vivo native fragments), and "calls" its first to last call.
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
        no-controls path cannot find enough bins with read sites, if
        `eta` is a dict with an unrecognized channel
        name, if `span` is not recognized, or if `norm_by_strand` is
        True but molecules with unmapped strands ('.') exist. See
        `MethPrintAnalysis.model_prob`.

    Warns
    -----
    UserWarning
        If the fraction of read sites with a call in the test sample
        lies outside the range spanned by explicit-mode controls. See
        `MethPrintAnalysis.model_prob`.

    Notes
    -----
    Per-chromosome (and, if `norm_by_strand`, per-strand) calibration
    diagnostics -- whether controls were used, fraction of informative
    positions per context, fraction of read sites carrying a call per
    source and context, and (no-controls path only) actual EM
    iterations run, final log-likelihood, fitted mixing fraction, and
    fitted theta_prot/theta_acc per context -- are stored in
    `exp.global_analysis[f"{prob_name}_calib"]`. The full per-position
    theta_prot/theta_acc/informative actually used to score each
    chromosome are stored in `exp.analysis[chrom][f"{prob_name}_theta"]`.
    """

    # Compute methylation probability via the calibrated log-odds model
    ana = MethPrintAnalysis()
    ana.model_prob(
        exp=exp,
        prob_name=prob_name,
        q_thres=q_thres,
        channels=channels,
        eta=eta,
        eta_max_lag=eta_max_lag,
        store_rho=store_rho,
        nu=nu,
        rho_leak=rho_leak,
        min_rel_contrast=min_rel_contrast,
        lnuc=lnuc,
        lbin=lbin,
        max_iters=max_iters,
        tol=tol,
        fill_edge=fill_edge,
        norm_by_strand=norm_by_strand,
        span=span,
        batch_size=batch_size,
        mask_name=mask_name,
    )

    # Save the results
    if out_file is not None:
        exp.save(out_file)

    return exp


def estimate_start_temp(
    exp: MethPrintExperiment,
    *,
    prob_name: str = "meth_prob",
    chroms: str | Iterable[str] | None = None,
    percentile: float = 90.0,
    batch_size: int = 20000,
    percentile_sample_size: int = 100000,
    seed: int | None = None,
    mask_name: str | None = None,
) -> float:
    """
    Estimate a data-driven `SimSettings.start_temp` value.

    Returns the `percentile`-th percentile of abs(logit(meth_prob))
    across `chroms`, via batched random subsampling, so sequence-
    energy barriers of that size are crossable early in the
    annealing schedule. Probabilities of exactly 0 or 1 (log odds
    beyond floating-point range) count as the largest finite log
    odds rather than infinity.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to estimate a temperature from. `prob_name`
        must already be in `exp.analysis[chrom]` for each chromosome.
    prob_name : str, default "meth_prob"
        The key to read from `exp.analysis[chrom]`.
    chroms : str or iterable of str, optional
        Chromosome(s) to include. If None (default), all of
        `exp.chroms` are used.
    percentile : float, default 90.0
        Percentile (0 to 100, exclusive of 100) of
        abs(logit(meth_prob)) to return.
    batch_size : int, default 20000
        Number of molecules read (and held in memory) per batch.
    percentile_sample_size : int, default 100000
        Approximate total number of molecules, split evenly across
        `chroms`, used to estimate the percentile.
    seed : int, optional
        Seed for the random number generator used for subsampling.
    mask_name : str, optional
        If given, excludes molecules dropped by
        `filter_dropout(mask_name=mask_name)` on the "test" source,
        without modifying `exp`.

    Returns
    -------
    float
        The suggested `start_temp` value.

    Raises
    ------
    KeyError
        If `mask_name` is given but no matching mask is found for a
        chromosome's "test" source.
    """
    manager = SimManager(verbose=False)
    return manager.estimate_start_temp(
        exp,
        prob_name=prob_name,
        chroms=chroms,
        percentile=percentile,
        batch_size=batch_size,
        percentile_sample_size=percentile_sample_size,
        seed=seed,
        mask_name=mask_name,
    )


def estimate_llink(
    exp: MethPrintExperiment,
    settings: str | Path | SimSettings,
    *,
    llinks: Iterable[int] = range(0, 65, 5),
    prob_name: str = "meth_prob",
    chroms: str | Iterable[str] | None = None,
    max_nmol: int | None = None,
    seed: int | None = None,
    batch_size: int = 20000,
    mask_name: str | None = None,
    nworker: int = 1,
    llink_evid_name: str | None = None,
    fill_nan: str | float | None = None,
) -> dict[str, int]:
    """
    Estimate `SimSettings.llink` per chromosome by maximum likelihood.

    For each chromosome, returns the candidate that maximizes the sum
    over molecules of log Z(data, llink) - log Z(no data, llink), the
    marginal likelihood of the data, computed exactly with the transfer
    matrix at temperature 1. `exp` is not modified unless
    `llink_evid_name` is given, and the value only takes effect once
    placed in the settings:

    >>> best = catella.estimate_llink(exp, settings)
    >>> settings = dataclasses.replace(settings, llink=best)
    >>> catella.run(settings=settings, ...)

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to estimate from. `prob_name` must already be in
        `exp.analysis[chrom]` for each chromosome.
    settings : str or Path or SimSettings
        A settings object, or a path to a settings file. Provides
        `nucbp`, `elink`, `mu` and `emax`, which should match those used
        for sampling. `llink` is ignored.
    llinks : iterable of int, default range(0, 65, 5)
        Candidate linker lengths (bp).
    prob_name : str, default "meth_prob"
        The key to read from `exp.analysis[chrom]`.
    chroms : str or iterable of str, optional
        Chromosome(s) to include. If None (default), all of `exp.chroms`
        are used.
    max_nmol : int, optional
        If given, use a random sample of at most this many molecules per
        chromosome, to reduce run time. All molecules are used if None
        (default).
    seed : int, optional
        Seed for the random sample of `max_nmol`.
    batch_size : int, default 20000
        Number of molecules read (and held in memory) per batch.
    mask_name : str, optional
        If given, excludes molecules dropped by
        `filter_dropout(mask_name=mask_name)` on the "test" source,
        without modifying `exp`.
    nworker : int, default 1
        Number of parallel processes to spawn. The estimate does not
        depend on it.
    llink_evid_name : str, optional
        If given, stores the evidence of each candidate in
        `exp.analysis[chrom][llink_evid_name]`, a DataFrame with columns
        `llink`, `evidence` and `nmol`, e.g. for plotting. `exp` is not
        modified if None (default).
    fill_nan : {"mean"}, float, or None, default None
        How to treat molecules that are nan at some but not all
        nucleosome start positions; see `run`. Molecules with no data
        (nan at every start position) are always left out.

    Returns
    -------
    dict of str to int
        Chromosome -> estimated `llink`.

    Raises
    ------
    ValueError
        If `llinks` is empty or holds a value that is not a non-negative
        integer, if `max_nmol` is not positive, if `nworker` is not a
        positive integer, if a chromosome has no usable molecules, if
        `fill_nan` is invalid, or if `fill_nan` is None and a molecule is
        nan at some but not all start positions.
    TypeError
        If `settings` is neither a path nor a SimSettings object.
    KeyError
        If `prob_name` or the mask of `mask_name` is missing for a
        chromosome.

    Warns
    -----
    UserWarning
        If `settings.end_temp` is not 1, if the estimate of a chromosome
        is the largest candidate (the maximum may lie beyond it), or if a
        chromosome uses fewer than 50 molecules.

    Notes
    -----
    `SimSettings.llink` acts as a soft minimum linker length rather than
    a preferred one, since the repulsion is zero for longer linkers.
    """
    manager = SimManager(nworker=nworker, verbose=False)
    return manager.estimate_llink(
        exp,
        settings,
        llinks=llinks,
        prob_name=prob_name,
        chroms=chroms,
        max_nmol=max_nmol,
        seed=seed,
        batch_size=batch_size,
        mask_name=mask_name,
        llink_evid_name=llink_evid_name,
        fill_nan=fill_nan,
    )


def filter_dropout(
    exp: MethPrintExperiment,
    *,
    which: str | None = None,
    channels: str | list | None = None,
    chroms: list | None = None,
    thres_min: float = 0.0,
    thres_max: float = 1.0,
    unmapped_strand: str = "union",
    method: str = "separate",
    mask_name: str = "dropout_mask",
) -> None:
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
    channels : str, list of str, or None, default None
        Channels to evaluate, as in `qc`. None uses one channel per
        `exp.mtase` label, or every assayed channel unmerged if both
        "GC" and "CG" are present.
    chroms : list of str, optional
        Subset of `exp.chroms` to evaluate. None (default) evaluates
        every chromosome.
    thres_min : float, default 0.0
        Min allowed no-signal fraction (per channel, or of the pooled
        total under `method="aggregate"`) to be kept.
    thres_max : float, default 1.0
        Max allowed no-signal fraction (per channel, or of the pooled
        total under `method="aggregate"`) to be kept.
    unmapped_strand : {"union", "drop", "+", "-"}, default "union"
        How to evaluate unmapped ('.') strand molecules: union of
        '+'/'-' position sets, always dropout, or treat as that
        strand.
    method : {"separate", "aggregate"}, default "separate"
        How multiple channels combine into `keep`. "separate": must
        clear `thres_min`/`thres_max` per channel. "aggregate": site
        counts pooled across channels into one fraction first.
        Irrelevant for a single channel.
    mask_name : str, default "dropout_mask"
        Key for the mask in `exp.analysis[chrom]`, as
        `f"{source}_{mask_name}"`.

    Raises
    ------
    ValueError
        If `mtase` is unset on `exp`, `channels` is invalid,
        `chroms` contains a chromosome not in
        `exp.chroms`, `thres_min` or `thres_max` is not in [0, 1],
        `thres_min` is greater than `thres_max`,
        `unmapped_strand`/`method` is invalid, a requested source is
        missing for some chromosome, or `refseq` is missing.
    """
    exp.filter_dropout(
        which=which,
        channels=channels,
        chroms=chroms,
        thres_min=thres_min,
        thres_max=thres_max,
        unmapped_strand=unmapped_strand,
        method=method,
        mask_name=mask_name,
    )


def qc(
    exp: MethPrintExperiment,
    *,
    channels: str | list | None = None,
    chroms: list | None = None,
    percentiles: float | list = (50,),
    mask_name: str | None = None,
    batch_size: int = 20000,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Summarize data quality per chromosome and channel.

    Thin wrapper around `MethPrintExperiment.qc`; see it for the
    columns of the returned table.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to summarize. The table is stored in
        `exp.global_analysis["qc"]`.
    channels : str, list of str, or None, default None
        Channels to summarize, as in `compute_model_prob`. None uses
        every assayed channel, unmerged.
    chroms : list of str, optional
        Subset of `exp.chroms`. None (default) uses all.
    percentiles : float or list of float, default (50,)
        Dropout percentiles to report, each in [0, 100].
    mask_name : str, optional
        Mask from `filter_dropout(mask_name=...)`. Masked molecules
        are left out of every column except `frac_masked_<src>`.
    batch_size : int, default 20000
        Molecules per batch when counting calls.
    overwrite : bool, default False
        Whether to replace an existing `exp.global_analysis["qc"]`.

    Returns
    -------
    pd.DataFrame
        One row per (chrom, channel), plus a `channel="all"` row per
        chrom if there is more than one channel.

    Raises
    ------
    ValueError
        If `channels` or `chroms` is invalid, `percentiles` is empty
        or outside [0, 100], the table exists and `overwrite` is
        False, or a chromosome has no `refseq` or no test data.
    KeyError
        If `mask_name` is not found for some source.
    """
    return exp.qc(
        channels=channels,
        chroms=chroms,
        percentiles=percentiles,
        mask_name=mask_name,
        batch_size=batch_size,
        overwrite=overwrite,
    )


def plot_dropout_ecdf(
    exp: MethPrintExperiment,
    *,
    chroms: str | list,
    which: str | None = None,
    channels: str | list | None = None,
    unmapped_strand: str = "union",
    pooled: bool = True,
    split_by: str | None = None,
    out_file: str | Path | None = None,
    show: bool = True,
):
    """
    Plot the empirical CDF of dropout rate per group.

    Computes `exp.dropout_fractions(...)`, restricted to `chroms`, and
    plots the result via `MethPlot.plot_dropout_ecdf`. For each
    matching group, at threshold `t` the plotted value is
    `100 * mean(dropout_frac <= t)` over that group's molecules.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment to evaluate.
    chroms : str or list of str
        Chromosome(s) to plot.
    which : {"test", "meth", "unmeth"} or None, default None
        Source to evaluate and plot. None uses every source present
        for each chromosome.
    channels : str, list of str, or None, default None
        Channels to evaluate, as in `filter_dropout`.
    unmapped_strand : {"union", "drop", "+", "-"}, default "union"
        How to evaluate unmapped ('.') strand molecules.
    pooled : bool, default True
        Whether to plot the pooled `channel="all"` groups.
    split_by : {"chrom", "source", "channel"} or None, default None
        Key field to split into panels, one per value. None draws a
        single panel.
    out_file : str or pathlib.Path, optional
        Path to save the generated figure. Directories are created if
        they do not exist.
    show : bool, default True
        Whether to display the plot using `plt.show()`.

    Raises
    ------
    ValueError
        If `mtase` is unset, `channels` is invalid,
        `unmapped_strand` is invalid, a source is missing for some
        chromosome, or `refseq` is missing (from
        `exp.dropout_fractions`); or if `chroms` holds an unknown
        chromosome or one with no matching entries in the result, or
        `split_by` is invalid.
    """
    chroms = [chroms] if isinstance(chroms, str) else list(chroms)
    fractions = exp.dropout_fractions(
        which=which,
        channels=channels,
        chroms=chroms,
        unmapped_strand=unmapped_strand,
    )
    methplot = MethPlot()
    methplot.plot_dropout_ecdf(
        fractions,
        chroms,
        which=which,
        pooled=pooled,
        split_by=split_by,
        out_file=out_file,
        show=show,
    )


def run(
    *,
    chroms: str | Iterable[str],
    nsim: int,
    settings: str | Path | SimSettings,
    meth_prob: np.ndarray | Mapping[str, np.ndarray | pd.DataFrame],
    out_dir: str | Path,
    dataset_name: str = "results",
    method: str = "mcmc",
    out_types: str | Iterable[str] = "all",
    seed: int | None = None,
    mols: IndexType | Mapping[str, IndexType] = slice(None),
    store_eseq: bool = True,
    use_median_eseq_mu: bool = False,
    fill_nan: str | float | None = None,
    nworker: int = 1,
    verbose: bool = True,
) -> SimDataset:
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
        Number of independent simulation runs per molecule (`method=
        'mcmc'`), or exact samples per molecule (`method='tm'`; may be
        0, since the exact occupancy/mean-N/variance quantities require
        no samples at all).
    settings : str or Path or SimSettings
        Simulation parameters. Can be a path to a configuration file or a
        `SimSettings` object. `llink` can be a mapping of chromosome to
        value, which must cover all of `chroms`.
    meth_prob : np.ndarray or Mapping[str, np.ndarray | pd.DataFrame]
        Probability of methylation. If multiple chromosomes are provided, this
        must be a mapping of {chrom_name: data}. Data can be NumPy arrays or
        Pandas DataFrames. Molecules that are nan at every nucleosome start
        position (e.g., molecules masked by `filter_dropout`) have no data
        and are not simulated; see `SimDataset.no_data`. Molecules that are
        nan at only some start positions are handled by `fill_nan`.
    out_dir : str or Path
        Directory or file prefix where simulation results will be stored.
    method : {"mcmc", "tm"}, default "mcmc"
        Which engine to run. For 'tm', `settings.end_temp` is the
        temperature the exact equilibrium distribution is evaluated
        at; `settings.start_temp`/`cool_option`/`nsweep`/`print_freq`
        and `out_types` are ignored.
    out_types : str | Iterable[str], default 'all'
        Types of data to record. Options include 'energy', 'position',
        'temp', or 'all'. Ignored for `method='tm'`.
    seed : int, optional
        Seed for the random number generator to ensure reproducibility.
    mols : IndexType | Mapping[str, IndexType], default slice(None
        Specific molecule indices to subset for the simulation.
    store_eseq : bool, default True
        Whether to store the sequence-specific nucleosome binding energy
        landscape derived from the methylation data to output dataset file.
    use_median_eseq_mu : bool : default False
        Whether to modify the chemical potential parameter so that it is
        equal to the mean of the methlyation energy.
    fill_nan : {"mean"}, float, or None, default None
        How to treat a molecule that is nan at some but not all
        nucleosome start positions. If None (default), raise
        `ValueError`. If "mean", fill each nan with the molecule's mean
        over its other start positions. If a number in [0, 1], fill each
        nan with that probability (0.5 adds no information). `meth_prob`
        is not modified, and the value is stored as `SimDataset.nan_fill`.
    nworker : int, default 1
        Number of parallel processes to spawn.
    verbose : bool, default True
        If True, print progress updates to the console.

    Returns
    -------
    SimDataset
        A container object providing access to the generated simulation
        results.

    Raises
    ------
    ValueError
        If `fill_nan` is invalid, or is None and a molecule is nan at some
        but not all nucleosome start positions.
    """
    manager = SimManager(nworker=nworker, verbose=verbose)
    dataset = manager.run(
        chroms=chroms,
        nsim=nsim,
        settings=settings,
        meth_prob=meth_prob,
        method=method,
        out_types=out_types,
        out_dir=out_dir,
        dataset_name=dataset_name,
        seed=seed,
        mols=mols,
        store_eseq=store_eseq,
        use_median_eseq_mu=use_median_eseq_mu,
        fill_nan=fill_nan,
    )
    return dataset


def compute_occup(
    dataset: SimDataset,
    *,
    time: int | None = None,
    chroms: str | Iterable[str] | None = None,
    occup_name: str = "occup",
    mean_nnuc_name: str | None = "mean_nnuc",
    compute_var: bool = False,
    var_nnuc_name: str = "var_nnuc",
    var_occup_name: str = "var_occup",
    record_time: bool = False,
    batch_size: int | None = None,
    nworker: int = 1,
):
    """
    Compute the nucleosome occupancy and mean nucleosome count.

    For `dataset.method == "mcmc"`, these are averages over the
    simulation runs at a given time point; for `"tm"`, they are computed
    exactly via the transfer-matrix solver, with no time axis, solving
    each molecule only once for all quantities. Results are stored in
    `dataset.analysis[chrom]`; molecules with no data (see
    `SimDataset.no_data`) get nan.

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset.
    time : int, optional
        The time point (snapshot) to use. If None, the final frame is
        used. `method='mcmc'` only; must be None for
        `dataset.method == "tm"`.
    chroms : str or iterable of str, optional
        The chromosome(s) to process. If None (default), all of
        `dataset.chroms`.
    occup_name : str, default "occup"
        The key of the per-bp occupancy (molecules by base pairs).
    mean_nnuc_name : str or None, default "mean_nnuc"
        The key of the mean nucleosome count of each molecule. If None,
        it is not computed.
    compute_var : bool, default False
        If True, also compute the exact variance of the nucleosome
        count and per-bp occupancy. Requires `dataset.method == "tm"`.
    var_nnuc_name : str, default "var_nnuc"
        The key of the nucleosome count variance. Only used if
        `compute_var` is True.
    var_occup_name : str, default "var_occup"
        The key of the per-bp occupancy variance. Only used if
        `compute_var` is True.
    record_time : bool, default False
        If True, the time point is appended to the keys (e.g.,
        "occup_t_100"). `method='mcmc'` only.
    batch_size : int, optional
        Number of molecules per batch. For "mcmc", the number held in
        memory at a time (default 20000); for "tm", the number per
        worker job (default 64).
    nworker : int, default 1
        Number of worker processes. Values above 1 are only supported
        for `dataset.method == "tm"`.

    Raises
    ------
    ValueError
        If `time` or `record_time` is given for a `dataset.method ==
        "tm"` dataset, if `compute_var` is True but `dataset.method !=
        "tm"`, or if `nworker` is not positive or above 1 for a non-"tm"
        dataset.
    """
    SimAnalysis().compute_occup(
        dataset,
        time=time,
        chroms=chroms,
        name=occup_name,
        mean_nnuc_name=mean_nnuc_name,
        compute_var=compute_var,
        var_nnuc_name=var_nnuc_name,
        var_occup_name=var_occup_name,
        record_time=record_time,
        batch_size=batch_size,
        nworker=nworker,
    )


def compute_config(
    dataset: SimDataset,
    *,
    method: str = "best_match",
    gamma: float | None = None,
    chroms: str | Iterable[str] | None = None,
    name: str | None = None,
    nworker: int = 1,
) -> None:
    """
    Call one nucleosome configuration per molecule.

    Solves each molecule exactly with the transfer matrix and stores a
    single configuration (start positions), together with its energy and
    log posterior probability. Read the result with `extract_config`.
    Only available for `dataset.method == "tm"` datasets.

    Parameters
    ----------
    dataset : SimDataset
        A "tm" dataset with a stored energy landscape.
    method : {"best_match", "most_likely"}, default "best_match"
        "best_match" calls the configuration that best matches the
        occupancy profile (maximum expected accuracy), penalized by the
        linker repulsion so that it never contains linkers the model
        forbids; "most_likely" calls the single most probable
        configuration (maximum a posteriori), which tends to be more
        densely packed.
    gamma : float, optional
        For "best_match" only: the weight of correctly called occupied
        sites relative to correctly called empty sites; larger values
        call more nucleosomes. 1.0 if None. Must be positive.
    chroms : str or iterable of str, optional
        The chromosome(s) to process. If None (default), all of
        `dataset.chroms`.
    name : str, optional
        The prefix of the stored keys `{name}_position_flat`,
        `{name}_position_offset` and `{name}_energy` in
        `dataset.analysis[chrom]`; `method` if None. An existing result
        under `name` is replaced.
    nworker : int, default 1
        Number of worker processes.

    Raises
    ------
    ValueError
        If `method` is unknown, if `gamma` is given with "most_likely" or
        is not positive, if `dataset.method != "tm"` or it has no stored
        energy landscape, or if `nworker` is not positive.

    Notes
    -----
    Molecules with no data (see `SimDataset.no_data`) get no positions
    and nan energy; `extract_config` returns None for them.
    """
    if method not in ("best_match", "most_likely"):
        raise ValueError(
            f"method must be 'best_match' or 'most_likely', got {method!r}"
        )
    if name is None:
        name = method
    ana = SimAnalysis()
    if method == "most_likely":
        if gamma is not None:
            raise ValueError("gamma is only used with method='best_match'")
        ana.compute_most_likely_config(
            dataset, chroms=chroms, name=name, nworker=nworker
        )
        return
    ana.compute_best_match_config(
        dataset,
        chroms=chroms,
        name=name,
        gamma=1.0 if gamma is None else gamma,
        nworker=nworker,
    )


def compute_linker_hist(
    dataset: SimDataset,
    *,
    lmax: int = 200,
    window: tuple[int, int] | Mapping | None = None,
    trim: int = 0,
    chroms: str | Iterable[str] | None = None,
    name: str = "linker_hist",
    batch_size: int | None = None,
    nworker: int = 1,
):
    """
    Compute the exact linker-length histogram of each molecule.

    A linker is the DNA between two consecutive nucleosomes; the
    stretches between the fiber ends and the outermost nucleosomes are
    not linkers. For each molecule, the expected number of linkers of
    each length under the posterior is computed exactly via the
    transfer-matrix solver, with no sampling. The values are expected
    counts, not normalized. Requires `dataset.method == "tm"`. Results
    are stored in `dataset.analysis[chrom]`; molecules with no data (see
    `SimDataset.no_data`) get nan rows. Use `extract_linker_stats` for
    the mean and standard deviation of the linker lengths, and
    `plot_linker_hist` to plot the histograms.

    The histogram reflects both the spacing of the nucleosomes and the
    uncertainty of their positions, which widens it; a strongly
    positioned array (e.g., the 601) shows how much. The results assume
    that each molecule is read along the whole fiber; linkers in regions
    filled by `fill_nan` follow the prior.

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset, with a stored energy landscape.
    lmax : int, default 200
        The longest linker length (bp) with its own bin; longer linkers
        are pooled into the last column ("tail").
    window : tuple of int or dict, optional
        The region `(xlo, xhi)` (bp, half-open, in the coordinates of the
        stored start positions) in which linkers are counted: both
        nucleosomes of a linker must lie inside it. A dict keyed by chrom
        gives one window per chrom (chroms not in it use the whole fiber;
        extra keys are ignored). If None (default), the whole fiber.
    trim : int, default 0
        Number of base pairs removed from each end of the window, e.g.,
        to drop nucleosomes near the fiber ends.
    chroms : str or iterable of str, optional
        The chromosome(s) to process. If None (default), all of
        `dataset.chroms`.
    name : str, default "linker_hist"
        The key of the histogram: an `H5Array` of shape (nmol, lmax + 2)
        with columns "0" to str(lmax) and "tail". The window and settings
        used are stored under `f"{name}_settings"`.
    batch_size : int, optional
        Number of molecules per worker job (default 64).
    nworker : int, default 1
        Number of worker processes.

    Raises
    ------
    TypeError
        If `lmax` or `trim` is not an integer.
    ValueError
        If `dataset.method != "tm"`, the dataset has no stored energy
        landscape, `lmax` or `trim` is negative, `window` is invalid or
        outside the fiber, the window after trimming cannot hold two
        nucleosomes, or `nworker` is not positive.

    Warns
    -----
    UserWarning
        If the dataset was created with `fill_nan`, since linkers in
        filled regions are counted.
    """
    SimAnalysis().compute_linker_hist(
        dataset,
        lmax=lmax,
        window=window,
        trim=trim,
        chroms=chroms,
        name=name,
        batch_size=batch_size,
        nworker=nworker,
    )


def extract_linker_stats(
    dataset: SimDataset,
    *,
    chroms: str | Iterable[str] | None = None,
    name: str = "linker_hist",
    mols: Iterable[int] | dict | None = None,
    pooled: bool = False,
) -> pd.DataFrame:
    """
    Compute the mean and standard deviation of the linker lengths.

    Uses the histograms stored by `compute_linker_hist`, so the values
    are exact posterior averages. Only linkers up to `lmax` enter; the
    tail bin is excluded, since its lengths are not resolved. The spread
    reflects both the irregularity of the spacing and the uncertainty of
    the nucleosome positions; compare with a strongly positioned array
    (e.g., the 601) to separate the two.

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset holding the histograms.
    chroms : str or iterable of str, optional
        The chromosome(s) to use. If None (default), all of
        `dataset.chroms`.
    name : str, default "linker_hist"
        The key of the histograms, as passed to `compute_linker_hist`.
    mols : iterable of int or dict, optional
        The molecules to use (duplicates are counted once). If None
        (default), all molecules. With several chroms, a dict keyed by
        chrom (chroms not in it use all molecules; extra keys are
        ignored).
    pooled : bool, default False
        If False, one row per molecule. If True, one row per chromosome,
        from the linkers of all selected molecules pooled together, so
        that each linker counts equally. The pooled standard deviation
        also includes the differences between the mean linker lengths of
        the molecules, so it is not the mean of the per-molecule values.

    Returns
    -------
    pd.DataFrame
        Columns "mean" and "sd" (bp) and "nlink", the expected number of
        linkers up to `lmax`. Indexed by (chrom, mol) if `pooled` is
        False, otherwise by chrom. Molecules with no data get nan;
        molecules or chromosomes with no linkers get nan mean and sd.

    Raises
    ------
    KeyError
        If no histogram is stored under `name` for a chromosome.
    ValueError
        If `mols` is not a dict with several chroms or is out of range.
    """
    return SimAnalysis().extract_linker_stats(
        dataset, chroms=chroms, name=name, mols=mols, pooled=pooled
    )


def extract_config(
    dataset: SimDataset,
    *,
    chrom: str,
    name: str | None = None,
    run: int | None = None,
    time: int | None = None,
    mols: Iterable[int] | None = None,
    nworker: int = 1,
) -> list[np.ndarray | None]:
    """
    Extract one nucleosome configuration per molecule.

    With `name`, reads the configurations stored by `compute_config`.
    Without it, reads the raw configurations: run `run` at time `time`
    of an MCMC dataset, or exact sample `run` of a TM dataset. Nothing
    is computed.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the configurations.
    chrom : str
        The chromosome of the molecules.
    name : str, optional
        The `name` given to `compute_config` (e.g., "best_match"). If
        None (default), raw configurations are read.
    run : int, optional
        The MCMC run or TM sample to read; 0 if None. Only used if
        `name` is None.
    time : int, optional
        The MCMC time point to read; the final frame if None. Only used
        for MCMC datasets with `name` None.
    mols : iterable of int, optional
        The molecules to read, in this order. If None (default), all
        molecules of `chrom`.
    nworker : int, default 1
        Number of threads used to read raw MCMC files.

    Returns
    -------
    list of np.ndarray or None
        The nucleosome start positions of each molecule, in increasing
        order, or None for a molecule with no data (see
        `SimDataset.no_data`) or no file on disk. An empty array is a
        configuration without nucleosomes.

    Raises
    ------
    ValueError
        If `run` or `time` is given with `name`, if `time` is given for a
        TM dataset, if a TM dataset stores no samples (`nsim` is 0) and
        `name` is None, or if `time` is not a recorded time point.
    KeyError
        If no configurations are stored under `name` for `chrom`.
    IndexError
        If a molecule or `run` is out of range.
    """
    return SimAnalysis().extract_config(
        dataset,
        chrom=chrom,
        name=name,
        run=run,
        time=time,
        mols=mols,
        nworker=nworker,
    )


def plot_config(
    dataset: SimDataset,
    *,
    chroms: str | Iterable[str],
    name: str | None = None,
    mols: Iterable[int] | dict | None = None,
    run: int | None = None,
    time: int | None = None,
    max_rows: int | None = None,
    downsample_how: str = "stride",
    xscale: int = 1000,
    cmap: str | None = None,
    out_file: str | Path | None = None,
    plot_mean: bool = False,
    plot_eseq: bool = False,
    link_mat: np.ndarray | pd.DataFrame | dict | None = None,
    sharex: bool = False,
    show: bool = True,
):
    """
    Plot one nucleosome configuration per molecule.

    Each row is a molecule, showing the bases covered by the nucleosomes
    (1) and the linkers (0) of one configuration read with
    `extract_config`: the calls stored under `name` by `compute_config`,
    or the raw configuration of run `run` at time `time` (MCMC) or of
    exact sample `run` (TM). Molecules without data are drawn in gray.
    For many configurations of one molecule, see
    `SimPlot.plot_samples`. Each chromosome gets its own panel; several
    panels are laid out side by side (at most 4 per row) in one figure,
    aligned and sharing one legend or colorbar.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the configurations.
    chroms : str or iterable of str
        Chromosome(s) to plot, one panel each, in reading order.
    name : str, optional
        The `name` given to `compute_config`. If None (default), raw
        configurations are shown.
    mols : iterable of int or dict, optional
        Molecules to show, in this order (all if None). The y-axis counts
        rows, not molecule indices. With several chroms, a dict keyed by
        chrom (chroms not in it show all molecules).
    run : int, optional
        MCMC run or TM sample to show; 0 if None. Only used if `name` is
        None.
    time : int, optional
        MCMC time point to show; the final frame if None. Only used for
        MCMC datasets with `name` None.
    max_rows : int, optional
        If given, show at most this many rows, merged as `downsample_how`
        says.
    downsample_how : {"stride", "mean", "max"}, default "stride"
        How to reduce the rows to `max_rows`: "stride" keeps every k-th
        molecule (real configurations), "mean" shows the covered share of
        each group of molecules, and "max" whether any of them is
        covered.
    xscale : int, default 1000
        Spatial scaling factor (a power of 10) for the x-axis labels.
    cmap : str, optional
        Matplotlib colormap name. If None, the `SimPlot` default is used.
    out_file : str or Path, optional
        Path to save the figure. If None, it is not saved.
    plot_mean : bool, default False
        Whether to plot, below the heatmap, the share of the shown
        molecules with a nucleosome at each base (on a 0 to 1 scale),
        i.e., the mean of their configurations. Molecules without data
        are ignored. Computed before `max_rows` merging, so each shown
        molecule counts once.
    plot_eseq : bool, default False
        Whether to plot the sequence-specific binding energy, averaged
        over the shown molecules, below the heatmap (and below the mean,
        if both are plotted).
    link_mat : np.ndarray, pd.DataFrame, or dict, optional
        Linkage matrix of these molecules (as returned or stored by
        `sort_by_linkage`). If given, the molecules are put in its leaf
        order before `max_rows` downsampling, and its dendrogram is drawn
        alongside. Cannot be combined with `mols`. With several chroms, a
        dict keyed by chrom, e.g. the result of `sort_by_linkage` (chroms
        not in it are not sorted; extra keys are ignored).
    sharex : bool, default False
        If True, all panels span the length of the longest chrom, so they
        share one bp scale. If False, each panel spans its own chrom.
    show : bool, default True
        Whether to display the figure.

    Raises
    ------
    ValueError
        If `chroms` is empty, if `max_rows` or `downsample_how` is
        invalid, if `xscale` is not a power of 10, if `plot_eseq` is True
        and the dataset stores no energies, if `mols` or `link_mat` is
        not a dict with several chroms, if `link_mat` is combined with
        `mols` or does not match the number of molecules, or for invalid
        arguments of `extract_config`.
    KeyError
        If no calls are stored under `name` for a chrom.
    IndexError
        If a molecule or `run` is out of range.

    Warns
    -----
    UserWarning
        If rendering the plot is estimated to need more than 1 GB of
        memory (pass `max_rows` or `mols` to reduce it), or if more than
        10 chroms are plotted in one figure.
    """
    SimPlot().plot_config(
        dataset,
        chroms=chroms,
        name=name,
        mols=mols,
        run=run,
        time=time,
        max_rows=max_rows,
        downsample_how=downsample_how,
        xscale=xscale,
        cmap=cmap,
        out_file=out_file,
        plot_mean=plot_mean,
        plot_eseq=plot_eseq,
        link_mat=link_mat,
        sharex=sharex,
        show=show,
    )


def plot_occup(
    dataset: SimDataset,
    *,
    chroms: str | Iterable[str],
    time: int | None = None,
    out_file: str | Path | None = None,
    occup_name: str = "occup",
    mols: Iterable[int] | dict | None = None,
    xscale: int = 1000,
    cmap: str | None = None,
    plot_mean: bool = False,
    plot_eseq: bool = False,
    link_mat: np.ndarray | dict | None = None,
    sharex: bool = False,
    show: bool = True,
):
    """
    Visualize nucleosome occupancy profiles for one or more chromosomes.

    Generate a plot showing the probability of nucleosome occupancy across
    the genomic coordinates based on previously computed analysis within
    the SimDataset. Each chromosome gets its own panel; several panels
    are laid out side by side (at most 4 per row) in one figure, aligned
    and sharing one colorbar.

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset containing the computed occupancy data.
    chroms : str or iterable of str
        The chromosome(s) to plot, one panel each, in reading order.
    time : int, optional
        The time step of interest if occupancy needs to be computed.
    out_file : str | Path, optional
        Path where the generated plot will be saved. If None, the plot
        is not saved to disk.
    occup_name : str, default "occup"
        The key or name of the occupancy data to retrieve from the dataset.
        This should match the `occup_name` used by `compute_occup`.
    mols : iterable of int or dict, optional
        Molecule indices to display, restricting the heatmap (and, if
        `plot_eseq` is True, the sequence-energy average) to this
        subset. If None (default), all molecules are shown. With several
        chroms, a dict keyed by chrom (chroms not in it show all
        molecules).
    xscale : int, default 1000
        Spatial scaling factor (must be a power of 10) for the x-axis
        labels.
    cmap : str, optional
        Matplotlib colormap name to use for this plot. If None, the
        `SimPlot` default is used.
    plot_mean : bool, default False
        Whether to plot the mean occupancy across the shown molecules
        (one value per bp, on a 0 to 1 scale) below the heatmap.
    plot_eseq : bool, default False
        Whether to plot the underlying sequence-specific nucleosome binding
        energy, averaged across all molecules, below the heatmap (and
        below the mean, if both are plotted).
    link_mat : np.ndarray, pd.DataFrame, or dict, optional
        Linkage matrix of these molecules (as returned or stored by
        `sort_by_linkage`). If given, the rows are put in its leaf order
        and its dendrogram is drawn alongside. Cannot be combined with
        `mols`. If None (default), rows are in molecule order. With
        several chroms, a dict keyed by chrom, e.g. the result of
        `sort_by_linkage` (chroms not in it are not sorted; extra keys
        are ignored).
    sharex : bool, default False
        If True, all panels span the length of the longest chrom, so
        they share one bp scale. If False, each panel spans its own
        chrom.
    show : bool, default True
        If True, invokes the active plotting backend to display the
        figure immediately.

    Raises
    ------
    ValueError
        If `chroms` is empty, if the occupancy is missing and no `time`
        is provided for computation, or if `mols` or `link_mat` is not
        a dict with several chroms.

    Warns
    -----
    UserWarning
        If more than 10 chroms are plotted in one figure, or if
        rendering is estimated to need more than 1 GB of memory.

    Notes
    -----
    If `occup_name` is missing, the occupancy is computed first (as
    `compute_occup` does, without the mean nucleosome count).
    """
    simplot = SimPlot()
    simplot.plot_occup(
        chroms=chroms,
        dataset=dataset,
        time=time,
        occup_name=occup_name,
        mols=mols,
        xscale=xscale,
        cmap=cmap,
        out_file=out_file,
        plot_mean=plot_mean,
        plot_eseq=plot_eseq,
        link_mat=link_mat,
        sharex=sharex,
        show=show,
    )


def plot_linker_hist(
    dataset: SimDataset,
    *,
    chroms: str | Iterable[str],
    name: str = "linker_hist",
    mols: Iterable[int] | dict | None = None,
    panels: bool = False,
    sharey: bool = True,
    density: bool = True,
    plot_stats: bool = True,
    logy: bool = False,
    out_file: str | Path | None = None,
    show: bool = True,
):
    """
    Plot the pooled linker-length histogram of one or more chromosomes.

    Uses the histograms stored by `compute_linker_hist`, pooled over the
    selected molecules of each chromosome. Only linkers up to `lmax` are
    shown (the tail bin is left out). By default, all chromosomes are
    drawn as filled steps on one axis, which suits a few chromosomes;
    with `panels`, each gets its own panel, laid out side by side (at
    most 4 per row).

    Parameters
    ----------
    dataset : SimDataset
        The simulation dataset holding the histograms.
    chroms : str or iterable of str
        The chromosome(s) to plot, in order.
    name : str, default "linker_hist"
        The key of the histograms, as passed to `compute_linker_hist`.
    mols : iterable of int or dict, optional
        The molecules to pool. If None (default), all molecules. With
        several chroms, a dict keyed by chrom (chroms not in it use all
        molecules; extra keys are ignored).
    panels : bool, default False
        If True, one panel per chromosome; otherwise all on one axis.
    sharey : bool, default True
        Whether the panels share the y-axis. Only used if `panels` is
        True.
    density : bool, default True
        If True, plot the fraction of linkers (up to `lmax`) of each
        length, so that chromosomes with different numbers of molecules
        can be compared. If False, plot the expected number of linkers of
        each length per molecule with data.
    plot_stats : bool, default True
        Whether to show the mean and standard deviation of the linker
        lengths (as from `extract_linker_stats` with `pooled=True`), in
        the legend or, with `panels`, in each panel.
    logy : bool, default False
        Whether to use a logarithmic y-axis.
    out_file : str or Path, optional
        Path where the generated plot will be saved. If None, the plot is
        not saved to disk.
    show : bool, default True
        If True, invokes the active plotting backend to display the
        figure immediately.

    Raises
    ------
    KeyError
        If no histogram is stored under `name` for a chromosome.
    ValueError
        If `chroms` is empty, or `mols` is not a dict with several chroms
        or is out of range.

    Warns
    -----
    UserWarning
        If more than 10 chroms are plotted in one figure, or a chromosome
        has no linkers up to `lmax` (it is then not plotted).
    """
    SimPlot().plot_linker_hist(
        dataset,
        chroms=chroms,
        name=name,
        mols=mols,
        panels=panels,
        sharey=sharey,
        density=density,
        plot_stats=plot_stats,
        logy=logy,
        out_file=out_file,
        show=show,
    )


def plot_meth(
    data: H5Array | pd.DataFrame | np.ndarray | None = None,
    *,
    value: str = "prob",
    exp: MethPrintExperiment | None = None,
    chroms: str | Iterable[str] | None = None,
    raw_which: str | None = None,
    data_name: str | None = None,
    mols: int | Sequence[int] | dict | None = None,
    mask_name: str | None = None,
    emax: float | None = None,
    max_rows: int | None = None,
    downsample_how: str = "mean",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
    cbar_label: str | None = None,
    out_file: str | Path | None = None,
    link_mat: np.ndarray | dict | None = None,
    xscale: int = 1000,
    sharex: bool = False,
    show: bool = True,
):
    """
    Plot a methylation heatmap, as probability or energy, of one matrix
    or of one or more chromosomes of an experiment.

    Plots at full resolution -- for large data, pass `max_rows` to
    downsample first, or downsample it yourself (`utils.downsample`,
    works uniformly for `H5Array`, `pd.DataFrame`, or `np.ndarray`) and
    pass the reduced result. Several chromosomes get one panel each,
    laid out side by side (at most 4 per row) in one figure, aligned and
    sharing one colorbar on a common color scale.

    Parameters
    ----------
    data : H5Array, pd.DataFrame, or np.ndarray, optional
        One dense signal matrix to plot (rows=molecules, columns=bp
        position), e.g. `exp.analysis[chrom]["meth_prob"]` or an array
        not stored in an experiment. Required unless `exp` and `chroms`
        are given instead.
    value : {"prob", "energy"}, default "prob"
        "prob" plots the values as given; "energy" plots `logit(data)`,
        the same quantity `catella.simulation.engine` treats as physical
        energy, on a color scale centered at zero. "energy" requires
        probabilities in [0, 1] (`data` or `data_name`) and cannot be
        combined with `raw_which`/`mols`/`mask_name`.
    exp : MethPrintExperiment, optional
        Experiment to read from, together with `chroms` and exactly one
        of `raw_which`/`data_name`. Not combined with `data`.
    chroms : str or iterable of str, optional
        Chromosome(s) to plot from `exp`, one panel each, in reading
        order.
    raw_which : {"test", "unmeth", "meth"}, optional
        Which raw table to plot per chromosome. Calls
        `exp.to_dense(chrom, which=raw_which, mols=mols,
        mask_name=mask_name)` internally.
    data_name : str, optional
        Key of a dense array to plot per chromosome, read from
        `exp.analysis[chrom][data_name]` (e.g., "meth_prob").
    mols : int, sequence of int, or dict, optional
        Molecule index/indices to include; only with `raw_which`. See
        `MethPrintExperiment.to_dense`. With several chroms, a dict keyed
        by chrom (chroms not in it show all molecules).
    mask_name : str, optional
        Dropout mask name to apply; only with `raw_which`. See
        `MethPrintExperiment.to_dense`.
    emax : float, optional
        For "energy" only: clamp energy to [-emax, emax] (matching
        `SimSettings.emax`'s clamping in `NucPosMCMC.setEnergy`) and
        use it as the default `vmin`/`vmax`. If None, energy is
        unclamped and the default range is inferred from the finite
        values.
    max_rows : int, optional
        If given, each panel is downsampled to at most this many rows
        (via `utils.downsample`) before plotting.
    downsample_how : {"mean", "sum", "min", "max", "stride"},
        default "mean"
        How to collapse rows when `max_rows` is given. See
        `utils.downsample`.
    vmin : float, optional
        Lower bound for the color scale. If None, inferred from the data
        (over all panels). For "energy", combined with `vmax` into a
        symmetric half-range so the color scale stays centered at zero.
    vmax : float, optional
        Upper bound for the color scale. If None, inferred from the data
        (over all panels). See `vmin` for "energy".
    cmap : str, optional
        Matplotlib colormap name to use for this plot. If None, the
        `MethPlot` default is used for "prob" and "RdBu_r" for
        "energy".
    cbar_label : str, optional
        Label drawn next to the colorbar. If None, "Methylation prob."
        for "prob" and "Energy [$k_BT$]" for "energy".
    out_file : str | Path, optional
        Path where the generated plot will be saved. If None, the plot
        is not saved to disk.
    link_mat : np.ndarray, pd.DataFrame, or dict, optional
        Linkage matrix of these molecules (as returned or stored by
        `sort_by_linkage`). If given, the rows are put in its leaf order
        before `max_rows` downsampling and its dendrogram is drawn
        alongside. Cannot be combined with `mols`. With several chroms, a
        dict keyed by chrom, e.g. the result of `sort_by_linkage` (chroms
        not in it are not sorted; extra keys are ignored).
    xscale : int, default 1000
        Spatial scaling factor (a power of 10) for the x-axis labels.
    sharex : bool, default False
        If True, all panels span the widest chromosome, so they share
        one bp scale. If False, each panel spans its own.
    show : bool, default True
        Whether to display the figure using the active plotting
        backend.

    Raises
    ------
    ValueError
        If `value` is not "prob" or "energy"; if `xscale` is not a power
        of 10; if neither `data` nor
        `exp` and `chroms` with exactly one of `raw_which`/`data_name`
        are given; if `data` is combined with any of `exp`/`chroms`/
        `raw_which`/`data_name`/`mols`/`mask_name`; if `mols` or
        `mask_name` is given without `raw_which`; if "energy" is
        combined with `raw_which`/`mols`/`mask_name`; if `emax` is given
        with "prob"; or if `mols` or `link_mat` is not a dict with
        several chroms.

    Warns
    -----
    UserWarning
        If more than 10 chroms are plotted in one figure, or if
        rendering is estimated to need more than 1 GB of memory.
    """
    if value not in ("prob", "energy"):
        raise ValueError(f'value must be "prob" or "energy", got {value!r}')
    kwargs = {
        "exp": exp,
        "chroms": chroms,
        "data_name": data_name,
        "max_rows": max_rows,
        "downsample_how": downsample_how,
        "vmin": vmin,
        "vmax": vmax,
        "cmap": cmap,
        "out_file": out_file,
        "link_mat": link_mat,
        "xscale": xscale,
        "sharex": sharex,
        "show": show,
    }
    if cbar_label is not None:
        kwargs["cbar_label"] = cbar_label
    methplot = MethPlot()
    if value == "prob":
        if emax is not None:
            raise ValueError('emax is only valid with value="energy"')
        methplot.plot_meth_prob(
            data,
            raw_which=raw_which,
            mols=mols,
            mask_name=mask_name,
            **kwargs,
        )
        return
    if any(v is not None for v in (raw_which, mols, mask_name)):
        raise ValueError(
            'value="energy" does not support raw_which, mols or mask_name'
        )
    methplot.plot_meth_energy(data, emax=emax, **kwargs)


def downsample(
    *,
    data: H5Array | pd.DataFrame | np.ndarray,
    max_rows: int,
    how: str = "mean",
    batch_size: int = 20000,
) -> np.ndarray:
    """
    Collapse rows of a dense array to at most `max_rows`.

    Public wrapper around `utils.downsample`, for reducing plot data
    ahead of `plot_occup`/`plot_meth`.

    Parameters
    ----------
    data : H5Array, pd.DataFrame, or np.ndarray
        2D data to collapse (rows=molecules, columns=bp position).
    max_rows : int
        Target number of rows.
    how : {"mean", "sum", "min", "max", "stride"}, default "mean"
        How to collapse groups of consecutive rows into one row.
    batch_size : int, default 20000
        Number of rows reduced at a time within a bin, when `how`
        aggregates bins. Ignored for "stride".

    Returns
    -------
    np.ndarray
        Array of shape (min(nrow, max_rows), ncol). If nrow <=
        `max_rows`, the data are returned as given. Otherwise, "mean"
        keeps a float dtype and gives float64 for other input; "sum"
        follows `np.nansum` (e.g., int8 gives int64); "min", "max", and
        "stride" keep the input dtype.

    Raises
    ------
    ValueError
        If `how` is not a recognized option, or if `max_rows` or
        `batch_size` is not positive.
    """
    return utils.downsample(data, max_rows, how=how, batch_size=batch_size)


def sort_by_linkage(
    *,
    dataset: SimDataset | None = None,
    exp: MethPrintExperiment | None = None,
    chroms: str | Iterable[str] | None = None,
    data_name: str | None = None,
    config_name: str | None = None,
    raw_which: str | None = None,
    mask_name: str | None = None,
    store_link_mat: bool = True,
    link_mat_name: str | None = None,
    metric: str = "euclidean",
    method: str = "ward",
    batch_size: int = 20000,
    fill_nan: str | float | None = None,
) -> dict[str, np.ndarray]:
    """
    Cluster molecules by similarity, for sorted plots.

    Computes a hierarchical-clustering linkage per chromosome; nothing is
    reordered or copied. Pass the linkage as `link_mat` to a plot
    (`plot_occup`, `plot_config`, `plot_meth`),
    which puts its rows in the linkage's leaf order. Several plots of
    the same molecules can share one linkage:

    >>> lm = catella.sort_by_linkage(dataset=ds, data_name="occup")
    >>> catella.plot_occup(ds, chroms=c, link_mat=lm[c])
    >>> catella.plot_config(ds, chroms=c, name="best_match", link_mat=lm)

    Dispatches to `SimAnalysis.sort_by_linkage` (if `dataset` is given)
    or `MethPrintAnalysis.sort_by_linkage` (if `exp` is given) -- exactly
    one of the two must be provided.

    Parameters
    ----------
    dataset : SimDataset, optional
        Simulation dataset to cluster. Mutually exclusive with `exp`.
    exp : MethPrintExperiment, optional
        Methylation experiment to cluster. Mutually exclusive with
        `dataset`.
    chroms : str or iterable of str, optional
        Chromosome(s) to process. If None (default), all chromosomes
        in `dataset.chroms` (or `exp.chroms`) are processed.
    data_name : str, optional
        Key of the analysis array to cluster. If None, the underlying
        method's own default is used ("occup" for `dataset`,
        "test_smoothed" for `exp`).
    config_name : str, optional
        Only used with `dataset`. The `name` of configurations stored by
        `compute_config`; molecules are clustered by their footprints
        (the base pairs covered by their calls) instead of an analysis
        array. Cannot be combined with `data_name`. Molecules with no
        data need `fill_nan`.
    raw_which : {"test", "meth", "unmeth"}, optional
        Only used with `exp`. If given, clusters raw long-form data
        instead of an `exp.analysis` entry.
    mask_name : str, optional
        Only used with `exp` and `raw_which`. If given, molecules
        flagged as dropout by a prior `filter_dropout(mask_name=
        mask_name)` call are all-nan rows (which need `fill_nan`).
    store_link_mat : bool, default True
        Whether to also persist each chromosome's linkage matrix into
        `.analysis[chrom][link_mat_name]`.
    link_mat_name : str, optional
        Key used to store the linkage matrix if `store_link_mat` is
        True. Defaults to `f"{data_name}_linkage"` (or
        `f"{config_name}_linkage"`, `f"{raw_which}_linkage"`).
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
        If neither or both of `dataset`/`exp` are given, if `mask_name`
        is given with `dataset` or `config_name` with `exp`, if both
        `data_name` and `config_name` are given, or if some molecule has
        no data (all nan) and `fill_nan` is None.
    """
    if (dataset is None) == (exp is None):
        raise ValueError("exactly one of 'dataset' or 'exp' must be given")
    if exp is not None and config_name is not None:
        raise ValueError("'config_name' is only used with 'dataset'")
    kwargs = {
        "store_link_mat": store_link_mat,
        "link_mat_name": link_mat_name,
        "metric": metric,
        "method": method,
        "batch_size": batch_size,
        "fill_nan": fill_nan,
    }
    if data_name is not None:
        kwargs["data_name"] = data_name
    if chroms is not None:
        kwargs["chroms"] = chroms
    if dataset is not None:
        if mask_name is not None:
            raise ValueError("'mask_name' is only used with 'exp'")
        return SimAnalysis().sort_by_linkage(
            dataset=dataset, config_name=config_name, **kwargs
        )
    if raw_which is not None:
        kwargs["raw_which"] = raw_which
    if mask_name is not None:
        kwargs["mask_name"] = mask_name
    return MethPrintAnalysis().sort_by_linkage(exp=exp, **kwargs)
