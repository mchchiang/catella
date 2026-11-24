# cli.py

import enum
import json
from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

import catella
from catella.experiment.methdata import MethPrintExperiment
from catella.simulation.results import SimDataset

app = typer.Typer(help="Nucleosome Positioning from Methylation Footprinting")


class SourceKind(str, enum.Enum):
    """Which object type an HDF5 file holds, for commands that operate on
    either a MethPrintExperiment or a SimDataset."""

    experiment = "experiment"
    dataset = "dataset"


def _load_source(source_file: Path, kind: "SourceKind"):
    if kind == SourceKind.experiment:
        return MethPrintExperiment.load(source_file)
    return SimDataset.load(source_file)


# Helper to parse a comma-separated option value into a list. Used as
# a `callback` (not `parser`) so it runs once on the final string
# value; `typer.Option(parser=...)` on a list[T]-typed option instead
# treats each flag occurrence as one parsed item and wraps the results
# in an outer list, which double-wraps a parser that already returns a
# list from a single occurrence.
def csv_parser[T](
    target_type: type[T],
) -> Callable[[str | None], list[T] | None]:
    def parser(value: str | None) -> list[T] | None:
        if value is None:
            return None
        if not value:
            return []
        return [target_type(item.strip()) for item in value.split(",")]

    return parser


def fill_nan_parser(value: str | None):
    if value is None:
        return None
    if value == "mean":
        return value
    try:
        return float(value)
    except ValueError:
        raise typer.BadParameter("fill_nan must be 'mean' or a number")


def eta_parser(value: str | None):
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise typer.BadParameter(
            "eta must be a number, or a JSON object of channel name "
            'to number, e.g. \'{"A": 0.9, "HCG": 0.7}\''
        )


@app.command(name="load_raw")
def load_raw(
    chromsize: Annotated[
        Path,
        typer.Argument(
            help="Chromosome sizes file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    test_file: Annotated[
        Path,
        typer.Argument(
            help="Primary methylation file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    out_file: Annotated[
        Path,
        typer.Argument(
            help="Path to save results", file_okay=True, dir_okay=False
        ),
    ],
    unmeth_file: Annotated[
        Path | None,
        typer.Option(
            help="Unmethylated control file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    meth_file: Annotated[
        Path | None,
        typer.Option(
            help="Methylated control file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    fasta_file: Annotated[
        Path | None,
        typer.Option(
            help="Reference FASTA file (required by compute_model_prob)",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    mtase: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str), help="Methyltransferase(s): A, CG, GC"
        ),
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str),
            help="Chromosomes to load, or a BED file of regions",
        ),
    ] = None,
    wrap: Annotated[
        bool, typer.Option(help="Wrap relative to center")
    ] = False,
    ignore_strand: Annotated[
        bool,
        typer.Option(
            help="Ignore recorded strand for mtase context filtering"
        ),
    ] = False,
    colidx: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(int),
            help="ModKit column indices (6, or 8 with alignment start/end)",
        ),
    ] = None,
    max_nmol: Annotated[
        int | None, typer.Option(help="Max molecules per region")
    ] = None,
    balance_strand: Annotated[
        bool,
        typer.Option(help="Sample +/- strand molecules 50:50 (max-nmol)"),
    ] = False,
    seed: Annotated[
        int | None, typer.Option(help="Seed for molecule sampling")
    ] = None,
    tmp_dir: Annotated[
        Path | None,
        typer.Option(
            help="Scratch directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    chunk_size: Annotated[
        int, typer.Option(help="Rows read per streamed chunk")
    ] = 1000000,
    max_cached_regions: Annotated[
        int, typer.Option(help="Max regions' raw data kept in memory")
    ] = 1,
    nworker: Annotated[
        int, typer.Option(help="Threads for parallel file reading")
    ] = 1,
    min_gap: Annotated[
        int,
        typer.Option(
            min=0, help="Min deletion (bp) recorded as a read gap; 0 = off"
        ),
    ] = 20,
    gaps_bed: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str),
            help="BED12 file(s) of read blocks (bamtobed -bed12 -splitD)",
        ),
    ] = None,
    min_overlap: Annotated[
        int,
        typer.Option(min=0, help="Min read overlap (bp) with a region"),
    ] = 0,
):
    """
    Load raw methylation footprinting data into a MethPrintExperiment.
    """
    is_bed = regions is not None and len(regions) == 1
    if is_bed and Path(regions[0]).is_file():
        regions = regions[0]
    exp = catella.load_raw(
        chromsize=chromsize,
        test_file=test_file,
        unmeth_file=unmeth_file,
        meth_file=meth_file,
        fasta_file=fasta_file,
        mtase=mtase,
        regions=regions,
        wrap=wrap,
        ignore_strand=ignore_strand,
        colidx=colidx,
        max_nmol=max_nmol,
        balance_strand=balance_strand,
        seed=seed,
        chunk_size=chunk_size,
        tmp_dir=tmp_dir,
        max_cached_regions=max_cached_regions,
        nworker=nworker,
        min_gap=min_gap or None,
        gaps_bed=gaps_bed,
        min_overlap=min_overlap,
    )
    exp.save(out_file)


@app.command(name="filter_dropout")
def filter_dropout(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    which: Annotated[
        str | None,
        typer.Option(
            help="test, meth, or unmeth; default evaluates every source"
        ),
    ] = None,
    channels: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str),
            help="Channels as in qc, e.g. A,GC or GCH,GCG",
        ),
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str), help="Subset of regions to evaluate"
        ),
    ] = None,
    thres_min: Annotated[
        float, typer.Option(help="Min allowed no-signal fraction")
    ] = 0.0,
    thres_max: Annotated[
        float, typer.Option(help="Max allowed no-signal fraction")
    ] = 1.0,
    unmapped_strand: Annotated[
        str, typer.Option(help="union, drop, +, or -")
    ] = "union",
    method: Annotated[
        str, typer.Option(help="separate or aggregate")
    ] = "separate",
    mask_name: Annotated[
        str, typer.Option(help="Key for the stored mask")
    ] = "dropout_mask",
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite source_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    span: Annotated[
        str,
        typer.Option(help="Molecule stretch counted: full, calls or read"),
    ] = "full",
):
    """
    Flag molecules with poor coverage at methylatable positions.
    """
    exp = MethPrintExperiment.load(source_file)
    catella.filter_dropout(
        exp=exp,
        which=which,
        channels=channels,
        regions=regions,
        thres_min=thres_min,
        thres_max=thres_max,
        unmapped_strand=unmapped_strand,
        method=method,
        mask_name=mask_name,
        span=span,
    )
    exp.save(out_file)


@app.command(name="qc")
def qc(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    channels: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str),
            help="Channels as in compute_model_prob, e.g. GCH,GCG or GC",
        ),
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str), help="Subset of regions to check"
        ),
    ] = None,
    percentiles: Annotated[
        str,
        typer.Option(
            callback=csv_parser(float),
            help="Dropout percentiles, e.g. 25,50,75",
        ),
    ] = "50",
    mask_name: Annotated[
        str | None,
        typer.Option(help="Dropout mask from filter_dropout"),
    ] = None,
    batch_size: Annotated[
        int,
        typer.Option(min=1, help="Molecules per batch when counting calls"),
    ] = 20000,
    tsv_file: Annotated[
        Path | None,
        typer.Option(
            help="Optional path to also save the table as TSV",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite source_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    overwrite: Annotated[
        bool,
        typer.Option(
            help="Replace an existing qc table"
        ),
    ] = False,
    span: Annotated[
        str,
        typer.Option(help="Molecule stretch counted: full, calls or read"),
    ] = "full",
):
    """
    Summarize data quality per region and channel (model-free).
    """
    exp = MethPrintExperiment.load(source_file)
    df = catella.qc(
        exp=exp,
        channels=channels,
        regions=regions,
        percentiles=percentiles,
        mask_name=mask_name,
        batch_size=batch_size,
        overwrite=overwrite,
        span=span,
    )
    typer.echo(df.to_csv(sep="\t", index=False), nl=False)
    if tsv_file is not None:
        df.to_csv(tsv_file, sep="\t", index=False)
    exp.save(out_file)


@app.command(name="plot_dropout_ecdf")
def plot_dropout_ecdf(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    regions: Annotated[
        str,
        typer.Argument(
            callback=csv_parser(str),
            help="Region(s) to plot (comma-separated)",
        ),
    ],
    which: Annotated[
        str | None, typer.Option(help="test, meth, or unmeth")
    ] = None,
    channels: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str),
            help="Channels as in qc, e.g. A,GC or GCH,GCG",
        ),
    ] = None,
    unmapped_strand: Annotated[
        str, typer.Option(help="union, drop, +, or -")
    ] = "union",
    pooled: Annotated[
        bool, typer.Option(help="Plot the pooled 'all' channel")
    ] = True,
    split_by: Annotated[
        str | None,
        typer.Option(help="Split into panels by region, source, or channel"),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to save the figure", file_okay=True, dir_okay=False
        ),
    ] = None,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True,
    span: Annotated[
        str,
        typer.Option(help="Molecule stretch counted: full, calls or read"),
    ] = "full",
):
    """
    Plot the empirical CDF of dropout rate per group.
    """
    exp = MethPrintExperiment.load(source_file)
    catella.plot_dropout_ecdf(
        exp=exp,
        regions=regions,
        which=which,
        channels=channels,
        unmapped_strand=unmapped_strand,
        pooled=pooled,
        split_by=split_by,
        out_file=out_file,
        show=show,
        span=span,
    )


@app.command(name="compute_empirical_prob")
def compute_empirical_prob(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    lnuc: Annotated[
        int, typer.Option(help="Nucleosome footprint / window size (bp)")
    ] = 147,
    prob_name: Annotated[
        str, typer.Option(help="Key for storing methylation probabilities")
    ] = "meth_prob",
    clip_low: Annotated[
        float | None, typer.Option(help="Lower clip percentile")
    ] = 0.1,
    clip_high: Annotated[
        float | None, typer.Option(help="Upper clip percentile")
    ] = 99.9,
    norm_by_strand: Annotated[
        bool | None, typer.Option(help="Normalize by strand")
    ] = False,
    fill_edge: Annotated[
        float, typer.Option(help="Probability for the unfilled trailing edge")
    ] = float("nan"),
    batch_size: Annotated[
        int, typer.Option(min=1, help="Molecules processed per batch")
    ] = 20000,
    percentile_sample_size: Annotated[
        int, typer.Option(help="Sample size for percentile estimation")
    ] = 100000,
    seed: Annotated[
        int | None, typer.Option(help="Seed for percentile subsampling")
    ] = None,
    mask_name: Annotated[
        str | None,
        typer.Option(help="Key of a prior filter_dropout mask to apply"),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite source_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Compute methylation probabilities via empirical normalization.
    """
    exp = MethPrintExperiment.load(source_file)
    catella.compute_empirical_prob(
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
    exp.save(out_file)


@app.command(name="compute_model_prob")
def compute_model_prob(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    prob_name: Annotated[
        str, typer.Option(help="Key for storing methylation probabilities")
    ] = "meth_prob",
    q_thres: Annotated[
        float | None,
        typer.Option(
            help="Binarize scores: 1 if at least this value, else 0. "
            "Omit to use the scores as they are."
        ),
    ] = None,
    channels: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str),
            help="Channel(s) to use as evidence: A, GCH, HCG, GCG, "
            "or merged GC (GCH+GCG) or CG (HCG+GCG) for in vitro data "
            "with a single C mtase (comma-separated). Omit to use all "
            "assayed channels, unmerged.",
        ),
    ] = None,
    eta: Annotated[
        str | None,
        typer.Option(
            callback=eta_parser,
            help="Log-odds correction for correlated sites. "
            "A number applies to every selected channel; a JSON object "
            "pins only the named channels, e.g. "
            '\'{"A": 0.9, "HCG": 0.7}\'. Omit to '
            "auto-estimate every channel (default).",
        ),
    ] = None,
    eta_max_lag: Annotated[
        int, typer.Option(help="Max lag (bp) for eta auto-estimation")
    ] = 10,
    store_rho: Annotated[
        bool | None,
        typer.Option(
            help="Store the lag-k autocorrelation rho(k) used in eta "
            "estimation"
        ),
    ] = False,
    nu: Annotated[
        float | None,
        typer.Option(
            help="Pseudo-count shrinkage strength; fitted by empirical "
            "Bayes if not given (controls only)"
        ),
    ] = None,
    rho_leak: Annotated[
        float | None,
        typer.Option(
            help="Leak fraction toward accessible-state rate; fitted "
            "if not given (controls only)"
        ),
    ] = None,
    min_rel_contrast: Annotated[
        float,
        typer.Option(
            help="Min relative contrast (kappa - tau) / kappa of a "
            "context after an EM fit to be informative"
        ),
    ] = 0.2,
    lnuc: Annotated[
        int, typer.Option(help="Nucleosome footprint / window size (bp)")
    ] = 147,
    lbin: Annotated[
        int, typer.Option(help="EM bin width (bp) (no-controls fit)")
    ] = 30,
    max_iters: Annotated[
        int, typer.Option(help="Max EM iterations (no-controls fit)")
    ] = 200,
    tol: Annotated[
        float, typer.Option(help="EM log-likelihood convergence tolerance")
    ] = 1e-8,
    fill_edge: Annotated[
        float, typer.Option(help="Probability for the unfilled trailing edge")
    ] = float("nan"),
    norm_by_strand: Annotated[
        bool | None, typer.Option(help="Normalize by strand")
    ] = False,
    span: Annotated[
        str,
        typer.Option(
            help="Molecule extent for reading uncalled sites: 'full' "
            "(whole fiber), 'read' (read extent) or 'calls' (first to "
            "last call)"
        ),
    ] = "full",
    batch_size: Annotated[
        int, typer.Option(min=1, help="Molecules processed per batch")
    ] = 20000,
    mask_name: Annotated[
        str | None,
        typer.Option(help="Key of a prior filter_dropout mask to apply"),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite source_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Compute methylation probabilities via a calibrated log-odds model.
    """
    exp = MethPrintExperiment.load(source_file)
    catella.compute_model_prob(
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
    exp.save(out_file)


@app.command()
def run(
    regions: Annotated[
        str, typer.Option(callback=csv_parser(str), help="Region names")
    ],
    nsim: Annotated[int, typer.Option(help="Simulations per molecule")],
    settings: Annotated[
        Path,
        typer.Option(
            help="Simulation configuration file directory",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    exp_file: Annotated[
        Path,
        typer.Option(
            help="Experiment HDF5 file directory",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    out_dir: Annotated[
        Path,
        typer.Option(
            help="Output directory",
            exists=False,
            file_okay=False,
            dir_okay=True,
            readable=True,
        ),
    ],
    dataset_name: Annotated[
        str, typer.Option(help="Dataset file name (no extension)")
    ] = "results",
    method: Annotated[
        str, typer.Option(help="Simulation engine: 'mcmc' or 'tm'")
    ] = "mcmc",
    out_types: Annotated[
        str | None,
        typer.Option(callback=csv_parser(str), help="energy,position..."),
    ] = "all",
    seed: Annotated[int | None, typer.Option(help="Random seed")] = None,
    mols: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(int),
            help="Molecule indices to simulate (all if omitted)",
        ),
    ] = None,
    nworker: Annotated[
        int | None, typer.Option(help="Number of processes")
    ] = 1,
    store_eseq: Annotated[
        bool | None,
        typer.Option(help="Store sequence-specific energy landscape"),
    ] = True,
    use_median_eseq_mu: Annotated[
        bool | None,
        typer.Option(help="Shift mu to median of sequence-specific energy"),
    ] = False,
    fill_nan: Annotated[
        str | None,
        typer.Option(
            callback=fill_nan_parser,
            help="'mean' or a probability in [0, 1]; fills molecules "
            "that are nan at some nucleosome start positions (error "
            "if omitted)",
        ),
    ] = None,
    verbose: Annotated[
        bool | None, typer.Option(help="Print progress")
    ] = True,
):
    """
    Execute a parallelized methylation simulation.
    """
    exp_data = MethPrintExperiment.load(exp_file)
    meth_prob = {
        region: exp_data.analysis[region]["meth_prob"]
        for region in exp_data.regions
    }
    return catella.run(
        regions=regions,
        nsim=nsim,
        settings=settings,
        meth_prob=meth_prob,
        out_dir=out_dir,
        dataset_name=dataset_name,
        method=method,
        out_types=out_types,
        seed=seed,
        mols=mols if mols is not None else slice(None),
        nworker=nworker,
        store_eseq=store_eseq,
        use_median_eseq_mu=use_median_eseq_mu,
        fill_nan=fill_nan,
        verbose=verbose,
    )


@app.command(name="compute_occup")
def compute_occup(
    dataset_file: Annotated[
        Path,
        typer.Argument(
            help="Simulation dataset HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    time: Annotated[
        int | None, typer.Option(help="Simulation time point")
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str), help="Regions (all if omitted)"
        ),
    ] = None,
    occup_name: Annotated[
        str, typer.Option(help="Key to store occupancy")
    ] = "occup",
    mean_nnuc_name: Annotated[
        str, typer.Option(help="Key to store mean nucleosome count")
    ] = "mean_nnuc",
    compute_var: Annotated[
        bool,
        typer.Option(help="Also compute exact variance (method='tm' only)"),
    ] = False,
    var_nnuc_name: Annotated[
        str, typer.Option(help="Key to store nucleosome count variance")
    ] = "var_nnuc",
    var_occup_name: Annotated[
        str, typer.Option(help="Key to store occupancy variance")
    ] = "var_occup",
    nworker: Annotated[
        int, typer.Option(help="Number of processes (method='tm' only)")
    ] = 1,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite dataset_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Compute the nucleosome occupancy and mean nucleosome count.
    """
    dataset = SimDataset.load(dataset_file)
    catella.compute_occup(
        dataset=dataset,
        time=time,
        regions=regions,
        occup_name=occup_name,
        mean_nnuc_name=mean_nnuc_name,
        compute_var=compute_var,
        var_nnuc_name=var_nnuc_name,
        var_occup_name=var_occup_name,
        nworker=nworker,
    )
    dataset.save(out_file)


@app.command(name="compute_config")
def compute_config(
    dataset_file: Annotated[
        Path,
        typer.Argument(
            help="Simulation dataset HDF5 file (method='tm')",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    method: Annotated[
        str, typer.Option(help="'best_match' or 'most_likely'")
    ] = "best_match",
    gamma: Annotated[
        float | None,
        typer.Option(help="best_match weight of occupied sites (default 1)"),
    ] = None,
    regions: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str), help="Regions (all if omitted)"
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(help="Key prefix to store the calls (default: method)"),
    ] = None,
    nworker: Annotated[int, typer.Option(help="Number of processes")] = 1,
    tsv_file: Annotated[
        Path | None,
        typer.Option(
            help="Write the calls as TSV: region, mol, starts "
            "(comma-joined; empty if none, NA if no data)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite dataset_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Call one nucleosome configuration per molecule.
    """
    dataset = SimDataset.load(dataset_file)
    catella.compute_config(
        dataset,
        method=method,
        gamma=gamma,
        regions=regions,
        name=name,
        nworker=nworker,
    )
    if tsv_file is not None:
        key = method if name is None else name
        rows = []
        for region in regions or dataset.regions:
            configs = catella.extract_config(
                dataset, region=region, name=key
            )
            for mol, starts in enumerate(configs):
                text = (
                    "NA"
                    if starts is None
                    else ",".join(str(int(s)) for s in starts)
                )
                rows.append({"region": region, "mol": mol, "starts": text})
        pd.DataFrame(rows, columns=["region", "mol", "starts"]).to_csv(
            tsv_file, sep="\t", index=False
        )
    dataset.save(out_file)


@app.command()
def downsample(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment or dataset HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    region: Annotated[str, typer.Argument(help="Region name")],
    key: Annotated[str, typer.Argument(help="Analysis key to downsample")],
    max_rows: Annotated[int, typer.Argument(help="Target number of rows")],
    kind: Annotated[
        SourceKind,
        typer.Option(
            help="Whether source_file holds a "
            "MethPrintExperiment or a SimDataset"
        ),
    ],
    how: Annotated[
        str, typer.Option(help="mean, sum, min, max, or stride")
    ] = "mean",
    batch_size: Annotated[
        int, typer.Option(min=1, help="Rows read per streamed chunk")
    ] = 20000,
    out_key: Annotated[
        str | None,
        typer.Option(
            help="Key to store the result (default: '<key>_downsampled')"
        ),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite source_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Downsample an analysis array and store the result under a new key.
    """
    obj = _load_source(source_file, kind)
    data = obj.analysis[region][key]
    result = catella.downsample(
        data=data, max_rows=max_rows, how=how, batch_size=batch_size
    )
    name = out_key if out_key is not None else f"{key}_downsampled"
    obj.analysis[region][name] = pd.DataFrame(result)
    obj.save(out_file)


@app.command(name="sort_by_linkage")
def sort_by_linkage(
    source_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment or dataset HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    kind: Annotated[
        SourceKind,
        typer.Option(
            help="Whether source_file holds a "
            "MethPrintExperiment or a SimDataset"
        ),
    ],
    regions: Annotated[
        str | None,
        typer.Option(
            callback=csv_parser(str), help="Region(s) to process"
        ),
    ] = None,
    data_name: Annotated[
        str | None, typer.Option(help="Analysis key to cluster")
    ] = None,
    config_name: Annotated[
        str | None,
        typer.Option(
            help="Name of compute_config calls to cluster by their "
            "footprints; 'dataset' kind only"
        ),
    ] = None,
    raw_which: Annotated[
        str | None,
        typer.Option(help="test, meth, or unmeth; 'experiment' kind only"),
    ] = None,
    mask_name: Annotated[
        str | None,
        typer.Option(
            help="Dropout mask key from "
            "filter_dropout to apply; requires "
            "raw_which, 'experiment' kind "
            "only"
        ),
    ] = None,
    store_link_mat: Annotated[
        bool, typer.Option(help="Persist the linkage matrix")
    ] = True,
    link_mat_name: Annotated[
        str | None, typer.Option(help="Key to store the linkage matrix")
    ] = None,
    metric: Annotated[str, typer.Option(help="Distance metric")] = "euclidean",
    method: Annotated[str, typer.Option(help="Linkage method")] = "ward",
    batch_size: Annotated[
        int, typer.Option(min=1, help="Rows processed per batch")
    ] = 20000,
    fill_nan: Annotated[
        str | None,
        typer.Option(
            callback=fill_nan_parser,
            help="'mean' or a number; how to "
            "handle nan values before "
            "clustering",
        ),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Output file (default: overwrite source_file)",
            file_okay=True,
            dir_okay=False,
        ),
    ] = None,
):
    """
    Cluster molecules by similarity and store the linkage matrix.

    Plots put their rows in its leaf order via --link-mat-name.
    """
    obj = _load_source(source_file, kind)
    if kind == SourceKind.experiment:
        catella.sort_by_linkage(
            exp=obj,
            regions=regions,
            data_name=data_name,
            config_name=config_name,
            raw_which=raw_which,
            mask_name=mask_name,
            store_link_mat=store_link_mat,
            link_mat_name=link_mat_name,
            metric=metric,
            method=method,
            batch_size=batch_size,
            fill_nan=fill_nan,
        )
    else:
        catella.sort_by_linkage(
            dataset=obj,
            regions=regions,
            data_name=data_name,
            config_name=config_name,
            store_link_mat=store_link_mat,
            link_mat_name=link_mat_name,
            metric=metric,
            method=method,
            batch_size=batch_size,
            fill_nan=fill_nan,
        )
    obj.save(out_file)


@app.command(name="plot_config")
def plot_config(
    regions: Annotated[
        str,
        typer.Argument(
            callback=csv_parser(str),
            help="Region(s) to plot (comma-separated)",
        ),
    ],
    dataset_file: Annotated[
        Path,
        typer.Argument(
            help="Simulation dataset HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    name: Annotated[
        str | None,
        typer.Option(
            help="Key prefix of stored calls (raw configurations if omitted)"
        ),
    ] = None,
    mols: Annotated[
        list[int] | None,
        typer.Option(help="Molecule index to display (repeatable)"),
    ] = None,
    run: Annotated[
        int | None, typer.Option(help="MCMC run or TM sample (default 0)")
    ] = None,
    time: Annotated[
        int | None,
        typer.Option(help="MCMC time point (default: final frame)"),
    ] = None,
    max_rows: Annotated[
        int | None, typer.Option(help="Show at most this many rows")
    ] = None,
    downsample_how: Annotated[
        str, typer.Option(help="'stride', 'mean' or 'max'")
    ] = "stride",
    xscale: Annotated[
        int,
        typer.Option(help="Spatial scale factor (power of 10) for the x-axis"),
    ] = 1000,
    cmap: Annotated[
        str | None, typer.Option(help="Matplotlib colormap name")
    ] = None,
    plot_mean: Annotated[
        bool,
        typer.Option(help="Plot the share of molecules with a nucleosome"),
    ] = False,
    plot_eseq: Annotated[
        bool, typer.Option(help="Plot the mean sequence-specific energy")
    ] = False,
    link_mat_name: Annotated[
        str | None,
        typer.Option(
            help="Key of a linkage matrix (from sort_by_linkage) to order "
            "the rows by and draw as a dendrogram"
        ),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to save the figure", file_okay=True, dir_okay=False
        ),
    ] = None,
    sharex: Annotated[
        bool,
        typer.Option(help="Put all panels on the longest region's scale"),
    ] = False,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True,
):
    """
    Plot one nucleosome configuration per molecule, for one or more
    regions.
    """
    if mols and len(regions) > 1:
        raise typer.BadParameter("--mols requires a single region")
    dataset = SimDataset.load(dataset_file)
    link_mat = (
        {r: dataset.analysis[r][link_mat_name] for r in regions}
        if link_mat_name is not None
        else None
    )
    catella.plot_config(
        dataset,
        regions=regions,
        name=name,
        mols=mols,
        run=run,
        time=time,
        max_rows=max_rows,
        downsample_how=downsample_how,
        xscale=xscale,
        cmap=cmap,
        plot_mean=plot_mean,
        plot_eseq=plot_eseq,
        link_mat=link_mat,
        out_file=out_file,
        sharex=sharex,
        show=show,
    )


@app.command(name="plot_occup")
def plot_occup(
    regions: Annotated[
        str,
        typer.Argument(
            callback=csv_parser(str),
            help="Region(s) to plot (comma-separated)",
        ),
    ],
    dataset_file: Annotated[
        Path,
        typer.Argument(
            help="Simulation dataset HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    time: Annotated[
        int | None,
        typer.Option(
            help="Time step to compute occupancy at, if not already present"
        ),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to save the figure", file_okay=True, dir_okay=False
        ),
    ] = None,
    occup_name: Annotated[
        str, typer.Option(help="Key of the occupancy data")
    ] = "occup",
    mols: Annotated[
        list[int] | None,
        typer.Option(help="Molecule index to display (repeatable)"),
    ] = None,
    xscale: Annotated[
        int,
        typer.Option(help="Spatial scale factor (power of 10) for the x-axis"),
    ] = 1000,
    cmap: Annotated[
        str | None, typer.Option(help="Matplotlib colormap name")
    ] = None,
    plot_mean: Annotated[
        bool, typer.Option(help="Plot the mean occupancy across molecules")
    ] = False,
    plot_eseq: Annotated[
        bool, typer.Option(help="Plot the sequence-specific binding energy")
    ] = False,
    link_mat_name: Annotated[
        str | None,
        typer.Option(
            help="Key of a linkage matrix (e.g. "
            "from sort_by_linkage) to draw as a "
            "dendrogram"
        ),
    ] = None,
    sharex: Annotated[
        bool,
        typer.Option(help="Put all panels on the longest region's scale"),
    ] = False,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True,
):
    """
    Visualize nucleosome occupancy for one or more regions.
    """
    if mols and len(regions) > 1:
        raise typer.BadParameter("--mols requires a single region")
    dataset = SimDataset.load(dataset_file)
    link_mat = (
        {
            r: dataset.analysis[r][link_mat_name].to_numpy()
            for r in regions
        }
        if link_mat_name is not None
        else None
    )
    catella.plot_occup(
        regions=regions,
        dataset=dataset,
        time=time,
        out_file=out_file,
        occup_name=occup_name,
        mols=mols,
        xscale=xscale,
        cmap=cmap,
        plot_mean=plot_mean,
        plot_eseq=plot_eseq,
        link_mat=link_mat,
        sharex=sharex,
        show=show,
    )


@app.command(name="plot_meth")
def plot_meth(
    exp_file: Annotated[
        Path,
        typer.Argument(
            help="Experiment HDF5 file",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    regions: Annotated[
        str,
        typer.Argument(
            callback=csv_parser(str),
            help="Region(s) to plot (comma-separated)",
        ),
    ],
    key: Annotated[
        str | None,
        typer.Option(
            help="Analysis key to plot. Mutually exclusive with --raw-which"
        ),
    ] = None,
    value: Annotated[
        str,
        typer.Option(
            help="'prob' to plot values as given, or 'energy' to plot "
            "logit(values) centered at zero (needs --key)"
        ),
    ] = "prob",
    raw_which: Annotated[
        str | None,
        typer.Option(
            help="Raw data source ('test', "
            "'unmeth', or 'meth') to plot "
            "directly via to_dense(). Mutually "
            "exclusive with --key"
        ),
    ] = None,
    mask_name: Annotated[
        str | None,
        typer.Option(
            help="Dropout mask name to apply "
            "when plotting via --raw-which (see "
            "to_dense)"
        ),
    ] = None,
    emax: Annotated[
        float | None,
        typer.Option(
            help="With --value energy: clamp energy to [-emax, emax] and "
            "use it as the default color scale range"
        ),
    ] = None,
    max_rows: Annotated[
        int | None,
        typer.Option(
            help="Downsample to at most this many rows before plotting"
        ),
    ] = None,
    downsample_how: Annotated[
        str, typer.Option(help="How to collapse rows when --max-rows is given")
    ] = "mean",
    vmin: Annotated[
        float | None,
        typer.Option(
            help="Lower color scale bound; with --value energy, combined "
            "with --vmax into a symmetric range centered at zero"
        ),
    ] = None,
    vmax: Annotated[
        float | None, typer.Option(help="Upper color scale bound")
    ] = None,
    cmap: Annotated[
        str | None, typer.Option(help="Matplotlib colormap name")
    ] = None,
    cbar_label: Annotated[
        str | None,
        typer.Option(help="Colorbar label; default depends on --value"),
    ] = None,
    out_file: Annotated[
        Path | None,
        typer.Option(
            help="Path to save the figure", file_okay=True, dir_okay=False
        ),
    ] = None,
    link_mat_name: Annotated[
        str | None,
        typer.Option(
            help="Key of a linkage matrix (e.g. "
            "from sort_by_linkage) to draw as a "
            "dendrogram"
        ),
    ] = None,
    xscale: Annotated[
        int,
        typer.Option(help="Spatial scale factor (power of 10) for the x-axis"),
    ] = 1000,
    sharex: Annotated[
        bool,
        typer.Option(help="Put all panels on the widest region's scale"),
    ] = False,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True,
):
    """
    Plot a methylation heatmap, as probability or energy, for one or
    more regions.
    """
    if (key is None) == (raw_which is None):
        raise typer.BadParameter("specify exactly one of --key or --raw-which")
    if value == "energy" and raw_which is not None:
        raise typer.BadParameter("--value energy requires --key")
    exp = MethPrintExperiment.load(exp_file)
    link_mat = (
        {r: exp.analysis[r][link_mat_name].to_numpy() for r in regions}
        if link_mat_name is not None
        else None
    )
    catella.plot_meth(
        value=value,
        exp=exp,
        regions=regions,
        raw_which=raw_which,
        data_name=key,
        mask_name=mask_name if raw_which is not None else None,
        emax=emax,
        max_rows=max_rows,
        downsample_how=downsample_how,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        cbar_label=cbar_label,
        out_file=out_file,
        link_mat=link_mat,
        xscale=xscale,
        sharex=sharex,
        show=show,
    )


# Click adapter of `app`, used by sphinx-click for the CLI reference docs.
click_app = typer.main.get_command(app)


if __name__ == "__main__":
    app()
