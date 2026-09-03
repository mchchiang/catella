# cli.py

import enum
import json
import catella
import typer
import pandas as pd
from pathlib import Path
from typing import Annotated, Type, TypeVar, Callable, Optional
from catella.experiment.methdata import MethPrintExperiment
from catella.simulation.results import SimDataset

app = typer.Typer(help="Nucleosome Positioning Monte Carlo Simulation Suite")


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
T = TypeVar("T")
def csv_parser(
        target_type: Type[T]) -> Callable[[Optional[str]],
                                          Optional[list[T]]]:
    def parser(value: Optional[str]) -> Optional[list[T]]:
        if value is None: return None
        if not value: return []
        return [target_type(item.strip()) for item in value.split(",")]
    return parser

def fill_nan_parser(value: Optional[str]):
    if value is None: return None
    if value == "mean": return value
    try:
        return float(value)
    except ValueError:
        raise typer.BadParameter("fill_nan must be 'mean' or a number")

def eta_parser(value: Optional[str]):
    if value is None: return None
    try:
        return float(value)
    except ValueError:
        pass
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        raise typer.BadParameter(
            "eta must be a number, or a JSON object of channel name "
            "to number, e.g. '{\"M6A\": 0.9, \"HCG\": 0.7}'")

@app.command(name="preprocess_empirical")
def preprocess_empirical(
    chromsize: Annotated[Path, typer.Argument(help="Chromosome sizes file",
                                              exists=True, file_okay=True,
                                              dir_okay=False, readable=True)],
    test_file: Annotated[Path, typer.Argument(help="Primary methylation file",
                                              exists=True, file_okay=True,
                                              dir_okay=False, readable=True)],
    out_file: Annotated[Path, typer.Argument(help="Path to save results",
                                             file_okay=True, dir_okay=False)],
    unmeth_file: Annotated[
        Optional[Path], typer.Option(help="Unmethylated control file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    meth_file: Annotated[
        Optional[Path], typer.Option(help="Methylated control file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    fasta_file: Annotated[
        Optional[Path], typer.Option(help="Reference FASTA file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    mtase: Annotated[
        Optional[str],
        typer.Option(callback=csv_parser(str),
                    help="Methyltransferase(s): A, CG, GC")] = None,
    binsize: Annotated[int, typer.Option(help="Window size (bp)")] = 147,
    wrap: Annotated[
        bool, typer.Option(help="Wrap relative to center")] = False,
    colidx: Annotated[
        Optional[str],
        typer.Option(callback=csv_parser(int), help="ModKit column indices")
    ] = None,
    max_nmol: Annotated[
        Optional[int], typer.Option(help="Max molecules per chromosome")
    ] = None,
    seed: Annotated[
        Optional[int], typer.Option(help="Seed for molecule sampling")
    ] = None,
    clip_low: Annotated[
        Optional[float], typer.Option(help="Lower clip percentile")] = 0.1,
    clip_high: Annotated[
        Optional[float], typer.Option(help="Upper clip percentile")] = 99.9,
    norm_by_strand: Annotated[
        Optional[bool], typer.Option(help="Normalize by strand")] = False,
    batch_size: Annotated[
        int, typer.Option(help="Molecules processed per batch")] = 20000,
    percentile_sample_size: Annotated[
        int, typer.Option(help="Sample size for percentile estimation")
    ] = 100000,
    tmp_dir: Annotated[
        Optional[Path], typer.Option(help="Scratch directory",
                                     exists=True, file_okay=False,
                                     dir_okay=True)] = None,
    chunk_size: Annotated[
        int, typer.Option(help="Rows read per streamed chunk")] = 1000000,
    max_cached_chroms: Annotated[
        int, typer.Option(help="Max chromosomes' raw data kept in "
                          "memory")] = 1):
    """
    Preprocess raw methylation data using empirical (percentile/
    control-based) normalization.
    """
    return catella.preprocess_empirical(
        chromsize=chromsize, test_file=test_file, out_file=out_file,
        unmeth_file=unmeth_file, meth_file=meth_file, fasta_file=fasta_file,
        mtase=mtase, binsize=binsize, wrap=wrap, colidx=colidx,
        max_nmol=max_nmol, seed=seed, clip_low=clip_low,
        clip_high=clip_high, norm_by_strand=norm_by_strand,
        batch_size=batch_size,
        percentile_sample_size=percentile_sample_size, tmp_dir=tmp_dir,
        chunk_size=chunk_size, max_cached_chroms=max_cached_chroms)


@app.command(name="preprocess_model")
def preprocess_model(
    chromsize: Annotated[Path, typer.Argument(help="Chromosome sizes file",
                                              exists=True, file_okay=True,
                                              dir_okay=False, readable=True)],
    test_file: Annotated[Path, typer.Argument(help="Primary methylation file",
                                              exists=True, file_okay=True,
                                              dir_okay=False, readable=True)],
    fasta_file: Annotated[Path, typer.Argument(help="Reference FASTA file",
                                               exists=True, file_okay=True,
                                               dir_okay=False,
                                               readable=True)],
    out_file: Annotated[Path, typer.Argument(help="Path to save results",
                                             file_okay=True, dir_okay=False)],
    unmeth_file: Annotated[
        Optional[Path], typer.Option(help="Unmethylated control file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    meth_file: Annotated[
        Optional[Path], typer.Option(help="Methylated control file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    mtase: Annotated[
        Optional[str],
        typer.Option(callback=csv_parser(str),
                    help="Methyltransferase(s): A, CG, GC")] = None,
    wrap: Annotated[
        bool, typer.Option(help="Wrap relative to center")] = False,
    colidx: Annotated[
        Optional[str],
        typer.Option(callback=csv_parser(int), help="ModKit column indices")
    ] = None,
    max_nmol: Annotated[
        Optional[int], typer.Option(help="Max molecules per chromosome")
    ] = None,
    seed: Annotated[
        Optional[int], typer.Option(help="Seed for molecule sampling")
    ] = None,
    pi0: Annotated[
        float, typer.Option(help="Prior probability a site is "
                            "methylated")] = 0.5,
    eta: Annotated[
        Optional[str],
        typer.Option(callback=eta_parser,
                    help="Log-odds correction for correlated sites. "
                    "A number applies to every channel; a JSON object "
                    "pins only the named channels, e.g. "
                    '\'{"M6A": 0.9, "HCG": 0.7}\'. Omit to '
                    "auto-estimate every channel (default).")] = None,
    eta_max_lag: Annotated[
        int, typer.Option(help="Max lag (bp) for eta "
                          "auto-estimation")] = 10,
    nu: Annotated[
        float, typer.Option(help="Pseudo-count shrinkage strength")] = 10.0,
    rho_leak: Annotated[
        float, typer.Option(help="Leak fraction toward accessible-state "
                            "rate")] = 0.1,
    min_gap: Annotated[
        float, typer.Option(help="Min accessible/protected rate gap to "
                            "be informative")] = 0.05,
    l_nuc: Annotated[
        int, typer.Option(help="Nucleosome footprint / window size "
                          "(bp)")] = 147,
    n_min: Annotated[
        int, typer.Option(help="Min context-eligible sites per window "
                          "(no-controls fit)")] = 10,
    iters: Annotated[
        int, typer.Option(help="Max EM iterations (no-controls "
                          "fit)")] = 200,
    init_prot: Annotated[
        float, typer.Option(help="Initial protected-state rate "
                            "(no-controls fit)")] = 0.05,
    init_acc: Annotated[
        float, typer.Option(help="Initial accessible-state rate "
                            "(no-controls fit)")] = 0.95,
    tol: Annotated[
        float, typer.Option(help="EM log-likelihood convergence "
                            "tolerance")] = 1e-8,
    fill_edge: Annotated[
        float, typer.Option(help="Probability for the unfilled "
                            "trailing edge")] = float("nan"),
    norm_by_strand: Annotated[
        Optional[bool], typer.Option(help="Normalize by strand")] = False,
    batch_size: Annotated[
        int, typer.Option(help="Molecules processed per batch")] = 20000,
    tmp_dir: Annotated[
        Optional[Path], typer.Option(help="Scratch directory",
                                     exists=True, file_okay=False,
                                     dir_okay=True)] = None,
    chunk_size: Annotated[
        int, typer.Option(help="Rows read per streamed chunk")] = 1000000,
    max_cached_chroms: Annotated[
        int, typer.Option(help="Max chromosomes' raw data kept in "
                          "memory")] = 1):
    """
    Preprocess raw methylation data using a calibrated Bayesian
    log-odds model.
    """
    return catella.preprocess_model(
        chromsize=chromsize, test_file=test_file, fasta_file=fasta_file,
        out_file=out_file, unmeth_file=unmeth_file, meth_file=meth_file,
        mtase=mtase, wrap=wrap, colidx=colidx, max_nmol=max_nmol,
        seed=seed, pi0=pi0, eta=eta, eta_max_lag=eta_max_lag, nu=nu,
        rho_leak=rho_leak, min_gap=min_gap, l_nuc=l_nuc, n_min=n_min,
        iters=iters, init_prot=init_prot, init_acc=init_acc, tol=tol,
        fill_edge=fill_edge, norm_by_strand=norm_by_strand,
        batch_size=batch_size, tmp_dir=tmp_dir, chunk_size=chunk_size,
        max_cached_chroms=max_cached_chroms)


@app.command()
def run(
    chroms: Annotated[
        str, typer.Option(callback=csv_parser(str),
                          help="Chromosome names")],
    nsim: Annotated[int, typer.Option(help="Simulations per molecule")],
    settings: Annotated[
        Path, typer.Option(help="Simulation configuration file directory",
                           exists=True, file_okay=True, dir_okay=False,
                           readable=True)],
    exp_file: Annotated[
        Path, typer.Option(help="Experiment HDF5 file directory",
                           exists=True, file_okay=True, dir_okay=False,
                           readable=True)],
    out_dir: Annotated[Path, typer.Option(help="Output directory",
                                          exists=False, file_okay=False,
                                          dir_okay=True, readable=True)],
    dataset_name: Annotated[
        str, typer.Option(help="Dataset file name (no extension)")
    ] = "results",
    out_types: Annotated[
        Optional[str], typer.Option(callback=csv_parser(str),
                                    help="energy,position...")] = "all",
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    mols: Annotated[
        Optional[str],
        typer.Option(callback=csv_parser(int),
                    help="Molecule indices to simulate (all if omitted)")
    ] = None,
    nworker: Annotated[
        Optional[int], typer.Option(help="Number of processes")] = 1,
    store_eseq: Annotated[
        Optional[bool], typer.Option(help="Store sequence-specific energy "
                                     "landscape")] = True,
    use_zero_point_mu: Annotated[
        Optional[bool], typer.Option(help="Shift mu to median of sequence-"
                                     "specific energy")] = False,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Print progress")] = True):
    """
    Execute a parallelized methylation simulation.
    """
    exp_data = MethPrintExperiment.load(exp_file)
    meth_prob = {chrom: exp_data.analysis[chrom]["meth_prob"]
                 for chrom in exp_data.chroms}
    return catella.run(
        chroms=chroms,
        nsim=nsim,
        settings=settings,
        meth_prob=meth_prob,
        out_dir=out_dir,
        dataset_name=dataset_name,
        out_types=out_types,
        seed=seed,
        mols=mols if mols is not None else slice(None),
        nworker=nworker,
        store_eseq=store_eseq,
        use_zero_point_mu=use_zero_point_mu,
        verbose=verbose
    )


@app.command()
def analyze(
    dataset_file: Annotated[
        Path, typer.Argument(help="Simulation dataset HDF5 file",
                             exists=True, file_okay=True, dir_okay=False,
                             readable=True)],
    time: Annotated[
        Optional[int], typer.Option(help="Simulation time point")] = None,
    occup_name: Annotated[
        str, typer.Option(help="Key to store occupancy")] = "occup",
    mean_nnuc_name: Annotated[
        str, typer.Option(help="Key to store mean nucleosome count")
    ] = "mean_nnuc",
    out_file: Annotated[
        Optional[Path], typer.Option(help="Output file (default: "
                                     "overwrite dataset_file)",
                                     file_okay=True, dir_okay=False)
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Allow overwriting out_file if it is "
                           "already backed by H5Array analysis data. "
                           "Ignored (always allowed) when out_file is "
                           "not given")] = False):
    """
    Compute post-simulation occupancy and nucleosome count statistics.
    """
    dataset = SimDataset.load(dataset_file)
    catella.analyze(dataset=dataset, time=time, occup_name=occup_name,
                 mean_nnuc_name=mean_nnuc_name)
    if out_file is not None:
        dataset.save(out_file, overwrite=overwrite)
    else:
        dataset.save(overwrite=True)


@app.command()
def downsample(
    source_file: Annotated[
        Path, typer.Argument(help="Experiment or dataset HDF5 file",
                             exists=True, file_okay=True, dir_okay=False,
                             readable=True)],
    chrom: Annotated[str, typer.Argument(help="Chromosome name")],
    key: Annotated[str, typer.Argument(help="Analysis key to downsample")],
    max_rows: Annotated[int, typer.Argument(help="Target number of rows")],
    kind: Annotated[
        SourceKind, typer.Option(help="Whether source_file holds a "
                                 "MethPrintExperiment or a SimDataset")],
    how: Annotated[
        str, typer.Option(help="mean, sum, min, max, or stride")] = "mean",
    batch_size: Annotated[
        int, typer.Option(help="Rows read per streamed chunk")] = 20000,
    out_key: Annotated[
        Optional[str], typer.Option(help="Key to store the result "
                                    "(default: '<key>_downsampled')")
    ] = None,
    out_file: Annotated[
        Optional[Path], typer.Option(help="Output file (default: "
                                     "overwrite source_file)",
                                     file_okay=True, dir_okay=False)
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Allow overwriting out_file if it is "
                           "already backed by H5Array analysis data. "
                           "Ignored (always allowed) when out_file is "
                           "not given")] = False):
    """
    Downsample an analysis array and store the result under a new key.
    """
    obj = _load_source(source_file, kind)
    data = obj.analysis[chrom][key]
    result = catella.downsample(data=data, max_rows=max_rows, how=how,
                              batch_size=batch_size)
    name = out_key if out_key is not None else f"{key}_downsampled"
    obj.analysis[chrom][name] = pd.DataFrame(result)
    if out_file is not None:
        obj.save(out_file, overwrite=overwrite)
    else:
        obj.save(overwrite=True)


@app.command(name="sort_by_linkage")
def sort_by_linkage(
    source_file: Annotated[
        Path, typer.Argument(help="Experiment or dataset HDF5 file",
                             exists=True, file_okay=True, dir_okay=False,
                             readable=True)],
    kind: Annotated[
        SourceKind, typer.Option(help="Whether source_file holds a "
                                 "MethPrintExperiment or a SimDataset")],
    chroms: Annotated[
        Optional[list[str]], typer.Option(parser=csv_parser(str),
                                          help="Chromosome(s); 'dataset' "
                                          "kind only")] = None,
    data_name: Annotated[
        Optional[str], typer.Option(help="Analysis key to sort")] = None,
    raw_which: Annotated[
        Optional[str], typer.Option(help="test, meth, or unmeth; "
                                    "'experiment' kind only")] = None,
    sorted_name: Annotated[
        Optional[str], typer.Option(help="Key to store the sorted "
                                    "result")] = None,
    store_link_mat: Annotated[
        bool, typer.Option(help="Persist the linkage matrix")] = True,
    link_mat_name: Annotated[
        Optional[str], typer.Option(help="Key to store the linkage "
                                    "matrix")] = None,
    metric: Annotated[
        str, typer.Option(help="Distance metric")] = "euclidean",
    method: Annotated[str, typer.Option(help="Linkage method")] = "ward",
    batch_size: Annotated[
        int, typer.Option(help="Rows processed per batch")] = 20000,
    fill_nan: Annotated[
        Optional[str], typer.Option(callback=fill_nan_parser,
                                    help="'mean' or a number; how to "
                                    "handle nan values before "
                                    "clustering")] = None,
    out_file: Annotated[
        Optional[Path], typer.Option(help="Output file (default: "
                                     "overwrite source_file)",
                                     file_okay=True, dir_okay=False)
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Allow overwriting out_file if it is "
                           "already backed by H5Array analysis data. "
                           "Ignored (always allowed) when out_file is "
                           "not given")] = False):
    """
    Sort molecules by hierarchical-clustering similarity.
    """
    obj = _load_source(source_file, kind)
    if kind == SourceKind.experiment:
        catella.sort_by_linkage(exp=obj, data_name=data_name,
                              raw_which=raw_which, sorted_name=sorted_name,
                              store_link_mat=store_link_mat,
                              link_mat_name=link_mat_name, metric=metric,
                              method=method, batch_size=batch_size,
                              fill_nan=fill_nan)
    else:
        catella.sort_by_linkage(dataset=obj, chroms=chroms,
                              data_name=data_name, sorted_name=sorted_name,
                              store_link_mat=store_link_mat,
                              link_mat_name=link_mat_name, metric=metric,
                              method=method, batch_size=batch_size,
                              fill_nan=fill_nan)
    if out_file is not None:
        obj.save(out_file, overwrite=overwrite)
    else:
        obj.save(overwrite=True)


@app.command(name="plot_occup")
def plot_occup(
    chrom: Annotated[str, typer.Argument(help="Chromosome identifier")],
    dataset_file: Annotated[
        Path, typer.Argument(help="Simulation dataset HDF5 file",
                             exists=True, file_okay=True, dir_okay=False,
                             readable=True)],
    out_file: Annotated[
        Optional[Path], typer.Option(help="Path to save the figure",
                                     file_okay=True, dir_okay=False)] = None,
    occup_name: Annotated[
        str, typer.Option(help="Key of the occupancy data")] = "occup",
    plot_eseq: Annotated[
        bool, typer.Option(help="Plot the sequence-specific binding "
                           "energy")] = False,
    link_mat_name: Annotated[
        Optional[str], typer.Option(help="Key of a linkage matrix (e.g. "
                                    "from sort_by_linkage) to draw as a "
                                    "dendrogram")] = None,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True):
    """
    Visualize nucleosome occupancy for a chromosome.
    """
    dataset = SimDataset.load(dataset_file)
    link_mat = dataset.analysis[chrom][link_mat_name].to_numpy() \
        if link_mat_name is not None else None
    catella.plot_occup(chrom=chrom, dataset=dataset, out_file=out_file,
                     occup_name=occup_name, plot_eseq=plot_eseq,
                     link_mat=link_mat, show=show)


@app.command(name="plot_nuc_pos")
def plot_nuc_pos(
    chrom: Annotated[str, typer.Argument(help="Chromosome identifier")],
    mol: Annotated[int, typer.Argument(help="Molecule index")],
    run: Annotated[int, typer.Argument(help="Simulation run index")],
    dataset_file: Annotated[
        Path, typer.Argument(help="Simulation dataset HDF5 file",
                             exists=True, file_okay=True, dir_okay=False,
                             readable=True)],
    out_file: Annotated[
        Optional[Path], typer.Option(help="Path to save the figure",
                                     file_okay=True, dir_okay=False)] = None,
    plot_eseq: Annotated[
        bool, typer.Option(help="Plot the sequence-specific binding "
                           "energy")] = False,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True):
    """
    Plot nucleosome positions over time for one simulation run.
    """
    dataset = SimDataset.load(dataset_file)
    catella.plot_nuc_pos(chrom=chrom, mol=mol, run=run, dataset=dataset,
                       out_file=out_file, plot_eseq=plot_eseq, show=show)


@app.command(name="plot_energy")
def plot_energy(
    chrom: Annotated[str, typer.Argument(help="Chromosome identifier")],
    mol: Annotated[int, typer.Argument(help="Molecule index")],
    run: Annotated[int, typer.Argument(help="Simulation run index")],
    dataset_file: Annotated[
        Path, typer.Argument(help="Simulation dataset HDF5 file",
                             exists=True, file_okay=True, dir_okay=False,
                             readable=True)],
    out_file: Annotated[
        Optional[Path], typer.Option(help="Path to save the figure",
                                     file_okay=True, dir_okay=False)] = None,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True):
    """
    Plot total system energy over time for one simulation run.
    """
    dataset = SimDataset.load(dataset_file)
    catella.plot_energy(chrom=chrom, mol=mol, run=run, dataset=dataset,
                      out_file=out_file, show=show)


@app.command(name="plot_methmap")
def plot_methmap(
    exp_file: Annotated[
        Path, typer.Argument(help="Experiment HDF5 file", exists=True,
                             file_okay=True, dir_okay=False,
                             readable=True)],
    chrom: Annotated[str, typer.Argument(help="Chromosome identifier")],
    key: Annotated[
        str, typer.Option(help="Analysis key to plot")] = "meth_prob",
    vmin: Annotated[
        Optional[float], typer.Option(help="Lower color scale bound")
    ] = None,
    vmax: Annotated[
        Optional[float], typer.Option(help="Upper color scale bound")
    ] = None,
    out_file: Annotated[
        Optional[Path], typer.Option(help="Path to save the figure",
                                     file_okay=True, dir_okay=False)] = None,
    link_mat_name: Annotated[
        Optional[str], typer.Option(help="Key of a linkage matrix (e.g. "
                                    "from sort_by_linkage) to draw as a "
                                    "dendrogram")] = None,
    show: Annotated[bool, typer.Option(help="Display the figure")] = True):
    """
    Plot a methylation heatmap.
    """
    exp = MethPrintExperiment.load(exp_file)
    data = exp.analysis[chrom][key]
    link_mat = exp.analysis[chrom][link_mat_name].to_numpy() \
        if link_mat_name is not None else None
    catella.plot_methmap(data=data, vmin=vmin, vmax=vmax, out_file=out_file,
                       link_mat=link_mat, show=show)


if __name__ == "__main__":
    app()
