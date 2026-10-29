# run_bench.py

import itertools
import json
import multiprocessing
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import catella
import h5py
import numpy as np
import pandas as pd
import synthgen
import typer
from catella.simulation.config import SimSettings
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

# Benchmark configuration; fixed before any test dataset is scored
NBP = 5000
TRIM = 150
NMOL = 100
QUICK_NMOL = 20
TUNE_SEED = 0
TEST_SEEDS = (1, 2, 3)
BASE_PARAMS = {
    "p_linker": 0.7,
    "p_nuc": 0.02,
    "signal_sep": 2.0,
    "linker_mean": 40.0,
}
CONDITIONS = {
    "default": {},
    "linker20": {"linker_mean": 20.0},
    "plinker04": {"p_linker": 0.4},
    "signal15": {"signal_sep": 1.5},
    "varfoot": {"lnuc_range": (120, 160)},
    "overconf": {"report_sep": 4.0},
}
GRIDS = {
    "catella": {"llink": [0, 10, 30, 50]},
    "ft": {"nucleosome_length": [65, 85], "combined_length": [90, 110]},
    "fiberhmm": {"nuc_min_size": [70, 100], "split_min_llr": [2.0, 6.0]},
    "fiberhmm_pre": {
        "nuc_min_size": [70, 100],
        "split_min_llr": [2.0, 6.0],
    },
}


def dataset_specs(quick: bool = False) -> list[dict]:
    """
    List the benchmark datasets to generate.

    Parameters
    ----------
    quick : bool, default False
        If True, only the tuning dataset and one test dataset of the
        "default" condition, for smoke testing.

    Returns
    -------
    list of dict
        One dict per dataset with keys "cond" ("tune" for the tuning
        dataset, otherwise a key of `CONDITIONS`), "seed", and "params"
        (keyword arguments for `synthgen.make_dataset`).
    """
    conds = {"default": {}} if quick else CONDITIONS
    seeds = TEST_SEEDS[:1] if quick else TEST_SEEDS
    nmol = QUICK_NMOL if quick else NMOL
    common = {"nbp": NBP, "nmol": nmol, "controls": True}
    common["double_strand"] = True
    specs = [
        {
            "cond": "tune",
            "seed": TUNE_SEED,
            "params": {**common, **BASE_PARAMS, "seed": TUNE_SEED},
        }
    ]
    for cond, override in conds.items():
        for seed in seeds:
            params = {**common, **BASE_PARAMS, **override, "seed": seed}
            specs.append({"cond": cond, "seed": seed, "params": params})
    return specs


def dataset_dir(out: str | Path, cond: str, seed: int) -> Path:
    """
    Directory of one benchmark dataset.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    cond : str
        Condition name, or "tune" for the tuning dataset.
    seed : int
        Dataset seed.

    Returns
    -------
    Path
        `<out>/data/<cond>/s<seed>`.
    """
    return Path(out) / "data" / cond / f"s{seed}"


def run_in_env(env: str, args: list[str], cwd: str | Path | None = None):
    """
    Run a command inside a conda environment.

    Uses the executable in `$CONDA_EXE`, else `$MAMBA_EXE`, else
    `conda` on the `PATH`.

    Parameters
    ----------
    env : str
        Conda environment name.
    args : list of str
        Command and its arguments.
    cwd : str or Path, optional
        Working directory.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with captured text output.
    """
    conda = os.environ.get("CONDA_EXE") or os.environ.get("MAMBA_EXE")
    conda = conda or "conda"
    proc = subprocess.run(
        [conda, "run", "-n", env, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed in env {env!r}: {' '.join(args)}; "
            f"{proc.stderr.strip()}"
        )
    return proc


def sam_to_bam(sam: str | Path, env: str) -> Path:
    """
    Convert a SAM file to a sorted, indexed BAM file with samtools.

    Parameters
    ----------
    sam : str or Path
        Input SAM file.
    env : str
        Conda environment providing samtools.

    Returns
    -------
    Path
        The BAM file written next to `sam`.
    """
    sam = Path(sam)
    bam = sam.with_suffix(".bam")
    run_in_env(env, ["samtools", "sort", "-o", str(bam), str(sam)])
    run_in_env(env, ["samtools", "index", str(bam)])
    return bam


def _run_tasks(func, args_list, nworker):
    if nworker <= 1:
        return [func(*args) for args in args_list]
    # spawn on all platforms; fork is unsafe with HDF5/OpenMP state
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nworker, mp_context=ctx) as pool:
        futures = [pool.submit(func, *args) for args in args_list]
        return [f.result() for f in futures]


def _make_one(out, spec, env, convert):
    ddir = dataset_dir(out, spec["cond"], spec["seed"])
    paths = synthgen.make_dataset(ddir, **spec["params"])
    if convert:
        for key in ("sam", "meth_sam", "unmeth_sam"):
            sam_to_bam(paths[key], env)
    return str(ddir)


def make_data(
    out: str | Path,
    *,
    quick: bool = False,
    env: str = "fiberseq",
    convert: bool = True,
    nworker: int = 1,
) -> list[dict]:
    """
    Generate all benchmark datasets.

    Also writes `<out>/datasets.json`, recording each dataset's
    condition, seed, parameters, and directory, along with the tuning
    grids and the git commit of the code that generated them.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    quick : bool, default False
        Generate the reduced dataset list; see `dataset_specs`.
    env : str, default "fiberseq"
        Conda environment providing samtools.
    convert : bool, default True
        If True, convert every SAM file to a sorted, indexed BAM file.
    nworker : int, default 1
        Number of processes generating datasets in parallel.

    Returns
    -------
    list of dict
        The dataset specs, each with an added "dir" key.
    """
    out = Path(out)
    specs = dataset_specs(quick)
    dirs = _run_tasks(
        _make_one, [(out, spec, env, convert) for spec in specs], nworker
    )
    for spec, ddir in zip(specs, dirs):
        spec["dir"] = ddir
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).parent,
        capture_output=True,
        text=True,
        check=False,
    )
    record = {
        "git_commit": git.stdout.strip(),
        "quick": quick,
        "grids": GRIDS,
        "datasets": specs,
    }
    with open(out / "datasets.json", "w") as fh:
        json.dump(record, fh, indent=2)
    return specs


@dataclass
class Calls:
    """
    Nucleosome calls of one tool on one dataset.

    Attributes
    ----------
    mol_ids : np.ndarray of str
        Molecule ids, one per row.
    occ : np.ndarray of float
        (nmol, nbp) per-bp nucleosome occupancy: 0/1 for interval calls,
        a probability for catella.
    count : np.ndarray of float
        Number of called (or expected, for catella) nucleosomes per
        molecule.
    """

    mol_ids: np.ndarray
    occ: np.ndarray
    count: np.ndarray


def intervals_to_calls(
    intervals: pd.DataFrame, mol_ids: list[str], nbp: int
) -> Calls:
    """
    Convert nucleosome intervals to calls.

    Parameters
    ----------
    intervals : pandas.DataFrame
        Columns `mol_id`, `start`, `end` (0-based, half-open). Intervals
        are clipped to `[0, nbp)`.
    mol_ids : list of str
        Molecules to include, in row order; molecules without intervals
        get no nucleosomes.
    nbp : int
        Molecule length in bp.

    Returns
    -------
    Calls
        Binary occupancy and interval counts.
    """
    row = {m: i for i, m in enumerate(mol_ids)}
    occ = np.zeros((len(mol_ids), nbp), dtype=np.float32)
    count = np.zeros(len(mol_ids))
    cols = intervals[["mol_id", "start", "end"]]
    for mol_id, start, end in cols.itertuples(index=False):
        if mol_id not in row:
            continue
        occ[row[mol_id], max(start, 0) : min(end, nbp)] = 1
        count[row[mol_id]] += 1
    return Calls(np.asarray(mol_ids), occ, count)


def _int_list(field):
    text = str(field).strip(",")
    if text in ("", ".", "nan"):
        return []
    return [int(v) for v in text.split(",")]


def parse_ft_extract(path: str | Path) -> pd.DataFrame:
    """
    Read nucleosome intervals from an `ft extract --all` table.

    Parameters
    ----------
    path : str or Path
        Output of `ft extract --all`, with columns `fiber`,
        `ref_nuc_starts`, and `ref_nuc_lengths`.

    Returns
    -------
    pandas.DataFrame
        Columns `mol_id`, `start`, `end` in reference coordinates.
    """
    table = pd.read_csv(path, sep="\t")
    cols = table[["fiber", "ref_nuc_starts", "ref_nuc_lengths"]]
    rows = []
    for fiber, starts, lengths in cols.itertuples(index=False):
        for start, length in zip(_int_list(starts), _int_list(lengths)):
            rows.append((fiber, start, start + length))
    return pd.DataFrame(rows, columns=["mol_id", "start", "end"])


def read_truth(ddir: str | Path) -> pd.DataFrame:
    """
    Read a dataset's ground-truth nucleosome positions.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.

    Returns
    -------
    pandas.DataFrame
        Columns `mol_id`, `start`, `end`.
    """
    return pd.read_csv(
        Path(ddir) / "synth.truth.bed",
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "mol_id"],
    )[["mol_id", "start", "end"]]


def _test_mol_ids(ddir):
    tsv = pd.read_csv(Path(ddir) / "synth.tsv", sep="\t", usecols=["read_id"])
    return list(dict.fromkeys(tsv["read_id"]))


def _ft_calls(bam, ddir, env):
    table = Path(bam).with_suffix(".tsv.gz")
    run_in_env(env, ["ft", "extract", str(bam), "--all", str(table)])
    return intervals_to_calls(
        parse_ft_extract(table), _test_mol_ids(ddir), NBP
    )


def run_ft(
    ddir: str | Path,
    workdir: str | Path,
    *,
    env: str = "fiberseq",
    nucleosome_length: int | None = None,
    combined_length: int | None = None,
) -> Calls:
    """
    Call nucleosomes with `ft add-nucleosomes`.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory (with `synth.bam`).
    workdir : str or Path
        Directory for intermediate files.
    env : str, default "fiberseq"
        Conda environment providing fibertools.
    nucleosome_length, combined_length : int, optional
        Values for `--nucleosome-length` and
        `--combined-nucleosome-length`; tool defaults if None.

    Returns
    -------
    Calls
    """
    out = Path(workdir) / "ft.bam"
    args = ["ft", "add-nucleosomes", "--ml", "128"]
    if nucleosome_length is not None:
        args += ["--nucleosome-length", str(nucleosome_length)]
    if combined_length is not None:
        args += ["--combined-nucleosome-length", str(combined_length)]
    run_in_env(env, [*args, str(Path(ddir) / "synth.bam"), str(out)])
    return _ft_calls(out, ddir, env)


def train_fiberhmm(
    ddir: str | Path,
    *,
    env: str = "fiberseq",
    iterations: int = 10,
) -> Path:
    """
    Train a FiberHMM model on a dataset, using its controls.

    Emission tables come from the methylated (accessible) and
    unmethylated (inaccessible) controls; the model is then trained on
    the test reads. No ground truth is used. The model is cached in the
    dataset directory and reused on later calls.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory (with `synth.bam`, `synth.meth.bam`,
        `synth.unmeth.bam`).
    env : str, default "fiberseq"
        Conda environment providing FiberHMM.
    iterations : int, default 10
        Training iterations (random initializations).

    Returns
    -------
    Path
        The trained model file.
    """
    ddir = Path(ddir)
    model = ddir / "fiberhmm_model" / "best-model.json"
    if model.exists():
        return model
    probs = ddir / "fiberhmm_probs"
    run_in_env(
        env,
        [
            "fiberhmm-probs",
            "-a",
            str(ddir / "synth.meth.bam"),
            "-u",
            str(ddir / "synth.unmeth.bam"),
            "-o",
            str(probs),
            "--mode",
            "pacbio-fiber",
            "-p",
            "128",
            "-k",
            "3",
        ],
    )
    tables = probs / "tables"
    run_in_env(
        env,
        [
            "fiberhmm-train",
            "-i",
            str(ddir / "synth.bam"),
            "-p",
            str(next(tables.glob("*_accessible_*_k3.tsv"))),
            str(next(tables.glob("*_inaccessible_*_k3.tsv"))),
            "-o",
            str(model.parent),
            "-k",
            "3",
            "-c",
            str(iterations),
            "--prob-threshold",
            "128",
        ],
    )
    return model


def run_fiberhmm(
    ddir: str | Path,
    workdir: str | Path,
    *,
    env: str = "fiberseq",
    retrain: bool = True,
    nuc_min_size: int | None = None,
    split_min_llr: float | None = None,
    iterations: int = 10,
) -> Calls:
    """
    Call nucleosomes with `fiberhmm-call`.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    env : str, default "fiberseq"
        Conda environment providing FiberHMM and fibertools.
    retrain : bool, default True
        If True, use a model trained on this dataset (see
        `train_fiberhmm`); otherwise the pretrained Hia5 PacBio model.
    nuc_min_size : int, optional
        Value for `--nuc-min-size`; tool default if None.
    split_min_llr : float, optional
        Value for `--split-min-llr`; tool default if None.
    iterations : int, default 10
        Training iterations if `retrain`.

    Returns
    -------
    Calls
    """
    out = Path(workdir) / "fiberhmm.bam"
    args = ["fiberhmm-call", "-i", str(Path(ddir) / "synth.bam")]
    args += ["-o", str(out), "--prob-threshold", "128", "-c", "2"]
    if retrain:
        model = train_fiberhmm(ddir, env=env, iterations=iterations)
        args += ["-m", str(model)]
    else:
        args += ["--enzyme", "hia5", "--seq", "pacbio"]
    if nuc_min_size is not None:
        args += ["--nuc-min-size", str(nuc_min_size)]
    if split_min_llr is not None:
        args += ["--split-min-llr", str(split_min_llr)]
    run_in_env(env, args)
    return _ft_calls(out, ddir, env)


def run_catella(
    ddir: str | Path,
    workdir: str | Path,
    *,
    llink: int = 50,
    elink: float = 25.0,
    mu: float = 0.0,
    emax: float | None = None,
    seed: int = 1,
) -> Calls:
    """
    Compute nucleosome occupancy with catella's exact (TM) solver.

    Uses both controls for the methylation probability calibration and
    reads both strands of each molecule (`ignore_strand=True`). By
    default, `emax` is estimated from the data.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    llink : int, default 50
        `SimSettings.llink`.
    elink : float, default 25.0
        `SimSettings.elink`.
    mu : float, default 0.0
        `SimSettings.mu`.
    emax : float, optional
        `SimSettings.emax`. If None (default), the 99th percentile of
        abs(logit(meth_prob)) from `catella.estimate_start_temp`.
    seed : int, default 1
        Seed passed to `catella.run` and `catella.estimate_start_temp`.

    Returns
    -------
    Calls
        Occupancy probabilities and expected nucleosome counts.
    """
    ddir = Path(ddir)
    exp = catella.load_raw(
        chromsize=ddir / "synth.size",
        test_file=ddir / "synth.tsv",
        unmeth_file=ddir / "synth.unmeth.tsv",
        meth_file=ddir / "synth.meth.tsv",
        fasta_file=ddir / "synth.fa",
        mtase="A",
        ignore_strand=True,
        tmp_dir=workdir,
    )
    exp = catella.compute_model_prob(exp=exp)
    chrom = exp.chroms[0]
    if emax is None:
        emax = catella.estimate_start_temp(exp, percentile=99.0, seed=seed)
    settings = SimSettings(
        nucbp=147, llink=llink, elink=elink, mu=mu, end_temp=1.0, emax=emax
    )
    sim_dir = Path(tempfile.mkdtemp(dir=workdir)) / "sim"
    sim = catella.run(
        chroms=chrom,
        nsim=0,
        settings=settings,
        meth_prob={chrom: exp.analysis[chrom]["meth_prob"]},
        out_dir=sim_dir,
        method="tm",
        seed=seed,
        verbose=False,
    )
    catella.analyze(sim)
    occ = np.asarray(sim.analysis[chrom]["occup"][:], dtype=np.float32)
    count = np.asarray(sim.analysis[chrom]["mean_nnuc"][:]).ravel()
    mol_ids = np.asarray(exp.raw[chrom].test_mol_id)
    return Calls(mol_ids, occ, count)


def save_calls(calls: Calls, path: str | Path) -> Path:
    """
    Save calls to an HDF5 file.

    The file holds datasets `mol_ids` (bytes), `occ` (float16,
    gzip-compressed), and `count`.

    Parameters
    ----------
    calls : Calls
        Calls to save.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("mol_ids", data=calls.mol_ids.astype("S"))
        fh.create_dataset(
            "occ", data=calls.occ.astype(np.float16), compression="gzip"
        )
        fh.create_dataset("count", data=calls.count)
    return path


def mcc(truth: np.ndarray, pred: np.ndarray) -> float:
    """
    Matthews correlation coefficient of two boolean arrays.

    Parameters
    ----------
    truth, pred : np.ndarray of bool
        True and predicted labels, same shape.

    Returns
    -------
    float
        MCC in [-1, 1]; 0 if either array is constant.
    """
    tp = float(np.sum(truth & pred))
    tn = float(np.sum(~truth & ~pred))
    fp = float(np.sum(~truth & pred))
    fn = float(np.sum(truth & ~pred))
    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return (tp * tn - fp * fn) / denom if denom else 0.0


def score_calls(
    calls: Calls,
    truth: pd.DataFrame,
    *,
    nbp: int = NBP,
    trim: int = TRIM,
) -> pd.DataFrame:
    """
    Score nucleosome calls against the ground truth, per molecule.

    Parameters
    ----------
    calls : Calls
        Calls to score.
    truth : pandas.DataFrame
        Ground truth with columns `mol_id`, `start`, `end`.
    nbp : int, default `NBP`
        Molecule length in bp.
    trim : int, default `TRIM`
        Bases excluded at each molecule end when computing the MCC.

    Returns
    -------
    pandas.DataFrame
        Columns `mol_id`; `mcc`, the MCC of occupancy >= 0.5 against
        the true nucleosome coverage over `[trim, nbp - trim)`; and
        `count_err`, the called (or expected) minus the true number of
        nucleosomes on the whole molecule.
    """
    true = intervals_to_calls(truth, list(calls.mol_ids), nbp)
    inner = slice(trim, nbp - trim)
    rows = []
    for i, mol_id in enumerate(calls.mol_ids):
        rows.append(
            (
                mol_id,
                mcc(true.occ[i, inner] >= 0.5, calls.occ[i, inner] >= 0.5),
                calls.count[i] - true.count[i],
            )
        )
    return pd.DataFrame(rows, columns=["mol_id", "mcc", "count_err"])


def load_calls(path: str | Path) -> Calls:
    """
    Load calls saved by `save_calls`.

    Parameters
    ----------
    path : str or Path
        File written by `save_calls`.

    Returns
    -------
    Calls
    """
    with h5py.File(path, "r") as fh:
        return Calls(
            fh["mol_ids"][:].astype(str),
            fh["occ"][:].astype(np.float32),
            fh["count"][:],
        )


TOOLS = {
    "catella": (run_catella, {}),
    "ft": (run_ft, {}),
    "fiberhmm": (run_fiberhmm, {"retrain": True}),
    "fiberhmm_pre": (run_fiberhmm, {"retrain": False}),
}


def run_tool(
    tool: str,
    ddir: str | Path,
    workdir: str | Path,
    params: dict,
    env: str = "fiberseq",
) -> Calls:
    """
    Run one tool on one dataset.

    Parameters
    ----------
    tool : str
        A key of `TOOLS`.
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    params : dict
        Keyword arguments for the tool's runner (e.g., a grid point).
    env : str, default "fiberseq"
        Conda environment for the external tools.

    Returns
    -------
    Calls
    """
    runner, fixed = TOOLS[tool]
    if tool == "catella":
        return runner(ddir, workdir, **fixed, **params)
    return runner(ddir, workdir, env=env, **fixed, **params)


def grid_combos(grid: dict[str, list]) -> list[dict]:
    """
    All parameter combinations of a grid.

    Parameters
    ----------
    grid : dict of str to list
        Parameter name -> candidate values.

    Returns
    -------
    list of dict
        One dict per combination, in `itertools.product` order.
    """
    keys = list(grid)
    return [
        dict(zip(keys, values))
        for values in itertools.product(*(grid[k] for k in keys))
    ]


def _read_record(out):
    with open(Path(out) / "datasets.json") as fh:
        return json.load(fh)


def _tune_one(out, ddir, tool, params, env):
    work = Path(out) / "work"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as wd:
        calls = run_tool(tool, ddir, wd, params, env)
    scores = score_calls(calls, read_truth(ddir))
    return {
        "tool": tool,
        "params": json.dumps(params),
        "mcc": scores["mcc"].mean(),
        "count_err": scores["count_err"].mean(),
    }


def tune(
    out: str | Path, *, env: str = "fiberseq", nworker: int = 1
) -> dict[str, dict]:
    """
    Pick each tool's parameters on the tuning dataset.

    Every combination in `GRIDS` is run on the tuning dataset only and
    scored by mean MCC; the best combination per tool is kept. Writes
    `<out>/tune_scores.csv` (all combinations) and
    `<out>/tuned_params.json` (the chosen ones).

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `make_data`).
    env : str, default "fiberseq"
        Conda environment for the external tools.
    nworker : int, default 1
        Number of grid combinations run in parallel. If greater than 1,
        the tuning dataset's FiberHMM model is trained first so that
        parallel runs share it.

    Returns
    -------
    dict of str to dict
        Tool -> chosen parameters.
    """
    out = Path(out)
    datasets = _read_record(out)["datasets"]
    spec = next(s for s in datasets if s["cond"] == "tune")
    ddir = dataset_dir(out, spec["cond"], spec["seed"])
    if nworker > 1 and "fiberhmm" in TOOLS:
        train_fiberhmm(ddir, env=env)
    tasks = [
        (out, ddir, tool, params, env)
        for tool in TOOLS
        for params in grid_combos(GRIDS[tool])
    ]
    table = pd.DataFrame(_run_tasks(_tune_one, tasks, nworker))
    table.to_csv(out / "tune_scores.csv", index=False)
    best = table.loc[table.groupby("tool", sort=False)["mcc"].idxmax()]
    tuned = {r.tool: json.loads(r.params) for r in best.itertuples()}
    with open(out / "tuned_params.json", "w") as fh:
        json.dump(tuned, fh, indent=2)
    return tuned


def _markdown_table(table):
    cols = [str(c) for c in table.columns]
    lines = ["| " + " | ".join(cols) + " |"]
    lines.append("|" + "---|" * len(cols))
    for row in table.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def summarize(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Summarize per-molecule results by condition.

    Molecule scores are first averaged within each dataset; datasets
    are the unit of replication.

    Parameters
    ----------
    results : pandas.DataFrame
        Columns `tool`, `variant`, `cond`, `seed`, `mcc`, `count_err`
        (one row per molecule).

    Returns
    -------
    per_tool : pandas.DataFrame
        Per (`tool`, `variant`, `cond`): mean and SD over datasets of
        `mcc` and `count_err`, and the number of datasets `n`.
    paired : pandas.DataFrame
        Per (`variant`, `cond`, `other` tool): mean and SD over datasets
        of the MCC difference catella minus `other`, and `wins`, the
        number of datasets where catella has the higher MCC, out of `n`.
    """
    keys = ["tool", "variant", "cond", "seed"]
    ds = results.groupby(keys, sort=False)[["mcc", "count_err"]].mean()
    ds = ds.reset_index()
    per_tool = (
        ds.groupby(["tool", "variant", "cond"], sort=False)
        .agg(
            mcc_mean=("mcc", "mean"),
            mcc_sd=("mcc", "std"),
            count_err_mean=("count_err", "mean"),
            count_err_sd=("count_err", "std"),
            n=("mcc", "size"),
        )
        .reset_index()
    )
    wide = ds.pivot_table(
        index=["variant", "cond", "seed"], columns="tool", values="mcc"
    )
    rows = []
    if "catella" in wide.columns:
        others = [t for t in wide.columns if t != "catella"]
        for (variant, cond), grp in wide.groupby(level=[0, 1], sort=False):
            for other in others:
                diff = (grp["catella"] - grp[other]).dropna()
                rows.append(
                    {
                        "variant": variant,
                        "cond": cond,
                        "other": other,
                        "diff_mean": diff.mean(),
                        "diff_sd": diff.std(),
                        "wins": int((diff > 0).sum()),
                        "n": len(diff),
                    }
                )
    return per_tool, pd.DataFrame(rows)


def unit_result_path(out: str | Path, tool: str, cond: str, seed: int) -> Path:
    """
    Result file of one evaluation unit (one tool on one dataset).

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    tool : str
        A key of `TOOLS`.
    cond : str
        Condition name.
    seed : int
        Dataset seed.

    Returns
    -------
    Path
        `<out>/results/<tool>/<cond>/s<seed>.csv`.
    """
    return Path(out) / "results" / tool / cond / f"s{seed}.csv"


def evaluate_unit(
    out: str | Path,
    spec: dict,
    tool: str,
    tuned: dict,
    env: str = "fiberseq",
    overwrite: bool = False,
) -> Path:
    """
    Run one tool on one test dataset, with tuned and default parameters.

    Saves the calls of each variant to
    `<out>/calls/<tool>_<variant>/<cond>/s<seed>.h5` and the
    per-molecule scores of both variants to `unit_result_path`.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    spec : dict
        Dataset spec from `datasets.json` (keys "cond" and "seed"); the
        dataset directory is found from `out` with `dataset_dir`.
    tool : str
        A key of `TOOLS`.
    tuned : dict
        Tuned parameters of `tool`.
    env : str, default "fiberseq"
        Conda environment for the external tools.
    overwrite : bool, default False
        If False and the result file exists, skip the unit.

    Returns
    -------
    Path
        The result file.
    """
    out = Path(out)
    cond, seed = spec["cond"], spec["seed"]
    path = unit_result_path(out, tool, cond, seed)
    if path.exists() and not overwrite:
        return path
    ddir = dataset_dir(out, cond, seed)
    truth = read_truth(ddir)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    frames = []
    for variant, params in (("tuned", tuned), ("default", {})):
        with tempfile.TemporaryDirectory(dir=work) as wd:
            calls = run_tool(tool, ddir, wd, params, env)
        save_calls(
            calls, out / "calls" / f"{tool}_{variant}" / cond / f"s{seed}.h5"
        )
        scores = score_calls(calls, truth)
        scores.insert(0, "seed", seed)
        scores.insert(0, "cond", cond)
        scores.insert(0, "variant", variant)
        scores.insert(0, "tool", tool)
        frames.append(scores)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)
    return path


def _check_choices(name, values, allowed):
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {name} {unknown}; expected values in {sorted(allowed)}"
        )


def evaluate(
    out: str | Path,
    *,
    env: str = "fiberseq",
    nworker: int = 1,
    tools: list[str] | None = None,
    conds: list[str] | None = None,
    seeds: list[int] | None = None,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Run tools on test datasets and summarize all results so far.

    Each (test dataset, tool) pair is one unit (see `evaluate_unit`);
    units run in parallel and existing unit results are skipped unless
    `overwrite`. Afterward, `write_summary` combines every unit result
    present, including those from earlier or separate runs.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `tune`).
    env : str, default "fiberseq"
        Conda environment for the external tools.
    nworker : int, default 1
        Number of units run in parallel.
    tools : list of str, optional
        Restrict to these tools (keys of `TOOLS`); all if None.
    conds : list of str, optional
        Restrict to these conditions; all if None.
    seeds : list of int, optional
        Restrict to these dataset seeds; all if None.
    overwrite : bool, default False
        If True, rerun units whose results already exist.

    Returns
    -------
    pandas.DataFrame
        Per-molecule results of all units present, as returned by
        `write_summary`.
    """
    out = Path(out)
    with open(out / "tuned_params.json") as fh:
        tuned = json.load(fh)
    specs = [s for s in _read_record(out)["datasets"] if s["cond"] != "tune"]
    _check_choices("tools", tools or [], set(TOOLS))
    _check_choices("conds", conds or [], {s["cond"] for s in specs})
    _check_choices("seeds", seeds or [], {s["seed"] for s in specs})
    units = [
        (out, spec, tool, tuned[tool], env, overwrite)
        for spec in specs
        if (conds is None or spec["cond"] in conds)
        and (seeds is None or spec["seed"] in seeds)
        for tool in TOOLS
        if tools is None or tool in tools
    ]
    _run_tasks(evaluate_unit, units, nworker)
    return write_summary(out)


def write_summary(out: str | Path) -> pd.DataFrame:
    """
    Combine all unit results and write the summary tables.

    Writes `<out>/results.csv` (per-molecule results of every unit
    present), `<out>/summary_per_tool.csv`, `<out>/summary_paired.csv`,
    and `<out>/summary.md`.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.

    Returns
    -------
    pandas.DataFrame
        Per-molecule results.
    """
    out = Path(out)
    files = sorted((out / "results").glob("*/*/*.csv"))
    if not files:
        raise ValueError(f"no unit results found in {out / 'results'}")
    results = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    results.to_csv(out / "results.csv", index=False)
    tuned_path = out / "tuned_params.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    per_tool, paired = summarize(results)
    per_tool.to_csv(out / "summary_per_tool.csv", index=False)
    paired.to_csv(out / "summary_paired.csv", index=False)
    text = [
        "# Benchmark summary",
        "",
        "## Per tool (mean and SD over datasets)",
        "",
        _markdown_table(per_tool),
        "",
        "## catella minus other tool (MCC)",
        "",
        _markdown_table(paired),
        "",
        "## Tuned parameters",
        "",
        "```",
        json.dumps(tuned, indent=2),
        "```",
        "",
    ]
    (out / "summary.md").write_text("\n".join(text))
    return results


TOOL_STYLES = {
    "truth": ("truth", "#52514e"),
    "catella": ("catella", "#2a78d6"),
    "ft": ("ft", "#eb6834"),
    "fiberhmm_pre": ("FiberHMM (pretrained)", "#1baf7a"),
    "fiberhmm": ("FiberHMM (retrained)", "#eda100"),
}


def _mod_qual_rows(ddir, mol_ids, nbp):
    tsv = pd.read_csv(Path(ddir) / "synth.tsv", sep="\t")
    rows = np.full((len(mol_ids), nbp), np.nan)
    index = {m: i for i, m in enumerate(mol_ids)}
    for mol_id, grp in tsv.groupby("read_id", sort=False):
        if mol_id in index:
            pos = grp["ref_position"].to_numpy()
            rows[index[mol_id], pos] = grp["mod_qual"].to_numpy()
    return rows


def _draw_track(ax, label, occ, color):
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=7)
    ax.set_yticks([])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if occ is None:
        return
    ax.axvspan(0, TRIM, color="#eeeeee", zorder=0)
    ax.axvspan(NBP - TRIM, NBP, color="#eeeeee", zorder=0)
    x = np.arange(len(occ))
    ax.fill_between(x, occ, step="mid", color=color, alpha=0.85, lw=0)
    ax.set_ylim(0, 1)


def plot_occupancy(
    out: str | Path,
    cond: str,
    seed: int,
    *,
    variant: str = "tuned",
    tools: list[str] | None = None,
    nmol: int | None = None,
    per_page: int = 4,
    path: str | Path | None = None,
) -> Path:
    """
    Plot methylation probability and nucleosome occupancy per molecule.

    For each molecule of one test dataset, draws the per-site
    `mod_qual` of the test data, the true nucleosome coverage, and the
    occupancy of each tool, as stacked tracks along the molecule. The
    `TRIM` bases at each end, excluded from scoring, are shaded. Tools
    without saved calls for this dataset are left out.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `evaluate`).
    cond : str
        Condition name.
    seed : int
        Dataset seed.
    variant : {"tuned", "default"}, default "tuned"
        Parameter variant whose calls are plotted.
    tools : list of str, optional
        Tools to plot, in track order; all of `TOOLS`, in `TOOL_STYLES`
        order, if None.
    nmol : int, optional
        Plot only the first `nmol` molecules of the dataset; all if None.
    per_page : int, default 4
        Molecules per A4 page.
    path : str or Path, optional
        Output PDF path; defaults to
        `<out>/plots/occupancy_<cond>_s<seed>_<variant>.pdf`.

    Returns
    -------
    Path
        The PDF written.
    """
    if nmol is not None and nmol <= 0:
        raise ValueError(f"nmol must be positive, got {nmol}")
    if per_page <= 0:
        raise ValueError(f"per_page must be positive, got {per_page}")
    out = Path(out)
    ddir = dataset_dir(out, cond, seed)
    if tools is None:
        tools = [t for t in TOOL_STYLES if t in TOOLS]
    _check_choices("tools", tools, set(TOOLS))
    calls = {}
    for tool in tools:
        h5 = out / "calls" / f"{tool}_{variant}" / cond / f"s{seed}.h5"
        if h5.exists():
            calls[tool] = load_calls(h5)
    if not calls:
        raise ValueError(
            f"no saved calls for cond {cond!r}, seed {seed}, "
            f"variant {variant!r}; run evaluate first"
        )
    mol_ids = _test_mol_ids(ddir)[:nmol]
    true = intervals_to_calls(read_truth(ddir), mol_ids, NBP)
    qual = _mod_qual_rows(ddir, mol_ids, NBP)
    rows = {
        tool: {m: i for i, m in enumerate(c.mol_ids)}
        for tool, c in calls.items()
    }
    ntrack = len(calls) + 2
    if path is None:
        path = out / "plots" / f"occupancy_{cond}_s{seed}_{variant}.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for start in range(0, len(mol_ids), per_page):
            page = mol_ids[start : start + per_page]
            fig = Figure(figsize=(8.27, 11.69))
            axes = fig.subplots(per_page * ntrack, 1, sharex=True)
            for ax in axes[len(page) * ntrack :]:
                ax.set_visible(False)
            for k, mol_id in enumerate(page):
                i = start + k
                top = axes[k * ntrack]
                image = top.imshow(
                    qual[i][None, :],
                    aspect="auto",
                    cmap="Purples",
                    vmin=0,
                    vmax=1,
                    interpolation="nearest",
                    extent=(-0.5, NBP - 0.5, 0, 1),
                )
                top.set_title(mol_id, fontsize=8, loc="left")
                _draw_track(top, "mod_qual", None, None)
                label, color = TOOL_STYLES["truth"]
                _draw_track(axes[k * ntrack + 1], label, true.occ[i], color)
                for j, tool in enumerate(calls, 2):
                    label, color = TOOL_STYLES[tool]
                    occ = None
                    if mol_id in rows[tool]:
                        occ = calls[tool].occ[rows[tool][mol_id]]
                    _draw_track(axes[k * ntrack + j], label, occ, color)
            axes[len(page) * ntrack - 1].set_xlabel("position (bp)")
            fig.suptitle(
                f"Methylation probability and nucleosome occupancy: {cond}, "
                f"seed {seed}, {variant} parameters\n"
                "(shaded ends excluded from scoring)",
                fontsize=9,
            )
            fig.tight_layout(rect=(0, 0.03, 1, 0.97), h_pad=0.2)
            cax = fig.add_axes((0.35, 0.012, 0.3, 0.008))
            bar = fig.colorbar(image, cax=cax, orientation="horizontal")
            bar.set_label(
                "mod_qual = P(methylated | signal); white = no site",
                fontsize=7,
            )
            bar.ax.tick_params(labelsize=6)
            pdf.savefig(fig)
    return path


app = typer.Typer(help="Benchmark catella against fibertools and FiberHMM")


@app.command(name="make_data")
def make_data_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    quick: Annotated[
        bool, typer.Option(help="Generate a small smoke-test set")
    ] = False,
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = "fiberseq",
    nworker: Annotated[
        int, typer.Option(help="Number of parallel processes")
    ] = 1,
):
    """Generate the benchmark datasets."""
    specs = make_data(out, quick=quick, env=env, nworker=nworker)
    typer.echo(f"wrote {len(specs)} datasets to {out}")


@app.command(name="tune")
def tune_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = "fiberseq",
    nworker: Annotated[
        int,
        typer.Option(
            help="Number of parallel processes (FiberHMM uses 2 threads)"
        ),
    ] = 1,
):
    """Pick each tool's parameters on the tuning dataset."""
    tuned = tune(out, env=env, nworker=nworker)
    typer.echo(json.dumps(tuned, indent=2))


@app.command(name="evaluate")
def evaluate_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = "fiberseq",
    nworker: Annotated[
        int,
        typer.Option(
            help="Number of parallel processes (FiberHMM uses 2 threads)"
        ),
    ] = 1,
    tool: Annotated[
        list[str] | None, typer.Option(help="Only this tool (repeatable)")
    ] = None,
    cond: Annotated[
        list[str] | None,
        typer.Option(help="Only this condition (repeatable)"),
    ] = None,
    seed: Annotated[
        list[int] | None, typer.Option(help="Only this seed (repeatable)")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Rerun units with existing results")
    ] = False,
):
    """Run tools on the test datasets and write the summary."""
    evaluate(
        out,
        env=env,
        nworker=nworker,
        tools=tool,
        conds=cond,
        seeds=seed,
        overwrite=overwrite,
    )
    typer.echo(f"wrote {Path(out) / 'summary.md'}")


@app.command(name="summarize")
def summarize_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
):
    """Combine all unit results and write the summary."""
    write_summary(out)
    typer.echo(f"wrote {Path(out) / 'summary.md'}")


@app.command(name="plot_occupancy")
def plot_occupancy_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    cond: Annotated[str, typer.Option(help="Condition name")],
    seed: Annotated[int, typer.Option(help="Dataset seed")],
    variant: Annotated[
        str, typer.Option(help="Parameter variant: tuned or default")
    ] = "tuned",
    tool: Annotated[
        list[str] | None, typer.Option(help="Only this tool (repeatable)")
    ] = None,
    nmol: Annotated[
        int | None, typer.Option(help="Plot only the first NMOL molecules")
    ] = None,
    per_page: Annotated[
        int, typer.Option(help="Molecules per page")
    ] = 4,
):
    """Plot methylation probability and occupancy per molecule."""
    path = plot_occupancy(
        out,
        cond,
        seed,
        variant=variant,
        tools=tool,
        nmol=nmol,
        per_page=per_page,
    )
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
