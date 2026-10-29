# run_bench.py

import itertools
import json
import os
import subprocess
import tempfile
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

# Benchmark configuration; fixed before any test dataset is scored
NBP = 5000
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
    "catella": {"llink": [0, 30], "mu": [-1.0, 1.0]},
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
    conda = os.environ.get("CONDA_EXE", "conda")
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


def make_data(
    out: str | Path,
    *,
    quick: bool = False,
    env: str = "fiberseq",
    convert: bool = True,
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

    Returns
    -------
    list of dict
        The dataset specs, each with an added "dir" key.
    """
    out = Path(out)
    specs = dataset_specs(quick)
    for spec in specs:
        ddir = dataset_dir(out, spec["cond"], spec["seed"])
        paths = synthgen.make_dataset(ddir, **spec["params"])
        if convert:
            for key in ("sam", "meth_sam", "unmeth_sam"):
                sam_to_bam(paths[key], env)
        spec["dir"] = str(ddir)
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
    trim: int = 150,
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
    trim : int, default 150
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


def tune(out: str | Path, *, env: str = "fiberseq") -> dict[str, dict]:
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

    Returns
    -------
    dict of str to dict
        Tool -> chosen parameters.
    """
    out = Path(out)
    datasets = _read_record(out)["datasets"]
    spec = next(s for s in datasets if s["cond"] == "tune")
    truth = read_truth(spec["dir"])
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for tool in TOOLS:
        for params in grid_combos(GRIDS[tool]):
            with tempfile.TemporaryDirectory(dir=work) as wd:
                calls = run_tool(tool, spec["dir"], wd, params, env)
            scores = score_calls(calls, truth)
            rows.append(
                {
                    "tool": tool,
                    "params": json.dumps(params),
                    "mcc": scores["mcc"].mean(),
                    "count_err": scores["count_err"].mean(),
                }
            )
    table = pd.DataFrame(rows)
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


def evaluate(out: str | Path, *, env: str = "fiberseq") -> pd.DataFrame:
    """
    Run all tools on all test datasets and summarize.

    Each tool is run with its tuned parameters (`tuned_params.json`)
    and with its default parameters. Writes calls to
    `<out>/calls/<tool>_<variant>/<cond>/s<seed>.h5`, per-molecule
    scores to `<out>/results.csv`, and the summary tables to
    `<out>/summary_per_tool.csv`, `<out>/summary_paired.csv`, and
    `<out>/summary.md`.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `tune`).
    env : str, default "fiberseq"
        Conda environment for the external tools.

    Returns
    -------
    pandas.DataFrame
        Per-molecule results.
    """
    out = Path(out)
    with open(out / "tuned_params.json") as fh:
        tuned = json.load(fh)
    datasets = _read_record(out)["datasets"]
    specs = [s for s in datasets if s["cond"] != "tune"]
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    frames = []
    for spec in specs:
        truth = read_truth(spec["dir"])
        for tool in TOOLS:
            for variant, params in (("tuned", tuned[tool]), ("default", {})):
                with tempfile.TemporaryDirectory(dir=work) as wd:
                    calls = run_tool(tool, spec["dir"], wd, params, env)
                name = f"{tool}_{variant}/{spec['cond']}/s{spec['seed']}.h5"
                save_calls(calls, out / "calls" / name)
                scores = score_calls(calls, truth)
                scores.insert(0, "seed", spec["seed"])
                scores.insert(0, "cond", spec["cond"])
                scores.insert(0, "variant", variant)
                scores.insert(0, "tool", tool)
                frames.append(scores)
    results = pd.concat(frames, ignore_index=True)
    results.to_csv(out / "results.csv", index=False)
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


app = typer.Typer(help="Benchmark catella against fibertools and FiberHMM")


@app.command(name="make-data")
def make_data_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    quick: Annotated[
        bool, typer.Option(help="Generate a small smoke-test set")
    ] = False,
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = "fiberseq",
):
    """Generate the benchmark datasets."""
    specs = make_data(out, quick=quick, env=env)
    typer.echo(f"wrote {len(specs)} datasets to {out}")


@app.command(name="tune")
def tune_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = "fiberseq",
):
    """Pick each tool's parameters on the tuning dataset."""
    tuned = tune(out, env=env)
    typer.echo(json.dumps(tuned, indent=2))


@app.command(name="evaluate")
def evaluate_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = "fiberseq",
):
    """Run all tools on the test datasets and write the summary."""
    evaluate(out, env=env)
    typer.echo(f"wrote {Path(out) / 'summary.md'}")


if __name__ == "__main__":
    app()
