# run_synth.py

import itertools
import json
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyfaidx
import synthgen
import typer
from callers import (
    ENV,
    TOOLS,
    Calls,
    _dataset_length,
    _test_mol_ids,
    intervals_to_calls,
    load_calls,
    run_tool,
    sam_to_bam,
    save_calls,
    tool_versions,
    train_fiberhmm,
)
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from plotting import (
    TOOL_STYLES,
    _mod_qual_rows,
    _occupancy_figure,
    _occupancy_per_page,
    _style_axes,
)

# Benchmark configuration; fixed before any test dataset is scored
NBP = 5000
TRIM = 150
NMOL = 100
QUICK_NMOL = 20
TUNE_SEED = 0
TEST_SEEDS = (1, 2, 3)
BASE_PARAMS = {
    "p_linker": 0.5,
    "p_nuc": 0.02,
    "signal_sep": 2.0,
    "linker_mean": 40.0,
    "linker_shape": 3.0,
}
CONDITIONS = {
    "default": {},
    "linker20": {"linker_mean": 20.0},
    "plinker035": {"p_linker": 0.35},
    "signal15": {"signal_sep": 1.5},
    "varfoot": {"lnuc_range": (120, 160)},
    "biased": {"report_bias": 3.0},
}
GRIDS = {
    "catella": {},
    "ft": {"nucleosome_length": [65, 85], "combined_length": [90, 110]},
    "fiberhmm": {"nuc_min_size": [70, 100], "split_min_llr": [2.0, 6.0]},
    "fiberhmm_pre": {
        "nuc_min_size": [70, 100],
        "split_min_llr": [2.0, 6.0],
    },
    "hiddenfoot": {"nuclen": [127, 137, 147, 157]},
}


def dataset_specs(quick: bool = False) -> list[dict]:
    """
    List the benchmark datasets to generate.

    Parameters
    ----------
    quick : bool, default False
        If True, only the tuning dataset and one test dataset of the
        "default" condition, each with fewer molecules, for a quick trial
        run.

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
    # Index now so that parallel tool runs only read the index
    pyfaidx.Faidx(str(ddir / "synth.fa")).close()
    if convert:
        for key in ("sam", "meth_sam", "unmeth_sam"):
            sam_to_bam(paths[key], env)
    return str(ddir)


def make_data(
    out: str | Path,
    *,
    quick: bool = False,
    env: str = ENV,
    convert: bool = True,
    nworker: int = 1,
) -> list[dict]:
    """
    Generate all benchmark datasets.

    Also writes `<out>/datasets.json`, recording each dataset's
    condition, seed, parameters, and directory, along with the tuning
    grids.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    quick : bool, default False
        Generate the reduced dataset list; see `dataset_specs`.
    env : str, default "catella"
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
    record = {
        "quick": quick,
        "grids": GRIDS,
        "datasets": specs,
    }
    with open(out / "datasets.json", "w") as fh:
        json.dump(record, fh, indent=2)
    return specs


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
        comment="#",
        names=["chrom", "start", "end", "mol_id"],
    )[["mol_id", "start", "end"]]


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


def brier(truth: np.ndarray, prob: np.ndarray) -> float:
    """
    Brier score of predicted probabilities against binary labels.

    Parameters
    ----------
    truth : np.ndarray of bool or {0, 1}
        True labels.
    prob : np.ndarray of float
        Predicted probabilities in [0, 1], same shape as `truth`.

    Returns
    -------
    float
        Mean squared difference between `prob` and `truth`; 0 is
        perfect, and lower is better.
    """
    diff = np.asarray(prob, dtype=float) - np.asarray(truth, dtype=float)
    return float(np.mean(diff**2))


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
        Bases excluded at each molecule end when computing the MCC and
        the Brier score.

    Returns
    -------
    pandas.DataFrame
        Columns `mol_id`; `mcc`, the MCC of occupancy >= 0.5 against
        the true nucleosome coverage over `[trim, nbp - trim)`;
        `count_err`, the called (or expected) minus the true number of
        nucleosomes on the whole molecule; and `brier`, the Brier score
        of the occupancy against the true coverage over the same bases
        as `mcc`.
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
                brier(true.occ[i, inner] >= 0.5, calls.occ[i, inner]),
            )
        )
    return pd.DataFrame(
        rows, columns=["mol_id", "mcc", "count_err", "brier"]
    )


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
    out: str | Path,
    *,
    env: str = ENV,
    nworker: int = 1,
    tools: list[str] | None = None,
) -> dict[str, dict]:
    """
    Pick each tool's parameters on the tuning dataset.

    Every combination in `GRIDS` is run on the tuning dataset only and
    scored by mean MCC; the best combination per tool is kept. Writes
    `<out>/tune_scores.tsv` (all combinations) and
    `<out>/tuned_params.json` (the chosen ones). With `tools`, the
    existing entries of the other tools in both files are kept.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `make_data`).
    env : str, default "catella"
        Conda environment for the external tools.
    nworker : int, default 1
        Number of grid combinations run in parallel. If greater than 1,
        the tuning dataset's FiberHMM model is trained first so that
        parallel runs share it.
    tools : list of str, optional
        Restrict to these keys of `TOOLS`; all if None.

    Returns
    -------
    dict of str to dict
        Tool -> chosen parameters, including the kept entries.
    """
    out = Path(out)
    _check_choices("tools", tools or [], set(TOOLS))
    selected = list(TOOLS) if tools is None else tools
    datasets = _read_record(out)["datasets"]
    spec = next(s for s in datasets if s["cond"] == "tune")
    ddir = dataset_dir(out, spec["cond"], spec["seed"])
    if nworker > 1 and "fiberhmm" in selected:
        train_fiberhmm(ddir, env=env)
    tasks = [
        (out, ddir, tool, params, env)
        for tool in selected
        for params in grid_combos(GRIDS[tool])
    ]
    table = pd.DataFrame(_run_tasks(_tune_one, tasks, nworker))
    best = table.loc[table.groupby("tool", sort=False)["mcc"].idxmax()]
    tuned = {r.tool: json.loads(r.params) for r in best.itertuples()}
    if tools is not None:
        kept = pd.read_csv(out / "tune_scores.tsv", sep="\t")
        table = pd.concat([kept[~kept["tool"].isin(tools)], table])
        old = json.loads((out / "tuned_params.json").read_text())
        tuned = {**old, **tuned}
    table.to_csv(out / "tune_scores.tsv", sep="\t", index=False)
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
        Columns `tool`, `variant`, `cond`, `seed`, `mcc`, `count_err`,
        and optionally `brier` (one row per molecule).

    Returns
    -------
    per_tool : pandas.DataFrame
        Per (`tool`, `variant`, `cond`): mean and SD over datasets of
        `mcc`, `count_err` and, if present, `brier`, and the number of
        datasets `n`.
    paired : pandas.DataFrame
        Per (`variant`, `cond`, `other` tool): mean and SD over datasets
        of the MCC difference catella minus `other`, and `wins`, the
        number of datasets where catella has the higher MCC, out of `n`.
    """
    keys = ["tool", "variant", "cond", "seed"]
    metrics = [
        m for m in ("mcc", "count_err", "brier") if m in results.columns
    ]
    ds = results.groupby(keys, sort=False)[metrics].mean().reset_index()
    aggs = {}
    for m in metrics:
        aggs[f"{m}_mean"] = (m, "mean")
        aggs[f"{m}_sd"] = (m, "std")
    aggs["n"] = ("mcc", "size")
    per_tool = (
        ds.groupby(["tool", "variant", "cond"], sort=False)
        .agg(**aggs)
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
        `<out>/results/<tool>/<cond>/s<seed>.tsv`.
    """
    return Path(out) / "results" / tool / cond / f"s{seed}.tsv"


def evaluate_unit(
    out: str | Path,
    spec: dict,
    tool: str,
    tuned: dict,
    env: str = ENV,
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
    env : str, default "catella"
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
        frames.append(_label_scores(calls, truth, tool, variant, cond, seed))
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(path, sep="\t", index=False)
    return path


def _label_scores(calls, truth, tool, variant, cond, seed):
    scores = score_calls(calls, truth)
    scores.insert(0, "seed", seed)
    scores.insert(0, "cond", cond)
    scores.insert(0, "variant", variant)
    scores.insert(0, "tool", tool)
    return scores


def _check_choices(name, values, allowed):
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(
            f"unknown {name} {unknown}; expected values in {sorted(allowed)}"
        )


def evaluate(
    out: str | Path,
    *,
    env: str = ENV,
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
    `overwrite`. The tool versions (see `tool_versions`) are written to
    `<out>/versions.json`. Afterward, `write_summary` combines every
    unit result present, including those from earlier or separate runs.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `tune`).
    env : str, default "catella"
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
    with open(out / "versions.json", "w") as fh:
        json.dump(tool_versions(env), fh, indent=2)
    _run_tasks(evaluate_unit, units, nworker)
    return write_summary(out)


def rescore(out: str | Path) -> pd.DataFrame:
    """
    Score the saved calls again and rewrite the summary.

    For every unit result present (see `unit_result_path`), reloads the
    saved calls of both variants and the ground truth, reruns
    `score_calls`, and overwrites the unit result. No tool is run. Use
    this after a change to `score_calls`.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `evaluate`).

    Returns
    -------
    pandas.DataFrame
        Per-molecule results of all units, as returned by
        `write_summary`.

    Raises
    ------
    ValueError
        If no unit results exist, or the saved calls of a unit are
        missing.
    """
    out = Path(out)
    files = sorted((out / "results").glob("*/*/s*.tsv"))
    if not files:
        raise ValueError(f"no unit results found in {out / 'results'}")
    for path in files:
        tool, cond = path.parent.parent.name, path.parent.name
        seed = int(path.stem[1:])
        truth = read_truth(dataset_dir(out, cond, seed))
        frames = []
        for variant in ("tuned", "default"):
            h5 = out / "calls" / f"{tool}_{variant}" / cond / f"s{seed}.h5"
            if not h5.exists():
                raise ValueError(f"saved calls not found: {h5}")
            frames.append(
                _label_scores(
                    load_calls(h5), truth, tool, variant, cond, seed
                )
            )
        pd.concat(frames, ignore_index=True).to_csv(
            path, sep="\t", index=False
        )
    return write_summary(out)


def write_summary(out: str | Path) -> pd.DataFrame:
    """
    Combine all unit results and write the summary tables.

    Writes `<out>/results.tsv` (per-molecule results of every unit
    present), `<out>/summary_per_tool.tsv`, `<out>/summary_paired.tsv`,
    and `<out>/summary.md`, which lists the tool versions if
    `<out>/versions.json` exists.

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
    files = sorted((out / "results").glob("*/*/*.tsv"))
    if not files:
        raise ValueError(f"no unit results found in {out / 'results'}")
    results = pd.concat(
        [pd.read_csv(f, sep="\t") for f in files], ignore_index=True
    )
    results.to_csv(out / "results.tsv", sep="\t", index=False)
    tuned_path = out / "tuned_params.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
    per_tool, paired = summarize(results)
    per_tool.to_csv(out / "summary_per_tool.tsv", sep="\t", index=False)
    paired.to_csv(out / "summary_paired.tsv", sep="\t", index=False)
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
    versions_path = out / "versions.json"
    if versions_path.exists():
        versions = json.loads(versions_path.read_text())
        text += ["## Tool versions", ""]
        text += [f"- {tool}: {ver}" for tool, ver in versions.items()]
        text += [""]
    (out / "summary.md").write_text("\n".join(text))
    return results


def plot_occupancy(
    out: str | Path,
    cond: str,
    seed: int,
    *,
    variant: str = "tuned",
    tools: list[str] | None = None,
    nmol: int | None = None,
    per_page: int | None = None,
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
    per_page : int, optional
        Molecules per A4 page; at most as many as fit on the page with
        every track at a readable height. If None (default), that
        maximum.
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
    if per_page is not None and per_page <= 0:
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
    nbp = _dataset_length(ddir)
    true = intervals_to_calls(read_truth(ddir), mol_ids, nbp)
    qual = _mod_qual_rows(ddir, mol_ids, nbp)
    rows = {
        tool: {m: i for i, m in enumerate(c.mol_ids)}
        for tool, c in calls.items()
    }
    ntrack = len(calls) + 2
    max_per_page = _occupancy_per_page(ntrack)
    if per_page is None:
        per_page = max_per_page
    elif per_page > max_per_page:
        raise ValueError(
            f"per_page must be at most {max_per_page} for {ntrack} tracks "
            f"per molecule, got {per_page}"
        )
    title = (
        f"Methylation probability and nucleosome occupancy: {cond}, "
        f"seed {seed}, {variant} parameters\n"
        "(shaded ends excluded from scoring)"
    )
    molecules = []
    for i, mol_id in enumerate(mol_ids):
        label, color = TOOL_STYLES["truth"]
        tracks = [(label, true.occ[i], color)]
        for tool in calls:
            label, color = TOOL_STYLES[tool]
            occ = None
            if mol_id in rows[tool]:
                occ = calls[tool].occ[rows[tool][mol_id]]
            tracks.append((label, occ, color))
        molecules.append((mol_id, qual[i], tracks))
    if path is None:
        path = out / "plots" / f"occupancy_{cond}_s{seed}_{variant}.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        for start in range(0, len(molecules), per_page):
            page = molecules[start : start + per_page]
            pdf.savefig(_occupancy_figure(page, title, nbp, TRIM))
    return path


def _dot_series(ax, x, values, color, *, hollow=False):
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.04, 0.04, size=len(values))
    face = "none" if hollow else color
    ax.scatter(x + jitter, values, s=10, facecolors=face, edgecolors=color,
               alpha=0.6, linewidths=0.8)
    sd = np.std(values, ddof=1) if len(values) > 1 else 0.0
    ax.errorbar(x, np.mean(values), yerr=sd, fmt="o", ms=4, color=color,
                mfc=face, capsize=2, lw=1)


def _dodged_panel(ax, ds, metric, tools, conds, ylabel):
    width = 0.8 / max(len(tools), 1)
    for j, tool in enumerate(tools):
        label, color = TOOL_STYLES[tool]
        offset = (j - (len(tools) - 1) / 2) * width
        for i, cond in enumerate(conds):
            sel = ds[(ds["tool"] == tool) & (ds["cond"] == cond)]
            if len(sel):
                _dot_series(ax, i + offset, sel[metric].to_numpy(), color)
        ax.scatter([], [], color=color, label=label, s=12)
    ax.set_xticks(range(len(conds)), conds, rotation=30, ha="right")
    ax.set_xlim(-0.5, len(conds) - 0.5)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.legend(fontsize=7, frameon=False, ncols=len(tools), loc="lower left",
              bbox_to_anchor=(0, 1.0))
    _style_axes(ax)


def _param_label(params):
    return ", ".join(f"{k}={v}" for k, v in json.loads(params).items())


def plot_summary(out: str | Path, path: str | Path | None = None) -> Path:
    """
    Plot summary figures of the benchmark comparison.

    Reads `<out>/results.tsv` (see `write_summary`) and, if present,
    `<out>/tune_scores.tsv`. Each dataset contributes one point, the
    mean over its molecules; error bars are mean +- SD over datasets.
    Pages, one figure each (skipped if their data are missing):

    1. MCC by condition and tool (tuned parameters).
    2. Count error by condition and tool (tuned parameters).
    3. Brier score by condition and tool (tuned parameters).
    4. MCC difference, catella minus each other tool, by condition.
    5. MCC with tuned and default parameters, per tool.
    6. Tuning-set MCC of every grid combination, per tool.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `evaluate` or `summarize`).
    path : str or Path, optional
        Output PDF path; defaults to `<out>/plots/summary.pdf`.

    Returns
    -------
    Path
        The PDF written.
    """
    out = Path(out)
    results_path = out / "results.tsv"
    if not results_path.exists():
        raise ValueError(f"{results_path} not found; run summarize first")
    results = pd.read_csv(results_path, sep="\t")
    keys = ["tool", "variant", "cond", "seed"]
    metrics = [
        m for m in ("mcc", "count_err", "brier") if m in results.columns
    ]
    ds = results.groupby(keys)[metrics].mean().reset_index()
    tools = [t for t in TOOL_STYLES if t in set(ds["tool"])]
    conds = [c for c in CONDITIONS if c in set(ds["cond"])]
    tuned = ds[ds["variant"] == "tuned"]
    if path is None:
        path = out / "plots" / "summary.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    size = (8.27, 11.69)
    with PdfPages(path) as pdf:
        for metric, ylabel, title in (
            ("mcc", "MCC", "MCC by condition (tuned parameters)"),
            (
                "count_err",
                "called - true nucleosomes",
                "Nucleosome count error by condition (tuned parameters)",
            ),
            (
                "brier",
                "Brier score (lower is better)",
                "Brier score by condition (tuned parameters)",
            ),
        ):
            if metric not in ds.columns:
                continue
            fig = Figure(figsize=size)
            ax = fig.add_subplot(2, 1, 1)
            _dodged_panel(ax, tuned, metric, tools, conds, ylabel)
            if metric == "count_err":
                ax.axhline(0, color="#52514e", lw=0.8)
            fig.suptitle(title, fontsize=10)
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)

        others = [t for t in tools if t != "catella"]
        if "catella" in tools and others:
            wide = tuned.pivot_table(
                index=["cond", "seed"], columns="tool", values="mcc"
            )
            fig = Figure(figsize=size)
            axes = fig.subplots(len(others), 1, sharex=True, squeeze=False)
            for ax, other in zip(axes[:, 0], others):
                label, color = TOOL_STYLES[other]
                for i, cond in enumerate(conds):
                    if cond not in wide.index.get_level_values(0):
                        continue
                    diff = (wide.loc[cond, "catella"] - wide.loc[cond, other])
                    diff = diff.dropna().to_numpy()
                    if len(diff):
                        _dot_series(ax, i, diff, color)
                ax.axhline(0, color="#52514e", lw=0.8)
                ax.set_ylabel(f"catella - {label}", fontsize=8)
                _style_axes(ax)
            axes[-1, 0].set_xticks(
                range(len(conds)), conds, rotation=30, ha="right"
            )
            axes[-1, 0].set_xlim(-0.5, len(conds) - 0.5)
            fig.suptitle(
                "MCC difference, catella minus other tool (tuned)",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)

        variants = [v for v in ("tuned", "default") if v in set(ds["variant"])]
        fig = Figure(figsize=size)
        axes = fig.subplots(len(tools), 1, sharex=True, squeeze=False)
        for ax, tool in zip(axes[:, 0], tools):
            label, color = TOOL_STYLES[tool]
            for k, variant in enumerate(variants):
                offset = (k - (len(variants) - 1) / 2) * 0.3
                sel = ds[(ds["tool"] == tool) & (ds["variant"] == variant)]
                for i, cond in enumerate(conds):
                    vals = sel.loc[sel["cond"] == cond, "mcc"].to_numpy()
                    if len(vals):
                        _dot_series(
                            ax, i + offset, vals, color,
                            hollow=variant == "default",
                        )
                face = "none" if variant == "default" else color
                ax.scatter(
                    [], [], s=12, edgecolors=color, facecolors=face,
                    label=variant,
                )
            ax.set_ylabel(f"{label}\nMCC", fontsize=8)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
            _style_axes(ax)
        axes[-1, 0].set_xticks(
            range(len(conds)), conds, rotation=30, ha="right"
        )
        axes[-1, 0].set_xlim(-0.5, len(conds) - 0.5)
        fig.suptitle(
            "MCC with tuned (filled) and default (hollow) parameters",
            fontsize=10,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)

        tune_path = out / "tune_scores.tsv"
        if tune_path.exists():
            scores = pd.read_csv(tune_path, sep="\t")
            tune_tools = [t for t in TOOL_STYLES if t in set(scores["tool"])]
            fig = Figure(figsize=size)
            axes = fig.subplots(len(tune_tools), 1, squeeze=False)
            for ax, tool in zip(axes[:, 0], tune_tools):
                label, color = TOOL_STYLES[tool]
                sel = scores[scores["tool"] == tool]
                x = np.arange(len(sel))
                best = sel["mcc"].to_numpy().argmax()
                ax.plot(x, sel["mcc"], color=color, lw=1, marker="o", ms=4)
                ax.plot(x[best], sel["mcc"].iloc[best], marker="o", ms=8,
                        mfc="none", mec="#0b0b0b")
                labels = [_param_label(p) for p in sel["params"]]
                ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=6)
                ax.set_ylabel(f"{label}\nMCC", fontsize=8)
                _style_axes(ax)
            fig.suptitle(
                "Tuning-set MCC per parameter combination (circled: chosen)",
                fontsize=10,
            )
            fig.tight_layout(rect=(0, 0, 1, 0.97))
            pdf.savefig(fig)
    return path


def reliability(
    calls: Calls,
    truth: pd.DataFrame,
    *,
    nbins: int = 10,
    nbp: int = NBP,
    trim: int = TRIM,
) -> pd.DataFrame:
    """
    Bin occupancy against true coverage, for a reliability diagram.

    Bases over `[trim, nbp - trim)` of every molecule are pooled and
    binned by occupancy into `nbins` equal bins of [0, 1]; occupancy 1
    falls in the last bin.

    Parameters
    ----------
    calls : Calls
        Calls to bin.
    truth : pandas.DataFrame
        Ground truth with columns `mol_id`, `start`, `end`.
    nbins : int, default 10
        Number of bins.
    nbp : int, default `NBP`
        Molecule length in bp.
    trim : int, default `TRIM`
        Bases excluded at each molecule end.

    Returns
    -------
    pandas.DataFrame
        One row per bin, with columns `lo` and `hi`, the bin edges; `n`,
        the number of bases; `occ_sum`, the sum of their occupancy; and
        `y_sum`, the number of them inside a true nucleosome.
    """
    if nbins <= 0:
        raise ValueError(f"nbins must be positive, got {nbins}")
    true = intervals_to_calls(truth, list(calls.mol_ids), nbp)
    inner = slice(trim, nbp - trim)
    occ = np.asarray(calls.occ[:, inner], dtype=float).ravel()
    y = (true.occ[:, inner] >= 0.5).ravel()
    idx = np.clip((occ * nbins).astype(int), 0, nbins - 1)
    edges = np.linspace(0, 1, nbins + 1)
    return pd.DataFrame(
        {
            "lo": edges[:-1],
            "hi": edges[1:],
            "n": np.bincount(idx, minlength=nbins),
            "occ_sum": np.bincount(idx, weights=occ, minlength=nbins),
            "y_sum": np.bincount(idx, weights=y, minlength=nbins),
        }
    )


def plot_reliability(
    out: str | Path,
    *,
    conds: list[str] | None = None,
    tools: list[str] | None = None,
    variant: str = "tuned",
    nbins: int = 10,
    min_n: int = 1000,
    path: str | Path | None = None,
) -> Path:
    """
    Plot reliability diagrams of the benchmark occupancy.

    One panel per condition. For each tool, the bins of `reliability`
    are pooled over the test datasets (seeds) of the condition, and the
    fraction of bases inside a true nucleosome is plotted against the
    mean occupancy of each bin. A calibrated tool lies on the diagonal.
    Tools or conditions without saved calls are left out.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `evaluate`).
    conds : list of str, optional
        Conditions, one panel each; ["default", "biased"] if None.
    tools : list of str, optional
        Tools to plot; ["catella", "hiddenfoot"] if None.
    variant : {"tuned", "default"}, default "tuned"
        Parameter variant whose calls are plotted.
    nbins : int, default 10
        Number of occupancy bins.
    min_n : int, default 1000
        Bins with fewer bases are not plotted.
    path : str or Path, optional
        Output PDF path; defaults to
        `<out>/plots/reliability_<variant>.pdf`.

    Returns
    -------
    Path
        The PDF written.
    """
    out = Path(out)
    conds = ["default", "biased"] if conds is None else conds
    tools = ["catella", "hiddenfoot"] if tools is None else tools
    _check_choices("conds", conds, set(CONDITIONS))
    _check_choices("tools", tools, set(TOOLS))
    panels = {}
    for cond in conds:
        curves = {}
        for tool in tools:
            cdir = out / "calls" / f"{tool}_{variant}" / cond
            files = sorted(cdir.glob("s*.h5"))
            if not files:
                continue
            bins = None
            for h5 in files:
                ddir = dataset_dir(out, cond, int(h5.stem[1:]))
                b = reliability(
                    load_calls(h5),
                    read_truth(ddir),
                    nbins=nbins,
                    nbp=_dataset_length(ddir),
                )
                if bins is None:
                    bins = b
                else:
                    bins[["n", "occ_sum", "y_sum"]] += b[
                        ["n", "occ_sum", "y_sum"]
                    ]
            curves[tool] = (bins, len(files))
        if curves:
            panels[cond] = curves
    if not panels:
        raise ValueError(
            f"no saved calls for conds {conds}, tools {tools}, variant "
            f"{variant!r}; run evaluate first"
        )
    if path is None:
        path = out / "plots" / f"reliability_{variant}.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(8.27, 11.69))
    ncol = 2
    nrow = max(-(-len(panels) // ncol), 2)
    for k, (cond, curves) in enumerate(panels.items()):
        ax = fig.add_subplot(nrow, ncol, k + 1)
        ax.plot([0, 1], [0, 1], color="#52514e", lw=0.8, ls="--")
        for tool, (bins, nseed) in curves.items():
            label, color = TOOL_STYLES[tool]
            keep = bins[bins["n"] >= min_n]
            ax.plot(
                keep["occ_sum"] / keep["n"],
                keep["y_sum"] / keep["n"],
                color=color,
                marker="o",
                ms=4,
                lw=1,
                label=f"{label} ({nseed} seeds)",
            )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.set_xlabel("mean stated occupancy", fontsize=8)
        ax.set_ylabel("fraction inside a true nucleosome", fontsize=8)
        ax.set_title(cond, fontsize=9)
        ax.legend(fontsize=7, frameon=False, loc="upper left")
        _style_axes(ax)
    fig.suptitle(
        f"Reliability of occupancy ({variant} parameters; bins with "
        f"fewer than {min_n} bases not shown)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    with PdfPages(path) as pdf:
        pdf.savefig(fig)
    return path


app = typer.Typer(help="Benchmark catella against fibertools and FiberHMM")


@app.command(name="make_data")
def make_data_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    quick: Annotated[
        bool, typer.Option(help="Generate a small set for a quick trial run")
    ] = False,
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = ENV,
    nworker: Annotated[
        int, typer.Option(help="Number of parallel processes")
    ] = 1,
):
    """Generate the benchmark datasets."""
    specs = make_data(out, quick=quick, env=env, nworker=nworker)
    typer.echo(f"wrote {len(specs)} datasets to {out}")


def _set_hiddenfoot(hiddenfoot):
    # Spawned workers inherit the environment
    if hiddenfoot:
        os.environ["HIDDENFOOT"] = os.path.abspath(hiddenfoot)


@app.command(name="tune")
def tune_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = ENV,
    nworker: Annotated[
        int,
        typer.Option(
            help="Number of parallel processes (FiberHMM and HiddenFoot "
            "use 2 threads)"
        ),
    ] = 1,
    hiddenfoot: Annotated[
        str | None,
        typer.Option(help="HiddenFoot executable; default $HIDDENFOOT"),
    ] = None,
    tool: Annotated[
        list[str] | None, typer.Option(help="Only this tool (repeatable)")
    ] = None,
):
    """Pick each tool's parameters on the tuning dataset."""
    _set_hiddenfoot(hiddenfoot)
    tuned = tune(out, env=env, nworker=nworker, tools=tool)
    typer.echo(json.dumps(tuned, indent=2))


@app.command(name="evaluate")
def evaluate_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = ENV,
    nworker: Annotated[
        int,
        typer.Option(
            help="Number of parallel processes (FiberHMM and HiddenFoot "
            "use 2 threads)"
        ),
    ] = 1,
    hiddenfoot: Annotated[
        str | None,
        typer.Option(help="HiddenFoot executable; default $HIDDENFOOT"),
    ] = None,
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
    _set_hiddenfoot(hiddenfoot)
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
    """Combine all unit results and write the summary and its plots."""
    write_summary(out)
    typer.echo(f"wrote {Path(out) / 'summary.md'}")
    typer.echo(f"wrote {plot_summary(out)}")


@app.command(name="rescore")
def rescore_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
):
    """Score the saved calls again, without running tools."""
    rescore(out)
    typer.echo(f"wrote {Path(out) / 'summary.md'}")


@app.command(name="plot_summary")
def plot_summary_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
):
    """Plot summary figures of the benchmark comparison."""
    typer.echo(f"wrote {plot_summary(out)}")


@app.command(name="plot_reliability")
def plot_reliability_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    cond: Annotated[
        list[str] | None,
        typer.Option(help="Condition (repeatable); default and biased"),
    ] = None,
    tool: Annotated[
        list[str] | None,
        typer.Option(help="Tool (repeatable); catella and hiddenfoot"),
    ] = None,
    variant: Annotated[
        str, typer.Option(help="Parameter variant: tuned or default")
    ] = "tuned",
):
    """Plot reliability diagrams of the occupancy."""
    path = plot_reliability(out, conds=cond, tools=tool, variant=variant)
    typer.echo(f"wrote {path}")


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
        int | None,
        typer.Option(help="Molecules per page; default as many as fit"),
    ] = None,
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
