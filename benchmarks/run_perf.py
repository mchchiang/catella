# run_perf.py

import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import psutil
import run_synth
import typer
from callers import ENV, NO_ENV_TOOLS, TOOLS, env_bin_dir, tool_versions
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from matplotlib.ticker import FormatStrFormatter, LogLocator
from plotting import TOOL_STYLES, _style_axes

WORKER = Path(__file__).resolve().parent / "perf_worker.py"
# Scaling axes: molecule count at BASE_NBP, length at BASE_NMOL
NMOL_AXIS = (25, 50, 100, 200, 400)
NBP_AXIS = (2000, 5000, 10000, 20000)
BASE_NMOL = 100
BASE_NBP = 5000
QUICK_NMOL_AXIS = (10, 20)
QUICK_NBP_AXIS = (2000, 5000)
QUICK_BASE_NMOL = 10
SEED = 1
NREP = 3
# FiberHMM files cached in a dataset directory by `train_fiberhmm`
FIBERHMM_CACHE = ("fiberhmm_model", "fiberhmm_probs")
METRICS = ("wall", "cpu", "fit_time", "call_time", "peak_mem", "tool_mem")
NOTES = (
    (
        "Times are end to end per tool run: input loading, the tool "
        "itself, and the parsing of its output (ft extract tables, "
        "HiddenFoot windows)."
    ),
    (
        "fit_time is the time spent fitting to the dataset: FiberHMM "
        "training (retrained model) and the catella llink fit; "
        "call_time = wall - fit_time. HiddenFoot fits and calls in one "
        "step, so it has no split."
    ),
    (
        "peak_mem is the peak summed memory of the worker process and "
        "all its child processes; tool_mem = peak_mem - base_mem, the "
        "memory above the worker's baseline (Python with catella "
        "imported). Memory is the physical footprint on macOS (RSS "
        "there also counts freed pages the system has not reclaimed "
        "yet) and the RSS elsewhere; see the mem_metric column. Summed "
        "memory counts shared libraries once per process, so tools "
        "running several processes (catella with nworker > 1) are "
        "slightly overestimated."
    ),
    (
        "FiberHMM training (fiberhmm-probs, fiberhmm-train) has no "
        "thread option; numerical library threads are capped by "
        "environment variables."
    ),
    "Tables show the median over repeats.",
)
# Thread pools of numerical libraries, capped to the tool thread count
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

app = typer.Typer(help="Benchmark the runtime and memory use of each tool")


def dataset_name(nmol: int, nbp: int) -> str:
    """
    Name of the scaling dataset with a given size.

    Parameters
    ----------
    nmol : int
        Number of molecules.
    nbp : int
        Molecule length in bp.

    Returns
    -------
    str
        `n<nmol>_l<nbp>`, used as the condition name in
        `run_synth.dataset_dir`.
    """
    return f"n{nmol}_l{nbp}"


def perf_specs(quick: bool = False) -> list[dict]:
    """
    List the points of both scaling axes.

    The "nmol" axis varies the molecule count at `BASE_NBP`; the "nbp"
    axis varies the length at `BASE_NMOL`. Datasets use the default
    synthetic benchmark condition (`run_synth.BASE_PARAMS`) with both
    controls, both strands and seed `SEED`.

    Parameters
    ----------
    quick : bool, default False
        If True, use the short axes `QUICK_NMOL_AXIS` and
        `QUICK_NBP_AXIS` (the latter at `QUICK_BASE_NMOL`), for a quick
        trial run.

    Returns
    -------
    list of dict
        One dict per axis point with keys "axis" ("nmol" or "nbp"),
        "value", "cond" (see `dataset_name`; shared by points with the
        same dataset) and "params" (keyword arguments for
        `synthgen.make_dataset`).
    """
    nmols = QUICK_NMOL_AXIS if quick else NMOL_AXIS
    nbps = QUICK_NBP_AXIS if quick else NBP_AXIS
    base_nmol = QUICK_BASE_NMOL if quick else BASE_NMOL
    points = [("nmol", n, n, BASE_NBP) for n in nmols]
    points += [("nbp", n, base_nmol, n) for n in nbps]
    specs = []
    for axis, value, nmol, nbp in points:
        params = {
            **run_synth.BASE_PARAMS,
            "nbp": nbp,
            "nmol": nmol,
            "controls": True,
            "double_strand": True,
            "seed": SEED,
        }
        specs.append(
            {
                "axis": axis,
                "value": value,
                "cond": dataset_name(nmol, nbp),
                "params": params,
            }
        )
    return specs


def make_data(
    out: str | Path,
    *,
    quick: bool = False,
    env: str = ENV,
    convert: bool = True,
    nworker: int = 1,
) -> list[dict]:
    """
    Generate the scaling datasets.

    Each distinct dataset is generated once, in
    `run_synth.dataset_dir(out, cond, SEED)`. Also writes
    `<out>/perf_datasets.json` with the axis points.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    quick : bool, default False
        Use the short axes; see `perf_specs`.
    env : str, default "catella"
        Conda environment providing samtools.
    convert : bool, default True
        If True, convert every SAM file to a sorted, indexed BAM file.
    nworker : int, default 1
        Number of processes generating datasets in parallel.

    Returns
    -------
    list of dict
        The axis points of `perf_specs`, each with an added "dir" key.
    """
    out = Path(out)
    specs = perf_specs(quick)
    unique = {}
    for spec in specs:
        unique.setdefault(
            spec["cond"],
            {"cond": spec["cond"], "seed": SEED, "params": spec["params"]},
        )
    tasks = [(out, spec, env, convert) for spec in unique.values()]
    dirs = run_synth._run_tasks(run_synth._make_one, tasks, nworker)
    by_cond = dict(zip(unique, dirs))
    for spec in specs:
        spec["dir"] = by_cond[spec["cond"]]
    record = {"quick": quick, "seed": SEED, "datasets": specs}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "perf_datasets.json", "w") as fh:
        json.dump(record, fh, indent=2)
    return specs


def measure_tool(
    tool: str,
    ddir: str | Path,
    params: dict,
    *,
    env: str = ENV,
    nthread: int = 2,
    interval: float = 0.05,
    workdir: str | Path | None = None,
) -> dict:
    """
    Measure one tool run on one dataset in a fresh worker process.

    The worker (`perf_worker.py`) times `callers.run_tool` only, so
    interpreter startup and imports are excluded. The thread pools of
    numerical libraries are capped to `nthread` (see `THREAD_ENV_VARS`).

    Parameters
    ----------
    tool : str
        A key of `callers.TOOLS`.
    ddir : str or Path
        Dataset directory.
    params : dict
        Keyword arguments for the tool's runner, including its thread
        option.
    env : str, default "catella"
        Conda environment for the external tools.
    nthread : int, default 2
        Value of the thread-count environment variables.
    interval : float, default 0.05
        Seconds between memory samples.
    workdir : str or Path, optional
        Directory for intermediate files; a temporary directory, removed
        afterward, if None.

    Returns
    -------
    dict
        "wall", "cpu", "peak_mem", "base_mem", "mem_metric" and
        "fit_time"; see
        `perf_worker.measure`.

    Raises
    ------
    ValueError
        If `nthread` is not positive.
    RuntimeError
        If the worker fails.
    """
    if nthread <= 0:
        raise ValueError(f"nthread must be positive, got {nthread}")
    with tempfile.TemporaryDirectory(dir=workdir) as tmp:
        tmp = Path(tmp)
        spec = {
            "tool": tool,
            "ddir": str(ddir),
            "workdir": str(tmp),
            "params": params,
            "env": env,
            "interval": interval,
        }
        spec_path = tmp / "spec.json"
        spec_path.write_text(json.dumps(spec))
        result_path = tmp / "result.json"
        child_env = {**os.environ}
        child_env.update({name: str(nthread) for name in THREAD_ENV_VARS})
        proc = subprocess.run(
            [sys.executable, str(WORKER), str(spec_path), str(result_path)],
            capture_output=True,
            text=True,
            check=False,
            env=child_env,
        )
        if proc.returncode != 0 or not result_path.exists():
            tail = "\n".join(proc.stderr.strip().splitlines()[-20:])
            raise RuntimeError(f"measuring {tool} failed; {tail}")
        return json.loads(result_path.read_text())


def thread_params(tool: str, nthread: int) -> dict:
    """
    Runner keyword arguments that set a tool's thread count.

    Parameters
    ----------
    tool : str
        A key of `callers.TOOLS`.
    nthread : int
        Number of threads (worker processes for catella and
        `fiberhmm-call`).

    Returns
    -------
    dict
        Keyword arguments for the tool's runner. FiberHMM training has
        no thread option.

    Raises
    ------
    ValueError
        If `tool` is unknown.
    """
    if tool == "catella":
        return {"nworker": nthread}
    if tool in ("ft", "hiddenfoot"):
        return {"threads": nthread}
    if tool in ("fiberhmm", "fiberhmm_pre"):
        return {"threads": nthread, "io_threads": nthread}
    raise ValueError(
        f"unknown tool {tool!r}; expected one of {sorted(TOOLS)}"
    )


def result_path(
    out: str | Path, tool: str, axis: str, value: int, rep: int
) -> Path:
    """
    Result file of one measured run.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.
    tool : str
        A key of `callers.TOOLS`.
    axis : {"nmol", "nbp"}
        Scaling axis.
    value : int
        Axis value.
    rep : int
        Repeat index.

    Returns
    -------
    Path
        `<out>/results/<tool>/<axis>_<value>_r<rep>.json`.
    """
    return Path(out) / "results" / tool / f"{axis}_{value}_r{rep}.json"


def machine_info() -> dict:
    """
    Describe the machine the benchmark runs on.

    Returns
    -------
    dict
        Keys "platform", "machine", "cpu", "physical_cores",
        "logical_cores", "memory_gb" and "python".
    """
    cpu = platform.processor()
    if sys.platform == "darwin":
        proc = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True,
            text=True,
            check=False,
        )
        cpu = proc.stdout.strip() or cpu
    elif Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": cpu,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "memory_gb": round(psutil.virtual_memory().total / 1024**3, 1),
        "python": platform.python_version(),
    }


def _clear_fiberhmm_cache(ddir):
    for name in FIBERHMM_CACHE:
        for path in Path(ddir).glob(f"{name}*"):
            shutil.rmtree(path, ignore_errors=True)


def _is_done(path):
    if not path.exists():
        return False
    return "error" not in json.loads(path.read_text())


def run(
    out: str | Path,
    *,
    env: str = ENV,
    nthread: int = 2,
    nrep: int | None = None,
    tools: list[str] | None = None,
    tuned: str | Path | None = None,
    overwrite: bool = False,
    interval: float = 0.05,
) -> list[Path]:
    """
    Measure every tool on every scaling dataset, one run at a time.

    Runs are ordered by repeat, then axis point, then tool. A run whose
    result file exists is skipped unless `overwrite`; a failed run is
    saved with an "error" key and run again next time. Before each
    retrained FiberHMM run, its cached model is removed so that
    training is always timed. External tools are run directly from
    their conda environment (see `callers.run_in_env`). Also writes
    `<out>/versions.json` and `<out>/machine.json`.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `make_data`).
    env : str, default "catella"
        Conda environment for the external tools.
    nthread : int, default 2
        Threads per tool; see `thread_params`.
    nrep : int, optional
        Repeats per run; `NREP`, or 1 for quick datasets, if None.
    tools : list of str, optional
        Restrict to these tools (keys of `callers.TOOLS`); all if None.
    tuned : str or Path, optional
        `tuned_params.json` of a synthetic benchmark (`run_synth.py`);
        tool defaults if None.
    overwrite : bool, default False
        If True, rerun runs whose results exist.
    interval : float, default 0.05
        Seconds between memory samples.

    Returns
    -------
    list of Path
        Result files of the runs measured in this call.
    """
    out = Path(out)
    with open(out / "perf_datasets.json") as fh:
        record = json.load(fh)
    tools = list(TOOLS) if tools is None else tools
    run_synth._check_choices("tools", tools, set(TOOLS))
    if nrep is None:
        nrep = 1 if record["quick"] else NREP
    tuned_params = {}
    if tuned is not None:
        with open(tuned) as fh:
            tuned_params = json.load(fh)
    if any(t not in NO_ENV_TOOLS for t in tools):
        os.environ.setdefault("BENCH_TOOL_BIN", str(env_bin_dir(env)))
    with open(out / "versions.json", "w") as fh:
        json.dump(tool_versions(env), fh, indent=2)
    with open(out / "machine.json", "w") as fh:
        json.dump(machine_info(), fh, indent=2)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    written = []
    for rep in range(nrep):
        for spec in record["datasets"]:
            for tool in tools:
                path = result_path(out, tool, spec["axis"], spec["value"], rep)
                if _is_done(path) and not overwrite:
                    continue
                if tool == "fiberhmm":
                    _clear_fiberhmm_cache(spec["dir"])
                params = {
                    **tuned_params.get(tool, {}),
                    **thread_params(tool, nthread),
                }
                result = {
                    "tool": tool,
                    "axis": spec["axis"],
                    "value": spec["value"],
                    "rep": rep,
                    "nmol": spec["params"]["nmol"],
                    "nbp": spec["params"]["nbp"],
                    "nthread": nthread,
                    "params": params,
                }
                try:
                    result.update(
                        measure_tool(
                            tool,
                            spec["dir"],
                            params,
                            env=env,
                            nthread=nthread,
                            interval=interval,
                            workdir=work,
                        )
                    )
                except RuntimeError as err:
                    result["error"] = str(err)
                    typer.echo(f"warning: {err}", err=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(result, indent=2))
                written.append(path)
    return written


def collect(out: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read every run result.

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.

    Returns
    -------
    results : pandas.DataFrame
        One row per successful run: tool, axis, value, rep, nmol, nbp,
        nthread, mem_metric, and the measures "wall", "cpu",
        "fit_time", "call_time" (wall - fit_time), "peak_mem",
        "base_mem" and "tool_mem" (peak_mem - base_mem). Times in
        seconds, memory in MB; fit_time and call_time are NaN for tools
        with no fitting step.
    errors : pandas.DataFrame
        One row per failed run: tool, axis, value, rep and error.

    Raises
    ------
    ValueError
        If no result files are found, or if a result file predates the
        "peak_mem" measure.
    """
    files = sorted((Path(out) / "results").glob("*/*.json"))
    if not files:
        raise ValueError(f"no run results found in {Path(out) / 'results'}")
    rows, errors = [], []
    for path in files:
        result = json.loads(path.read_text())
        result.pop("params", None)
        if "peak_rss" in result:
            raise ValueError(
                f"{path} has the old peak_rss measure; rerun with "
                "--overwrite"
            )
        (errors if "error" in result else rows).append(result)
    keys = ["tool", "axis", "value", "rep"]
    results = pd.DataFrame(rows)
    if not results.empty:
        results["fit_time"] = results["fit_time"].astype(float)
        results["call_time"] = results["wall"] - results["fit_time"]
        results["tool_mem"] = results["peak_mem"] - results["base_mem"]
        results = results.sort_values(keys, ignore_index=True)
    errors = pd.DataFrame(errors, columns=[*keys, "error"])
    return results, errors


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    """
    Take the median of each measure over repeats.

    Parameters
    ----------
    results : pandas.DataFrame
        Output of `collect`.

    Returns
    -------
    pandas.DataFrame
        One row per (axis, value, tool) with nmol, nbp, the number of
        repeats "nrep", and the median of each of `METRICS`.
    """
    keys = ["axis", "value", "tool", "nmol", "nbp"]
    grouped = results.groupby(keys, sort=True)
    table = grouped[list(METRICS)].median()
    table.insert(0, "nrep", grouped.size())
    return table.reset_index()


def _read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def write_summary(out: str | Path) -> pd.DataFrame:
    """
    Combine all run results and write the summary files.

    Writes `<out>/perf.tsv` (every successful run) and `<out>/perf.md`
    (median tables per axis, failed runs, notes, machine and tool
    versions).

    Parameters
    ----------
    out : str or Path
        Benchmark output directory.

    Returns
    -------
    pandas.DataFrame
        Medians, as returned by `summarize`.
    """
    out = Path(out)
    results, errors = collect(out)
    if results.empty:
        raise ValueError(f"every run in {out / 'results'} failed")
    results.to_csv(out / "perf.tsv", sep="\t", index=False)
    table = summarize(results)
    nthread = sorted(results["nthread"].unique().tolist())
    text = ["# Runtime and memory benchmark", ""]
    metrics = sorted(results["mem_metric"].unique().tolist())
    text += [f"Threads per tool: {', '.join(map(str, nthread))}", ""]
    text += [f"Memory measure: {', '.join(metrics)}", ""]
    for axis, label in (("nmol", "molecule count"), ("nbp", "length")):
        sub = table[table["axis"] == axis].drop(columns="axis")
        if sub.empty:
            continue
        text += [f"## Scaling with {label}", ""]
        text += [run_synth._markdown_table(sub), ""]
    if not errors.empty:
        text += ["## Failed runs", ""]
        for row in errors.itertuples(index=False):
            reason = row.error.strip().splitlines()[-1]
            text.append(
                f"- {row.tool} {row.axis}={row.value} rep {row.rep}: {reason}"
            )
        text += [""]
    text += ["## Notes", ""] + [f"- {note}" for note in NOTES] + [""]
    for name, title in (("machine", "Machine"), ("versions", "Tool versions")):
        info = _read_json(out / f"{name}.json")
        if info:
            text += [f"## {title}", ""]
            text += [f"- {key}: {value}" for key, value in info.items()]
            text += [""]
    (out / "perf.md").write_text("\n".join(text))
    return table


_AXIS_LABELS = {"nmol": "Molecules", "nbp": "Molecule length (bp)"}
_FIT_COLORS = {"fit_time": "#52514e", "call_time": "#b8b6b0"}


def _scaling_page(results, tools):
    fig = Figure(figsize=(8.27, 11.69))
    axes = fig.subplots(2, 2, gridspec_kw={"hspace": 0.3, "wspace": 0.3})
    rows = (("wall", "Wall time (s)"), ("tool_mem", "Peak memory (MB)"))
    for i, (metric, ylabel) in enumerate(rows):
        for j, axis in enumerate(("nmol", "nbp")):
            ax = axes[i, j]
            sub = results[results["axis"] == axis]
            for tool in tools:
                pts = sub[sub["tool"] == tool].groupby("value")[metric]
                if pts.ngroups == 0:
                    continue
                med, lo, hi = pts.median(), pts.min(), pts.max()
                label, color = TOOL_STYLES.get(tool, (tool, "#52514e"))
                ax.vlines(med.index, lo, hi, color=color, linewidth=1)
                ax.plot(
                    med.index,
                    med.to_numpy(),
                    color=color,
                    linewidth=1.5,
                    marker="o",
                    markersize=4,
                    markeredgecolor="white",
                    markeredgewidth=0.8,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_yscale("log")
            values = sorted(sub["value"].unique())
            ax.set_xticks(values, [str(v) for v in values])
            ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%g"))
            ax.minorticks_off()
            ax.set_xlabel(_AXIS_LABELS[axis], fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            _style_axes(ax)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=len(labels),
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.95),
    )
    fig.suptitle("Runtime and memory scaling (median, min-max over repeats)")
    return fig


def _fit_page(results, tools):
    sub = results[(results["axis"] == "nmol") & results["fit_time"].notna()]
    tools = [t for t in tools if t in set(sub["tool"])]
    if not tools:
        return None
    fig = Figure(figsize=(8.27, 11.69))
    axes = np.atleast_1d(
        fig.subplots(len(tools), 1, gridspec_kw={"hspace": 0.4})
    )
    for ax, tool in zip(axes, tools):
        med = sub[sub["tool"] == tool].groupby("value")[
            ["fit_time", "call_time"]
        ].median()
        x = np.arange(len(med))
        bottom = np.zeros(len(med))
        for metric, name in (("fit_time", "fit"), ("call_time", "call")):
            ax.bar(
                x,
                med[metric],
                bottom=bottom,
                width=0.6,
                color=_FIT_COLORS[metric],
                edgecolor="white",
                linewidth=1,
                label=name,
            )
            bottom = bottom + med[metric].to_numpy()
        for xi, top, call in zip(x, bottom, med["call_time"]):
            ax.annotate(
                f"call {call:.2g} s",
                (xi, top),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="#52514e",
            )
        ax.set_xticks(x, [str(v) for v in med.index])
        ax.set_xlabel(_AXIS_LABELS["nmol"], fontsize=8)
        ax.set_ylabel("Wall time (s)", fontsize=8)
        ax.set_title(TOOL_STYLES.get(tool, (tool,))[0], fontsize=9)
        ax.legend(frameon=False, fontsize=7)
        _style_axes(ax)
    fig.suptitle("Fit and call time (median over repeats)")
    return fig


def plot_perf(out: str | Path, path: str | Path | None = None) -> Path:
    """
    Plot the runtime and memory benchmark.

    Reads `<out>/perf.tsv` (see `write_summary`). Pages:

    1. Wall time and peak memory above the worker baseline
       (`tool_mem`) against molecule count and against length, log-log,
       one line per tool (median, min-max over repeats).
    2. Fit and call time against molecule count, one panel per tool
       with a fitting step (skipped if there is none).

    Parameters
    ----------
    out : str or Path
        Benchmark output directory (after `summarize`).
    path : str or Path, optional
        Output PDF path; defaults to `<out>/plots/perf.pdf`.

    Returns
    -------
    Path
        The PDF written.

    Raises
    ------
    ValueError
        If `<out>/perf.tsv` does not exist.
    """
    out = Path(out)
    tsv = out / "perf.tsv"
    if not tsv.exists():
        raise ValueError(f"{tsv} not found; run summarize first")
    results = pd.read_csv(tsv, sep="\t")
    present = set(results["tool"])
    tools = [t for t in TOOL_STYLES if t in present]
    tools += sorted(present - set(tools))
    path = out / "plots" / "perf.pdf" if path is None else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(path) as pdf:
        pdf.savefig(_scaling_page(results, tools))
        fit = _fit_page(results, tools)
        if fit is not None:
            pdf.savefig(fit)
    return path


@app.command(name="make_data")
def make_data_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    quick: Annotated[
        bool, typer.Option(help="Use short axes for a quick trial run")
    ] = False,
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = ENV,
    nworker: Annotated[
        int, typer.Option(help="Number of parallel processes")
    ] = 1,
):
    """Generate the scaling datasets."""
    specs = make_data(out, quick=quick, env=env, nworker=nworker)
    ndata = len({spec["cond"] for spec in specs})
    typer.echo(f"wrote {ndata} datasets to {out}")


@app.command(name="run")
def run_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = ENV,
    nthread: Annotated[
        int, typer.Option(help="Threads (or worker processes) per tool")
    ] = 2,
    nrep: Annotated[
        int | None,
        typer.Option(help=f"Repeats per run; default {NREP} (1 if quick)"),
    ] = None,
    tool: Annotated[
        list[str] | None, typer.Option(help="Only this tool (repeatable)")
    ] = None,
    tuned: Annotated[
        Path | None,
        typer.Option(help="tuned_params.json from run_synth.py"),
    ] = None,
    hiddenfoot: Annotated[
        str | None,
        typer.Option(help="HiddenFoot executable; default $HIDDENFOOT"),
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Rerun runs whose results exist")
    ] = False,
):
    """Measure the runtime and memory use of each tool."""
    run_synth._set_hiddenfoot(hiddenfoot)
    paths = run(
        out,
        env=env,
        nthread=nthread,
        nrep=nrep,
        tools=tool,
        tuned=tuned,
        overwrite=overwrite,
    )
    typer.echo(f"measured {len(paths)} runs in {out}")


@app.command(name="summarize")
def summarize_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
):
    """Write perf.tsv and perf.md from the run results."""
    write_summary(out)
    typer.echo(f"wrote {out / 'perf.md'}")


@app.command(name="plot")
def plot_cmd(
    out: Annotated[Path, typer.Argument(help="Benchmark output directory")],
):
    """Plot runtime and memory scaling."""
    typer.echo(f"wrote {plot_perf(out)}")


if __name__ == "__main__":
    app()
