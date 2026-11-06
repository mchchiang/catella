# run_perf.py

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

import run_synth
import typer
from callers import ENV

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
        "wall", "cpu", "peak_rss", "base_rss" and "fit_time"; see
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


if __name__ == "__main__":
    app()
