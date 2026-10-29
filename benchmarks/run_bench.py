# run_bench.py

import json
import os
import subprocess
from pathlib import Path
from typing import Annotated

import synthgen
import typer

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
    "ft": {"nucleosome_length": [60, 90], "combined_length": [85, 115]},
    "fiberhmm": {"nuc_min_size": [70, 100], "split_min_llr": [2.0, 6.0]},
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


if __name__ == "__main__":
    app()
