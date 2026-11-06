# run_perf.py

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import typer
from callers import ENV

WORKER = Path(__file__).resolve().parent / "perf_worker.py"
# Thread pools of numerical libraries, capped to the tool thread count
THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

app = typer.Typer(help="Benchmark the runtime and memory use of each tool")


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


if __name__ == "__main__":
    app()
