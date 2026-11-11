# _parallel.py

import multiprocessing as mp
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from itertools import islice

from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)


def _threaded_map(func, items, *, nworker=1, batch_size=1000):
    """
    Apply `func` to each item, in threads if `nworker > 1`.

    Items are submitted in batches of `batch_size` so that at most one
    batch of results is pending at a time.

    Parameters
    ----------
    func : callable
        Called as `func(item)`.
    items : iterable
        The items to process; consumed once.
    nworker : int, default 1
        The number of threads. If 1, items are processed in order in the
        calling thread.
    batch_size : int, default 1000
        The number of items submitted at a time when `nworker > 1`.

    Yields
    ------
    tuple
        `(item, func(item))`, in input order if `nworker == 1`, otherwise
        in completion order within each batch.

    Raises
    ------
    Exception
        Any exception raised by `func`.
    """
    if nworker <= 1:
        for item in items:
            yield item, func(item)
        return
    items = iter(items)
    with ThreadPoolExecutor(max_workers=nworker) as executor:
        while batch := list(islice(items, batch_size)):
            futures = {executor.submit(func, item): item for item in batch}
            for future in as_completed(futures):
                yield futures[future], future.result()


def _process_map(func, items, *, nworker=1, on_crash=None):
    """
    Apply `func` to each item, in spawned worker processes if `nworker > 1`.

    At most `2 * nworker` items are in flight at a time, so `items` can be
    a lazy generator of any length.

    Parameters
    ----------
    func : callable
        Called as `func(item)`; must be picklable when `nworker > 1`.
    items : iterable
        The items to process; consumed once.
    nworker : int, default 1
        The number of worker processes. If 1, items are processed in
        order in the calling process.
    on_crash : callable, optional
        Called as `on_crash(in_flight)` with the list of items still in
        flight, including the failed one, if collecting a result from a
        worker raises. Only used when `nworker > 1`.

    Yields
    ------
    tuple
        `(item, func(item))`, in input order if `nworker == 1`, otherwise
        in completion order.

    Raises
    ------
    Exception
        Any exception raised by `func` or by the worker pool.
    """
    if nworker <= 1:
        for item in items:
            yield item, func(item)
        return
    items = iter(items)
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nworker, mp_context=ctx) as pool:
        first = islice(items, 2 * nworker)
        in_flight = {pool.submit(func, item): item for item in first}
        while in_flight:
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                try:
                    result = future.result()
                except Exception:
                    if on_crash is not None:
                        on_crash(list(in_flight.values()))
                    raise
                yield in_flight.pop(future), result
            for item in islice(items, len(done)):
                in_flight[pool.submit(func, item)] = item


def _progress_bar(verbose):
    """
    Create the progress bar used for simulation batches.

    Parameters
    ----------
    verbose : bool
        If False, the bar is disabled.

    Returns
    -------
    rich.progress.Progress
        The progress bar.
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        disable=not verbose,
    )


def _job_label(chrom, mol, run=None):
    """
    Describe a simulation job for messages.

    Parameters
    ----------
    chrom : str
        The chromosome.
    mol : int
        The molecule index.
    run : int, optional
        The run index; omitted for per-molecule jobs.

    Returns
    -------
    str
        For example "chrom=chr1, mol=3, run=0".
    """
    label = f"chrom={chrom}, mol={mol}"
    return label if run is None else f"{label}, run={run}"


def _report_crash(console, dataset, jobs):
    """
    Print the on-disk status of the jobs in flight when a pool crashed.

    Parameters
    ----------
    console : rich.console.Console
        Where to print.
    dataset : SimDataset
        The dataset the jobs write to, used to classify each run.
    jobs : iterable of MCMCJob or TMJob
        The jobs in flight.
    """
    console.print(
        "[red]Error[/red] Worker pool crashed; aborting the remaining batch."
    )
    for job in jobs:
        run = getattr(job, "run", None)
        try:
            status = dataset._classify_run(job.chrom, job.mol, run or 0)
            status = status or "ok"
        except Exception:  # noqa: BLE001
            status = "unknown"
        console.print(f"  {_job_label(job.chrom, job.mol, run)}: {status}")
