# perf_worker.py

import json
import resource
import sys
import threading
import time

import callers
import psutil

MB = 1024 * 1024


def tree_rss(proc: psutil.Process) -> int:
    """
    Resident memory of a process and all its descendants.

    Parameters
    ----------
    proc : psutil.Process
        Root process.

    Returns
    -------
    int
        Summed RSS in bytes; processes that exit while being read count
        as 0.
    """
    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += p.memory_info().rss
        except psutil.Error:
            pass
    return total


class PeakRSS:
    """
    Track the peak summed RSS of a process tree in a background thread.

    Use as a context manager; the peak is read from `peak` afterward.

    Parameters
    ----------
    pid : int, optional
        Root process; the current process if None.
    interval : float, default 0.05
        Seconds between samples.

    Attributes
    ----------
    peak : int
        Largest summed RSS seen, in bytes.
    """

    def __init__(self, pid: int | None = None, interval: float = 0.05):
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while True:
            self.peak = max(self.peak, tree_rss(self.proc))
            if self._stop.wait(self.interval):
                return

    def __enter__(self):
        self.peak = tree_rss(self.proc)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, tree_rss(self.proc))
        return False


def _cpu_seconds():
    total = 0.0
    for who in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN):
        usage = resource.getrusage(who)
        total += usage.ru_utime + usage.ru_stime
    return total


def measure(spec: dict) -> dict:
    """
    Run one tool on one dataset and measure its resource use.

    Parameters
    ----------
    spec : dict
        Keys "tool", "ddir", "workdir", "params" and "env", passed to
        `callers.run_tool`; optional "interval" (seconds between memory
        samples, default 0.05).

    Returns
    -------
    dict
        "wall" (s), "cpu" (s, user + system of this process and its
        finished children), "peak_rss" (MB, peak summed RSS of the
        process tree), "base_rss" (MB, RSS just before the tool started)
        and "fit_time" (s, or None; see `callers.Calls`).
    """
    interval = spec.get("interval", 0.05)
    base = tree_rss(psutil.Process())
    cpu0 = _cpu_seconds()
    with PeakRSS(interval=interval) as rss:
        start = time.perf_counter()
        calls = callers.run_tool(
            spec["tool"],
            spec["ddir"],
            spec["workdir"],
            spec["params"],
            spec["env"],
        )
        wall = time.perf_counter() - start
    return {
        "wall": wall,
        "cpu": _cpu_seconds() - cpu0,
        "peak_rss": rss.peak / MB,
        "base_rss": base / MB,
        "fit_time": calls.fit_time,
    }


def main(argv: list[str] | None = None) -> None:
    """
    Read a spec from a JSON file, run `measure`, and write the result.

    Parameters
    ----------
    argv : list of str, optional
        `[spec_path, result_path]`; `sys.argv[1:]` if None.
    """
    spec_path, result_path = argv if argv is not None else sys.argv[1:]
    with open(spec_path) as fh:
        spec = json.load(fh)
    result = measure(spec)
    with open(result_path, "w") as fh:
        json.dump(result, fh, indent=2)


if __name__ == "__main__":
    main()
