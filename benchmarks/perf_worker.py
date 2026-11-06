# perf_worker.py

import ctypes
import json
import os
import resource
import sys
import threading
import time

import callers
import psutil

MB = 1024 * 1024
# On macOS, RSS keeps freed pages the system has not yet reclaimed, so
# the physical footprint (the memory the kernel charges) is used instead
MEMORY_METRIC = "footprint" if sys.platform == "darwin" else "rss"
_RUSAGE_INFO_V4 = 4
_RUSAGE_FIELDS = (
    "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups",
    "pageins", "wired_size", "resident_size", "phys_footprint",
    "proc_start_abstime", "proc_exit_abstime", "child_user_time",
    "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups",
    "child_pageins", "child_elapsed_abstime", "diskio_bytesread",
    "diskio_byteswritten", "cpu_time_qos_default",
    "cpu_time_qos_maintenance", "cpu_time_qos_background",
    "cpu_time_qos_utility", "cpu_time_qos_legacy",
    "cpu_time_qos_user_initiated", "cpu_time_qos_user_interactive",
    "billed_system_time", "serviced_system_time", "logical_writes",
    "lifetime_max_phys_footprint", "instructions", "cycles",
    "billed_energy", "serviced_energy", "interval_max_phys_footprint",
    "runnable_time",
)


class _RusageInfoV4(ctypes.Structure):
    # Layout of `struct rusage_info_v4` (macOS), padded for safety
    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        *((name, ctypes.c_uint64) for name in _RUSAGE_FIELDS),
        ("padding", ctypes.c_uint64 * 32),
    ]


_libproc = None


def _rusage(pid):
    global _libproc
    if _libproc is None:
        _libproc = ctypes.CDLL("/usr/lib/libproc.dylib")
    info = _RusageInfoV4()
    err = _libproc.proc_pid_rusage(
        int(pid), _RUSAGE_INFO_V4, ctypes.byref(info)
    )
    if err != 0:
        raise psutil.NoSuchProcess(pid)
    return info


def process_memory(proc: psutil.Process) -> int:
    """
    Memory in use by one process.

    Parameters
    ----------
    proc : psutil.Process
        The process.

    Returns
    -------
    int
        Bytes: the physical footprint on macOS, else the RSS (see
        `MEMORY_METRIC`).

    Raises
    ------
    psutil.Error
        If the process cannot be read (e.g., it has exited).
    """
    if MEMORY_METRIC == "footprint":
        return _rusage(proc.pid).phys_footprint
    return proc.memory_info().rss


def lifetime_peak_memory() -> int:
    """
    Peak memory of the current process over its lifetime so far.

    Returns
    -------
    int
        Bytes: the lifetime peak physical footprint on macOS; 0 on other
        platforms, where it is not tracked.
    """
    if MEMORY_METRIC == "footprint":
        return _rusage(os.getpid()).lifetime_max_phys_footprint
    return 0


def tree_memory(proc: psutil.Process) -> int:
    """
    Memory in use by a process and all its descendants.

    Parameters
    ----------
    proc : psutil.Process
        Root process.

    Returns
    -------
    int
        Summed `process_memory` in bytes; processes that exit while
        being read count as 0.
    """
    total = 0
    try:
        procs = [proc, *proc.children(recursive=True)]
    except psutil.Error:
        return 0
    for p in procs:
        try:
            total += process_memory(p)
        except psutil.Error:
            pass
    return total


class PeakMemory:
    """
    Track the peak memory of a process tree in a background thread.

    Samples `tree_memory`. Use as a context manager; the peak is read
    from `peak` afterward.

    Parameters
    ----------
    pid : int, optional
        Root process; the current process if None.
    interval : float, default 0.05
        Seconds between samples.

    Attributes
    ----------
    peak : int
        Largest summed memory seen, in bytes.
    """

    def __init__(self, pid: int | None = None, interval: float = 0.05):
        self.proc = psutil.Process(pid)
        self.interval = interval
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        while True:
            self.peak = max(self.peak, tree_memory(self.proc))
            if self._stop.wait(self.interval):
                return

    def __enter__(self):
        self.peak = tree_memory(self.proc)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, tree_memory(self.proc))
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
        finished children), "peak_mem" (MB, peak summed memory of the
        process tree; on macOS at least the worker's own lifetime peak,
        which catches spikes between samples), "base_mem" (MB, memory
        just before the tool started), "mem_metric" (`MEMORY_METRIC`)
        and "fit_time" (s, or None; see `callers.Calls`).
    """
    interval = spec.get("interval", 0.05)
    base = tree_memory(psutil.Process())
    cpu0 = _cpu_seconds()
    with PeakMemory(interval=interval) as mem:
        start = time.perf_counter()
        calls = callers.run_tool(
            spec["tool"],
            spec["ddir"],
            spec["workdir"],
            spec["params"],
            spec["env"],
        )
        wall = time.perf_counter() - start
    peak = max(mem.peak, lifetime_peak_memory())
    return {
        "wall": wall,
        "cpu": _cpu_seconds() - cpu0,
        "peak_mem": peak / MB,
        "base_mem": base / MB,
        "mem_metric": MEMORY_METRIC,
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
