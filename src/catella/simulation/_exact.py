# _exact.py

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM

from catella.h5_array import H5Array
from catella.simulation._io import _pack_jagged
from catella.simulation._parallel import _process_map
from catella.simulation.config import SimSettings

# Unlike `run` (one molecule per job, workers write their own files), the
# parent writes every result, so molecules are sent in batches.
TM_BATCH_SIZE = 64


@dataclass(frozen=True, kw_only=True)
class TMBatchJob:
    """
    Inputs for the exact analysis of a batch of molecules.

    Each molecule is solved once, then every getter is called on the
    solved model. All fields are self-contained so an instance can be
    sent as-is to a freshly spawned worker process.
    """

    chrom: str
    """The chromosome of the molecules."""

    start: int
    """The index of the first molecule of the batch."""

    settings: SimSettings
    """The settings, resolved for `chrom`."""

    nbp: int
    """The number of base pairs of each molecule."""

    eseq: np.ndarray
    """The binding energy of each molecule, shape (nmol_batch, nbp)."""

    getters: tuple[tuple[str, dict], ...]
    """The `(method_name, kwargs)` of each `NucPosTM` getter to call."""


@dataclass(frozen=True)
class TMStat:
    """
    A quantity computed from each solved transfer-matrix model.
    """

    name: str
    """The key of the result in `dataset.analysis`."""

    getter: str
    """The name of the `NucPosTM` method that returns the value."""

    kwargs: dict = field(default_factory=dict)
    """Keyword arguments passed to the getter."""

    per_bp: bool = False
    """If True, the getter returns one value per base pair."""


def _require_eseq(dataset):
    """
    Check that `dataset` stores the binding energy of each molecule.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to check.

    Raises
    ------
    ValueError
        If the dataset has no stored energy landscape.
    """
    if not dataset.has_eseq:
        raise ValueError(
            "exact computation requires the per-molecule energy "
            "landscape; recreate the dataset with store_eseq=True"
        )


def _require_tm(dataset, caller):
    """
    Check that `dataset` supports exact transfer-matrix analysis.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to check.
    caller : str
        The name of the calling method, used in the error message.

    Raises
    ------
    ValueError
        If `dataset.method != "tm"`, or the dataset has no stored energy
        landscape.
    """
    if dataset.method != "tm":
        raise ValueError(f"{caller} requires a method='tm' dataset")
    _require_eseq(dataset)


def _reject_time(time, record_time):
    """
    Check that no time point is requested for a TM dataset.

    Parameters
    ----------
    time : int or None
        The requested time point.
    record_time : bool
        Whether the time point was to be added to the result name.

    Raises
    ------
    ValueError
        If `time` is given or `record_time` is True.
    """
    if time is not None or record_time:
        raise ValueError(
            "'time'/'record_time' are not meaningful for method='tm' "
            "datasets (no time axis)"
        )


def _chrom_settings(dataset, chrom):
    """
    Return the settings of `dataset` resolved for one chromosome.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the settings.
    chrom : str
        The chromosome.

    Returns
    -------
    SimSettings
        The resolved settings.
    """
    return SimSettings(**dict(dataset.settings)).for_chrom(chrom)


def _build_tm(settings, nbp, eseq):
    """
    Build and solve the transfer matrix of one molecule.

    The model is solved at `settings.end_temp`, the temperature of the
    exact equilibrium distribution.

    Parameters
    ----------
    settings : SimSettings
        The settings, resolved for the chromosome of the molecule.
    nbp : int
        The number of base pairs of the molecule.
    eseq : np.ndarray
        The binding energy of each base pair.

    Returns
    -------
    NucPosTM
        The solved model.
    """
    tm = NucPosTM(
        settings.nucbp,
        nbp,
        settings.llink,
        settings.elink,
        settings.mu,
        1,
    )
    tm.setEnergy(np.asarray(eseq, dtype=np.float64).tolist(), settings.emax)
    tm.solve(settings.end_temp)
    return tm


def _solved_tm(dataset, chrom, mol):
    """
    Build and solve the transfer matrix of one molecule of a dataset.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the settings and binding energy.
    chrom : str
        The chromosome of the molecule.
    mol : int
        The molecule index.

    Returns
    -------
    NucPosTM
        The solved model.
    """
    return _build_tm(
        _chrom_settings(dataset, chrom),
        dataset.nbp[chrom],
        dataset.eseq[chrom][mol],
    )


def _run_tm_batch(job):
    """
    Solve each molecule of a batch once and call every getter on it.

    Parameters
    ----------
    job : TMBatchJob
        The batch to process.

    Returns
    -------
    list of tuple
        One tuple per molecule, holding the result of each getter in the
        order of `job.getters`.
    """
    results = []
    for row in job.eseq:
        tm = _build_tm(job.settings, job.nbp, row)
        results.append(
            tuple(
                _lists_to_arrays(getattr(tm, name)(**kwargs))
                for name, kwargs in job.getters
            )
        )
    return results


def _lists_to_arrays(value):
    """
    Convert the lists in a getter result to arrays.

    Arrays are much cheaper than lists of floats to send between
    processes.

    Parameters
    ----------
    value : object
        A getter result: a scalar, a list, or a tuple of these.

    Returns
    -------
    object
        `value` with each list, including those inside a tuple, replaced
        by an array.
    """
    if isinstance(value, list):
        return np.asarray(value)
    if isinstance(value, tuple):
        return tuple(_lists_to_arrays(v) for v in value)
    return value


def _iter_tm_batches(dataset, chroms, getters, batch_size, nworker):
    """
    Solve every molecule of `chroms` once, in batches, possibly in parallel.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset holding the settings and binding energy.
    chroms : iterable of str
        The chromosomes to process.
    getters : sequence of (str, dict)
        The `(method_name, kwargs)` of each `NucPosTM` getter to call.
    batch_size : int
        The number of molecules per job.
    nworker : int
        The number of worker processes. If 1, batches run in order in the
        calling process.

    Yields
    ------
    tuple
        `(chrom, start, results)`, where `results` holds one tuple of
        getter results per molecule, starting at molecule `start`.
        Batches arrive in completion order when `nworker > 1`.

    Raises
    ------
    ValueError
        If `batch_size` or `nworker` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if nworker < 1:
        raise ValueError(f"nworker must be positive, got {nworker}")
    getters = tuple((name, dict(kwargs)) for name, kwargs in getters)

    def jobs():
        for chrom in chroms:
            settings = _chrom_settings(dataset, chrom)
            eseq = dataset.eseq[chrom]
            for start in range(0, dataset.nmol[chrom], batch_size):
                yield TMBatchJob(
                    chrom=chrom,
                    start=start,
                    settings=settings,
                    nbp=dataset.nbp[chrom],
                    eseq=np.array(eseq[start : start + batch_size]),
                    getters=getters,
                )

    for job, results in _process_map(_run_tm_batch, jobs(), nworker=nworker):
        yield job.chrom, job.start, results


def _tm_stats(
    dataset, chroms, stats, *, batch_size=TM_BATCH_SIZE, nworker=1
):
    """
    Store several exact quantities, solving each molecule only once.

    Per-molecule quantities are stored as DataFrames and per-bp ones as
    H5Arrays. Each batch is written as soon as it arrives, so peak memory
    scales with `2 * nworker * batch_size` molecules.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom][stat.name]`.
    chroms : iterable of str
        The chromosomes to process.
    stats : sequence of TMStat
        The quantities to compute.
    batch_size : int, default TM_BATCH_SIZE
        The number of molecules per job.
    nworker : int, default 1
        The number of worker processes.

    Raises
    ------
    ValueError
        If `batch_size` or `nworker` is not positive.
    """
    chroms = list(chroms)
    stats = list(stats)
    per_bp = any(s.per_bp for s in stats)
    tmp_dir = dataset.resolve_tmp_dir() if per_bp else None
    out = {}
    for chrom in chroms:
        shape = (dataset.nmol[chrom], dataset.nbp[chrom])
        out[chrom] = [
            H5Array.create(shape, dtype=np.float64, dir=tmp_dir)
            if s.per_bp
            else np.empty(shape[0], dtype=np.float64)
            for s in stats
        ]
    getters = [(s.getter, s.kwargs) for s in stats]
    batches = _iter_tm_batches(dataset, chroms, getters, batch_size, nworker)
    for chrom, start, results in batches:
        stop = start + len(results)
        for i, (stat, arr) in enumerate(zip(stats, out[chrom])):
            block = np.array([r[i] for r in results], dtype=np.float64)
            if stat.per_bp:
                arr.write_batch(start, stop, block)
            else:
                arr[start:stop] = block
    for chrom in chroms:
        for stat, arr in zip(stats, out[chrom]):
            value = arr if stat.per_bp else pd.DataFrame(arr)
            dataset.analysis[chrom][stat.name] = value


def _tm_per_mol(dataset, chroms, name, getter, *, nworker=1):
    """
    Store one exact value per molecule as a DataFrame.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom][name]`.
    chroms : iterable of str
        The chromosomes to process.
    name : str
        The key of the result.
    getter : callable
        A `NucPosTM` method returning a float, such as
        `NucPosTM.getMeanN`.
    nworker : int, default 1
        The number of worker processes.
    """
    stats = [TMStat(name, getter.__name__)]
    _tm_stats(dataset, chroms, stats, nworker=nworker)


def _tm_per_bp(dataset, chroms, name, batch_size, getter, *, nworker=1):
    """
    Store one exact value per molecule and base pair as an H5Array.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom][name]`.
    chroms : iterable of str
        The chromosomes to process.
    name : str
        The key of the result.
    batch_size : int
        The number of molecules per job.
    getter : callable
        A `NucPosTM` method returning one value per base pair, such as
        `NucPosTM.getOccup`.
    nworker : int, default 1
        The number of worker processes.
    """
    _tm_stats(
        dataset,
        chroms,
        [TMStat(name, getter.__name__, per_bp=True)],
        batch_size=batch_size,
        nworker=nworker,
    )


def _tm_configs(dataset, chroms, name, get_config, *, kwargs=None, nworker=1):
    """
    Store one nucleosome configuration per molecule.

    For each chromosome, writes `{name}_position_flat` and
    `{name}_position_offset` (the packed positions) and `{name}_energy`
    (the energy and log posterior probability of each configuration).

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom]`.
    chroms : iterable of str
        The chromosomes to process.
    name : str
        The prefix of the result keys.
    get_config : callable
        A `NucPosTM` method returning `(positions, energy)`, such as
        `NucPosTM.getMostLikelyConfig`.
    kwargs : dict, optional
        Keyword arguments passed to `get_config`.
    nworker : int, default 1
        The number of worker processes.
    """
    chroms = list(chroms)
    end_temp = SimSettings(**dict(dataset.settings)).end_temp
    getters = [(get_config.__name__, kwargs or {}), ("getLogZ", {})]
    positions = {chrom: [None] * dataset.nmol[chrom] for chrom in chroms}
    energy = {chrom: np.empty(dataset.nmol[chrom]) for chrom in chroms}
    log_z = {chrom: np.empty(dataset.nmol[chrom]) for chrom in chroms}
    batches = _iter_tm_batches(
        dataset, chroms, getters, TM_BATCH_SIZE, nworker
    )
    for chrom, start, results in batches:
        for mol, ((pos, e), lz) in enumerate(results, start):
            positions[chrom][mol] = np.asarray(pos, dtype=np.int32)
            energy[chrom][mol] = e
            log_z[chrom][mol] = lz
    for chrom in chroms:
        log_prob = -energy[chrom] / end_temp - log_z[chrom]
        flat, offset = _pack_jagged(positions[chrom])
        analysis = dataset.analysis[chrom]
        analysis[f"{name}_position_flat"] = pd.DataFrame({"position": flat})
        analysis[f"{name}_position_offset"] = pd.DataFrame({"offset": offset})
        analysis[f"{name}_energy"] = pd.DataFrame(
            {"energy": energy[chrom], "log_prob": log_prob}
        )
