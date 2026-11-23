# _exact.py

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM

from catella._jagged import _pack_jagged
from catella.h5_array import H5Array
from catella.simulation._inputs import _start_positions
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

    region: str
    """The region of the molecules."""

    start: int
    """The index of the first molecule of the batch."""

    settings: SimSettings
    """The settings, resolved for `region`."""

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

    ncol: int | None = None
    """If given, the getter returns this many values per molecule."""

    columns: tuple | None = None
    """Column labels of a `per_bp` or `ncol` result; None for none."""

    def __post_init__(self):
        if self.per_bp and self.ncol is not None:
            raise ValueError("per_bp and ncol are mutually exclusive")

    def width(self, nbp):
        """
        Return the number of values per molecule, or None for a scalar.

        Parameters
        ----------
        nbp : int
            The number of base pairs of each molecule.

        Returns
        -------
        int or None
            `nbp` for a `per_bp` stat, `ncol` for a fixed-width one, and
            None for a scalar.
        """
        return nbp if self.per_bp else self.ncol


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


def _region_settings(dataset, region):
    """
    Return the settings of `dataset` resolved for one region.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the settings.
    region : str
        The region.

    Returns
    -------
    SimSettings
        The resolved settings.
    """
    return SimSettings(**dict(dataset.settings)).for_region(region)


def _build_tm(settings, nbp, eseq):
    """
    Build and solve the transfer matrix of one molecule.

    The model is solved at `settings.end_temp`, the temperature of the
    exact equilibrium distribution.

    Parameters
    ----------
    settings : SimSettings
        The settings, resolved for the region of the molecule.
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


def _solved_tm(dataset, region, mol):
    """
    Build and solve the transfer matrix of one molecule of a dataset.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the settings and binding energy.
    region : str
        The region of the molecule.
    mol : int
        The molecule index.

    Returns
    -------
    NucPosTM
        The solved model.
    """
    return _build_tm(
        _region_settings(dataset, region),
        dataset.nbp[region],
        dataset.eseq[region][mol],
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
    list of tuple or None
        One tuple per molecule, holding the result of each getter in the
        order of `job.getters`, or None for a molecule with no data (nan
        energy at every nucleosome start position), which is not solved.
    """
    results = []
    for row in job.eseq:
        if np.isnan(_start_positions(row, job.settings.nucbp)).all():
            results.append(None)
            continue
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


def _iter_tm_batches(dataset, regions, getters, batch_size, nworker):
    """
    Solve every molecule of `regions` once, in batches, possibly in parallel.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset holding the settings and binding energy.
    regions : iterable of str
        The regions to process.
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
        `(region, start, results)`, where `results` holds one tuple of
        getter results per molecule (None for a molecule with no data),
        starting at molecule `start`. Batches arrive in completion order
        when `nworker > 1`.

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
        for region in regions:
            settings = _region_settings(dataset, region)
            eseq = dataset.eseq[region]
            for start in range(0, dataset.nmol[region], batch_size):
                yield TMBatchJob(
                    region=region,
                    start=start,
                    settings=settings,
                    nbp=dataset.nbp[region],
                    eseq=np.array(eseq[start : start + batch_size]),
                    getters=getters,
                )

    for job, results in _process_map(_run_tm_batch, jobs(), nworker=nworker):
        yield job.region, job.start, results


def _tm_stats(
    dataset, regions, stats, *, batch_size=TM_BATCH_SIZE, nworker=1
):
    """
    Store several exact quantities, solving each molecule only once.

    Per-molecule quantities are stored as DataFrames, and per-bp and
    fixed-width ones as H5Arrays. Each batch is written as soon as it
    arrives, so peak memory scales with `2 * nworker * batch_size`
    molecules. Molecules with no data get nan.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[region][stat.name]`.
    regions : iterable of str
        The regions to process.
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
    regions = list(regions)
    stats = list(stats)
    vector = any(s.per_bp or s.ncol is not None for s in stats)
    tmp_dir = dataset.resolve_tmp_dir() if vector else None
    out = {}
    for region in regions:
        nmol, nbp = dataset.nmol[region], dataset.nbp[region]
        out[region] = [
            np.empty(nmol, dtype=np.float64)
            if s.width(nbp) is None
            else H5Array.create(
                (nmol, s.width(nbp)),
                dtype=np.float64,
                dir=tmp_dir,
                columns=s.columns,
            )
            for s in stats
        ]
    getters = [(s.getter, s.kwargs) for s in stats]
    batches = _iter_tm_batches(dataset, regions, getters, batch_size, nworker)
    for region, start, results in batches:
        stop = start + len(results)
        for i, (stat, arr) in enumerate(zip(stats, out[region])):
            width = stat.width(dataset.nbp[region])
            fill = np.nan if width is None else np.full(width, np.nan)
            block = np.array(
                [fill if r is None else r[i] for r in results],
                dtype=np.float64,
            )
            if width is None:
                arr[start:stop] = block
            else:
                arr.write_batch(start, stop, block)
    for region in regions:
        for stat, arr in zip(stats, out[region]):
            value = arr if isinstance(arr, H5Array) else pd.DataFrame(arr)
            dataset.analysis[region][stat.name] = value


def _tm_per_mol(dataset, regions, name, getter, *, nworker=1):
    """
    Store one exact value per molecule as a DataFrame.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[region][name]`.
    regions : iterable of str
        The regions to process.
    name : str
        The key of the result.
    getter : callable
        A `NucPosTM` method returning a float, such as
        `NucPosTM.getMeanN`.
    nworker : int, default 1
        The number of worker processes.
    """
    stats = [TMStat(name, getter.__name__)]
    _tm_stats(dataset, regions, stats, nworker=nworker)


def _tm_per_bp(dataset, regions, name, batch_size, getter, *, nworker=1):
    """
    Store one exact value per molecule and base pair as an H5Array.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[region][name]`.
    regions : iterable of str
        The regions to process.
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
        regions,
        [TMStat(name, getter.__name__, per_bp=True)],
        batch_size=batch_size,
        nworker=nworker,
    )


def _tm_configs(dataset, regions, name, get_config, *, kwargs=None, nworker=1):
    """
    Store one nucleosome configuration per molecule.

    For each region, writes `{name}_position_flat` and
    `{name}_position_offset` (the packed positions) and `{name}_energy`
    (the energy and log posterior probability of each configuration).
    Molecules with no data get no positions and nan energy.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[region]`.
    regions : iterable of str
        The regions to process.
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
    regions = list(regions)
    end_temp = SimSettings(**dict(dataset.settings)).end_temp
    getters = [(get_config.__name__, kwargs or {}), ("getLogZ", {})]
    positions = {region: [None] * dataset.nmol[region] for region in regions}
    energy = {region: np.empty(dataset.nmol[region]) for region in regions}
    log_z = {region: np.empty(dataset.nmol[region]) for region in regions}
    batches = _iter_tm_batches(
        dataset, regions, getters, TM_BATCH_SIZE, nworker
    )
    for region, start, results in batches:
        for mol, result in enumerate(results, start):
            if result is None:
                positions[region][mol] = np.zeros(0, dtype=np.int32)
                energy[region][mol] = log_z[region][mol] = np.nan
                continue
            (pos, e), lz = result
            positions[region][mol] = np.asarray(pos, dtype=np.int32)
            energy[region][mol] = e
            log_z[region][mol] = lz
    for region in regions:
        log_prob = -energy[region] / end_temp - log_z[region]
        flat, offset = _pack_jagged(positions[region])
        analysis = dataset.analysis[region]
        analysis[f"{name}_position_flat"] = pd.DataFrame({"position": flat})
        analysis[f"{name}_position_offset"] = pd.DataFrame({"offset": offset})
        analysis[f"{name}_energy"] = pd.DataFrame(
            {"energy": energy[region], "log_prob": log_prob}
        )
