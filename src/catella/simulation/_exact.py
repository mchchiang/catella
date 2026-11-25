# _exact.py

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM

from catella._jagged import _interval_mask, _pack_jagged
from catella.h5_array import H5Array
from catella.simulation._inputs import (
    _extent,
    _set_energy,
    _start_positions,
)
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

    edata: np.ndarray
    """The data energy of each molecule, shape (nmol_batch, nbp)."""

    getters: tuple[tuple[str | Callable, dict], ...]
    """The `(getter, kwargs)` of each getter to call: a `NucPosTM`
    method name, or a function `(tm, intervals, settings, **kwargs)`."""

    spanned: tuple[np.ndarray, np.ndarray] | None = None
    """The `(intervals, offset)` of the batch's molecules, with offsets
    from 0; None if every base pair is spanned."""

    occupy_read_ends: bool = True
    """If False, no nucleosome may leave a molecule's extent."""


@dataclass(frozen=True)
class TMStat:
    """
    A quantity computed from each solved transfer-matrix model.
    """

    name: str
    """The key of the result in `dataset.analysis`."""

    getter: str | Callable
    """The name of the `NucPosTM` method that returns the value, or a
    function `(tm, intervals, settings, **kwargs)`, where `intervals`
    holds the molecule's spanned intervals (None if every base pair is
    spanned) and `settings` the region's settings. A function must be
    defined at module level, so worker processes can import it."""

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


def _require_edata(dataset):
    """
    Check that `dataset` stores the data energy of each molecule.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to check.

    Raises
    ------
    ValueError
        If the dataset has no stored energy landscape.
    """
    if not dataset.has_edata:
        raise ValueError(
            "exact computation requires the per-molecule energy "
            "landscape; recreate the dataset with store_edata=True"
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
    _require_edata(dataset)


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


def _build_tm(settings, nbp, edata, extent=None):
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
    edata : np.ndarray
        The data energy of each base pair.
    extent : tuple of int, optional
        If given, no nucleosome may leave `[lo, hi)` (see
        `_set_energy`).

    Returns
    -------
    NucPosTM
        The solved model.
    """
    tm = NucPosTM(
        settings.lnuc,
        nbp,
        settings.llink,
        settings.elink,
        settings.mu,
        1,
    )
    _set_energy(tm, edata, settings.emax, settings.lnuc, extent)
    tm.solve(settings.end_temp)
    return tm


def _solved_tm(dataset, region, mol):
    """
    Build and solve the transfer matrix of one molecule of a dataset.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the settings and data energy.
    region : str
        The region of the molecule.
    mol : int
        The molecule index.

    Returns
    -------
    NucPosTM
        The solved model.
    """
    extent = None
    if not dataset.occupy_read_ends:
        extent = _extent(dataset.spanned[region], mol)
    return _build_tm(
        _region_settings(dataset, region),
        dataset.nbp[region],
        dataset.edata[region][mol],
        extent,
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
    for i, row in enumerate(job.edata):
        if np.isnan(_start_positions(row, job.settings.lnuc)).all():
            results.append(None)
            continue
        intervals = None
        if job.spanned is not None:
            spanned, offset = job.spanned
            intervals = spanned[offset[i] : offset[i + 1]]
        extent = None
        if not job.occupy_read_ends and job.spanned is not None:
            extent = _extent(job.spanned, i)
        tm = _build_tm(job.settings, job.nbp, row, extent)
        results.append(
            tuple(
                _lists_to_arrays(
                    getter(tm, intervals, job.settings, **kwargs)
                    if callable(getter)
                    else getattr(tm, getter)(**kwargs)
                )
                for getter, kwargs in job.getters
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


def _batch_intervals(spanned, start, stop):
    """
    Return the spanned intervals of molecules `start:stop`.

    Parameters
    ----------
    spanned : tuple of np.ndarray or None
        `(intervals, offset)` of a region (see `SimDataset.spanned`).
    start, stop : int
        The molecule range.

    Returns
    -------
    tuple of np.ndarray or None
        `(intervals, offset)` of the molecules, with offsets from 0, or
        None if `spanned` is None.
    """
    if spanned is None:
        return None
    intervals, offset = spanned
    lo, hi = offset[start], offset[stop]
    return np.array(intervals[lo:hi]), offset[start : stop + 1] - lo


def _nan_unspanned(block, spanned, start, stop):
    """
    Set the values of `block` at unspanned base pairs to nan, in place.

    Parameters
    ----------
    block : np.ndarray
        float, (stop - start, nbp), per-bp values of molecules
        `start:stop`.
    spanned : tuple of np.ndarray or None
        `(intervals, offset)` of the region (see `SimDataset.spanned`);
        nothing is changed if None.
    start, stop : int
        The molecule range of `block`.
    """
    if spanned is not None:
        mask = _interval_mask(*spanned, start, stop, block.shape[1])
        block[~mask] = np.nan


def _spanned_rows(spanned, mol_ids, nbp):
    """
    Spanned base pairs of the molecules `mol_ids`, in their order.

    Parameters
    ----------
    spanned : tuple of np.ndarray or None
        `(intervals, offset)` of the region (see `SimDataset.spanned`).
    mol_ids : sequence of int
        The molecules, in any order.
    nbp : int
        The number of base pairs.

    Returns
    -------
    np.ndarray or None
        bool, (len(mol_ids), nbp), or None if `spanned` is None.
    """
    if spanned is None:
        return None
    intervals, offset = spanned
    mol_ids = np.asarray(mol_ids, dtype=np.int64)
    counts = offset[mol_ids + 1] - offset[mol_ids]
    rows = np.concatenate(
        [np.arange(offset[m], offset[m + 1]) for m in mol_ids]
        + [np.zeros(0, dtype=np.int64)]
    )
    sub_offset = np.zeros(len(mol_ids) + 1, dtype=np.int64)
    np.cumsum(counts, out=sub_offset[1:])
    return _interval_mask(intervals[rows], sub_offset, 0, len(mol_ids), nbp)


def _spanned_fraction(intervals, nbp, lnuc):
    """
    Spanned share of the footprint of a nucleosome at each start position.

    Parameters
    ----------
    intervals : np.ndarray
        int, (n, 2), the molecule's spanned intervals.
    nbp : int
        The number of base pairs.
    lnuc : int
        The nucleosome footprint size.

    Returns
    -------
    np.ndarray
        float64, (nbp - lnuc + 1,): spanned bp in `[x, x + lnuc)`
        divided by `lnuc`.
    """
    edges = np.zeros(nbp + 1, dtype=np.int64)
    np.add.at(edges, intervals[:, 0], 1)
    np.add.at(edges, intervals[:, 1], -1)
    cum = np.zeros(nbp + 1, dtype=np.int64)
    np.cumsum(np.cumsum(edges)[:nbp] > 0, out=cum[1:])
    npos = nbp - lnuc + 1
    return (cum[lnuc : lnuc + npos] - cum[:npos]) / lnuc


def _mean_nnuc(tm, intervals, settings):
    """
    Mean nucleosome count on spanned DNA (a `TMStat` getter).

    Each nucleosome counts as the spanned share of its footprint; the
    plain mean `getMeanN` if every base pair is spanned.
    """
    if intervals is None:
        return tm.getMeanN()
    start_prob = np.asarray(tm.getStartProb())
    nbp = len(tm.getDataEnergy())
    return float(start_prob @ _spanned_fraction(intervals, nbp, settings.lnuc))


def _var_nnuc(tm, intervals, settings, dmu=1e-4):
    """
    Variance of the nucleosome count on spanned DNA (a `TMStat` getter).

    The count is as in `_mean_nnuc`; its variance is the second
    difference of log Z in a shift of the energy of each start position
    by its spanned share times `dmu` (as `getVarN` shifts `mu`); the
    plain `getVarN` if every base pair is spanned.
    """
    if intervals is None:
        return tm.getVarN(dmu)
    edata = np.asarray(tm.getDataEnergy(), dtype=np.float64)
    nbp = len(edata)
    frac = _spanned_fraction(intervals, nbp, settings.lnuc)
    shift = np.zeros(nbp)
    shift[: len(frac)] = frac * dmu

    def log_z(sign):
        model = NucPosTM(
            settings.lnuc,
            nbp,
            settings.llink,
            settings.elink,
            settings.mu,
            1,
        )
        # Energies are already clamped; inf keeps the shift unclamped
        model.setEnergy((edata - sign * shift).tolist(), np.inf)
        model.solve(settings.end_temp)
        return model.getLogZ()

    temp = settings.end_temp
    d2 = (log_z(1.0) - 2.0 * tm.getLogZ() + log_z(-1.0)) / dmu**2
    return temp * temp * d2


def _linker_hist(tm, intervals, settings, lmax, xlo, xhi, trim):
    """
    Linker histogram within the spanned intervals (a `TMStat` getter).

    Sums `getLinkerHist` over each spanned interval intersected with the
    window `[xlo, xhi)` (already trimmed), after removing `trim` bp
    from each interval end; intervals too short for two nucleosomes
    add nothing. The plain `getLinkerHist` if every base pair is
    spanned.
    """
    if intervals is None:
        return tm.getLinkerHist(lmax, xlo, xhi)
    total = np.zeros(lmax + 2)
    for a, b in intervals:
        lo, hi = max(int(a) + trim, xlo), min(int(b) - trim, xhi)
        if hi - lo >= 2 * settings.lnuc:
            total += np.asarray(tm.getLinkerHist(lmax, lo, hi))
    return total


def _iter_tm_batches(dataset, regions, getters, batch_size, nworker):
    """
    Solve every molecule of `regions` once, in batches, possibly in parallel.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset holding the settings and data energy.
    regions : iterable of str
        The regions to process.
    getters : sequence of (str or callable, dict)
        The `(getter, kwargs)` of each getter to call (see
        `TMBatchJob.getters`).
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
    getters = tuple((getter, dict(kwargs)) for getter, kwargs in getters)

    def jobs():
        for region in regions:
            settings = _region_settings(dataset, region)
            edata = dataset.edata[region]
            spanned = dataset.spanned[region]
            for start in range(0, dataset.nmol[region], batch_size):
                stop = min(start + batch_size, dataset.nmol[region])
                yield TMBatchJob(
                    region=region,
                    start=start,
                    settings=settings,
                    nbp=dataset.nbp[region],
                    spanned=_batch_intervals(spanned, start, stop),
                    occupy_read_ends=dataset.occupy_read_ends,
                    edata=np.array(edata[start : start + batch_size]),
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
    molecules. Molecules with no data get nan, and per-bp quantities are
    nan at unspanned base pairs.

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
            if stat.per_bp:
                _nan_unspanned(block, dataset.spanned[region], start, stop)
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
