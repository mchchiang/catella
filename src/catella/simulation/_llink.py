# _llink.py

import multiprocessing as mp
import warnings
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM
from scipy.special import logit

from catella.experiment._reads import _lookup_mask
from catella.simulation._inputs import _is_valid_llink, _resolve_settings

# Warn below this many molecules, as per-molecule estimates are noisy
_MIN_NMOL = 50

# Molecules per worker task; large enough that sending data is negligible
_CHUNK_NMOL = 200


def _resolve_llinks(llinks):
    """Validate candidate linker lengths and return them sorted and unique."""
    values = list(llinks)
    if not values:
        raise ValueError("llinks must not be empty")
    for value in values:
        if not _is_valid_llink(value):
            raise ValueError(
                f"llinks must be non-negative integers, got {value!r}"
            )
    return sorted({int(value) for value in values})


def _log_z(settings, llink, energy, nbp):
    """Log partition function of one fiber at temperature 1."""
    model = NucPosTM(
        settings.nucbp, nbp, llink, settings.elink, settings.mu, 1
    )
    model.setEnergy(energy, settings.emax)
    model.solve(1.0)
    return model.getLogZ()


def _chrom_rows(arr, keep, max_nmol, rng, batch_size):
    """
    Rows of one chromosome to use: kept, not all nan, and at most
    `max_nmol` of them, drawn at random.
    """
    usable = []
    for start in range(0, arr.shape[0], batch_size):
        stop = min(start + batch_size, arr.shape[0])
        block = np.asarray(arr[start:stop, :], dtype=np.float64)
        has_data = ~np.isnan(block).all(axis=1)
        if keep is not None:
            has_data &= keep[start:stop]
        usable.append(start + np.flatnonzero(has_data))
    rows = np.concatenate(usable) if usable else np.empty(0, dtype=np.int64)
    if max_nmol is not None and len(rows) > max_nmol:
        rows = np.sort(rng.choice(rows, size=max_nmol, replace=False))
    return rows


def _chrom_array(exp, chrom, prob_name):
    """Array of `prob_name` for one chromosome, as an array or H5Array."""
    arr = exp.analysis[chrom][prob_name]
    if isinstance(arr, pd.DataFrame):
        arr = arr.to_numpy()
    return arr


def _chunk_evidence(settings, candidates, block, log_z0):
    """
    Sum over the rows of `block` of log Z(data) - log Z(no data), one
    value per candidate.
    """
    nbp = block.shape[1]
    totals = np.zeros(len(candidates))
    for row in block:
        energy = logit(row).tolist()
        for i, ll in enumerate(candidates):
            totals[i] += _log_z(settings, ll, energy, nbp) - log_z0[i]
    return totals


def _chunk_jobs(exp, settings, candidates, prob_name, selected):
    """Yield `(chrom, block, log_z0)` for each chunk of selected rows."""
    for chrom, rows in selected.items():
        arr = _chrom_array(exp, chrom, prob_name)
        nbp = arr.shape[1]
        zero = [0.0] * nbp
        log_z0 = [_log_z(settings, ll, zero, nbp) for ll in candidates]
        for start in range(0, len(rows), _CHUNK_NMOL):
            chunk = rows[start : start + _CHUNK_NMOL]
            block = np.asarray(arr[chunk, :], dtype=np.float64)
            yield chrom, block, log_z0


def _run_chunks(jobs, settings, candidates, nworker):
    """Yield `(chrom, totals)` for each job, in submission order."""
    if nworker == 1:
        for chrom, block, log_z0 in jobs:
            yield chrom, _chunk_evidence(settings, candidates, block, log_z0)
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=nworker, mp_context=ctx) as pool:
        pending = deque()
        for chrom, block, log_z0 in jobs:
            future = pool.submit(
                _chunk_evidence, settings, candidates, block, log_z0
            )
            pending.append((chrom, future))
            if len(pending) >= 2 * nworker:
                done_chrom, done = pending.popleft()
                yield done_chrom, done.result()
        while pending:
            done_chrom, done = pending.popleft()
            yield done_chrom, done.result()


def _llink_evidence(
    exp,
    settings,
    *,
    llinks=range(0, 65, 5),
    prob_name="meth_prob",
    chroms=None,
    max_nmol=None,
    seed=None,
    batch_size=20000,
    mask_name=None,
    nworker=1,
):
    """
    Marginal likelihood of each candidate `llink`, per chromosome.

    The evidence of a chromosome is the sum over its molecules of
    log Z(data, llink) - log Z(no data, llink), at temperature 1. The
    second term is the normalization of the prior, which depends on
    `llink` and must not be dropped.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment; `prob_name` must be in `exp.analysis[chrom]`.
    settings : str or Path or SimSettings
        Provides `nucbp`, `elink`, `mu` and `emax`. `llink` is ignored.
    llinks : iterable of int, default range(0, 65, 5)
        Candidate linker lengths (bp).
    prob_name : str, default "meth_prob"
        The key to read from `exp.analysis[chrom]`.
    chroms : str or iterable of str, optional
        Chromosome(s) to include; all of `exp.chroms` if None.
    max_nmol : int, optional
        If given, use a random sample of at most this many molecules per
        chromosome.
    seed : int, optional
        Seed for the random sample of `max_nmol`.
    batch_size : int, default 20000
        Number of molecules read (and held in memory) per batch.
    mask_name : str, optional
        If given, excludes molecules dropped by
        `filter_dropout(mask_name=mask_name)` on the "test" source.
    nworker : int, default 1
        Number of parallel processes to spawn. The result does not
        depend on it.

    Returns
    -------
    pandas.DataFrame
        Columns `chrom`, `llink`, `evidence` and `nmol` (the number of
        molecules used), one row per chromosome and candidate.

    Raises
    ------
    ValueError
        If `llinks` is empty or holds a value that is not a non-negative
        integer, if `max_nmol` is not positive, if `nworker` is not a
        positive integer, or if a chromosome has no usable molecules.
    TypeError
        If `settings` is neither a path nor a SimSettings object.
    KeyError
        If `prob_name` or the mask of `mask_name` is missing.
    """
    settings = _resolve_settings(settings)
    candidates = _resolve_llinks(llinks)
    if max_nmol is not None and max_nmol <= 0:
        raise ValueError(f"max_nmol must be positive, got {max_nmol}")
    if (
        not isinstance(nworker, Integral)
        or isinstance(nworker, bool)
        or nworker < 1
    ):
        raise ValueError(
            f"nworker must be a positive integer, got {nworker!r}"
        )
    if chroms is None:
        chroms = exp.chroms
    elif isinstance(chroms, str):
        chroms = (chroms,)

    rng = np.random.default_rng(seed)
    selected = {}
    for chrom in chroms:
        arr = _chrom_array(exp, chrom, prob_name)
        keep = None
        if mask_name is not None:
            keep = np.asarray(
                _lookup_mask(exp, chrom, "test", mask_name), dtype=bool
            )
        rows = _chrom_rows(arr, keep, max_nmol, rng, batch_size)
        if len(rows) == 0:
            raise ValueError(f"chromosome {chrom} has no usable molecules")
        selected[chrom] = rows

    jobs = _chunk_jobs(exp, settings, candidates, prob_name, selected)
    totals = {chrom: np.zeros(len(candidates)) for chrom in selected}
    for chrom, part in _run_chunks(jobs, settings, candidates, nworker):
        totals[chrom] += part
    records = [
        {
            "chrom": chrom,
            "llink": ll,
            "evidence": float(totals[chrom][i]),
            "nmol": len(rows),
        }
        for chrom, rows in selected.items()
        for i, ll in enumerate(candidates)
    ]
    return pd.DataFrame(records)


def _estimate_llink(exp, settings, **kwargs):
    """
    Maximize `_llink_evidence` per chromosome; see `api.estimate_llink`.
    """
    settings = _resolve_settings(settings)
    if settings.end_temp != 1:
        warnings.warn(
            "settings.end_temp is ignored; llink evidence uses temperature 1",
            stacklevel=3,
        )
    table = _llink_evidence(exp, settings, **kwargs)
    largest = table["llink"].max()
    best = {}
    for chrom, sub in table.groupby("chrom", sort=False):
        pick = int(sub.loc[sub["evidence"].idxmax(), "llink"])
        best[chrom] = pick
        if pick == largest and sub["llink"].nunique() > 1:
            warnings.warn(
                f"chromosome {chrom}: best llink {pick} is the largest "
                "candidate; extend llinks",
                stacklevel=3,
            )
        nmol = int(sub["nmol"].iloc[0])
        if nmol < _MIN_NMOL:
            warnings.warn(
                f"chromosome {chrom}: llink estimated from {nmol} molecules; "
                "the estimate may be unreliable",
                stacklevel=3,
            )
    return best
