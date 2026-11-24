# _jobs.py

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from catella_cpp import Dump, NucPosMCMC
from scipy.special import logit

from catella.simulation.config import SimSettings


@dataclass(frozen=True, kw_only=True)
class MCMCJob:
    """
    Inputs for one MCMC run of a single molecule.

    Combines the physical parameters and annealing protocol
    (SimSettings) with the identifiers of the run. All fields are
    self-contained so an instance can be sent as-is to a freshly
    spawned worker process.
    """

    region: str
    """The name or identifier of the region being simulated."""

    mol: int
    """The index or identifier of the specific molecule within the
    region."""

    run: int
    """The iteration or replica index for this specific molecule-region
    pair."""

    seed: int
    """The specific random seed used for this execution. Note: This may differ
    from the batch seed if per-run offsets are applied.
    """

    seq_energy: bytes
    """Binary representation of the sequence-specific nucleosome binding
    energy derived from the methylation data."""

    out_type: Dump.OutputType
    """The format or scope of data to be recorded (e.g., Energy, Position,
    All)."""

    out_file: Path
    """The HDF5 file where the simulation results are stored."""

    settings: SimSettings
    """The physical constants and Monte Carlo protocol used for this run."""


@dataclass(frozen=True, kw_only=True)
class TMJob:
    """
    Inputs for the exact sampling of a single molecule.

    One transfer-matrix solve, followed by one exact sample per entry
    of `seeds`. All fields are self-contained so an instance can be
    sent as-is to a freshly spawned worker process.
    """

    region: str
    """The name or identifier of the region being simulated."""

    mol: int
    """The index or identifier of the specific molecule within the
    region."""

    seeds: tuple[int, ...]
    """One seed per sample; sample `i` is drawn after re-seeding the
    sampler with `seeds[i]`."""

    seq_energy: bytes
    """Binary representation of the sequence-specific nucleosome binding
    energy derived from the methylation data."""

    out_file: Path
    """The HDF5 file where all of the molecule's samples are stored."""

    settings: SimSettings
    """The physical constants used for this molecule."""


def _job_seed(master_seed, region_idx, mol, run):
    """
    Derive a deterministic, order-independent seed for a single run.

    Parameters
    ----------
    master_seed : int
        The batch-level seed the whole dataset was generated from.
    region_idx : int
        The index of the region within the dataset's region list.
    mol : int
        The molecule index.
    run : int
        The replicate run index.

    Returns
    -------
    int
        A seed depending only on its inputs, not on generation order.
    """
    return int(
        np.random.SeedSequence(
            entropy=master_seed, spawn_key=(region_idx, mol, run)
        ).generate_state(1)[0]
    )


def _resolve_master_seed(seed):
    """
    Return `seed`, or draw a fresh master seed if it is None.

    Parameters
    ----------
    seed : int or None
        The requested master seed.

    Returns
    -------
    int
        `seed`, or a random integer in [0, 2**63) that fits a signed
        64-bit HDF5 attribute.
    """
    if seed is None:
        return int(np.random.default_rng().integers(2**63))
    return seed


def _seed_table(seed, regions, nmol, nsim):
    """
    Precompute the seed of every run of every molecule.

    The table covers all molecules, not only selected ones, so it can be
    stored and used later to reproduce any single run.

    Parameters
    ----------
    seed : int
        The master seed.
    regions : sequence of str
        The regions, in dataset order.
    nmol : mapping of str to int
        The number of molecules of each region.
    nsim : int
        The number of runs per molecule.

    Returns
    -------
    dict of str to np.ndarray
        A uint32 array of shape `(nmol[region], nsim)` for each region,
        where entry `[mol, run]` is `_job_seed(seed, region_idx, mol, run)`.
    """
    table = {}
    for region_idx, region in enumerate(regions):
        arr = np.empty((nmol[region], nsim), dtype=np.uint32)
        for mol in range(nmol[region]):
            for run in range(nsim):
                arr[mol, run] = _job_seed(seed, region_idx, mol, run)
        table[region] = arr
    return table


def _check_seed_source(dataset, seed):
    """
    Check that rerun seeds can be resolved for `dataset`.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to rerun.
    seed : int or None
        The new master seed, or None to reuse the stored seeds.

    Raises
    ------
    ValueError
        If `seed` is None and `dataset` has no stored seed table.
    """
    if seed is None and dataset.seed_table is None:
        raise ValueError(
            "dataset predates seed persistence; pass seed= explicitly"
        )


def _run_seeds(dataset, region, mol, runs, seed=None):
    """
    Return the seeds of the given runs of one molecule.

    Parameters
    ----------
    dataset : SimDataset
        The dataset the runs belong to.
    region : str
        The region of the molecule.
    mol : int
        The molecule index.
    runs : iterable of int
        The run indices.
    seed : int, optional
        A new master seed to derive fresh seeds from. If None (default),
        the seeds stored in `dataset.seed_table` are used.

    Returns
    -------
    tuple of int
        One seed per entry of `runs`.
    """
    if seed is None:
        return tuple(int(dataset.seed_table[region][mol, run]) for run in runs)
    region_idx = dataset.regions.index(region)
    return tuple(_job_seed(seed, region_idx, mol, run) for run in runs)


def _eseq_bytes(meth_prob_row):
    """
    Encode one molecule's logit methylation probability for a job.

    Parameters
    ----------
    meth_prob_row : np.ndarray
        The methylation probability of each base pair of the molecule.

    Returns
    -------
    bytes
        The float64 buffer of `logit(meth_prob_row)`.
    """
    return np.asarray(logit(meth_prob_row), dtype=np.float64).tobytes()


def _compute_eseq(settings, meth_prob, regions):
    """
    Compute the sequence-specific nucleosome binding energy of molecules.

    Parameters
    ----------
    settings : SimSettings
        The physical constants used to build the energy model.
    meth_prob : mapping of str to np.ndarray
        The 2D methylation probability (molecules by base pairs) of each
        region.
    regions : iterable of str
        The regions to process.

    Returns
    -------
    dict of str to np.ndarray
        The energy of each molecule and base pair, with the same shape as
        `meth_prob[region]`.
    """
    eseq = {}
    for region in regions:
        prob = meth_prob[region]
        model = NucPosMCMC(
            settings.nucbp,
            prob.shape[1],
            settings.for_region(region).llink,
            settings.elink,
            settings.mu,
            0,
        )
        eseq[region] = np.empty(prob.shape)
        for i in range(prob.shape[0]):
            model.setEnergy(logit(prob[i]), settings.emax)
            eseq[region][i] = model.getSeqEnergy()
    return eseq


def _iter_selected(regions, nmol, mols):
    """
    Yield the selected molecules of each region.

    Parameters
    ----------
    regions : iterable of str
        The regions, in order.
    nmol : mapping of str to int
        The number of molecules of each region.
    mols : mapping of str to int, slice, or sequence of int
        The molecule selection of each region.

    Yields
    ------
    tuple of (str, int)
        The region and index of each selected molecule.
    """
    for region in regions:
        for mol in np.arange(nmol[region])[mols[region]].ravel():
            yield region, int(mol)


def _iter_mcmc_jobs(
    dataset, settings, meth_prob, targets, out_type, seed=None
):
    """
    Yield one MCMC job per target run.

    Parameters
    ----------
    dataset : SimDataset
        The dataset the runs belong to; gives output files and seeds.
    settings : SimSettings
        The settings of the runs, resolved per region.
    meth_prob : mapping of str to np.ndarray
        The 2D methylation probability of each region.
    targets : iterable of (str, int, int)
        The `(region, mol, run)` of each run.
    out_type : Dump.OutputType
        The data to record.
    seed : int, optional
        A new master seed. If None (default), stored seeds are reused;
        see `_run_seeds`.

    Yields
    ------
    MCMCJob
        The job of each target, in order.
    """
    last_mol = eseq = None
    for region, mol, run in targets:
        if (region, mol) != last_mol:
            eseq = _eseq_bytes(meth_prob[region][mol, :])
            last_mol = (region, mol)
        yield MCMCJob(
            region=region,
            mol=mol,
            run=run,
            settings=settings.for_region(region),
            seed=_run_seeds(dataset, region, mol, [run], seed)[0],
            seq_energy=eseq,
            out_type=out_type,
            out_file=dataset.mcmc_file(region, mol, run),
        )


def _iter_tm_jobs(dataset, settings, meth_prob, targets, seed=None):
    """
    Yield one TM job per target molecule, drawing `dataset.nsim` samples.

    Parameters
    ----------
    dataset : SimDataset
        The dataset the molecules belong to; gives output files and seeds.
    settings : SimSettings
        The settings of the molecules, resolved per region.
    meth_prob : mapping of str to np.ndarray
        The 2D methylation probability of each region.
    targets : iterable of (str, int)
        The `(region, mol)` of each molecule.
    seed : int, optional
        A new master seed. If None (default), stored seeds are reused;
        see `_run_seeds`.

    Yields
    ------
    TMJob
        The job of each target, in order.
    """
    runs = range(dataset.nsim)
    for region, mol in targets:
        yield TMJob(
            region=region,
            mol=mol,
            seeds=_run_seeds(dataset, region, mol, runs, seed),
            seq_energy=_eseq_bytes(meth_prob[region][mol, :]),
            out_file=dataset.tm_file(region, mol),
            settings=settings.for_region(region),
        )


def _mcmc_rerun_targets(dataset, only):
    """
    Return the MCMC runs to rerun.

    Parameters
    ----------
    dataset : SimDataset
        The MCMC dataset.
    only : iterable of (str, int, int) or None
        The runs to rerun. If None, all missing, corrupted, and truncated
        runs found by `dataset.find_incomplete_runs` are used.

    Returns
    -------
    list of (str, int, int)
        The `(region, mol, run)` of each run to rerun.
    """
    if only is not None:
        return list(only)
    incomplete = dataset.find_incomplete_runs()
    return (
        incomplete["missing"]
        + incomplete["corrupted"]
        + incomplete["truncated"]
    )


def _tm_rerun_targets(dataset, only):
    """
    Return the TM molecules whose samples to regenerate.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset.
    only : iterable of (str, int) or None
        The molecules to regenerate. If None, every molecule with a
        missing or corrupted sample found by `dataset.find_incomplete_runs`
        is used, once each and sorted.

    Returns
    -------
    list of (str, int)
        The `(region, mol)` of each molecule to regenerate.

    Raises
    ------
    ValueError
        If `only` contains an entry that is not a `(region, mol)` pair.
    """
    if only is None:
        incomplete = dataset.find_incomplete_runs()
        triplets = incomplete["missing"] + incomplete["corrupted"]
        return sorted({(region, mol) for region, mol, _ in triplets})
    targets = list(only)
    for target in targets:
        if len(target) != 2:
            raise ValueError(
                "for a 'tm' dataset, 'only' must contain (region, mol) "
                "pairs, not (region, mol, run) triplets; TM samples are "
                "regenerated at molecule granularity"
            )
    return targets
