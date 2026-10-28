# engine.py

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, replace
from itertools import islice
from pathlib import Path
from typing import Any, ClassVar

import catella_cpp as sim
import numpy as np
import pandas as pd
from catella_cpp import Dump, NucPosMCMC, NucPosTM
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeRemainingColumn,
)
from scipy.special import logit

from catella import utils
from catella.simulation.config import SimSettings
from catella.simulation.results import (
    SimDataset,
    _consolidate_tm_samples,
    _write_tm_samples,
)
from catella.utils import IndexType


# Store parameters for a specific simulation run
@dataclass(frozen=True, kw_only=True)
class MCMCJob:
    """
    Inputs for one MCMC run of a single molecule.

    Combines the physical parameters and annealing protocol
    (SimSettings) with the identifiers of the run. All fields are
    self-contained so an instance can be sent as-is to a freshly
    spawned worker process.
    """

    chrom: str
    """The name or identifier of the chromosome being simulated."""

    mol: int
    """The index or identifier of the specific molecule within the
    chromosome."""

    run: int
    """The iteration or replica index for this specific molecule-chromosome
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

    _out_map: ClassVar[dict[str, Dump.OutputType]] = {
        "energy": Dump.OutputType.Energy,
        "position": Dump.OutputType.Position,
        "temp": Dump.OutputType.Temp,
        "all": Dump.OutputType.All,
    }


@dataclass(frozen=True, kw_only=True)
class TMJob:
    """
    Inputs for the exact sampling of a single molecule.

    One transfer-matrix solve, followed by one exact sample per entry
    of `seeds`. All fields are self-contained so an instance can be
    sent as-is to a freshly spawned worker process.
    """

    chrom: str
    """The name or identifier of the chromosome being simulated."""

    mol: int
    """The index or identifier of the specific molecule within the
    chromosome."""

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


class SimManager:
    """
    Orchestrator for managing and executing nucleosome positioning simulations.

    The SimManager handles the setup, parallel execution, and data collection
    of Monte Carlo simulations across multiple chromosomes and molecules.

    Attributes
    ----------
    nworker : int, default 1
        Number of parallel processes to use.
    verbose : bool, default True
        If True, display a progress bar when running simulations.

    Raises
    ------
    ValueError
        If the number of parallel processes `nworker` is invalid.
    """

    def __init__(self, *, nworker: int = 1, verbose: bool = True):

        if nworker <= 0:
            raise ValueError("nworker must be greater than zero")

        self.nworker = nworker
        self.verbose = verbose

    @staticmethod
    def _job_seed(master_seed: int, chrom_idx: int, mol: int, run: int) -> int:
        """
        Derive a deterministic, order-independent seed for a single run.

        Parameters
        ----------
        master_seed : int
            The batch-level seed the whole dataset was generated from.
        chrom_idx : int
            The index of the chromosome within the dataset's chromosome
            list.
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
                entropy=master_seed, spawn_key=(chrom_idx, mol, run)
            ).generate_state(1)[0]
        )

    def run(
        self,
        *,
        chroms: str | Iterable[str],
        nsim: int,
        settings: str | Path | SimSettings,
        meth_prob: np.ndarray | Mapping[str, np.ndarray | pd.DataFrame],
        out_dir: str | Path,
        dataset_name: str = "results",
        method: str = "mcmc",
        out_types: str | Iterable[str] = "all",
        seed: int | None = None,
        mols: IndexType | Mapping[str, IndexType] = slice(None),
        store_eseq: bool = True,
        use_median_eseq_mu: bool = False,
    ) -> SimDataset:
        """
        Execute simulations in parallel across chromosomes and molecules.

        This method generates simulation parameters, spawns a process pool,
        and tracks progress as results are written to disk.

        Parameters
        ----------
        chroms : str or iterable of str
            The identifier(s) of chromosome(s) to include in the simulation.
        nsim : int
            Number of independent simulation runs per molecule
            (`method='mcmc'`), or exact samples per molecule
            (`method='tm'`; may be 0, since the exact occupancy/mean-N/
            variance quantities require no samples at all).
        settings : str or Path or SimSettings
            The physical constants and Monte Carlo protocol (e.g., nucbp,
            llink, mu) to apply to all runs in this batch. These can be
            read from a configuration file.
        meth_prob : np.ndarray or dict
            The probability of methylation. If multiple chromosomes are
            provided, this must be a mapping of {chrom_name: data}. Data can
            be NumPy arrays or Pandas DataFrames.
        out_dir : str or Path
            Directory where the simulation dataset will be initialized. The
            raw simulation data will be stored with a sub-directory called
            `raw_data`.
        dataset_name : str, default = "results"
            Name of the dataset HDF5 file storing the metadata and analysis of
            the simulation results. The file directory of this HDF5 file is
            `out_dir/{dataset_name}.h5` (e.g., `out_dir/results.h5`).
        method : {"mcmc", "tm"}, default "mcmc"
            Which engine to run. For 'tm', `settings.end_temp` is the
            temperature the exact equilibrium distribution is evaluated
            at; `settings.start_temp`/`cool_option`/`nsweep`/
            `print_freq` and `out_types` are ignored.
        out_types : str or iterable of str, default 'all'
            Types of data to record. Options include 'energy', 'position',
            'temp', or 'all'. Ignored for `method='tm'`.
        seed : int, optional
            Master seed used to generate independent seeds for each run. If
            None, a high-entropy seed is automatically generated using the
            NumPy default random generator.
        mols : IndexType or dict, default slice(None)
            Indices of molecules to simulate for each chromosome. Can be a
            single index/slice or a mapping of {chrom_name: indices}.
        store_eseq : bool, default True
            Whether to store the sequence-specific nucleosome binding energy
            landscape derived from the methylation data to output dataset file.
        use_median_eseq_mu : bool : default False
            Whether to modify the chemical potential parameter so that it is
            equal to the median of the sequence-specific nucleosome binding
            energy.

        Returns
        -------
        SimDataset
            A dataset object providing access to the newly generated results.

        Raises
        ------
        TypeError
            If input types for `chroms`, `meth_prob`, or `mols` are
            inconsistent.
        ValueError
            If the requested `out_types` are not recognized.
        FileExistsError
            If the specified `out_dir` already contains an existing dataset.
        """

        if method not in ("mcmc", "tm"):
            raise ValueError("'method' must be 'mcmc' or 'tm'")

        chroms = utils.normalize_chroms(chroms)

        # Normalize methylation data
        if isinstance(meth_prob, np.ndarray):
            if not len(chroms) == 1:
                raise TypeError(
                    "methylation probability array 'meth_prob' "
                    "provided but multiple chroms requested"
                )
            meth_prob = {chroms[0]: meth_prob}
        elif isinstance(meth_prob, Mapping):
            meth_prob = {chrom: meth_prob[chrom] for chrom in chroms}
        else:
            raise TypeError("'meth_prob' must be a numpy array or a mapping")

        # Normalize molecule indices (unparameterized base types, since
        # isinstance() rejects IndexType.__args__'s Sequence[int])
        if isinstance(mols, (int, slice, Sequence)):
            # Apply the same set of indices across all chromosomes
            mols = {chrom: mols for chrom in chroms}
        elif isinstance(mols, Mapping):
            mols = {chrom: mols[chrom] for chrom in chroms}
        else:
            raise TypeError("'mols' must be an IndexType or a mapping")

        # Check that the molecule indices are valid and compute total number
        # of molecules to simulate
        nmol = {}
        nbp = {}
        total_mol = 0
        for chrom in chroms:
            # Determine the maximum number of molecules and all molecule ids
            size = meth_prob[chrom].shape[0]
            nmol[chrom] = size

            # Convert any data frames to numpy arrays in meth_prob
            if isinstance(meth_prob[chrom], pd.DataFrame):
                meth_prob[chrom] = meth_prob[chrom].to_numpy()
            meth_prob[chrom] = np.atleast_2d(meth_prob[chrom])
            nbp[chrom] = meth_prob[chrom].shape[1]

            try:
                # Apply the index/slice to an 0-element view to trigger errors
                # without allocating memory for the full index list
                text_idx = np.empty(size, dtype=np.int8)[mols[chrom]]
                if text_idx.size == 0:
                    raise ValueError(
                        f"selection 'mols' for chromosome '{chrom}' is empty"
                    )
                total_mol += text_idx.size
            except IndexError as e:
                raise IndexError(
                    f"mols index for chromosome '{chrom}': {e}"
                ) from None

        # Read simulation settings from file if needed
        if isinstance(settings, (str | Path)):
            settings = SimSettings.load(settings)
        elif not isinstance(settings, SimSettings):
            raise TypeError(
                "'settings' must be either a path to the "
                "simulation setting file or a SimSettings object"
            )

        if method == "mcmc":
            # Check the MCMC-only protocol fields are present and valid
            # (they are optional on SimSettings so 'tm' configs can omit
            # them, so 'mcmc' must check for their presence itself).
            for field_name in (
                "start_temp", "cool_option", "nsweep", "print_freq"
            ):
                if getattr(settings, field_name) is None:
                    raise ValueError(
                        f"'{field_name}' is required for method='mcmc'"
                    )
            if settings.end_temp > settings.start_temp:
                raise ValueError("expect 'end_temp' <= 'start_temp'")
            if settings.cool_option not in SimSettings._cool_map:
                raise KeyError(
                    "invalid value for 'cool_option'; expect either "
                    "'linear', 'geometric', or 'constant'"
                )

            # Combine output types
            if isinstance(out_types, str):
                if out_types not in MCMCJob._out_map:
                    raise ValueError(
                        f"output type '{out_types}' is not a valid option"
                    )
                out_types = [out_types]
            elif not isinstance(out_types, Iterable):
                raise TypeError("'out_types' must be a str or an iterable")

            out_type = MCMCJob._out_map[out_types[0]]
            for i in range(1, len(out_types)):
                if out_types[i] not in MCMCJob._out_map:
                    raise ValueError(
                        f"output type '{out_types[i]}' is not a valid "
                        "option"
                    )
                otype = MCMCJob._out_map[out_types[i]]
                # Dump.OutputType is a scoped C++ enum class, so pybind11
                # doesn't bind '|' between two enum values directly;
                # combine via their underlying int values instead.
                out_type = Dump.OutputType(int(out_type) | int(otype))
        else:  # method == "tm"
            out_type = None

        # Compute the methylation energy landscape (also needed to compute
        # the median-based chemical potential when it is not stored)
        eseq = None
        if store_eseq or use_median_eseq_mu:
            eseq = {}
            for chrom in chroms:
                model = NucPosMCMC(
                    settings.nucbp,
                    nbp[chrom],
                    settings.llink,
                    settings.elink,
                    settings.mu,
                    0,
                )
                eseq[chrom] = np.empty(meth_prob[chrom].shape)
                for i in range(meth_prob[chrom].shape[0]):
                    model.setEnergy(logit(meth_prob[chrom][i]), settings.emax)
                    eseq[chrom][i] = model.getSeqEnergy()

        # Adjust the chemical potential if needed
        if use_median_eseq_mu:
            avg_eseq = np.empty(len(chroms))
            for i, chrom in enumerate(chroms):
                avg_eseq[i] = np.nanmedian(eseq[chrom])
            avg_eseq = np.mean(avg_eseq)
            print(f"Using median eseq mu: {avg_eseq}")
            settings = replace(settings, mu=avg_eseq)

        # Resolve the master seed and precompute a full per-run seed table
        # (covers every (chrom, mol, run) combination, independent of any
        # 'mols' filtering, so it can be persisted and later used to
        # reproduce or rerun any individual simulation exactly)
        if seed is None:
            rng = np.random.default_rng()
            seed = int(rng.bit_generator.seed_seq.entropy)
        seed_table = {}
        for chrom_idx, chrom in enumerate(chroms):
            arr = np.empty((nmol[chrom], nsim), dtype=np.uint32)
            for mol in range(nmol[chrom]):
                for run in range(nsim):
                    arr[mol, run] = SimManager._job_seed(
                        seed, chrom_idx, mol, run
                    )
            seed_table[chrom] = arr

        # Prepare the dataset object and save it immediately so a valid,
        # loadable manifest exists on disk even if the run is interrupted
        dataset = SimDataset.create(
            chroms=chroms,
            nmol=nmol,
            nsim=nsim,
            nbp=nbp,
            settings=settings,
            eseq=eseq if store_eseq else None,
            out_dir=out_dir,
            dataset_name=dataset_name,
            method=method,
            out_type=out_type,
            seed=seed,
            seed_table=seed_table,
        )
        dataset.save()

        # Generate the jobs: one per run for MCMC, one per molecule for TM
        def molecules():
            for chrom in chroms:
                molidxs = np.arange(nmol[chrom])[mols[chrom]]
                for molidx in molidxs:
                    eseq = np.asarray(
                        logit(meth_prob[chrom][molidx, :]), dtype=np.float64
                    ).tobytes()
                    yield chrom, int(molidx), eseq

        def mcmc_jobs():
            for chrom, idx, eseq in molecules():
                for run in range(nsim):
                    yield MCMCJob(
                        chrom=chrom,
                        mol=idx,
                        run=run,
                        settings=settings,
                        seed=int(seed_table[chrom][idx, run]),
                        seq_energy=eseq,
                        out_type=out_type,
                        out_file=dataset.mcmc_file(chrom, idx, run),
                    )

        def tm_jobs():
            for chrom, idx, eseq in molecules():
                yield TMJob(
                    chrom=chrom,
                    mol=idx,
                    seeds=tuple(int(s) for s in seed_table[chrom][idx, :]),
                    seq_energy=eseq,
                    out_file=dataset.tm_file(chrom, idx),
                    settings=settings,
                )

        if method == "mcmc":
            self._dispatch(
                dataset,
                mcmc_jobs(),
                total_mol * nsim,
                SimManager._run_mcmc_job,
            )
        elif nsim > 0:
            self._dispatch(
                dataset,
                tm_jobs(),
                total_mol,
                SimManager._run_tm_job,
                progress_desc="[cyan]Solving molecules ...",
            )

        # Save the dataset to file
        dataset.save()

        return dataset

    def rerun(
        self,
        dataset: SimDataset,
        *,
        meth_prob: np.ndarray | Mapping[str, np.ndarray | pd.DataFrame],
        only: Iterable[tuple[str, int, int]] | None = None,
        seed: int | None = None,
        out_type: Any | None = None,
    ) -> SimDataset:
        """
        Rerun specific simulations, by default the incomplete ones.

        Parameters
        ----------
        dataset : SimDataset
            The dataset to rerun simulations for. Modified in place.
        meth_prob : np.ndarray or dict
            The same methylation probability data originally passed to
            `run()`. This is never persisted in `dataset`, so it must be
            re-supplied here; given the same data, reruns reproduce the
            exact original simulation by default.
        only : iterable of (str, int, int) or (str, int), optional
            For `dataset.method == 'mcmc'`: the specific `(chrom, mol,
            run)` triplets to rerun. For `'tm'`: `(chrom, mol)` pairs --
            TM samples live in one combined file per molecule, so
            individual samples cannot be regenerated in isolation;
            passing a 3-tuple for a TM dataset raises `ValueError`. If
            None (default), `dataset.find_incomplete_runs()` is used
            (for 'tm', reduced to the distinct incomplete molecules).
        seed : int, optional
            If None (default), each rerun reproduces the exact original
            seed via `dataset.seed_table`. If given, reruns instead use
            fresh seeds derived from this new master seed -- a
            deliberately different simulation, not a reproduction.
        out_type : Dump.OutputType, optional
            The output type to use. If None (default), `dataset.out_type`
            is used. Required if the dataset predates `out_type` being
            persisted. Ignored for `dataset.method == 'tm'`.

        Returns
        -------
        SimDataset
            The same `dataset`, updated in place.

        Raises
        ------
        ValueError
            If `out_type` is not given and `dataset.out_type` is None, or
            if `seed` is not given and `dataset.seed_table` is None --
            both indicate the dataset predates that being persisted. Also
            raised if `only` contains a 3-tuple for a `'tm'` dataset.
        """
        if dataset.method == "tm":
            return self._rerun_tm(
                dataset, meth_prob=meth_prob, only=only, seed=seed
            )
        return self._rerun_mcmc(
            dataset,
            meth_prob=meth_prob,
            only=only,
            seed=seed,
            out_type=out_type,
        )

    def _rerun_mcmc(
        self,
        dataset: SimDataset,
        *,
        meth_prob: np.ndarray | Mapping[str, np.ndarray | pd.DataFrame],
        only: Iterable[tuple[str, int, int]] | None,
        seed: int | None,
        out_type: Any | None,
    ) -> SimDataset:
        """Rerun an MCMC dataset's runs, at run granularity (see
        `rerun`'s docstring)."""
        resolved_out_type = (
            out_type if out_type is not None else dataset.out_type
        )
        if resolved_out_type is None:
            raise ValueError(
                "dataset predates out_type persistence; "
                "pass out_type= explicitly"
            )

        if only is None:
            incomplete = dataset.find_incomplete_runs()
            targets = (
                incomplete["missing"]
                + incomplete["corrupted"]
                + incomplete["truncated"]
            )
        else:
            targets = list(only)

        if seed is None:
            if dataset.seed_table is None:
                raise ValueError(
                    "dataset predates seed persistence; pass seed= explicitly"
                )
            resolved_seeds = {
                (chrom, mol, run): int(dataset.seed_table[chrom][mol, run])
                for chrom, mol, run in targets
            }
        else:
            # Caller explicitly wants different (not reproduced) seeds.
            resolved_seeds = {
                (chrom, mol, run): SimManager._job_seed(
                    seed, dataset.chroms.index(chrom), mol, run
                )
                for chrom, mol, run in targets
            }

        settings = SimSettings(**dict(dataset.settings))

        # Normalize methylation data the same way run() does
        chroms = dataset.chroms
        if isinstance(meth_prob, np.ndarray):
            if not len(chroms) == 1:
                raise TypeError(
                    "methylation probability array 'meth_prob' "
                    "provided but dataset has multiple chroms"
                )
            meth_prob = {chroms[0]: meth_prob}
        elif isinstance(meth_prob, Mapping):
            meth_prob = {chrom: meth_prob[chrom] for chrom in chroms}
        else:
            raise TypeError("'meth_prob' must be a numpy array or a mapping")
        for chrom in chroms:
            if isinstance(meth_prob[chrom], pd.DataFrame):
                meth_prob[chrom] = meth_prob[chrom].to_numpy()
            meth_prob[chrom] = np.atleast_2d(meth_prob[chrom])

        def mcmc_jobs():
            for chrom, mol, run in targets:
                eseq = np.asarray(
                    logit(meth_prob[chrom][mol, :]), dtype=np.float64
                ).tobytes()
                yield MCMCJob(
                    chrom=chrom,
                    mol=mol,
                    run=run,
                    settings=settings,
                    seed=resolved_seeds[(chrom, mol, run)],
                    seq_energy=eseq,
                    out_type=resolved_out_type,
                    out_file=dataset.mcmc_file(chrom, mol, run),
                )

        self._dispatch(
            dataset,
            mcmc_jobs(),
            len(targets),
            SimManager._run_mcmc_job,
            progress_desc="[cyan]Rerunning simulations ...",
        )

        dataset.save()

        return dataset

    def _rerun_tm(
        self,
        dataset: SimDataset,
        *,
        meth_prob: np.ndarray | Mapping[str, np.ndarray | pd.DataFrame],
        only: Iterable[tuple[str, int]] | None,
        seed: int | None,
    ) -> SimDataset:
        """Regenerate a TM dataset's exact samples, at molecule
        granularity (see `rerun`'s docstring for why)."""
        if only is None:
            incomplete = dataset.find_incomplete_runs()
            triplets = incomplete["missing"] + incomplete["corrupted"]
            targets = sorted({(chrom, mol) for chrom, mol, _ in triplets})
        else:
            targets = list(only)
            for target in targets:
                if len(target) != 2:
                    raise ValueError(
                        "for a 'tm' dataset, 'only' must contain "
                        "(chrom, mol) pairs, not (chrom, mol, run) "
                        "triplets; TM samples are regenerated at "
                        "molecule granularity"
                    )

        nsim = dataset.nsim
        chroms = dataset.chroms

        if seed is None and dataset.seed_table is None:
            raise ValueError(
                "dataset predates seed persistence; pass seed= explicitly"
            )

        # Normalize methylation data the same way run() does
        if isinstance(meth_prob, np.ndarray):
            if not len(chroms) == 1:
                raise TypeError(
                    "methylation probability array 'meth_prob' "
                    "provided but dataset has multiple chroms"
                )
            meth_prob = {chroms[0]: meth_prob}
        elif isinstance(meth_prob, Mapping):
            meth_prob = {chrom: meth_prob[chrom] for chrom in chroms}
        else:
            raise TypeError("'meth_prob' must be a numpy array or a mapping")
        for chrom in chroms:
            if isinstance(meth_prob[chrom], pd.DataFrame):
                meth_prob[chrom] = meth_prob[chrom].to_numpy()
            meth_prob[chrom] = np.atleast_2d(meth_prob[chrom])

        settings = SimSettings(**dict(dataset.settings))

        def tm_jobs():
            for chrom, mol in targets:
                eseq = np.asarray(
                    logit(meth_prob[chrom][mol, :]), dtype=np.float64
                ).tobytes()
                if seed is None:
                    seeds = tuple(
                        int(s) for s in dataset.seed_table[chrom][mol, :]
                    )
                else:
                    seeds = tuple(
                        SimManager._job_seed(
                            seed, chroms.index(chrom), mol, run
                        )
                        for run in range(nsim)
                    )
                yield TMJob(
                    chrom=chrom,
                    mol=mol,
                    seeds=seeds,
                    seq_energy=eseq,
                    out_file=dataset.tm_file(chrom, mol),
                    settings=settings,
                )

        if nsim > 0:
            self._dispatch(
                dataset,
                tm_jobs(),
                len(targets),
                SimManager._run_tm_job,
                progress_desc="[cyan]Regenerating exact samples ...",
            )

        dataset.save()

        return dataset

    def _consolidate_tm_molecules(
        self,
        dataset: SimDataset,
        targets: Iterable[tuple[str, int]],
        nsim: int,
    ) -> None:
        """
        Merge each (chrom, mol)'s per-run temp files (as written by
        `_run_mcmc_job` in TM mode) into one combined file via
        `_consolidate_tm_samples`, then delete the temp files. A
        molecule with zero successfully-generated samples is left
        without a combined file (reported as 'missing' by
        `find_incomplete_runs`, matching the pre-consolidation state).
        """
        for chrom, mol in targets:
            temp_files = [
                dataset.mcmc_file(chrom, mol, run) for run in range(nsim)
            ]
            existing = [f for f in temp_files if f.exists()]
            if not existing:
                continue
            combined = dataset.tm_file(chrom, mol)
            combined.parent.mkdir(parents=True, exist_ok=True)
            _consolidate_tm_samples(existing, combined)
            for f in existing:
                f.unlink()
            try:
                temp_files[0].parent.rmdir()
            except OSError:
                pass

    def _dispatch(
        self,
        dataset: SimDataset,
        param_gen: Iterator[MCMCJob | TMJob],
        total: int,
        run_job: Callable[[MCMCJob | TMJob], tuple],
        progress_desc: str = "[cyan]Running simulations ...",
    ) -> None:
        """
        Dispatch a stream of jobs and report progress/errors.

        Parameters
        ----------
        dataset : SimDataset
            The dataset being populated. Used only to classify runs that
            were still in flight if the worker pool crashes.
        param_gen : Iterator[MCMCJob] or Iterator[TMJob]
            A (possibly lazy) generator of jobs to execute.
        total : int
            The total number of jobs `param_gen` will yield, used for the
            progress bar.
        run_job : callable
            Executes one job (`SimManager._run_mcmc_job` or
            `SimManager._run_tm_job`), returning `(chrom, mol, run,
            success, err_msg)`; `run` is None for per-molecule jobs.
        progress_desc : str, default "[cyan]Running simulations ..."
            The label shown next to the progress bar.
        """

        def _label(chrom, mol, run):
            label = f"chrom={chrom}, mol={mol}"
            return label if run is None else f"{label}, run={run}"

        # Progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            disable=not self.verbose,
        )

        with progress:
            main_task = progress.add_task(progress_desc, total=total)

            def _handle_result(result):
                chrom, mol, run, success, err_msg = result
                if not success:
                    progress.console.print(
                        "[red]Error[/red] Simulation failed "
                        f"({_label(chrom, mol, run)}): {err_msg}"
                    )
                progress.update(main_task, advance=1)

            if self.nworker > 1:
                ctx = mp.get_context("spawn")
                window = self.nworker * 2
                with ProcessPoolExecutor(
                    max_workers=self.nworker, mp_context=ctx
                ) as executor:
                    in_flight = {}
                    for param in islice(param_gen, window):
                        future = executor.submit(run_job, param)
                        in_flight[future] = param
                    while in_flight:
                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            param = in_flight[future]
                            try:
                                result = future.result()
                            except Exception:
                                progress.console.print(
                                    "[red]Error[/red] Worker pool crashed; "
                                    "aborting the remaining batch."
                                )
                                for p in in_flight.values():
                                    run = getattr(p, "run", None)
                                    try:
                                        status = (
                                            dataset._classify_run(
                                                p.chrom, p.mol, run or 0
                                            )
                                            or "ok"
                                        )
                                    except Exception:  # noqa: BLE001
                                        status = "unknown"
                                    progress.console.print(
                                        f"  {_label(p.chrom, p.mol, run)}: "
                                        f"{status}"
                                    )
                                raise
                            del in_flight[future]
                            _handle_result(result)
                        for param in islice(param_gen, len(done)):
                            future = executor.submit(run_job, param)
                            in_flight[future] = param
            else:  # nworker = 1
                for param in param_gen:
                    _handle_result(run_job(param))

    # Run a single MCMC simulation
    @staticmethod
    def _run_mcmc_job(p: MCMCJob):
        settings = p.settings
        try:
            # Create the output directory
            sim_dir = Path(p.out_file).parents[0]
            sim_dir.mkdir(exist_ok=True, parents=True)
            # Initialize the methylation energy landscape
            eseq_list = np.frombuffer(p.seq_energy, dtype=np.float64).tolist()
            nbp = len(eseq_list)
            cool_option = SimSettings._cool_map[settings.cool_option]
            # Create the cpp backend Monte Carlo simulation model
            model = NucPosMCMC(
                settings.nucbp,
                nbp,
                settings.llink,
                settings.elink,
                settings.mu,
                p.seed,
            )
            model.setEnergy(eseq_list, settings.emax)
            # For tracking all simulation data
            model.addTracker(
                sim.createDump(
                    settings.print_freq, str(p.out_file), p.out_type
                )
            )
            # Run the simulation
            model.run(
                settings.nsweep,
                settings.start_temp,
                settings.end_temp,
                cool_option,
            )
            return (p.chrom, p.mol, p.run, True, None)
        except Exception as e:  # noqa: BLE001
            return (p.chrom, p.mol, p.run, False, str(e))

    # Solve one molecule once and draw all of its exact samples
    @staticmethod
    def _run_tm_job(p: TMJob):
        settings = p.settings
        try:
            Path(p.out_file).parent.mkdir(exist_ok=True, parents=True)
            eseq_list = np.frombuffer(p.seq_energy, dtype=np.float64).tolist()
            nbp = len(eseq_list)
            model = NucPosTM(
                settings.nucbp,
                nbp,
                settings.llink,
                settings.elink,
                settings.mu,
                p.seeds[0],
            )
            model.setEnergy(eseq_list, settings.emax)
            model.solve(settings.end_temp)
            positions = []
            energies = []
            for s in p.seeds:
                model.setSeed(s)
                position, energy = model.sampleConfig()
                positions.append(np.asarray(position, dtype=np.int32))
                energies.append(energy)
            _write_tm_samples(
                p.out_file,
                nucbp=settings.nucbp,
                nbp=nbp,
                llink=settings.llink,
                elink=settings.elink,
                mu=settings.mu,
                logz=model.getLogZ(),
                positions=positions,
                energies=energies,
                seeds=p.seeds,
            )
            return (p.chrom, p.mol, None, True, None)
        except Exception as e:  # noqa: BLE001
            return (p.chrom, p.mol, None, False, str(e))
