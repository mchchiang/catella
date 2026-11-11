# engine.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import catella_cpp as sim
import h5py
import numpy as np
import pandas as pd
from catella_cpp import NucPosMCMC, NucPosTM

from catella import utils
from catella.simulation._inputs import (
    _check_llink,
    _check_mcmc_settings,
    _count_mols,
    _normalize_meth_prob,
    _normalize_mols,
    _parse_out_types,
    _resolve_settings,
)
from catella.simulation._io import (
    _write_mcmc_data,
    _write_mcmc_params,
    _write_tm_samples,
)
from catella.simulation._jobs import (
    MCMCJob,
    TMJob,
    _check_seed_source,
    _compute_eseq,
    _iter_mcmc_jobs,
    _iter_selected,
    _iter_tm_jobs,
    _mcmc_rerun_targets,
    _median_eseq_mu,
    _resolve_master_seed,
    _seed_table,
    _tm_rerun_targets,
)
from catella.simulation._parallel import (
    _job_label,
    _process_map,
    _progress_bar,
    _report_crash,
)
from catella.simulation.config import SimSettings
from catella.simulation.results import SimDataset
from catella.utils import IndexType


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
            read from a configuration file. `llink` can be a mapping of
            chromosome to value, which must cover all of `chroms`.
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
            None, a random seed in [0, 2**63) is drawn from a freshly
            seeded NumPy default random generator.
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
            If the requested `out_types` are not recognized, or if an
            `llink` value of a mapping is not a non-negative integer.
        KeyError
            If a mapping `llink` lacks a chromosome of `chroms`.
        FileExistsError
            If the specified `out_dir` already contains an existing dataset.
        """

        if method not in ("mcmc", "tm"):
            raise ValueError("'method' must be 'mcmc' or 'tm'")

        chroms = utils.normalize_chroms(chroms)
        meth_prob = _normalize_meth_prob(meth_prob, chroms)
        mols = _normalize_mols(mols, chroms)
        nmol = {chrom: meth_prob[chrom].shape[0] for chrom in chroms}
        nbp = {chrom: meth_prob[chrom].shape[1] for chrom in chroms}
        total_mol = _count_mols(mols, nmol)
        settings = _resolve_settings(settings)
        _check_llink(settings, chroms)
        if method == "mcmc":
            _check_mcmc_settings(settings)
            out_type = _parse_out_types(out_types)
        else:
            out_type = None

        eseq = None
        if store_eseq or use_median_eseq_mu:
            eseq = _compute_eseq(settings, meth_prob, chroms)
        if use_median_eseq_mu:
            mu = _median_eseq_mu(eseq)
            print(f"Using median eseq mu: {mu}")
            settings = replace(settings, mu=mu)
        seed = _resolve_master_seed(seed)

        # Save now so a loadable dataset exists even if the run is cut short
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
            seed_table=_seed_table(seed, chroms, nmol, nsim),
        )
        dataset.save()

        selected = _iter_selected(chroms, nmol, mols)
        if method == "mcmc":
            targets = (
                (chrom, mol, run)
                for chrom, mol in selected
                for run in range(nsim)
            )
            self._dispatch(
                dataset,
                _iter_mcmc_jobs(
                    dataset, settings, meth_prob, targets, out_type
                ),
                total_mol * nsim,
                SimManager._run_mcmc_job,
            )
        elif nsim > 0:
            self._dispatch(
                dataset,
                _iter_tm_jobs(dataset, settings, meth_prob, selected),
                total_mol,
                SimManager._run_tm_job,
                progress_desc="[cyan]Solving molecules ...",
            )

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

        targets = _mcmc_rerun_targets(dataset, only)
        _check_seed_source(dataset, seed)
        settings = SimSettings(**dict(dataset.settings))
        meth_prob = _normalize_meth_prob(meth_prob, dataset.chroms)
        jobs = _iter_mcmc_jobs(
            dataset, settings, meth_prob, targets, resolved_out_type, seed
        )
        self._dispatch(
            dataset,
            jobs,
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
        targets = _tm_rerun_targets(dataset, only)
        _check_seed_source(dataset, seed)
        meth_prob = _normalize_meth_prob(meth_prob, dataset.chroms)
        settings = SimSettings(**dict(dataset.settings))
        if dataset.nsim > 0:
            self._dispatch(
                dataset,
                _iter_tm_jobs(dataset, settings, meth_prob, targets, seed),
                len(targets),
                SimManager._run_tm_job,
                progress_desc="[cyan]Regenerating exact samples ...",
            )
        dataset.save()
        return dataset

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
        progress = _progress_bar(self.verbose)
        with progress:
            task = progress.add_task(progress_desc, total=total)
            results = _process_map(
                run_job,
                param_gen,
                nworker=self.nworker,
                on_crash=lambda jobs: _report_crash(
                    progress.console, dataset, jobs
                ),
            )
            for _, (chrom, mol, run, success, err_msg) in results:
                if not success:
                    progress.console.print(
                        "[red]Error[/red] Simulation failed "
                        f"({_job_label(chrom, mol, run)}): {err_msg}"
                    )
                progress.update(task, advance=1)

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
            dump = sim.createDump(settings.print_freq, p.out_type)
            model.addTracker(dump)
            # Open the file first so a bad output path fails before the run
            with h5py.File(p.out_file, "w") as f:
                _write_mcmc_params(
                    f,
                    nucbp=settings.nucbp,
                    nbp=nbp,
                    llink=settings.llink,
                    elink=settings.elink,
                    mu=settings.mu,
                    seed=p.seed,
                )
                model.run(
                    settings.nsweep,
                    settings.start_temp,
                    settings.end_temp,
                    cool_option,
                )
                _write_mcmc_data(f, dump.values())
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
