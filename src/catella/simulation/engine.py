# engine.py

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import catella_cpp as sim
import h5py
import numpy as np
import pandas as pd
from catella_cpp import NucPosMCMC, NucPosTM
from scipy.special import logit

from catella._batches import _row_batches
from catella._regions import _check_region_names, _normalize_regions
from catella.experiment.methdata import MethPrintExperiment
from catella.simulation._inputs import (
    _check_llink,
    _check_mcmc_settings,
    _count_mols,
    _normalize_meth_prob,
    _normalize_mols,
    _parse_out_types,
    _prepare_block,
    _resolve_settings,
    _set_energy,
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
    _compute_edata,
    _iter_mcmc_jobs,
    _iter_selected,
    _iter_tm_jobs,
    _mcmc_rerun_targets,
    _resolve_master_seed,
    _seed_table,
    _tm_rerun_targets,
)
from catella.simulation._llink import _estimate_llink
from catella.simulation._parallel import (
    _job_label,
    _process_map,
    _progress_bar,
    _report_crash,
)
from catella.simulation.config import SimSettings
from catella.simulation.results import SimDataset
from catella.utils import IndexType


def _prepare_meth_prob(meth_prob, regions, lnuc, fill_nan=None):
    """
    Normalize `meth_prob`, find the molecules with no data and the base
    pairs each molecule spans, and fill the partly-nan ones.

    Parameters
    ----------
    meth_prob : np.ndarray or mapping of str to np.ndarray or pd.DataFrame
        The methylation probability, as passed to `SimManager.run`.
    regions : sequence of str
        The regions to select.
    lnuc : int
        The nucleosome footprint size.
    fill_nan : None, "mean", or float, default None
        See `SimManager.run`.

    Returns
    -------
    meth_prob : dict of str to np.ndarray
        The 2D array of each region; a filled copy where needed.
    no_data : dict of str to np.ndarray
        bool, True for each molecule with no data (no spanned base
        pair).
    spanned : dict of str to tuple or None
        The spanned intervals of each region (see `_spanned_intervals`).

    Raises
    ------
    ValueError
        If `fill_nan` is invalid, or is None and a molecule is nan at
        some but not all nucleosome start positions.
    """
    meth_prob = _normalize_meth_prob(meth_prob, regions)
    no_data = {}
    spanned = {}
    for region in regions:
        meth_prob[region], no_data[region], spanned[region] = (
            _prepare_block(meth_prob[region], lnuc, region, fill_nan)
        )
    return meth_prob, no_data, spanned


def _rerun_meth_prob(dataset, meth_prob):
    """
    Prepare `meth_prob` for a rerun and check it matches `dataset`.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to rerun.
    meth_prob : np.ndarray or mapping of str to np.ndarray or pd.DataFrame
        The methylation probability passed to the rerun.

    Returns
    -------
    dict of str to np.ndarray
        The 2D array of each region.

    Raises
    ------
    ValueError
        If the molecules with no data differ from `dataset.no_data`, or
        the spanned intervals differ from `dataset.spanned`.
    """
    meth_prob, no_data, spanned = _prepare_meth_prob(
        meth_prob, dataset.regions, dataset.settings["lnuc"], dataset.nan_fill
    )
    for region in dataset.regions:
        if not np.array_equal(no_data[region], dataset.no_data[region]):
            raise ValueError(
                f"meth_prob does not match the dataset for region '{region}'; "
                "its molecules with no data differ from dataset.no_data"
            )
        if not _same_intervals(spanned[region], dataset.spanned[region]):
            raise ValueError(
                f"meth_prob does not match the dataset for region '{region}'; "
                "its spanned positions differ from dataset.spanned"
            )
    return meth_prob


def _same_intervals(a, b):
    """Return True if two `(intervals, offset)` pairs (or None) are equal."""
    if a is None or b is None:
        return a is None and b is None
    return all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))


def _check_rerun_targets(dataset, targets):
    """
    Check that no rerun target is a molecule with no data.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to rerun.
    targets : iterable of tuple
        `(region, mol, ...)` of each target.

    Raises
    ------
    ValueError
        If a target is a molecule skipped for having no data.
    """
    for region, mol, *_ in targets:
        if dataset.no_data[region][mol]:
            raise ValueError(
                f"molecule {mol} of region '{region}' has no data and cannot "
                "be rerun"
            )


class SimManager:
    """
    Orchestrator for managing and executing nucleosome positioning simulations.

    The SimManager handles the setup, parallel execution, and data collection
    of Monte Carlo simulations across multiple regions and molecules.

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
        regions: str | Iterable[str],
        nsim: int,
        settings: str | Path | SimSettings,
        meth_prob: np.ndarray | Mapping[str, np.ndarray | pd.DataFrame],
        out_dir: str | Path,
        dataset_name: str = "results",
        method: str = "mcmc",
        out_types: str | Iterable[str] = "all",
        seed: int | None = None,
        mols: IndexType | Mapping[str, IndexType] = slice(None),
        store_edata: bool = True,
        fill_nan: str | float | None = None,
        occupy_read_ends: bool = True,
    ) -> SimDataset:
        """
        Execute simulations in parallel across regions and molecules.

        This method generates simulation parameters, spawns a process pool,
        and tracks progress as results are written to disk.

        Parameters
        ----------
        regions : str or iterable of str
            The identifier(s) of region(s) to include in the simulation.
        nsim : int
            Number of independent simulation runs per molecule
            (`method='mcmc'`), or exact samples per molecule
            (`method='tm'`; may be 0, since the exact occupancy/mean-N/
            variance quantities require no samples at all).
        settings : str or Path or SimSettings
            The physical constants and Monte Carlo protocol (e.g., lnuc,
            llink, mu) to apply to all runs in this batch. These can be
            read from a configuration file. `llink` can be a mapping of
            region to value, which must cover all of `regions`.
        meth_prob : np.ndarray or dict
            The probability of methylation. If multiple regions are
            provided, this must be a mapping of {region_name: data}. Data can
            be NumPy arrays or Pandas DataFrames. Molecules that span no
            base pair (e.g., masked by `filter_dropout`) have no data and
            are not simulated; see `SimDataset.no_data`. Nan beyond the
            last start position (`nbp - lnuc`) is ignored.
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
            Indices of molecules to simulate for each region. Can be a
            single index/slice or a mapping of {region_name: indices}.
        store_edata : bool, default True
            Whether to store the data energy of each molecule (derived from
            the methylation data) in the output dataset file.
        fill_nan : {"mean"}, float, or None, default None
            How to treat a molecule of `meth_prob` that is nan at some but
            not all nucleosome start positions. If None (default), raise
            `ValueError`. If "mean", fill each nan with the molecule's
            mean over its other start positions. If a number in [0, 1],
            fill each nan with that probability (0.5 adds no information).
            Use 0.5 for windows a read does not cover (read ends with
            `span="read"`, long gaps). `meth_prob` itself is not
            modified, and the value is stored as `SimDataset.nan_fill`
            for reruns.
        occupy_read_ends : bool, default True
            Whether nucleosomes may occupy the unread DNA before and
            after each read. Set False for reads that cover their whole
            molecule (e.g., native fragments with `span="read"`): no
            nucleosome then leaves the molecule's extent. Interior gaps
            are not affected. Stored as `SimDataset.occupy_read_ends`.

        Returns
        -------
        SimDataset
            A dataset object providing access to the newly generated results.

        Raises
        ------
        TypeError
            If input types for `regions`, `meth_prob`, or `mols` are
            inconsistent.
        ValueError
            If the requested `out_types` are not recognized, if an
            `llink` value of a mapping is not a non-negative integer, if
            `fill_nan` is invalid, if `fill_nan` is None and a molecule
            of `meth_prob` is nan at some but not all nucleosome start
            positions, or if a region name is not safe as a directory
            name.
        KeyError
            If a mapping `llink` lacks a region of `regions`.
        FileExistsError
            If the specified `out_dir` already contains an existing dataset.
        """

        if method not in ("mcmc", "tm"):
            raise ValueError("'method' must be 'mcmc' or 'tm'")

        regions = _normalize_regions(regions)
        _check_region_names(regions)
        settings = _resolve_settings(settings)
        meth_prob, no_data, spanned = _prepare_meth_prob(
            meth_prob, regions, settings.lnuc, fill_nan
        )
        mols = _normalize_mols(mols, regions)
        nmol = {region: meth_prob[region].shape[0] for region in regions}
        nbp = {region: meth_prob[region].shape[1] for region in regions}
        total_mol = _count_mols(mols, nmol) - sum(
            int(np.count_nonzero(no_data[region][mols[region]]))
            for region in regions
        )
        _check_llink(settings, regions)
        if method == "mcmc":
            _check_mcmc_settings(settings)
            out_type = _parse_out_types(out_types)
        else:
            out_type = None

        edata = None
        if store_edata:
            edata = _compute_edata(settings, meth_prob, regions)
        seed = _resolve_master_seed(seed)

        # Save now so a loadable dataset exists even if the run is cut short
        dataset = SimDataset.create(
            regions=regions,
            nmol=nmol,
            nsim=nsim,
            nbp=nbp,
            settings=settings,
            edata=edata if store_edata else None,
            out_dir=out_dir,
            dataset_name=dataset_name,
            method=method,
            out_type=out_type,
            seed=seed,
            seed_table=_seed_table(seed, regions, nmol, nsim),
            no_data=no_data,
            nan_fill=fill_nan,
            spanned=spanned,
            occupy_read_ends=occupy_read_ends,
        )
        dataset.save()

        selected = (
            (region, mol)
            for region, mol in _iter_selected(regions, nmol, mols)
            if not no_data[region][mol]
        )
        if method == "mcmc":
            targets = (
                (region, mol, run)
                for region, mol in selected
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
            exact original simulation by default. Its molecules with no
            data must match `dataset.no_data`, and partly-nan molecules
            are filled as in the original run (`dataset.nan_fill`).
        only : iterable of (str, int, int) or (str, int), optional
            For `dataset.method == 'mcmc'`: the specific `(region, mol,
            run)` triplets to rerun. For `'tm'`: `(region, mol)` pairs --
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
            raised if `only` contains a 3-tuple for a `'tm'` dataset or
            a molecule with no data, or if the molecules with no data of
            `meth_prob` differ from `dataset.no_data`.
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

    def estimate_llink(
        self,
        exp: MethPrintExperiment,
        settings: str | Path | SimSettings,
        *,
        llinks: Iterable[int] = range(0, 65, 5),
        prob_name: str = "meth_prob",
        regions: str | Iterable[str] | None = None,
        max_nmol: int | None = None,
        seed: int | None = None,
        batch_size: int = 20000,
        mask_name: str | None = None,
        llink_evid_name: str | None = None,
        fill_nan: str | float | None = None,
        occupy_read_ends: bool = True,
    ) -> dict[str, int]:
        """
        Estimate `SimSettings.llink` per region by maximum likelihood.

        For each region, returns the candidate that maximizes the sum
        over molecules of log Z(data, llink) - log Z(no data, llink), the
        marginal likelihood of the data, computed exactly with the
        transfer matrix at temperature 1. Molecules are split across
        `nworker` processes. `exp` is not modified unless
        `llink_evid_name` is given, and the value only takes effect once
        placed in the settings:

        >>> best = manager.estimate_llink(exp, settings)
        >>> settings = dataclasses.replace(settings, llink=best)
        >>> manager.run(settings=settings, ...)

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment to estimate from. `prob_name` must already be
            in `exp.analysis[region]` for each region.
        settings : str or Path or SimSettings
            A settings object, or a path to a settings file. Provides
            `lnuc`, `elink`, `mu` and `emax`, which should match those
            used for sampling. `llink` is ignored.
        llinks : iterable of int, default range(0, 65, 5)
            Candidate linker lengths (bp).
        prob_name : str, default "meth_prob"
            The key to read from `exp.analysis[region]`.
        regions : str or iterable of str, optional
            Region(s) to include. If None (default), all of
            `exp.regions` are used.
        max_nmol : int, optional
            If given, use a random sample of at most this many molecules
            per region, to reduce run time. All molecules are used if
            None (default).
        seed : int, optional
            Seed for the random sample of `max_nmol`.
        batch_size : int, default 20000
            Number of molecules read (and held in memory) per batch.
        mask_name : str, optional
            If given, excludes molecules dropped by
            `filter_dropout(mask_name=mask_name)` on the "test" source,
            without modifying `exp`.
        llink_evid_name : str, optional
            If given, stores the evidence of each candidate in
            `exp.analysis[region][llink_evid_name]`, a DataFrame with
            columns `llink`, `evidence` and `nmol`, e.g. for plotting.
            `exp` is not modified if None (default).
        fill_nan : {"mean"}, float, or None, default None
            How to treat molecules that are nan at some but not all
            nucleosome start positions; see `run`. Molecules with no data
            (no spanned base pair) are always left out.
        occupy_read_ends : bool, default True
            See `run`; use the same value as for `run`.

        Returns
        -------
        dict of str to int
            Region -> estimated `llink`.

        Raises
        ------
        ValueError
            If `llinks` is empty or holds a value that is not a
            non-negative integer, if `max_nmol` is not positive, if a
            region has no usable molecules, if `fill_nan` is invalid,
            or if `fill_nan` is None and a molecule is nan at some but not
            all start positions.
        TypeError
            If `settings` is neither a path nor a SimSettings object.
        KeyError
            If `prob_name` or the mask of `mask_name` is missing for a
            region.

        Warns
        -----
        UserWarning
            If `settings.end_temp` is not 1, if the estimate of a
            region is the largest candidate (the maximum may lie
            beyond it), or if a region uses fewer than 50 molecules.

        Notes
        -----
        `SimSettings.llink` acts as a soft minimum linker length rather
        than a preferred one, since the repulsion is zero for longer
        linkers.
        """
        return _estimate_llink(
            exp,
            settings,
            llinks=llinks,
            prob_name=prob_name,
            regions=regions,
            max_nmol=max_nmol,
            seed=seed,
            batch_size=batch_size,
            mask_name=mask_name,
            nworker=self.nworker,
            llink_evid_name=llink_evid_name,
            fill_nan=fill_nan,
            occupy_read_ends=occupy_read_ends,
        )

    def estimate_start_temp(
        self,
        exp: MethPrintExperiment,
        *,
        prob_name: str = "meth_prob",
        regions: str | Iterable[str] | None = None,
        percentile: float = 90.0,
        batch_size: int = 20000,
        percentile_sample_size: int = 100000,
        seed: int | None = None,
        mask_name: str | None = None,
    ) -> float:
        """
        Estimate a data-driven `SimSettings.start_temp` value.

        Returns the `percentile`-th percentile of abs(logit(meth_prob))
        across `regions`, via batched random subsampling, so sequence-
        energy barriers of that size are crossable early in the
        annealing schedule. Probabilities of exactly 0 or 1 (log odds
        beyond floating-point range) count as the largest finite log
        odds rather than infinity. Runs in a single process.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment to estimate a temperature from. `prob_name`
            must already be in `exp.analysis[region]` for each region.
        prob_name : str, default "meth_prob"
            The key to read from `exp.analysis[region]`.
        regions : str or iterable of str, optional
            Region(s) to include. If None (default), all of
            `exp.regions` are used.
        percentile : float, default 90.0
            Percentile (0 to 100, exclusive of 100) of
            abs(logit(meth_prob)) to return.
        batch_size : int, default 20000
            Number of molecules read (and held in memory) per batch.
        percentile_sample_size : int, default 100000
            Approximate total number of molecules, split evenly across
            `regions`, used to estimate the percentile.
        seed : int, optional
            Seed for the random number generator used for subsampling.
        mask_name : str, optional
            If given, excludes molecules dropped by
            `filter_dropout(mask_name=mask_name)` on the "test" source,
            without modifying `exp`.

        Returns
        -------
        float
            The suggested `start_temp` value.

        Raises
        ------
        KeyError
            If `mask_name` is given but no matching mask is found for a
            region's "test" source.
        """
        if isinstance(regions, str):
            resolved_regions = (regions,)
        elif regions is not None:
            resolved_regions = tuple(regions)
        else:
            resolved_regions = exp.regions

        rng = np.random.default_rng(seed)
        target_per_region = max(
            1, percentile_sample_size // len(resolved_regions)
        )
        chunks = []
        for region in resolved_regions:
            arr = exp.analysis[region][prob_name]
            if isinstance(arr, pd.DataFrame):
                arr = arr.to_numpy()
            keep = None
            if mask_name is not None:
                keep = exp.is_kept(region, mask_name=mask_name)
            n_total = arr.shape[0]
            if n_total == 0:
                continue
            sample_frac = min(1.0, target_per_region / n_total)
            for start, stop in _row_batches(0, n_total, batch_size):
                batch = arr[start:stop, :]
                if keep is not None:
                    batch = batch[keep[start:stop]]
                if batch.shape[0] == 0:
                    continue
                n_take = min(
                    batch.shape[0],
                    max(1, round(batch.shape[0] * sample_frac)),
                )
                rows = rng.choice(batch.shape[0], size=n_take, replace=False)
                # Clip so saturated probabilities give finite log odds
                prob = np.clip(
                    np.asarray(batch[rows, :], dtype=np.float64),
                    np.finfo(np.float64).tiny,
                    1 - np.finfo(np.float64).epsneg,
                )
                chunks.append(np.abs(logit(prob)))

        sample = np.concatenate([c.ravel() for c in chunks])
        return float(np.nanpercentile(sample, percentile))

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
        _check_rerun_targets(dataset, targets)
        _check_seed_source(dataset, seed)
        settings = SimSettings(**dict(dataset.settings))
        meth_prob = _rerun_meth_prob(dataset, meth_prob)
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
        _check_rerun_targets(dataset, targets)
        _check_seed_source(dataset, seed)
        meth_prob = _rerun_meth_prob(dataset, meth_prob)
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
            `SimManager._run_tm_job`), returning `(region, mol, run,
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
            for _, (region, mol, run, success, err_msg) in results:
                if not success:
                    progress.console.print(
                        "[red]Error[/red] Simulation failed "
                        f"({_job_label(region, mol, run)}): {err_msg}"
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
            edata_list = np.frombuffer(p.edata, dtype=np.float64).tolist()
            nbp = len(edata_list)
            cool_option = SimSettings._cool_map[settings.cool_option]
            # Create the cpp backend Monte Carlo simulation model
            model = NucPosMCMC(
                settings.lnuc,
                nbp,
                settings.llink,
                settings.elink,
                settings.mu,
                p.seed,
            )
            _set_energy(
                model, edata_list, settings.emax, settings.lnuc, p.extent
            )
            # For tracking all simulation data
            dump = sim.createDump(settings.print_freq, p.out_type)
            model.addTracker(dump)
            # Open the file first so a bad output path fails before the run
            with h5py.File(p.out_file, "w") as f:
                _write_mcmc_params(
                    f,
                    lnuc=settings.lnuc,
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
            return (p.region, p.mol, p.run, True, None)
        except Exception as e:  # noqa: BLE001
            return (p.region, p.mol, p.run, False, str(e))

    # Solve one molecule once and draw all of its exact samples
    @staticmethod
    def _run_tm_job(p: TMJob):
        settings = p.settings
        try:
            Path(p.out_file).parent.mkdir(exist_ok=True, parents=True)
            edata_list = np.frombuffer(p.edata, dtype=np.float64).tolist()
            nbp = len(edata_list)
            model = NucPosTM(
                settings.lnuc,
                nbp,
                settings.llink,
                settings.elink,
                settings.mu,
                p.seeds[0],
            )
            _set_energy(
                model, edata_list, settings.emax, settings.lnuc, p.extent
            )
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
                lnuc=settings.lnuc,
                nbp=nbp,
                llink=settings.llink,
                elink=settings.elink,
                mu=settings.mu,
                logz=model.getLogZ(),
                positions=positions,
                energies=energies,
                seeds=p.seeds,
            )
            return (p.region, p.mol, None, True, None)
        except Exception as e:  # noqa: BLE001
            return (p.region, p.mol, None, False, str(e))
