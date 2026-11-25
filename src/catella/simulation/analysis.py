# analysis.py

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM

from catella import utils
from catella._batches import _row_batches
from catella._regions import _normalize_regions, _per_region
from catella._warnings import _warn
from catella.h5_array import H5Array
from catella.simulation._exact import (
    TM_BATCH_SIZE,
    TMStat,
    _nan_unspanned,
    _region_settings,
    _reject_time,
    _require_edata,
    _require_tm,
    _tm_configs,
    _tm_per_bp,
    _tm_per_mol,
    _tm_stats,
)
from catella.simulation._parallel import _threaded_map
from catella.simulation.results import SimDataset, SimMCMCData

MCMC_BATCH_SIZE = 20000


def _stored_configs(dataset, region, name, mol_ids):
    """
    Read the configurations stored under `name` for some molecules.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the configurations.
    region : str
        The region of the molecules.
    name : str
        The prefix of the stored keys.
    mol_ids : iterable of int
        The molecules to read, in order.

    Returns
    -------
    list of np.ndarray or None
        The start positions of each molecule, or None for a molecule
        with no data (nan energy).

    Raises
    ------
    KeyError
        If no configurations are stored under `name` for `region`.
    """
    analysis = dataset.analysis[region]
    if f"{name}_position_flat" not in analysis:
        raise KeyError(
            f"no configurations stored under '{name}' for region '{region}'"
        )
    flat = analysis[f"{name}_position_flat"]["position"].to_numpy()
    offset = analysis[f"{name}_position_offset"]["offset"].to_numpy()
    energy = analysis[f"{name}_energy"]["energy"].to_numpy()
    return [
        None
        if np.isnan(energy[mol])
        else flat[int(offset[mol]) : int(offset[mol + 1])]
        for mol in mol_ids
    ]


def _check_nworker(dataset, nworker):
    """
    Check the number of worker processes of an analysis.

    Parameters
    ----------
    dataset : SimDataset
        The dataset being analyzed.
    nworker : int
        The requested number of worker processes.

    Raises
    ------
    ValueError
        If `nworker` is not positive, or `nworker > 1` for a dataset
        whose method is not "tm".
    """
    if nworker < 1:
        raise ValueError(f"nworker must be positive, got {nworker}")
    if nworker > 1 and dataset.method != "tm":
        raise ValueError(
            "nworker > 1 requires a method='tm' dataset, "
            f"got method={dataset.method!r}"
        )


def _resolve_batch_size(dataset, batch_size):
    """
    Return `batch_size`, or the default for the method of `dataset`.

    Parameters
    ----------
    dataset : SimDataset
        The dataset being analyzed.
    batch_size : int or None
        The requested batch size.

    Returns
    -------
    int
        `batch_size` if given; otherwise `TM_BATCH_SIZE` for "tm" and
        `MCMC_BATCH_SIZE` for "mcmc" datasets.
    """
    if batch_size is not None:
        return batch_size
    return TM_BATCH_SIZE if dataset.method == "tm" else MCMC_BATCH_SIZE


def _check_nonneg_int(value, name):
    """
    Check that a value is a nonnegative integer.

    Parameters
    ----------
    value : object
        The value to check.
    name : str
        The argument name, for the error message.

    Raises
    ------
    TypeError
        If `value` is not an integer.
    ValueError
        If `value` is negative.
    """
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, got {value!r}")
    if value < 0:
        raise ValueError(f"{name} must be nonnegative, got {value}")


def _is_window(value):
    """Return True if `value` is a pair of integers."""
    return (
        isinstance(value, (tuple, list))
        and len(value) == 2
        and all(
            isinstance(v, Integral) and not isinstance(v, bool) for v in value
        )
    )


def _check_window(window):
    """
    Check a `window` argument of `compute_linker_hist`.

    Parameters
    ----------
    window : object
        None, a pair of integers, or a mapping from region to a pair
        of integers.

    Raises
    ------
    ValueError
        If `window` is none of these.
    """
    if window is None or _is_window(window):
        return
    if isinstance(window, Mapping) and all(
        _is_window(w) for w in window.values()
    ):
        return
    raise ValueError(
        "window must be a pair of integers or a dict of them by region, "
        f"got {window!r}"
    )


def _linker_window(window, region, nbp, lnuc, trim):
    """
    Return the window of a region in which linkers are counted.

    Parameters
    ----------
    window : None, tuple of int, or mapping of str to tuple of int
        A checked `window` argument; regions missing from a mapping
        use the full fiber.
    region : str
        The region.
    nbp : int
        The number of base pairs of the region.
    lnuc : int
        The nucleosome footprint size.
    trim : int
        The number of base pairs removed from each end of the window.

    Returns
    -------
    tuple of int
        The window `(xlo, xhi)` after trimming.

    Raises
    ------
    ValueError
        If the window is not inside the fiber, or cannot hold two
        nucleosomes after trimming.
    """
    if isinstance(window, Mapping):
        window = window.get(region)
    xlo, xhi = (0, nbp) if window is None else tuple(window)
    if not 0 <= xlo <= xhi <= nbp:
        raise ValueError(
            f"window must satisfy 0 <= xlo <= xhi <= {nbp} for region "
            f"'{region}', got ({xlo}, {xhi})"
        )
    xlo, xhi = xlo + trim, xhi - trim
    if xhi - xlo < 2 * lnuc:
        raise ValueError(
            "window after trim must hold two nucleosomes for region "
            f"'{region}', got ({xlo}, {xhi})"
        )
    return xlo, xhi


def _stored_linker_hist(dataset, region, name):
    """
    Return the linker histogram stored under `name` for a region.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the histogram.
    region : str
        The region.
    name : str
        The key of the histogram.

    Returns
    -------
    H5Array
        The histogram, molecules by linker-length bins.

    Raises
    ------
    KeyError
        If no histogram is stored under `name` for `region`.
    """
    analysis = dataset.analysis[region]
    if name not in analysis:
        raise KeyError(
            f"no linker histogram stored under '{name}' for region "
            f"'{region}'; run compute_linker_hist first"
        )
    return analysis[name]


def _mol_mask(mols, nmol, region):
    """
    Return a boolean mask of the selected molecules of a region.

    Parameters
    ----------
    mols : iterable of int or None
        The selected molecules (duplicates are counted once), or None
        for all.
    nmol : int
        The number of molecules of the region.
    region : str
        The region, for the error message.

    Returns
    -------
    np.ndarray or None
        bool, True for each selected molecule; None for all.

    Raises
    ------
    ValueError
        If a molecule index is out of range.
    """
    if mols is None:
        return None
    idx = np.asarray(list(mols), dtype=np.int64)
    if idx.size and (idx.min() < 0 or idx.max() >= nmol):
        raise ValueError(
            f"mols must be in [0, {nmol}) for region '{region}', got "
            f"{idx.min()} to {idx.max()}"
        )
    mask = np.zeros(nmol, dtype=bool)
    mask[idx] = True
    return mask


def _linker_moments(counts):
    """
    Return the mean, standard deviation, and number of linkers.

    Parameters
    ----------
    counts : np.ndarray
        2D, one row per histogram and one column per linker length
        0, 1, ..., lmax (the tail bin excluded).

    Returns
    -------
    mean, sd, nlink : np.ndarray
        Per row: the mean and standard deviation of the linker length
        (nan if there are no linkers) and the number of linkers.
    """
    lengths = np.arange(counts.shape[1], dtype=np.float64)
    nlink = counts.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = counts @ lengths / nlink
        second = counts @ lengths**2 / nlink
    mean[nlink == 0] = np.nan
    var = np.maximum(second - mean**2, 0.0)
    return mean, np.sqrt(var), nlink


def _iter_hist_rows(arr, mask, batch_size):
    """
    Yield the selected rows of a histogram array in batches.

    Parameters
    ----------
    arr : H5Array
        The histograms, one row per molecule.
    mask : np.ndarray or None
        bool, True for each selected molecule; None for all.
    batch_size : int
        Number of rows read per batch.

    Yields
    ------
    mol_ids : np.ndarray
        The molecule indices of the batch.
    rows : np.ndarray
        Their histograms.
    """
    nmol = arr.shape[0]
    for start, stop in _row_batches(0, nmol, batch_size):
        mol_ids = np.arange(start, stop)
        rows = arr.iloc[start:stop, :]
        if mask is not None:
            mol_ids, rows = mol_ids[mask[start:stop]], rows[mask[start:stop]]
        if mol_ids.size:
            yield mol_ids, rows


@dataclass(slots=True)
class NucFiberMap:
    """
    A mapping tool to project nucleosome coordinates onto a 1D DNA lattice.

    This class converts discrete start positions into a continuous binary
    occupancy representation, simulating the physical footprint of nucleosomes
    along a DNA fiber of fixed length.
    """

    lnuc: int
    """The footprint size (in base pairs) of a single nucleosome."""

    nbp: int
    """The total number of base pairs in the DNA or chromatin fiber."""

    def project(self, nucpos: np.ndarray) -> np.ndarray:
        """
        Expand left-aligned positions into a binary occupancy array.

        Generates a 1D array where each nucleosome's footprint is represented
        by a contiguous block of ones. This is achieved via broadcasting
        to ensure high performance on large genomic tracks.

        Parameters
        ----------
        nucpos : ndarray
            1D array of integers containing the left-aligned (start)
            coordinates of nucleosomes.

        Returns
        -------
        arr : ndarray
            A 1D array of type `int8` and length `nbp`. A value of 1
            indicates occupancy; 0 indicates empty DNA.

        Notes
        -----
        Positions falling partially or entirely outside the range [0, nbp)
        are automatically clipped or filtered to prevent indexing errors.
        """
        diff = np.zeros(self.nbp + 1, dtype=np.int32)
        starts = nucpos[(nucpos >= 0) & (nucpos < self.nbp)]
        ends = (starts + self.lnuc).clip(max=self.nbp)
        np.add.at(diff, starts, 1)
        np.add.at(diff, ends, -1)
        return (np.cumsum(diff)[:-1] > 0).astype(np.int8)

    def aggregate(
        self, samples: Iterable[np.ndarray], norm: bool = False
    ) -> np.ndarray:
        """
        Aggregate multiple nucleosome position sets into a single occupancy
        map.

        This method uses a difference array (prefix sum) algorithm to
        compute the total occupancy across all samples in a single linear
        pass. It is highly memory-efficient as it avoids generating dense
        intermediate lattices for each sample.

        Parameters
        ----------
        samples : iterable of ndarray
            An iterable where each element is an array of nucleosome start
            positions (integers). Each array represents one DNA fiber or
            simulation frame.
        norm : bool, default False
            If True, the result is divided by the number of samples to
            produce a probability map (range [0, 1]). If False, the result
            contains raw counts of overlapping footprints.

        Returns
        -------
        occupancy : ndarray
            A 1D array of length `nbp`. If `norm=False`, returns
            `int32` counts; if `normalize=True`, returns `float64`
            probabilities.
        """
        diff = np.zeros(self.nbp + 1, dtype=np.int32)
        for nucpos in samples:
            starts = nucpos[(nucpos >= 0) & (nucpos < self.nbp)]
            ends = (starts + self.lnuc).clip(max=self.nbp)
            np.add.at(diff, starts, 1)
            np.add.at(diff, ends, -1)
        occup = np.cumsum(diff)[:-1]

        # Normalize the data if needed
        if norm:
            return occup.astype(np.float64) / max(len(samples), 1)
        return occup

    def stack(self, samples: Iterable[np.ndarray]) -> np.ndarray:
        """
        Stack multiple nucleosome sets into a 2D occupancy matrix.

        Each sample is projected into its own row, creating a 2D representation
        where rows correspond to time/samples and columns correspond to
        genomic positions.

        Parameters
        ----------
        samples : Iterable[ndarray]
            An iterable of 1D arrays containing nucleosome start positions.
            Must be convertible to a list or have a known length to
            pre-allocate the matrix.

        Returns
        -------
        matrix : ndarray
            A 2D array of shape (n_samples, nbp) with `int8` values.
        """
        sample_list = list(samples)
        n_samples = len(sample_list)
        matrix = np.zeros((n_samples, self.nbp), dtype=np.int8)
        for i, nucpos in enumerate(sample_list):
            diff = np.zeros(self.nbp + 1, dtype=np.int32)
            starts = nucpos[(nucpos >= 0) & (nucpos < self.nbp)]
            ends = (starts + self.lnuc).clip(max=self.nbp)
            np.add.at(diff, starts, 1)
            np.add.at(diff, ends, -1)
            matrix[i, :] = (np.cumsum(diff)[:-1] > 0).astype(np.int8)
        return matrix


class SimAnalysis:
    """
    A collection of post-processing tools for simulation data.

    This class provides methods to calculate physical properties such as
    nucleosome occupancy from raw simulation trajectories. Results are
    stored directly in the provided dataset's analysis map.
    """

    def _final_time(self, dataset: SimDataset, regions) -> int:
        """Return the final time of the first simulation run on disk."""
        run_id = next(dataset.iter_runs(regions), None)
        if run_id is None:
            raise ValueError(
                f"no simulation files found for regions {list(regions)}"
            )
        return dataset.raw[run_id].time[-1]

    def compute_occup(
        self,
        dataset: SimDataset,
        *,
        time: int | None = None,
        regions: str | Iterable[str] | None = None,
        name: str = "occup",
        mean_nnuc_name: str | None = "mean_nnuc",
        compute_var: bool = False,
        var_nnuc_name: str = "var_nnuc",
        var_occup_name: str = "var_occup",
        record_time: bool = False,
        batch_size: int | None = None,
        nworker: int = 1,
    ):
        """
        Calculate nucleosome occupancy and the mean nucleosome count.

        For `dataset.method == "mcmc"`, occupancy is the fraction of
        simulation runs where a base pair is covered by a nucleosome,
        and the mean count is the average over runs, at a specific time
        point. For `"tm"`, both (and optionally their variances) are
        computed exactly via the transfer-matrix solver, with no time
        axis, solving each molecule only once. Molecules are processed
        in batches and streamed to a disk-backed array so that peak
        memory scales with `batch_size` rather than the total number of
        molecules. Occupancy is nan at base pairs a molecule does not
        span (see `SimDataset.spanned`).

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the raw simulation results.
        time : int, optional
            The simulation time point (snapshot) to analyze. If None, the
            final time frame will be used. `mcmc` only; must be None for
            `dataset.method == "tm"`.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "occup"
            The key used to store the occupancy `H5Array` in
            `dataset.analysis`.
        mean_nnuc_name : str or None, default "mean_nnuc"
            The key used to store the mean nucleosome count of each
            molecule (a DataFrame). If None, it is not computed.
        compute_var : bool, default False
            If True, also compute the exact variance of the nucleosome
            count and of the per-bp occupancy. `tm` only.
        var_nnuc_name : str, default "var_nnuc"
            The key used to store the nucleosome count variance. Only
            used if `compute_var` is True.
        var_occup_name : str, default "var_occup"
            The key used to store the per-bp occupancy variance. Only
            used if `compute_var` is True.
        record_time : bool, default False
            If True, the time point is appended to the storage names
            (e.g., "occup_t_100", "mean_nnuc_t_100"). `mcmc` only.
        batch_size : int, optional
            Number of molecules processed per batch. For `"mcmc"`, the
            number held in memory at a time (default 20000). For `"tm"`,
            the number per worker job (default 64); peak memory scales
            with `2 * nworker * batch_size` molecules.
        nworker : int, default 1
            Number of worker processes. Values above 1 are only supported
            for `"tm"` datasets.

        Raises
        ------
        ValueError
            If `time` or `record_time` is given for a `"tm"` dataset, if
            `compute_var` is True for a non-`"tm"` dataset, or (for
            `"tm"`) if the dataset has no stored energy landscape, or
            (for `"mcmc"`) if `time` is None and no simulation files
            exist for `regions`, or `nworker` is invalid.
        """
        _check_nworker(dataset, nworker)
        if compute_var:
            _require_tm(dataset, "compute_var=True")
        batch_size = _resolve_batch_size(dataset, batch_size)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        if dataset.method == "tm":
            _reject_time(time, record_time)
            _require_edata(dataset)
            stats = [TMStat(name, "getOccup", per_bp=True)]
            if mean_nnuc_name is not None:
                stats.append(TMStat(mean_nnuc_name, "getMeanN"))
            if compute_var:
                stats += [
                    TMStat(var_nnuc_name, "getVarN"),
                    TMStat(var_occup_name, "getVarOccup", per_bp=True),
                ]
            _tm_stats(
                dataset, regions, stats, batch_size=batch_size, nworker=nworker
            )
            return
        self._compute_occup_mcmc(
            dataset,
            time=time,
            regions=regions,
            name=name,
            record_time=record_time,
            batch_size=batch_size,
        )
        if mean_nnuc_name is not None:
            self._compute_mean_nnuc_mcmc(
                dataset,
                time=time,
                regions=regions,
                name=mean_nnuc_name,
                record_time=record_time,
            )

    def _compute_occup_mcmc(
        self, dataset, *, time, regions, name, record_time, batch_size
    ):
        nuc_map = {
            region: NucFiberMap(dataset.settings["lnuc"], dataset.nbp[region])
            for region in regions
        }
        if time is None:
            time = self._final_time(dataset, regions)

        def occup_agg(region, mol, nucpos):
            return nuc_map[region].aggregate(nucpos, norm=True)

        if record_time:
            name = f"{name}_t_{time}"
        tmp_dir = dataset.resolve_tmp_dir()
        for region in regions:
            nmol = dataset.nmol[region]
            nbp = dataset.nbp[region]
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            n_missing = 0
            for start, stop in _row_batches(0, nmol, batch_size):
                batch = dataset.extract(
                    time=time,
                    obs="position",
                    regions=[region],
                    mols=range(start, stop),
                    agg_func=occup_agg,
                )
                rows = batch[region][start:stop]
                block = np.full((stop - start, nbp), np.nan, dtype=np.float64)
                skipped = dataset.no_data[region][start:stop]
                for i, row in enumerate(rows):
                    if row is not None:
                        block[i] = row
                    elif not skipped[i]:
                        n_missing += 1
                _nan_unspanned(block, dataset.spanned[region], start, stop)
                out.write_batch(start, stop, block)
            if n_missing:
                print(
                    f"{n_missing} molecule(s) with no simulation data "
                    f"for region {region!r}; occupancy filled with NaN."
                )
            dataset.analysis[region][name] = out

    def compute_mean_nnuc(
        self,
        dataset: SimDataset,
        *,
        time: int | None = None,
        regions: str | Iterable[str] | None = None,
        name: str = "mean_nnuc",
        record_time: bool = False,
        nworker: int = 1,
    ):
        """
        Calculate the mean number of nucleosomes.

        For `dataset.method == "mcmc"`, this is a Monte Carlo average
        over simulation runs at a specific time point. For `"tm"`, it
        is computed exactly via the transfer-matrix solver, with no
        time axis.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the raw simulation results.
        time : int, optional
            The simulation time point (snapshot) to analyze. If None, the
            final time frame will be used. `mcmc` only; must be None for
            `dataset.method == "tm"`.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "mean_nnuc"
            The key name used to store the resulting DataFrame in
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name
            (e.g., "mean_nnuc_t_100"). `mcmc` only.
        nworker : int, default 1
            Number of worker processes. Values above 1 are only supported
            for `"tm"` datasets.

        Raises
        ------
        ValueError
            If `time` or `record_time` is given for a `"tm"` dataset, or
            (for `"tm"`) if the dataset has no stored energy landscape,
            or (for `"mcmc"`) if `time` is None and no simulation files
            exist for `regions`, or `nworker` is invalid.
        """
        _check_nworker(dataset, nworker)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        if dataset.method == "tm":
            _reject_time(time, record_time)
            _require_edata(dataset)
            _tm_per_mol(
                dataset, regions, name, NucPosTM.getMeanN, nworker=nworker
            )
            return
        self._compute_mean_nnuc_mcmc(
            dataset,
            time=time,
            regions=regions,
            name=name,
            record_time=record_time,
        )

    def _compute_mean_nnuc_mcmc(
        self, dataset, *, time, regions, name, record_time
    ):
        if time is None:
            time = self._final_time(dataset, regions)

        def mean_nnuc_agg(region, mol, nucpos):
            mean_nnuc = 0.0
            for x in nucpos:
                mean_nnuc += len(x)
            mean_nnuc /= float(len(nucpos))
            return mean_nnuc

        mean_nnuc = dataset.extract(
            time=time, obs="position", regions=regions, agg_func=mean_nnuc_agg
        )
        # Store the results
        if record_time:
            name = f"{name}_t_{time}"
        for region in regions:
            dataset.analysis[region][name] = pd.DataFrame(mean_nnuc[region])

    def compute_var_nnuc(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        name: str = "var_nnuc",
        nworker: int = 1,
    ):
        """
        Calculate the exact variance of the nucleosome count.

        Uncertainty has no MCMC equivalent computed here; only
        available for `dataset.method == "tm"` datasets.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "var_nnuc"
            The key name used to store the resulting DataFrame in
            `dataset.analysis`.
        nworker : int, default 1
            Number of worker processes.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, the dataset has no stored
            energy landscape, or `nworker` is not positive.
        """
        _require_tm(dataset, "compute_var_nnuc")
        _check_nworker(dataset, nworker)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        _tm_per_mol(dataset, regions, name, NucPosTM.getVarN, nworker=nworker)

    def compute_var_occup(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        name: str = "var_occup",
        batch_size: int | None = None,
        nworker: int = 1,
    ):
        """
        Calculate the exact per-bp variance of nucleosome occupancy.

        Closed form (occup*(1-occup)): a bp can be covered by at most
        one nucleosome, so occupancy there is Bernoulli. Uncertainty
        has no MCMC equivalent computed here; only available for
        `dataset.method == "tm"` datasets. Nan at base pairs a molecule
        does not span.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "var_occup"
            The key name used to store the resulting `H5Array` in
            `dataset.analysis`.
        batch_size : int, optional
            Number of molecules per worker job (default 64); peak memory
            scales with `2 * nworker * batch_size` molecules.
        nworker : int, default 1
            Number of worker processes.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, the dataset has no stored
            energy landscape, or `nworker` is not positive.
        """
        _require_tm(dataset, "compute_var_occup")
        _check_nworker(dataset, nworker)
        batch_size = _resolve_batch_size(dataset, batch_size)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        _tm_per_bp(
            dataset,
            regions,
            name,
            batch_size,
            NucPosTM.getVarOccup,
            nworker=nworker,
        )

    def compute_most_likely_config(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        name: str = "most_likely",
        nworker: int = 1,
    ):
        """
        Calculate the most likely nucleosome configuration of each molecule.

        This is the single most probable configuration under the posterior
        (maximum a posteriori, MAP), found exactly by a forward
        max-recursion through the transfer matrix. It ignores the
        positional uncertainty of the nucleosomes and tends to be more
        densely packed than typical posterior configurations; see
        `compute_best_match_config` for a call that agrees with the
        occupancy profile. Only available for `dataset.method == "tm"`
        datasets.

        Positions are stored in the flat/offset layout of the raw
        simulation data, together with the energy and log posterior
        probability of each configuration; use `extract_config` to read
        them.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "most_likely"
            Prefix of the keys used to store the results in
            `dataset.analysis`.
        nworker : int, default 1
            Number of worker processes.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, the dataset has no stored
            energy landscape, or `nworker` is not positive.
        """
        _require_tm(dataset, "compute_most_likely_config")
        _check_nworker(dataset, nworker)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        _tm_configs(
            dataset,
            regions,
            name,
            NucPosTM.getMostLikelyConfig,
            nworker=nworker,
        )

    def compute_best_match_config(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        name: str = "best_match",
        gamma: float = 1.0,
        nworker: int = 1,
    ):
        """
        Calculate the configuration that best matches the occupancy profile.

        For each molecule, this is the configuration of non-overlapping
        nucleosome footprints that maximizes the expected number of base
        pairs whose occupancy is called correctly under the posterior
        (maximum expected accuracy, MEA), minus the linker repulsion of
        the model. The penalty keeps out linkers that the model forbids
        (e.g., touching footprints), so the result is always a
        configuration that the model allows. With `gamma = 1`, it is
        the configuration closest to the occupancy profile thresholded
        at 0.5 among those the model allows. Only available for
        `dataset.method == "tm"` datasets. Results are stored as for
        `compute_most_likely_config`.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "best_match"
            Prefix of the keys used to store the results in
            `dataset.analysis`.
        gamma : float, default 1.0
            Weight of correctly called occupied sites relative to
            correctly called empty sites. A site favors being covered when
            its occupancy exceeds 1/(1 + gamma), so larger values call more
            nucleosomes. Must be positive.
        nworker : int, default 1
            Number of worker processes.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, the dataset has no stored energy
            landscape, `gamma` is not positive, or `nworker` is not
            positive.
        """
        if not gamma > 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        _require_tm(dataset, "compute_best_match_config")
        _check_nworker(dataset, nworker)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        _tm_configs(
            dataset,
            regions,
            name,
            NucPosTM.getBestMatchConfig,
            kwargs={"gamma": gamma},
            nworker=nworker,
        )

    def compute_linker_hist(
        self,
        dataset: SimDataset,
        *,
        lmax: int = 200,
        window: tuple[int, int] | Mapping | None = None,
        trim: int = 0,
        regions: str | Iterable[str] | None = None,
        name: str = "linker_hist",
        batch_size: int | None = None,
        nworker: int = 1,
    ):
        """
        Calculate the exact linker-length histogram of each molecule.

        A linker is the DNA between two consecutive nucleosomes, of length
        `b - a - lnuc` for nucleosomes starting at `a` and `b`; the
        stretches between the fiber ends and the outermost nucleosomes
        are not linkers. For each molecule, the expected number of
        linkers of each length under the posterior is computed exactly
        from the pair probabilities of the transfer matrix, with no
        sampling. The values are expected counts, not normalized: a row
        sums to the expected number of linkers in the window. Only
        linkers whose two nucleosome footprints lie inside the window
        are counted. Only available for `dataset.method == "tm"`
        datasets.

        The histogram reflects both the spacing of the nucleosomes and
        the uncertainty of their positions, which widens it; a strongly
        positioned array (e.g., the 601) shows how much. The results
        assume that each molecule is read along the whole fiber; linkers
        in regions filled by `fill_nan` follow the prior.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        lmax : int, default 200
            The longest linker length (bp) with its own bin; longer
            linkers are pooled into the last column ("tail").
        window : tuple of int or dict, optional
            The region `(xlo, xhi)` (bp, half-open, in the coordinates of
            the stored start positions) in which linkers are counted. A
            dict keyed by region gives one window per region (regions not in
            it use the whole fiber; extra keys are ignored). If None
            (default), the whole fiber.
        trim : int, default 0
            Number of base pairs removed from each end of the window,
            e.g., to drop nucleosomes near the fiber ends.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all regions
            in the dataset are analyzed.
        name : str, default "linker_hist"
            The key of the histogram in `dataset.analysis`: an `H5Array`
            of shape (nmol, lmax + 2) with columns "0" to str(lmax) and
            "tail". Molecules with no data get nan rows. The window and
            settings used are stored under `f"{name}_settings"` (a
            one-row DataFrame with lmax, xlo, xhi, and llink).
        batch_size : int, optional
            Number of molecules per worker job (default 64); peak memory
            scales with `2 * nworker * batch_size` molecules.
        nworker : int, default 1
            Number of worker processes.

        Raises
        ------
        TypeError
            If `lmax` or `trim` is not an integer.
        ValueError
            If `dataset.method != "tm"`, the dataset has no stored energy
            landscape, `lmax` or `trim` is negative, `window` is invalid
            or outside the fiber, the window after trimming cannot hold
            two nucleosomes, or `nworker` is not positive.

        Warns
        -----
        UserWarning
            If the dataset was created with `fill_nan`, since linkers in
            filled regions are counted.
        """
        _check_nonneg_int(lmax, "lmax")
        _check_nonneg_int(trim, "trim")
        _check_window(window)
        _require_tm(dataset, "compute_linker_hist")
        _check_nworker(dataset, nworker)
        batch_size = _resolve_batch_size(dataset, batch_size)
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        lnuc = dataset.settings["lnuc"]
        windows = {
            region: _linker_window(
                window, region, dataset.nbp[region], lnuc, trim
            )
            for region in regions
        }
        if dataset.nan_fill is not None:
            _warn(
                f"dataset was created with fill_nan={dataset.nan_fill!r}; "
                "linkers in filled regions, if any, are counted"
            )
        columns = (*(str(i) for i in range(lmax + 1)), "tail")
        for region in regions:
            xlo, xhi = windows[region]
            stat = TMStat(
                name,
                "getLinkerHist",
                {"lmax": lmax, "xlo": xlo, "xhi": xhi},
                ncol=lmax + 2,
                columns=columns,
            )
            _tm_stats(
                dataset,
                [region],
                [stat],
                batch_size=batch_size,
                nworker=nworker,
            )
            dataset.analysis[region][f"{name}_settings"] = pd.DataFrame(
                {
                    "lmax": [lmax],
                    "xlo": [xlo],
                    "xhi": [xhi],
                    "llink": [_region_settings(dataset, region).llink],
                }
            )

    def pool_linker_hist(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        name: str = "linker_hist",
        mols: Iterable[int] | dict | None = None,
        how: str = "sum",
        batch_size: int = 20000,
    ) -> pd.DataFrame:
        """
        Pool the linker histograms of molecules, per region.

        Reads the per-molecule histograms stored by `compute_linker_hist`
        in batches, so memory does not scale with the number of
        molecules. Molecules with no data (nan rows) are skipped.

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the histograms.
        regions : str or iterable of str, optional
            The region(s) to pool. If None (default), all regions
            in the dataset.
        name : str, default "linker_hist"
            The key of the histograms, as passed to `compute_linker_hist`.
        mols : iterable of int or dict, optional
            The molecules to pool (duplicates are counted once). If None
            (default), all molecules. With several regions, a dict keyed
            by region (regions not in it use all molecules; extra keys are
            ignored).
        how : {"sum", "mean"}, default "sum"
            "sum" adds the expected counts of the molecules, so that each
            linker counts equally; "mean" divides the sum by the number
            of molecules with data, giving the expected counts per
            molecule.
        batch_size : int, default 20000
            Number of molecules read (and held in memory) per batch.

        Returns
        -------
        pd.DataFrame
            One row per linker-length bin (labeled "0" to str(lmax), then
            "tail") and one column per region; nan for a region
            with no molecules with data under "mean".

        Raises
        ------
        KeyError
            If no histogram is stored under `name` for a region.
        ValueError
            If `how` is invalid, `mols` is not a dict with several regions
            or is out of range, or the regions have histograms with
            different bins.
        """
        if how not in ("sum", "mean"):
            raise ValueError(f"how must be 'sum' or 'mean', got {how!r}")
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        mols = _per_region(mols, regions, "mols")
        pooled = {}
        columns = None
        for region in regions:
            arr = _stored_linker_hist(dataset, region, name)
            if columns is None:
                columns = list(arr.columns)
            elif list(arr.columns) != columns:
                raise ValueError(
                    f"linker histograms under '{name}' have different "
                    "bins across regions; pool them one region at a time"
                )
            mask = _mol_mask(mols[region], arr.shape[0], region)
            pooled[region] = getattr(arr, how)(
                axis=0, batch_size=batch_size, row_mask=mask
            )
        return pd.DataFrame(pooled, index=pd.Index(columns))

    def extract_linker_stats(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        name: str = "linker_hist",
        mols: Iterable[int] | dict | None = None,
        pooled: bool = False,
        batch_size: int = 20000,
    ) -> pd.DataFrame:
        """
        Calculate the mean and standard deviation of the linker lengths.

        Uses the histograms stored by `compute_linker_hist`, so the
        values are exact posterior averages. Only linkers up to `lmax`
        enter; the tail bin is excluded, since its lengths are not
        resolved. The spread reflects both the irregularity of the
        spacing and the uncertainty of the nucleosome positions; compare
        with a strongly positioned array (e.g., the 601) to separate the
        two.

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the histograms.
        regions : str or iterable of str, optional
            The region(s) to use. If None (default), all regions
            in the dataset.
        name : str, default "linker_hist"
            The key of the histograms, as passed to `compute_linker_hist`.
        mols : iterable of int or dict, optional
            The molecules to use (duplicates are counted once). If None
            (default), all molecules. With several regions, a dict keyed
            by region (regions not in it use all molecules; extra keys are
            ignored).
        pooled : bool, default False
            If False, one row per molecule. If True, one row per
            region, from the linkers of all selected molecules pooled
            together, so that each linker counts equally. The pooled
            standard deviation also includes the differences between the
            mean linker lengths of the molecules, so it is not the mean
            of the per-molecule values.
        batch_size : int, default 20000
            Number of molecules read (and held in memory) per batch.

        Returns
        -------
        pd.DataFrame
            Columns "mean" and "sd" (bp) and "nlink", the expected number
            of linkers up to `lmax`. Indexed by (region, mol) if `pooled`
            is False, otherwise by region. Molecules with no data get nan;
            molecules or regions with no linkers get nan mean and sd.

        Raises
        ------
        KeyError
            If no histogram is stored under `name` for a region.
        ValueError
            If `mols` is not a dict with several regions or is out of
            range.
        """
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        mols = _per_region(mols, regions, "mols")
        columns = ["mean", "sd", "nlink"]
        if pooled:
            rows = []
            for region in regions:
                counts = self.pool_linker_hist(
                    dataset,
                    regions=region,
                    name=name,
                    mols=mols[region],
                    batch_size=batch_size,
                )[region].to_numpy()
                mean, sd, nlink = _linker_moments(counts[None, :-1])
                rows.append([mean[0], sd[0], nlink[0]])
            return pd.DataFrame(
                rows, index=pd.Index(regions, name="region"), columns=columns
            )
        frames = []
        for region in regions:
            arr = _stored_linker_hist(dataset, region, name)
            mask = _mol_mask(mols[region], arr.shape[0], region)
            for mol_ids, rows in _iter_hist_rows(arr, mask, batch_size):
                mean, sd, nlink = _linker_moments(rows[:, :-1])
                index = pd.MultiIndex.from_arrays(
                    [[region] * len(mol_ids), mol_ids], names=["region", "mol"]
                )
                frames.append(
                    pd.DataFrame(
                        {"mean": mean, "sd": sd, "nlink": nlink}, index=index
                    )
                )
        if not frames:
            index = pd.MultiIndex.from_arrays(
                [[], []], names=["region", "mol"]
            )
            return pd.DataFrame(columns=columns, index=index, dtype=float)
        return pd.concat(frames)

    def extract_config(
        self,
        dataset: SimDataset,
        *,
        region: str,
        name: str | None = None,
        run: int | None = None,
        time: int | None = None,
        mols: Iterable[int] | None = None,
        nworker: int = 1,
    ) -> list[np.ndarray | None]:
        """
        Extract one nucleosome configuration per molecule.

        With `name`, reads the configurations stored by
        `compute_most_likely_config` or `compute_best_match_config`.
        Without it, reads the raw configurations: run `run` at time
        `time` of an MCMC dataset, or exact sample `run` of a TM
        dataset. Nothing is computed.

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the configurations.
        region : str
            The region of the molecules.
        name : str, optional
            The prefix the configurations were stored under (e.g.,
            "most_likely" or "best_match"). If None (default), raw
            configurations are read.
        run : int, optional
            The MCMC run or TM sample to read; 0 if None. Only used if
            `name` is None.
        time : int, optional
            The MCMC time point to read; the final frame if None. Only
            used for MCMC datasets with `name` None.
        mols : iterable of int, optional
            The molecules to read, in this order. If None (default), all
            molecules of `region`.
        nworker : int, default 1
            Number of threads used to read raw MCMC files.

        Returns
        -------
        list of np.ndarray or None
            The nucleosome start positions of each molecule, in
            increasing order, or None for a molecule with no data (see
            `SimDataset.no_data`) or no file on disk. An empty array is
            a configuration without nucleosomes.

        Raises
        ------
        ValueError
            If `run` or `time` is given with `name`, if `time` is given
            for a TM dataset, if a TM dataset stores no samples
            (`nsim` is 0) and `name` is None, or if `time` is not a
            recorded time point of a run.
        KeyError
            If no configurations are stored under `name` for `region`.
        IndexError
            If a molecule or `run` is out of range.
        """
        nmol = dataset.nmol[region]
        mol_ids = range(nmol) if mols is None else [int(m) for m in mols]
        for mol in mol_ids:
            if not 0 <= mol < nmol:
                raise IndexError(f"mol must be in [0, {nmol}), got {mol}")
        if name is not None:
            if run is not None or time is not None:
                raise ValueError("run and time cannot be given with name")
            return _stored_configs(dataset, region, name, mol_ids)
        run = 0 if run is None else int(run)
        if dataset.method == "tm":
            if time is not None:
                raise ValueError(
                    "time is not meaningful for method='tm' datasets"
                )
            if dataset.nsim == 0:
                raise ValueError(
                    "no samples stored (nsim is 0); pass the name of a "
                    "compute_most_likely_config or compute_best_match_config "
                    "result"
                )
        if not 0 <= run < dataset.nsim:
            raise IndexError(f"run must be in [0, {dataset.nsim}), got {run}")
        if dataset.method == "tm":
            return [
                None
                if dataset.no_data[region][mol]
                or not dataset.tm_file(region, mol).exists()
                else np.asarray(dataset.raw[region, mol, run].position)
                for mol in mol_ids
            ]
        if time is None:
            time = self._final_time(dataset, [region])

        def read(mol):
            sim_file = dataset.mcmc_file(region, mol, run)
            if not sim_file.exists():
                return None
            try:
                return SimMCMCData._cached_extract(
                    str(sim_file.resolve()), int(time), "position"
                )
            except KeyError:
                raise ValueError(
                    f"time {time} is not recorded in run {run} of molecule "
                    f"{mol} of region '{region}'"
                ) from None

        out = dict(_threaded_map(read, mol_ids, nworker=nworker))
        return [out[mol] for mol in mol_ids]

    def sort_by_linkage(
        self,
        dataset: SimDataset,
        *,
        regions: str | Iterable[str] | None = None,
        data_name: str | None = None,
        config_name: str | None = None,
        store_link_mat: bool = True,
        link_mat_name: str | None = None,
        metric: str = "euclidean",
        method: str = "ward",
        batch_size: int = 20000,
        fill_nan: str | float | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Cluster molecules by similarity, for sorted plots.

        Computes a hierarchical-clustering linkage of the molecules, from
        an analysis array (`data_name`, e.g. the occupancy) or from the
        configurations stored by `compute_config` (`config_name`).
        Nothing is reordered or copied: pass the linkage as `link_mat` to
        `plot_occup` or `plot_config`, which put their rows in its leaf
        order, so any number of plots of the same molecules share one
        order.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the data to cluster.
        regions : str or iterable of str, optional
            The region(s) to process. If None (default), all
            regions in the dataset are processed.
        data_name : str, optional
            The key of the analysis array to cluster, looked up in
            `dataset.analysis[region]`; "occup" if neither `data_name` nor
            `config_name` is given.
        config_name : str, optional
            The `name` of configurations stored by `compute_config`.
            Molecules are clustered by their footprints (1 covered, 0
            linker): with the default ward/euclidean, the distance of two
            molecules is the square root of the number of base pairs where
            their calls differ. Molecules with no data are nan rows, so
            they need `fill_nan`. The footprints are streamed through a
            temporary float32 file in batches of `batch_size`.
        store_link_mat : bool, default True
            Whether to also persist the linkage matrix into
            `dataset.analysis[region][link_mat_name]`. If False, the
            linkage matrix is only returned, not stored.
        link_mat_name : str, optional
            The key used to store the linkage matrix if
            `store_link_mat` is True. If None (default),
            `f"{data_name}_linkage"` or `f"{config_name}_linkage"` is
            used.
        metric : str, default "euclidean"
            Distance metric, forwarded to `utils.compute_linkage`.
        method : str, default "ward"
            Linkage method, forwarded to `utils.compute_linkage`.
        batch_size : int, default 20000
            Number of rows processed (and held in memory) per batch.
        fill_nan : {"mean"}, float, or None, default None
            How to handle `nan` values, forwarded to
            `utils.compute_linkage`. Molecules with no data (all nan,
            see `SimDataset.no_data`) need it.

        Returns
        -------
        dict of str to np.ndarray
            A mapping from region name to that region's
            linkage matrix, to pass as `link_mat` to `SimPlot.plot_occup`
            or `SimPlot.plot_config`. Also stored in
            `dataset.analysis[region][link_mat_name]` if `store_link_mat`
            is True.

        Raises
        ------
        KeyError
            If `data_name` (or the configurations of `config_name`) is not
            found for any of the selected regions.
        ValueError
            If both `data_name` and `config_name` are given, or if some
            molecule has no data and `fill_nan` is None.

        Warns
        -----
        UserWarning
            If there are more molecules than `utils.compute_linkage`
            handles comfortably (its pairwise distances need memory
            quadratic in the number of molecules).
        """
        if data_name is not None and config_name is not None:
            raise ValueError("data_name and config_name cannot both be given")
        if config_name is None and data_name is None:
            data_name = "occup"
        base_name = data_name if config_name is None else config_name
        regions = _normalize_regions(regions, default_regions=dataset.regions)
        tmp_dir = dataset.resolve_tmp_dir()
        link_mats = {}
        for region in regions:
            if config_name is None:
                data = dataset.analysis[region][data_name]
            else:
                data = self._config_footprints(
                    dataset, region, config_name, batch_size, tmp_dir
                )
            try:
                _, link_mat = utils.compute_linkage(
                    data,
                    metric=metric,
                    method=method,
                    batch_size=batch_size,
                    dir=tmp_dir,
                    fill_nan=fill_nan,
                )
            finally:
                if config_name is not None:
                    data.close()
            if store_link_mat:
                lname = (
                    link_mat_name
                    if link_mat_name is not None
                    else f"{base_name}_linkage"
                )
                dataset.analysis[region][lname] = pd.DataFrame(link_mat)
            link_mats[region] = link_mat
        return link_mats

    def _config_footprints(self, dataset, region, name, batch_size, tmp_dir):
        """
        Write the stored configurations of `name` as a temporary footprint
        array (float32: 1 covered, 0 linker, nan no data).

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the configurations.
        region : str
            The region.
        name : str
            The `name` of the configurations.
        batch_size : int
            The number of molecules written per batch.
        tmp_dir : str
            The directory of the temporary file.

        Returns
        -------
        H5Array
            The footprints, molecules by base pairs; close it when done.
        """
        configs = self.extract_config(dataset, region=region, name=name)
        nbp = dataset.nbp[region]
        nuc_map = NucFiberMap(nbp=nbp, lnuc=dataset.settings["lnuc"])
        out = H5Array.create(
            (len(configs), nbp), dtype=np.float32, dir=tmp_dir
        )
        for start, stop in _row_batches(0, len(configs), batch_size):
            block = np.full((stop - start, nbp), np.nan, dtype=np.float32)
            for i, starts in enumerate(configs[start:stop]):
                if starts is not None:
                    block[i] = nuc_map.project(np.asarray(starts))
            out.write_batch(start, stop, block)
        return out
