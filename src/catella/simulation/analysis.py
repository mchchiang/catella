# analysis.py

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM

from catella import utils
from catella.h5_array import H5Array
from catella.simulation._io import _pack_jagged
from catella.simulation.config import SimSettings
from catella.simulation.results import SimDataset


@dataclass(slots=True)
class NucFiberMap:
    """
    A mapping tool to project nucleosome coordinates onto a 1D DNA lattice.

    This class converts discrete start positions into a continuous binary
    occupancy representation, simulating the physical footprint of nucleosomes
    along a DNA fiber of fixed length.
    """

    nucbp: int
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
        ends = (starts + self.nucbp).clip(max=self.nbp)
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
            ends = (starts + self.nucbp).clip(max=self.nbp)
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
            ends = (starts + self.nucbp).clip(max=self.nbp)
            np.add.at(diff, starts, 1)
            np.add.at(diff, ends, -1)
            matrix[i, :] = (np.cumsum(diff)[:-1] > 0).astype(np.int8)
        return matrix


class SimAnalysis:
    """
    A collection of post-processing tools for simulation data.

    This class provides methods to calculate physical properties such as
    nucleosome occupancy and DNA accessibility from raw simulation
    trajectories. Results are stored directly in the provided dataset's
    analysis map.
    """

    def _require_tm_eseq(self, dataset: SimDataset):
        if not dataset.has_eseq:
            raise ValueError(
                "exact computation requires the per-molecule energy "
                "landscape; recreate the dataset with store_eseq=True"
            )

    def _final_time(self, dataset: SimDataset, chroms) -> int:
        """Return the final time of the first simulation run on disk."""
        run_id = next(dataset.iter_runs(chroms), None)
        if run_id is None:
            raise ValueError(
                f"no simulation files found for chroms {list(chroms)}"
            )
        return dataset.raw[run_id].time[-1]

    def _solved_tm(self, dataset: SimDataset, chrom: str, mol: int):
        """Build and solve a NucPosTM for one molecule, at the
        temperature settings.end_temp evaluates the exact equilibrium
        distribution at."""
        settings = SimSettings(**dict(dataset.settings)).for_chrom(chrom)
        tm = NucPosTM(
            settings.nucbp,
            dataset.nbp[chrom],
            settings.llink,
            settings.elink,
            settings.mu,
            1,
        )
        tm.setEnergy(dataset.eseq[chrom][mol].tolist(), settings.emax)
        tm.solve(settings.end_temp)
        return tm

    def compute_occup(
        self,
        dataset: SimDataset,
        *,
        time: int | None = None,
        chroms: str | Iterable[str] | None = None,
        name: str = "occup",
        record_time: bool = False,
        batch_size: int = 20000,
    ):
        """
        Calculate nucleosome occupancy.

        For `dataset.method == "mcmc"`, occupancy is the fraction of
        simulation runs where a base pair is covered by a nucleosome,
        at a specific time point. For `"tm"`, occupancy is computed
        exactly via the transfer-matrix solver, with no time axis.
        Molecules are processed in batches and streamed to a
        disk-backed array so that peak memory scales with `batch_size`
        rather than the total number of molecules.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the raw simulation results.
        time : int, optional
            The simulation time point (snapshot) to analyze. If None, the
            final time frame will be used. `mcmc` only; must be None for
            `dataset.method == "tm"`.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "occup"
            The key name used to store the resulting `H5Array` in
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name
            (e.g., "occup_t_100"). `mcmc` only.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.

        Raises
        ------
        ValueError
            If `time` or `record_time` is given for a `"tm"` dataset, or
            (for `"tm"`) if the dataset has no stored energy landscape,
            or (for `"mcmc"`) if `time` is None and no simulation files
            exist for `chroms`.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        if dataset.method == "tm":
            if time is not None or record_time:
                raise ValueError(
                    "'time'/'record_time' are not meaningful for "
                    "method='tm' datasets (no time axis)"
                )
            self._compute_occup_tm(
                dataset, chroms=chroms, name=name, batch_size=batch_size
            )
            return
        self._compute_occup_mcmc(
            dataset,
            time=time,
            chroms=chroms,
            name=name,
            record_time=record_time,
            batch_size=batch_size,
        )

    def _compute_occup_mcmc(
        self, dataset, *, time, chroms, name, record_time, batch_size
    ):
        nuc_map = {
            chrom: NucFiberMap(dataset.settings["nucbp"], dataset.nbp[chrom])
            for chrom in chroms
        }
        if time is None:
            time = self._final_time(dataset, chroms)

        def occup_agg(chrom, mol, nucpos):
            return nuc_map[chrom].aggregate(nucpos, norm=True)

        if record_time:
            name = f"{name}_t_{time}"
        tmp_dir = dataset.resolve_tmp_dir()
        for chrom in chroms:
            nmol = dataset.nmol[chrom]
            nbp = dataset.nbp[chrom]
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            n_missing = 0
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = dataset.extract(
                    time=time,
                    obs="position",
                    chroms=[chrom],
                    mols=range(start, stop),
                    agg_func=occup_agg,
                )
                rows = batch[chrom][start:stop]
                block = np.full((stop - start, nbp), np.nan, dtype=np.float64)
                for i, row in enumerate(rows):
                    if row is not None:
                        block[i] = row
                    else:
                        n_missing += 1
                out.write_batch(start, stop, block)
            if n_missing:
                print(
                    f"{n_missing} molecule(s) with no simulation data "
                    f"for chrom {chrom!r}; occupancy filled with NaN."
                )
            dataset.analysis[chrom][name] = out

    def _compute_occup_tm(self, dataset, *, chroms, name, batch_size):
        self._require_tm_eseq(dataset)
        tmp_dir = dataset.resolve_tmp_dir()
        for chrom in chroms:
            nmol = dataset.nmol[chrom]
            nbp = dataset.nbp[chrom]
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                block = np.empty((stop - start, nbp), dtype=np.float64)
                for i, mol in enumerate(range(start, stop)):
                    tm = self._solved_tm(dataset, chrom, mol)
                    block[i] = tm.getOccup()
                out.write_batch(start, stop, block)
            dataset.analysis[chrom][name] = out

    def compute_mean_nnuc(
        self,
        dataset: SimDataset,
        *,
        time: int | None = None,
        chroms: str | Iterable[str] | None = None,
        name: str = "mean_nnuc",
        record_time: bool = False,
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
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "mean_nnuc"
            The key name used to store the resulting DataFrame in
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name
            (e.g., "mean_nnuc_t_100"). `mcmc` only.

        Raises
        ------
        ValueError
            If `time` or `record_time` is given for a `"tm"` dataset, or
            (for `"tm"`) if the dataset has no stored energy landscape,
            or (for `"mcmc"`) if `time` is None and no simulation files
            exist for `chroms`.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        if dataset.method == "tm":
            if time is not None or record_time:
                raise ValueError(
                    "'time'/'record_time' are not meaningful for "
                    "method='tm' datasets (no time axis)"
                )
            self._compute_mean_nnuc_tm(dataset, chroms=chroms, name=name)
            return
        self._compute_mean_nnuc_mcmc(
            dataset,
            time=time,
            chroms=chroms,
            name=name,
            record_time=record_time,
        )

    def _compute_mean_nnuc_mcmc(
        self, dataset, *, time, chroms, name, record_time
    ):
        if time is None:
            time = self._final_time(dataset, chroms)

        def mean_nnuc_agg(chrom, mol, nucpos):
            mean_nnuc = 0.0
            for x in nucpos:
                mean_nnuc += len(x)
            mean_nnuc /= float(len(nucpos))
            return mean_nnuc

        mean_nnuc = dataset.extract(
            time=time, obs="position", chroms=chroms, agg_func=mean_nnuc_agg
        )
        # Store the results
        if record_time:
            name = f"{name}_t_{time}"
        for chrom in chroms:
            dataset.analysis[chrom][name] = pd.DataFrame(mean_nnuc[chrom])

    def _compute_mean_nnuc_tm(self, dataset, *, chroms, name):
        self._require_tm_eseq(dataset)
        for chrom in chroms:
            nmol = dataset.nmol[chrom]
            values = np.empty(nmol, dtype=np.float64)
            for mol in range(nmol):
                tm = self._solved_tm(dataset, chrom, mol)
                values[mol] = tm.getMeanN()
            dataset.analysis[chrom][name] = pd.DataFrame(values)

    def compute_var_nnuc(
        self,
        dataset: SimDataset,
        *,
        chroms: str | Iterable[str] | None = None,
        name: str = "var_nnuc",
    ):
        """
        Calculate the exact variance of the nucleosome count.

        Uncertainty has no MCMC equivalent computed here; only
        available for `dataset.method == "tm"` datasets.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "var_nnuc"
            The key name used to store the resulting DataFrame in
            `dataset.analysis`.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, or the dataset has no stored
            energy landscape.
        """
        if dataset.method != "tm":
            raise ValueError(
                "compute_var_nnuc requires a method='tm' dataset"
            )
        self._require_tm_eseq(dataset)
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        for chrom in chroms:
            nmol = dataset.nmol[chrom]
            values = np.empty(nmol, dtype=np.float64)
            for mol in range(nmol):
                tm = self._solved_tm(dataset, chrom, mol)
                values[mol] = tm.getVarN()
            dataset.analysis[chrom][name] = pd.DataFrame(values)

    def compute_var_occup(
        self,
        dataset: SimDataset,
        *,
        chroms: str | Iterable[str] | None = None,
        name: str = "var_occup",
        batch_size: int = 20000,
    ):
        """
        Calculate the exact per-bp variance of nucleosome occupancy.

        Closed form (occup*(1-occup)): a bp can be covered by at most
        one nucleosome, so occupancy there is Bernoulli. Uncertainty
        has no MCMC equivalent computed here; only available for
        `dataset.method == "tm"` datasets.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "var_occup"
            The key name used to store the resulting `H5Array` in
            `dataset.analysis`.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, or the dataset has no stored
            energy landscape.
        """
        if dataset.method != "tm":
            raise ValueError(
                "compute_var_occup requires a method='tm' dataset"
            )
        self._require_tm_eseq(dataset)
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        tmp_dir = dataset.resolve_tmp_dir()
        for chrom in chroms:
            nmol = dataset.nmol[chrom]
            nbp = dataset.nbp[chrom]
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                block = np.empty((stop - start, nbp), dtype=np.float64)
                for i, mol in enumerate(range(start, stop)):
                    tm = self._solved_tm(dataset, chrom, mol)
                    block[i] = tm.getVarOccup()
                out.write_batch(start, stop, block)
            dataset.analysis[chrom][name] = out

    def compute_most_likely_config(
        self,
        dataset: SimDataset,
        *,
        chroms: str | Iterable[str] | None = None,
        name: str = "most_likely",
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
        probability of each configuration; use `get_config` to read one
        molecule.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "most_likely"
            Prefix of the keys used to store the results in
            `dataset.analysis`.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, or the dataset has no stored
            energy landscape.
        """
        self._compute_configs(
            dataset,
            chroms=chroms,
            name=name,
            caller="compute_most_likely_config",
            get_config=lambda tm: tm.getMostLikelyConfig(),
        )

    def compute_best_match_config(
        self,
        dataset: SimDataset,
        *,
        chroms: str | Iterable[str] | None = None,
        name: str = "best_match",
        gamma: float = 1.0,
    ):
        """
        Calculate the configuration that best matches the occupancy profile.

        For each molecule, this is the configuration of non-overlapping
        nucleosome footprints that maximizes the expected number of base
        pairs whose occupancy is called correctly under the posterior
        (maximum expected accuracy, MEA). With `gamma = 1`, it is the
        valid configuration closest to the occupancy profile thresholded
        at 0.5. Only available for `dataset.method == "tm"` datasets.
        Results are stored as for `compute_most_likely_config`.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the stored energy landscape.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "best_match"
            Prefix of the keys used to store the results in
            `dataset.analysis`.
        gamma : float, default 1.0
            Weight of correctly called occupied sites relative to
            correctly called empty sites. A site favors being covered when
            its occupancy exceeds 1/(1 + gamma), so larger values call more
            nucleosomes. Must be positive.

        Raises
        ------
        ValueError
            If `dataset.method != "tm"`, the dataset has no stored energy
            landscape, or `gamma` is not positive.
        """
        if not gamma > 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self._compute_configs(
            dataset,
            chroms=chroms,
            name=name,
            caller="compute_best_match_config",
            get_config=lambda tm: tm.getBestMatchConfig(gamma),
        )

    def _compute_configs(self, dataset, *, chroms, name, caller, get_config):
        if dataset.method != "tm":
            raise ValueError(f"{caller} requires a method='tm' dataset")
        self._require_tm_eseq(dataset)
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        for chrom in chroms:
            settings = SimSettings(**dict(dataset.settings)).for_chrom(chrom)
            nmol = dataset.nmol[chrom]
            positions = []
            energy = np.empty(nmol, dtype=np.float64)
            log_prob = np.empty(nmol, dtype=np.float64)
            for mol in range(nmol):
                tm = self._solved_tm(dataset, chrom, mol)
                pos, energy[mol] = get_config(tm)
                positions.append(np.asarray(pos, dtype=np.int32))
                log_prob[mol] = -energy[mol] / settings.end_temp - tm.getLogZ()
            flat, offset = _pack_jagged(positions)
            analysis = dataset.analysis[chrom]
            analysis[f"{name}_position_flat"] = pd.DataFrame(
                {"position": flat}
            )
            analysis[f"{name}_position_offset"] = pd.DataFrame(
                {"offset": offset}
            )
            analysis[f"{name}_energy"] = pd.DataFrame(
                {"energy": energy, "log_prob": log_prob}
            )

    def get_config(
        self, dataset: SimDataset, chrom: str, mol: int, *, name: str
    ) -> np.ndarray:
        """
        Get the stored configuration of one molecule.

        Reads the results of `compute_most_likely_config` or
        `compute_best_match_config`.

        Parameters
        ----------
        dataset : SimDataset
            The dataset holding the analysis results.
        chrom : str
            The chromosome of the molecule.
        mol : int
            The molecule index within `chrom`.
        name : str
            The prefix the configurations were stored under (e.g.,
            "most_likely" or "best_match").

        Returns
        -------
        np.ndarray
            The nucleosome start positions of the molecule, in increasing
            order.

        Raises
        ------
        KeyError
            If no configurations are stored under `name` for `chrom`.
        IndexError
            If `mol` is out of range.
        """
        analysis = dataset.analysis[chrom]
        flat = analysis[f"{name}_position_flat"]["position"].to_numpy()
        offset = analysis[f"{name}_position_offset"]["offset"].to_numpy()
        nmol = len(offset) - 1
        if not 0 <= mol < nmol:
            raise IndexError(f"mol must be in [0, {nmol}), got {mol}")
        return flat[int(offset[mol]) : int(offset[mol + 1])]

    def sort_by_linkage(
        self,
        dataset: SimDataset,
        *,
        chroms: str | Iterable[str] | None = None,
        data_name: str = "occup",
        sorted_name: str | None = None,
        store_link_mat: bool = True,
        link_mat_name: str | None = None,
        metric: str = "euclidean",
        method: str = "ward",
        batch_size: int = 20000,
        fill_nan: str | float | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Sort molecules by hierarchical-clustering similarity.

        Reorder the rows of a previously-computed analysis array so
        that similar molecules sit next to each other, matching the
        leaf order of a hierarchical-clustering dendrogram.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the analysis array to sort.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all
            chromosomes in the dataset are processed.
        data_name : str, default "occup"
            The key of the analysis array to sort, looked up in
            `dataset.analysis[chrom]`.
        sorted_name : str, optional
            The key used to store the sorted result. If None
            (default), `f"{data_name}_sorted"` is used.
        store_link_mat : bool, default True
            Whether to also persist the linkage matrix into
            `dataset.analysis[chrom][link_mat_name]`. If False, the
            linkage matrix is only returned, not stored.
        link_mat_name : str, optional
            The key used to store the linkage matrix if
            `store_link_mat` is True. If None (default),
            `f"{data_name}_linkage"` is used.
        metric : str, default "euclidean"
            Distance metric, forwarded to `utils.compute_linkage`.
        method : str, default "ward"
            Linkage method, forwarded to `utils.compute_linkage`.
        batch_size : int, default 20000
            Number of rows processed (and held in memory) per batch.
        fill_nan : {"mean"}, float, or None, default None
            How to handle `nan` values, forwarded to
            `utils.compute_linkage`.

        Returns
        -------
        dict of str to np.ndarray
            A mapping from chromosome name to that chromosome's
            linkage matrix, for optional immediate use (e.g. passing
            straight to `SimPlot.plot_occup`'s `link_mat` argument).
            Also persisted into `dataset.analysis[chrom][link_mat_name]`
            if `store_link_mat` is True.

        Raises
        ------
        KeyError
            If `data_name` is not found in `dataset.analysis[chrom]`
            for any of the selected chromosomes.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        tmp_dir = dataset.resolve_tmp_dir()
        link_mats = {}
        for chrom in chroms:
            data = dataset.analysis[chrom][data_name]
            order, link_mat = utils.compute_linkage(
                data,
                metric=metric,
                method=method,
                batch_size=batch_size,
                dir=tmp_dir,
                fill_nan=fill_nan,
            )
            if isinstance(data, H5Array):
                sorted_data = data.reorder_rows(
                    order, dir=tmp_dir, batch_size=batch_size
                )
            else:
                sorted_data = data.iloc[order]
            name = (
                sorted_name
                if sorted_name is not None
                else f"{data_name}_sorted"
            )
            dataset.analysis[chrom][name] = sorted_data
            if store_link_mat:
                lname = (
                    link_mat_name
                    if link_mat_name is not None
                    else f"{data_name}_linkage"
                )
                dataset.analysis[chrom][lname] = pd.DataFrame(link_mat)
            link_mats[chrom] = link_mat
        return link_mats
