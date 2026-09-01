# analysis.py

from catella.simulation.results import SimDataset
from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
import pandas as pd
from catella import utils
from catella.h5_array import H5Array

@dataclass(slots=True)
class NucFiberMap:
    """
    A mapping tool to project nucleosome coordinates onto a 1D DNA lattice.

    This class converts discrete start positions into a continuous binary 
    occupancy representation, simulating the physical footprint of nucleosomes 
    along a DNA fiber of fixed length.
    """
    
    nucbp : int
    """The footprint size (in base pairs) of a single nucleosome."""
    
    nbp : int
    """The total number of base pairs in the DNA or chromatin fiber."""

    def project(self, nucpos : np.ndarray) -> np.ndarray:
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
        diff = np.zeros(self.nbp+1, dtype=np.int32)
        starts = nucpos[(nucpos >= 0) & (nucpos < self.nbp)]
        ends = (starts + self.nucbp).clip(max=self.nbp)        
        np.add.at(diff, starts, 1)
        np.add.at(diff, ends, -1)
        return (np.cumsum(diff)[:-1] > 0).astype(np.int8)

    def aggregate(self,
                  samples: Iterable[np.ndarray],
                  norm: bool = False) -> np.ndarray:
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
        diff = np.zeros(self.nbp+1, dtype=np.int32)
        for nucpos in samples:
            starts = nucpos[(nucpos >= 0) & (nucpos < self.nbp)]
            ends = (starts + self.nucbp).clip(max=self.nbp)
            np.add.at(diff, starts, 1)
            np.add.at(diff, ends, -1)
        occup = np.cumsum(diff)[:-1]

        # Normalize the data if needed
        if norm:
            return occup.astype(np.float64) / max(len(samples),1)
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
            diff = np.zeros(self.nbp+1, dtype=np.int32)
            starts = nucpos[(nucpos >= 0) & (nucpos < self.nbp)]
            ends = (starts + self.nucbp).clip(max=self.nbp)            
            np.add.at(diff, starts, 1)
            np.add.at(diff, ends, -1)
            matrix[i,:] = (np.cumsum(diff)[:-1] > 0).astype(np.int8)
        return matrix
    
    
class SimAnalysis:
    """
    A collection of post-processing tools for simulation data.

    This class provides methods to calculate physical properties such as 
    nucleosome occupancy and DNA accessibility from raw simulation
    trajectories. Results are stored directly in the provided dataset's
    analysis map.
    """
    def compute_occup(self, *,
                      dataset : SimDataset,
                      time : int | None = None,
                      chroms : str | Iterable[str] | None = None,
                      name : str = "occup",
                      record_time : bool = False,
                      batch_size : int = 20000):
        """
        Calculate nucleosome occupancy at a specific time point.

        Occupancy is defined as the fraction of simulation runs where a
        base pair is covered by a nucleosome. Molecules are processed in
        batches and streamed to a disk-backed array so that peak memory
        scales with `batch_size` rather than the total number of
        molecules.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the raw simulation results.
        time : int, optional
            The simulation time point (snapshot) to analyze. If None, the
            final time frame will be used.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes
            in the dataset are analyzed.
        name : str, default "occup"
            The key name used to store the resulting `H5Array` in
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name
            (e.g., "occup_t_100").
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        nuc_map = {chrom:NucFiberMap(dataset.settings["nucbp"],
                                     dataset.nbp[chrom])
                   for chrom in chroms}
        if time is None:
            time = dataset.raw[chroms[0],0,0].time[-1]
        def occup_agg(chrom, mol, nucpos):
            return nuc_map[chrom].aggregate(nucpos, norm=True)

        if record_time: name = f"{name}_t_{time}"
        tmp_dir = dataset.resolve_tmp_dir()
        for chrom in chroms:
            nmol = dataset.nmol[chrom]
            nbp = dataset.nbp[chrom]
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = dataset.extract(time=time, obs="position",
                                        chroms=[chrom],
                                        mols=range(start, stop),
                                        agg_func=occup_agg)
                block = np.asarray(batch[chrom][start:stop])
                out.write_batch(start, stop, block)
            dataset.analysis[chrom][name] = out

    def compute_access(self, *,
                       dataset : SimDataset,
                       time : int | None = None,
                       chroms : str | Iterable[str] | None = None,
                       occup_name : str = "occup",
                       access_name : str = "access",
                       record_time : bool = False,
                       batch_size : int = 20000):
        """
        Calculate DNA accessibility based on nucleosome occupancy.

        Accessibility (A) is defined as 1.0 minus Occupancy (O). This
        method automatically triggers occupancy computation if the
        required data is not found in the dataset.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the raw simulation results.
        time : int, optional
            The simulation time point (snapshot) to analyse. If None, the
            final time frame will be used.
        chroms : str or iterable of str, optional
            The chromosome(s) to process.
        occup_name : str, default "occup"
            The name of the occupancy data to use or create.
        access_name : str, default "access"
            The key name used to store the resulting accessibility
            `H5Array` in `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name
            (e.g., "access_t_100").
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        if time is None:
            time = dataset.raw[chroms[0],0,0].time[-1]
        if record_time:
            access_name = f"{access_name}_t_{time}"
            occup_name = f"{occup_name}_t_{time}"
        # Compute nucleosome occupancy if not done so
        has_occup = True
        for chrom in chroms:
            if occup_name not in dataset.analysis[chrom]:
                has_occup = False
                break
        if not has_occup:
            self.compute_occup(dataset=dataset, time=time,
                                   chroms = chroms, name = occup_name,
                                   record_time = record_time,
                                   batch_size = batch_size)
        # Compute accessibility A = 1.0 - O
        tmp_dir = dataset.resolve_tmp_dir()
        for chrom in chroms:
            occup_arr = dataset.analysis[chrom][occup_name]
            nmol, nbp = occup_arr.shape
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                out.write_batch(start, stop, 1.0 - occup_arr[start:stop, :])
            dataset.analysis[chrom][access_name] = out

    def compute_mean_nnuc(self, *,
                          dataset : SimDataset,
                          time : int | None = None,
                          chroms : str | Iterable[str] | None = None,
                          name : str = "mean_nnuc",
                          record_time : bool = False):
        """
        Calculate the mean number of nucleosomes at a specific time point.

        Parameters
        ----------
        dataset : SimDataset
            The dataset containing the raw simulation results.
        time : int, optional
            The simulation time point (snapshot) to analyze. If None, the
            final time frame will be used.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes 
            in the dataset are analyzed.
        name : str, default "mean_nnuc"
            The key name used to store the resulting DataFrame in 
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name 
            (e.g., "mean_nnuc_t_100").
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        if time is None:
            time = dataset.raw[chroms[0],0,0].time[-1]        
        def mean_nnuc_agg(chrom, mol, nucpos):
            mean_nnuc = 0.0
            for x in nucpos:
                mean_nnuc += len(x)
            mean_nnuc /= float(len(nucpos))
            return mean_nnuc
        mean_nnuc = dataset.extract(time=time, obs="position",
                                    agg_func=mean_nnuc_agg)
        # Store the results
        if record_time: name = f"{name}_t_{time}"
        for chrom in chroms:
            dataset.analysis[chrom][name] = pd.DataFrame(mean_nnuc[chrom])

    def sort_by_linkage(self, *,
                        dataset : SimDataset,
                        chroms : str | Iterable[str] | None = None,
                        data_name : str = "occup",
                        sorted_name : str | None = None,
                        store_link_mat : bool = True,
                        link_mat_name : str | None = None,
                        metric : str = "euclidean",
                        method : str = "ward",
                        batch_size : int = 20000,
                        fill_nan : str | float | None = None
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
                data, metric=metric, method=method, batch_size=batch_size,
                dir=tmp_dir, fill_nan=fill_nan)
            if isinstance(data, H5Array):
                sorted_data = data.reorder_rows(order, dir=tmp_dir,
                                                batch_size=batch_size)
            else:
                sorted_data = data.iloc[order]
            name = sorted_name if sorted_name is not None \
                else f"{data_name}_sorted"
            dataset.analysis[chrom][name] = sorted_data
            if store_link_mat:
                lname = link_mat_name if link_mat_name is not None \
                    else f"{data_name}_linkage"
                dataset.analysis[chrom][lname] = pd.DataFrame(link_mat)
            link_mats[chrom] = link_mat
        return link_mats

