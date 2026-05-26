# analysis.py

from .results import SimDataset
from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
import pandas as pd
from .. import utils

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
                      record_time : bool = False):
        """
        Calculate nucleosome occupancy at a specific time point.

        Occupancy is defined as the fraction of simulation runs where a 
        base pair is covered by a nucleosome.

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
            The key name used to store the resulting DataFrame in 
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name 
            (e.g., "occup_t_100").
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        nuc_map = {chrom:NucFiberMap(dataset.settings["nucbp"],
                                     dataset.nbp[chrom])
                   for chrom in chroms}
        if time is None:
            time = dataset.raw[chroms[0],0,0].time[-1]
        def occup_agg(chrom, mol, nucpos):
            return nuc_map[chrom].aggregate(nucpos, norm=True)
        occup = dataset.extract(time=time, obs="position", agg_func=occup_agg)

        # Store the results 
        if record_time: name = f"{name}_t_{time}"
        for chrom in chroms:
            dataset.analysis[chrom][name] = pd.DataFrame(occup[chrom])

    def compute_access(self, *,
                       dataset : SimDataset,
                       time : int | None = None,                       
                       chroms : str | Iterable[str] | None = None,
                       occup_name : str = "occup",
                       access_name : str = "access",
                       record_time : bool = False):
        """
        Calculate DNA accessibility based on nucleosome occupancy.

        Accessibility (A) is defined as 1.0 minus Occupancy (O). This method 
        automatically triggers occupancy computation if the required data 
        is not found in the dataset.

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
            DataFrame in `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name 
            (e.g., "access_t_100").
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
                                   record_time = record_time)
        # Compute accessibility A = 1.0 - O
        for chrom in chroms:
            df_occup = dataset.analysis[chrom][occup_name]
            dataset.analysis[chrom][access_name] = 1.0-df_occup

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
        
