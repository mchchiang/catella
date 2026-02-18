# analysis.py

from .results import SimDataset
from pathlib import Path
from dataclasses import dataclass
from collections.abc import Iterable
import numpy as np
import pandas as pd
from .. import utils

class SimAnalysis:
    """
    A collection of post-processing tools for simulation data.

    This class provides methods to calculate physical properties such as 
    nucleosome occupancy and DNA accessibility from raw simulation
    trajectories.
    """
    def compute_occupancy(self,
                          nucbp : int,
                          time : int,
                          dataset : SimDataset,
                          chroms : str | Iterable[str] | None = None, 
                          name : str = "occup",
                          record_time : bool = False):
        """
        Calculate nucleosome occupancy at a specific time point.

        Occupancy is defined as the fraction of simulation runs where a 
        base pair is covered by a nucleosome. Results are stored directly 
        in the provided dataset's analysis map.

        Parameters
        ----------
        nucbp : int
            The footprint size (in base pairs) of a single nucleosome.
        time : int
            The simulation time point (snapshot) to analyse.
        dataset : SimDataset
            The dataset containing the raw simulation results.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all chromosomes 
            in the dataset are analysed.
        name : str, default "occup"
            The key name used to store the resulting DataFrame in 
            `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage name 
            (e.g., "occup_t_100").
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=dataset.chroms)
        # Get the size of each chromosome
        nbp = {chrom:int(dataset.raw[chrom,0,0].nbp) for chrom in chroms}
        def occup_agg(chrom, mol, nucpos):
            nsim = len(nucpos)
            occup = np.zeros(nbp[chrom])
            norm = 1.0/float(nsim)
            for p in nucpos: # Get the set of nucleosome positions for a run
                for x in p: # Aggregate these positions onto the empty fiber
                    occup[x:x+nucbp] += norm
            return occup
        occup = dataset.extract(time=time, obs="position", agg_func=occup_agg)

        # Store the results 
        if record_time: name = f"{name}_t_{time}"
        for chrom in chroms:
            dataset.analysis[chrom][name] = pd.DataFrame(occup[chrom])

    def compute_accessibility(self,
                              nucbp : int,
                              time : int,
                              dataset : SimDataset,
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
        nucbp : int
            The footprint size (in base pairs) of a single nucleosome.
        time : int
            The simulation time point (snapshot) to analyse.
        dataset : SimDataset
            The dataset containing the raw simulation results.
        chroms : str or iterable of str, optional
            The chromosome(s) to process.
        occup_name : str, default "occup"
            The name of the occupancy data to use or create.
        access_name : str, default "access"
            The key name used to store the resulting accessibility 
            DataFrame in `dataset.analysis`.
        record_time : bool, default False
            If True, the time point is appended to the storage names.
        """
        if record_time:
            access_name = f"{access_name}_t_{time}"
            occup_name = f"{occup_name}_t_{time}"
        chroms = utils.normalize_chroms(chroms)
        # Compute nucleosome occupancy if not done so
        has_occup = True
        for chrom in chroms:
            if not occup_name in dataset.analysis[chrom]:
                has_occup = False
                break
        if not has_occup:
            self.compute_occupancy(nucbp, time, dataset, chroms, occup_name,
                                   record_time)
        # Compute accessibilty A = 1.0 = O
        for chrom in chroms:
            df_occup = dataset.analysis[chrom][occup_name]
            dataset.analysis[chrom][access_name] = 1.0-df_occup
        
    def sort_by_occupancy(self,
                          dataset : SimDataset,
                          name : str = "occup"):
        pass
