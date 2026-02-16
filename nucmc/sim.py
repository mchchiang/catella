# sim.py

from typing import Tuple, Dict, Sequence
from pathlib import Path
from dataclasses import dataclass
from itertools import islice
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
import numpy as np
import pandas as pd
from nucmc_cpp import NucPosModel, Dump
import nucmc_cpp as sim
from .sim_data import SimDataset
from .util import IndexType

@dataclass
class SimParams:
    chrom : str
    mol : int
    run : int
    nsweep : int
    start_temp : float
    end_temp : float
    ninc_temp : int
    print_freq : int
    seed : int
    meth : np.ndarray
    out_type : Dump.OutputType
    out_path : str | Path

class SimManager:

    def __init__(self,
                 nucbp : int,
                 llink : int,
                 mu : float):
        
        self.nucbp = nucbp
        self.llink = llink
        self.mu = mu
        self._out_map = {"energy" : Dump.OutputType.Energy,
                         "position" : Dump.OutputType.Position,
                         "temp" : Dump.OutputType.Temp,
                         "all" : Dump.OutputType.All}
        
    def run(self,
            chroms : str | Iterable[str],
            nsim : int,
            nsweep : int,
            start_temp : float,
            end_temp : float,
            ninc_temp : int,
            print_freq : int,
            seed : int,
            meth : np.ndarray | Mapping[str,np.ndarray|pd.DataFrame],
            out_types : str | Iterable[str],    
            out_path : str | Path,
            nworker : int = 1,
            mols : IndexType | Mapping[str,IndexType] = slice(None),
            batch_size = 5000) \
            -> SimDataset:

        # Check that meth data and molecule iterators are valid
        if isinstance(chroms,Iterable):
            if not isinstance(meth, Mapping):
                raise TypeError("Methylation data are needed for all "
                                "chromosomes")
            if isinstance(mols, IndexType.__args__):
                # Apply the same indices across all chromosomes
                mols = {chrom:mols for chrom in chroms}
            elif not isinstance(mols, Mapping):
                raise TypeError("Molecule indices are needed for all "
                                "chromosomes")
        elif isinstance(chroms,str):
            chroms = [chroms]
            meth = {chroms[0]:meth}
            mols = {chroms[0]:mols}
        else:
            raise TypeError("chroms must be a str or an Iterable")

        # Convert any data frames to numpy arrays in meth
        for chrom in chroms:
            if isinstance(meth[chrom], pd.DataFrame):
                meth[chrom] = meth[chrom].to_numpy()

        # Create the output directory if needed
        out_path = Path(out_path)
        out_path.mkdir(exist_ok=True, parents=True)

        # Determine the maximum number of molecule and all moleculde ids
        nmol = {chrom:meth[chrom].shape[0] for chrom in chroms}
        molids = {chrom:np.arange(0,nmol[chrom]) for chrom in chroms}
        
        # Combine output types
        if isinstance(out_types,str):
            out_types = [out_types]
        elif not isinstance(out_types,Iterable):
            raise TypeError("out_types must be a str or an Iterable")
        
        out_type = self._out_map[out_types[0]]
        for i in range(1,len(out_types)):
            out_type |= self._out_map[out_types[i]]

        # Prepare the dataset object
        dataset = SimDataset.create_from_sim(chroms, nmol, nsim, out_path)
        
        # Generate random seeds
        def seed_generator(parent_seed):
            ss = np.random.SeedSequence(parent_seed)
            while True: yield ss.spawn(1)[0].generate_state(1)[0]
        seed_gen = seed_generator(seed)

        # Generate the parameter list
        def params_generator():
            for chrom in chroms:
                idxs = mols[chrom]
                for molidx in molids[chrom][idxs]:
                    for run in range(nsim):
                        sim_path = dataset.get_sim_path(chrom, molidx, run)
                        sim_seed = next(seed_gen)
                        data = meth[chrom][molidx,:]
                        yield SimParams(chrom, molidx, run, nsweep, start_temp,
                                        end_temp, ninc_temp, print_freq,
                                        sim_seed, data, out_type, sim_path), \
                                        (chrom, molidx, run)
        
        param_gen = params_generator()
        with ProcessPoolExecutor(max_workers=nworker) as executor:
            while True:
                # Grab a chunk of work
                batch = list(islice(param_gen, batch_size))
                if not batch: break
                # Submit onl this batch
                tasks : Dict[Future,Tuple] = {
                    executor.submit(self.run_job, params): sim_id
                    for params, sim_id in batch}
                # Process this batch as it completes
                for future in as_completed(tasks):
                    chrom, mol, run = tasks[future]
                    try: # Check if the task ran okay
                        future.result()
                    except Exception as e:
                        print(f"Simulation for chrom {chrom}, molecule {mol}, "
                              f"run {run} failed: {e}")
                # Clean up batch
                tasks.clear()
                
        return dataset
        
    # Run a single simulation
    def run_job(self, p : SimParams):

        print(f"Running simulation for chrom {p.chrom}, molecule {p.mol}, "
              f"run {p.run} ...")

        # Create the output directory
        Path(p.out_path).parents[0].mkdir(exist_ok=True, parents=True)
        
        # Create the cpp backend Monte Carlo simulation model
        model = NucPosModel(self.nucbp, len(p.meth), self.llink, self.mu,
                            p.seed)

        # Initialize the methylation energy landscape
        model.initMeth(p.meth)

        # For tracking all simulation data
        model.addTracker(
            sim.createDump(p.print_freq, str(p.out_path), p.out_type))

        # Run the simulation
        model.run(p.nsweep, p.start_temp, p.end_temp, p.ninc_temp)
        
        return p.out_path
