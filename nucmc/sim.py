# sim.py

from typing import Tuple, Dict, Sequence
from pathlib import Path
from dataclasses import dataclass
from itertools import islice
from collections.abc import Iterable, Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed, Future
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.progress import TimeRemainingColumn
import numpy as np
import pandas as pd
from nucmc_cpp import NucPosModel, Dump
import nucmc_cpp as sim
from .sim_data import SimDataset
from .util import IndexType
from . import util

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
            *,
            nworker : int = 1,
            mols : IndexType | Mapping[str,IndexType] = slice(None),
            batch_size = 5000,
            verbose : bool = True) -> SimDataset:

        # Check that meth data and molecule iterators are valid
        if isinstance(chroms,str):
            chroms = [chroms]
            meth = {chroms[0]:meth}
            mols = {chroms[0]:mols}
        elif isinstance(chroms,Iterable):
            if not isinstance(meth, Mapping):
                raise TypeError("Methylation data are needed for all "
                                "chromosomes")
            if isinstance(mols, IndexType.__args__):
                # Apply the same indices across all chromosomes
                mols = {chrom:mols for chrom in chroms}
            elif not isinstance(mols, Mapping):
                raise TypeError("Molecule indices are needed for all "
                                "chromosomes")
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

        # Determine the total number of simulations
        total_sim = 0
        for chrom in chroms:
            idxs = mols[chrom]
            total_sim += len(molids[chrom][idxs]) * nsim 

        # Progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            disable=not verbose)

        with progress:
            main_task = progress.add_task("[cyan]Running simulations ...",
                                          total=total_sim)
            with ProcessPoolExecutor(max_workers=nworker) as executor:
                while True:
                    # Grab a chunk of work
                    batch = list(islice(param_gen, batch_size))
                    if not batch: break
                    # Submit onl this batch
                    tasks : Dict[Future,Tuple] = {
                        executor.submit(self._run_job, params): sim_id
                        for params, sim_id in batch}
                    # Process this batch as it completes
                    for future in as_completed(tasks):
                        chrom, mol, run = tasks[future]
                        try: # Check if the task ran okay
                            future.result()
                        except Exception as e:
                            print(f"Simulation for chrom {chrom}, molecule "
                                  f"{mol}, run {run} failed: {e}")
                        progress.update(main_task, advance=1)
                    # Clean up batch
                    tasks.clear()                
        return dataset
        
    # Run a single simulation
    def _run_job(self, p : SimParams):
        # Create the output directory
        sim_dir = Path(p.out_path).parents[0]
        sim_dir.mkdir(exist_ok=True, parents=True)
        
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


class SimAnalysis:
    
    def compute_occupancy(self,
                          nucbp : int,
                          time : int,
                          dataset : SimDataset,
                          chroms : str | Iterable[str] | None = None, 
                          name : str = "occup",
                          record_time : bool = False):
        chroms = util.normalize_chroms(chroms, default_chroms=dataset.chroms)
        # Get the size of each chromosome
        nbp = {chrom:int(dataset.raw[chrom,0,0].nbp) for chrom in chroms}
        def occup_agg(chrom, mol, nucpos):
            print(chrom, mol, nbp[chrom], nucpos)
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
        if record_time:
            access_name = f"{access_name}_t_{time}"
            occup_name = f"{occup_name}_t_{time}"
        chroms = util.normalize_chroms(chroms)
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
