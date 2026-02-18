# engine.pyA

from typing import Tuple, Dict, List
from pathlib import Path
from dataclasses import dataclass
from itertools import islice
from collections.abc import Iterable, Mapping
import array
import multiprocess as mp
import platform
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.progress import TimeRemainingColumn
import numpy as np
import pandas as pd
from .results import SimDataset
from ..utils import IndexType
from nucmc_cpp import NucPosModel, Dump
import nucmc_cpp as sim

@dataclass
class SimParams:    
    chrom : str
    mol : int
    run : int
    nucbp : int
    llink : int
    mu : float
    nsweep : int
    start_temp : float
    end_temp : float
    ninc_temp : int
    print_freq : int
    seed : int
    meth : bytes
    out_type : Dump.OutputType
    out_path : str | Path

class SimManager:
    """
    Orchestrator for managing and executing chromatin fiber simulations.

    The SimManager handles the setup, parallel execution, and data collection 
    of Monte Carlo simulations across multiple chromosomes and molecules.

    Parameters
    ----------
    nucbp : int
        The number of base pairs occupied by a single nucleosome.
    llink : int
        The DNA linker length between nucleosomes.
    mu : float
        The chemical potential; energy gained by adding a nucleosome.
    """
    
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
            mols : IndexType | Mapping[str,IndexType] = slice(None),
            nworker : int = 1,
            mp_context : str | None = None,
            verbose : bool = True) -> SimDataset:
        """
        Execute simulations in parallel across chromosomes and molecules.

        This method generates simulation parameters, spawns a process pool, 
        and tracks progress as results are written to disk.

        Parameters
        ----------
        chroms : str or iterable of str
            The chromosome identifier(s) to simulate.
        nsim : int
            Number of independent simulation runs per molecule.
        nsweep : int
            Number of Monte Carlo sweeps to perform per simulation.
        start_temp : float
            Starting temperature for the simulation annealing.
        end_temp : float
            Ending temperature for the simulation annealing.
        ninc_temp : int
            Number of temperature increments between start and end.
        print_freq : int
            Frequency (in sweeps) at which to record simulation data.
        seed : int
            Master seed used to generate independent seeds for each run.
        meth : np.ndarray or dict
            Methylation data. If multiple chromosomes are provided, this 
            must be a mapping of {chrom_name: data}. Data can be NumPy 
            arrays or Pandas DataFrames.
        out_types : str or iterable of str
            Types of data to record. Options include 'energy', 'position', 
            'temp', or 'all'.
        out_path : str or pathlib.Path
            Directory where the simulation dataset will be initialized.
        mols : IndexType or dict, default slice(None)
            Indices of molecules to simulate for each chromosome. Can be a 
            single index/slice or a mapping of {chrom_name: indices}.
        nworker : int, default 1
            Number of parallel processes to use.
        mp_context : {'spawn', 'forkserver'}, optional
            The multiprocessing start method. Defaults to 'spawn' on macOS 
            and 'forkserver' on other platforms.
        verbose : bool, default True
            If True, displays a progress bar during execution.

        Returns
        -------
        SimDataset
            A dataset object providing access to the newly generated results.

        Raises
        ------
        TypeError
            If input types for `chroms`, `meth`, or `mols` are inconsistent.
        ValueError
            If an invalid `mp_context` is provided.
        """

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
                        data = meth[chrom][molidx,:].copy().tobytes()
                        yield SimParams(chrom, molidx, run, self.nucbp,
                                        self.llink, self.mu, nsweep,
                                        start_temp, end_temp, ninc_temp,
                                        print_freq, sim_seed, data, out_type,
                                        sim_path)
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
            if mp_context is None:
                mp_context = "spawn" if platform.system() == "Darwin" \
                    else "forkserver"
            elif mp_context != "spawn" and mp_context != "forkserver":
                raise ValueError("mp_context must be 'spawn' or 'forkserver'")
            ctx = mp.get_context(mp_context)
            with ctx.Pool(processes=nworker) as pool:
                iterator = pool.imap_unordered(SimManager._run_job, param_gen)
                while True:
                    try:
                        result = next(iterator)
                        progress.update(main_task, advance=1)
                    except StopIteration:
                        break
                    except Exception as e:
                        chrom, mol, run, _, e = result
                        print(f"Simulation for chrom {chrom}, molecule "
                              f"{mol}, run {run} failed: {e}")
                        progress.update(main_task, advance=1)
        return dataset
        
    # Run a single simulation
    @staticmethod
    def _run_job(p : SimParams):
        try: 
            # Create the output directory
            sim_dir = Path(p.out_path).parents[0]
            sim_dir.mkdir(exist_ok=True, parents=True)
            # Initialize the methylation energy landscape
            meth_list = array.array('d', p.meth).tolist()            
            # Create the cpp backend Monte Carlo simulation model
            model = NucPosModel(p.nucbp, len(meth_list), p.llink, p.mu, p.seed)
            model.initMeth(meth_list)            
            # For tracking all simulation data
            model.addTracker(
                sim.createDump(p.print_freq, str(p.out_path), p.out_type))
            # Run the simulation
            model.run(p.nsweep, p.start_temp, p.end_temp, p.ninc_temp)
            return (p.chrom, p.mol, p.run, True)
        except Exception as e:
            return (p.chrom, p.mol, p.run, False, str(e))
