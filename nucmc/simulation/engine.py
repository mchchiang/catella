# engine.py

from typing import Tuple, Dict, List, Self, Any
from pathlib import Path
from dataclasses import dataclass
from itertools import islice
from collections.abc import Iterable, Mapping
import multiprocess as mp
import platform
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.progress import TimeRemainingColumn
import numpy as np
import pandas as pd
from .results import SimDataset
from ..utils import IndexType
from .. import utils
from .config import SimSettings
from nucmc_cpp import NucPosModel, Dump
import nucmc_cpp as sim
import matplotlib.pyplot as plt

# Store parameters for a specific simulation run
@dataclass(frozen=True, kw_only=True)
class SimParams:
    """
    The complete execution manifest for a single chromatin fiber simulation.
    
    This class combines the physical 'recipe' (SimSettings) with specific 
    identifying metadata for a unique simulation run.
    """
    
    chrom : str
    """The name or identifier of the chromosome being simulated."""

    mol : int
    """The index or identifier of the specific molecule within the
    chromosome."""
    
    run : int
    """The iteration or replica index for this specific molecule-chromosome
    pair."""
    
    seed : int
    """The specific random seed used for this execution. Note: This may differ
    from the batch seed if per-run offsets are applied.
    """
    
    seq_prob : bytes
    """Binary representation of the sequence-specific nucleosome binding
    probability derived from the methylation data."""
    
    out_type : Dump.OutputType
    """The format or scope of data to be recorded (e.g., Energy, Position,
    All)."""

    out_file : Path
    """The HDF5 file where the simulation results are stored."""

    settings : SimSettings
    """The physical constants and Monte Carlo protocol used for this run."""

    override : Mapping[str,Any]
    """Override any of the parameter values in settings."""

    _out_map = {"energy" : Dump.OutputType.Energy,
                "position" : Dump.OutputType.Position,
                "temp" : Dump.OutputType.Temp,
                "all" : Dump.OutputType.All}    

class SimManager:
    """
    Orchestrator for managing and executing nucleosome positioning simulations.

    The SimManager handles the setup, parallel execution, and data collection 
    of Monte Carlo simulations across multiple chromosomes and molecules.

    Attributes
    ----------
    nworker : int, default 1
        Number of parallel processes to use.
    mp_context : {'spawn', 'forkserver'}, optional
        The multiprocessing start method. Default to 'spawn' on macOS 
        and 'forkserver' on other platforms.
    verbose : bool, default True
        If True, display a progress bar when running simulations.

    Raises
    ------
    ValueError
        If the number of parallel processes `nworker` is invalid.
    ValueError
        If an invalid `mp_context` is provided.
    """
    
    def __init__(self, *,
                 nworker : int = 1,
                 mp_context : str | None = None,
                 verbose : bool = True):

        if nworker <= 0:
            raise ValueError("nworker must be greater than zero.")
        
        if mp_context is None:
            mp_context = "spawn" if platform.system() == "Darwin" \
                else "forkserver"
        elif mp_context != "spawn" and mp_context != "forkserver":
            raise ValueError("mp_context must be 'spawn' or 'forkserver'")
        
        self.nworker = nworker
        self.mp_context = mp_context
        self.verbose = verbose

    def run(self, *,
            chroms : str | Iterable[str],
            nsim : int,
            settings : str | Path | SimSettings,
            meth_prob : np.ndarray | Mapping[str,np.ndarray|pd.DataFrame],
            out_dir : str | Path,
            dataset_name : str = "results",
            out_types : str | Iterable[str] = "all",
            seed : int | None = None,
            mols : IndexType | Mapping[str,IndexType] = slice(None),
            store_eseq : bool = True,
            use_zero_point_mu : bool = False
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
            Number of independent simulation runs per molecule.
        settings : str or Path or SimSettings
            The physical constants and Monte Carlo protocol (e.g., nucbp, llink,
            mu) to apply to all runs in this batch. These can be read from a
            configuration file.
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
        out_types : str or iterable of str, default 'all'
            Types of data to record. Options include 'energy', 'position', 
            'temp', or 'all'.        
        seed : int, optional
            Master seed used to generate independent seeds for each run. If
            None, a high-entropy seed is automatically generated using the
            NumPy default random generator.
        mols : IndexType or dict, default slice(None)
            Indices of molecules to simulate for each chromosome. Can be a 
            single index/slice or a mapping of {chrom_name: indices}.
        store_eseq : bool, default True
            Whether to store the sequence-specific nucleosome binding energy
            landscape derived from the methylation data to output dataset file.
        use_zero_point_mu : bool : default False
            Whether to modify the chemical potential parameter so that it is
            equal to the median of the squence-specific nucleosome binding
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
            If the requested `out_types` are not recognized.
        FileExistsError
            If the specified `out_dir` already contains an existing dataset.
        """

        chroms = utils.normalize_chroms(chroms)

        # Normalize methylation data
        if isinstance(meth_prob, np.ndarray):
            if not len(chroms) == 1:
                raise TypeError("Methylation probability array 'meth_prob' "
                                "provided but multiple chroms requested.")
            meth_prob = {chroms[0]:meth_prob}
        elif isinstance(meth_prob, Mapping):
            meth_prob = {chrom:meth_prob[chrom] for chrom in chroms}
        else:
            raise TypeError("'meth_prob' must be a numpy array or a mapping.")

        # Normalize molecule indices
        if isinstance(mols, IndexType.__args__):
            # Apply the same set of indices across all chromosomes
            mols = {chrom:mols for chrom in chroms}
        elif isinstance(mols, Mapping):
            mols = {chrom:mols[chrom] for chrom in chroms}
        else:
            raise TypeError("'mols' must be an IndexType or a mapping.")

        
        # Check that the molecule indices are valid and compute total number
        # of simulations required
        nmol = {}
        nbp = {}
        total_sim = 0
        for chrom in chroms:
            # Determine the maximum number of molecules and all molecule ids
            size = meth_prob[chrom].shape[0]
            nmol[chrom] = size
            
            # Convert any data frames to numpy arrays in meth_prob
            if isinstance(meth_prob[chrom], pd.DataFrame):
                meth_prob[chrom] = meth_prob[chrom].to_numpy()
            meth_prob[chrom] = np.atleast_2d(meth_prob[chrom])
            nbp[chrom] = meth_prob[chrom].shape[1]
            
            try:
                # Apply the index/slice to an 0-element view to trigger errors
                # without allocating memory for the full index list
                text_idx = np.empty(size, dtype=np.int8)[mols[chrom]]
                if text_idx.size == 0:
                    raise ValueError("Selection 'mols' for chromosome "
                                     f"'{chrom}' is empty.")
                total_sim += text_idx.size * nsim
            except IndexError as e:
                raise IndexError(f"mols index for chromosome '{chrom}': {e}") \
                    from None

        # Read simulation settings from file if needed
        if isinstance(settings, (str | Path)):
            settings = SimSettings.load(settings)
        elif not isinstance(settings, SimSettings):
            raise TypeError("'settings' must be either a path to the "
                            "simulation setting file or a SimSettings object.")

        # Check temperatures are valid
        if settings.end_temp > settings.start_temp:
            raise ValueError("Expect 'end_temp' <= 'start_temp'.")
        if settings.cool_option not in SimSettings._cool_map:
            raise KeyError("Invalid value for 'cool_option'. Expect either "
                           "'linear', 'geometric', or 'constant'.")
        
        # Combine output types
        if isinstance(out_types, str):
            if out_types not in SimParams._out_map:
                raise ValueError(f"Output type '{out_types}' is not a valid "
                                 "option.")
            out_types = [out_types]
        elif not isinstance(out_types, Iterable):
            raise TypeError("'out_types' must be a str or an iterable.")
        
        out_type = SimParams._out_map[out_types[0]]
        for i in range(1,len(out_types)):
            otype = SimParams._out_map[out_types[i]]
            if otype not in SimParams._out_map:
                raise ValueError(f"Output type '{otype}' is not a valid "
                                 "option.")
            out_type |= otype
        
        # Compute the methylation energy landscape
        eseq = None
        if store_eseq:
            eseq = {}
            for chrom in chroms:
                model = NucPosModel(settings.nucbp, nbp[chrom], settings.llink,
                                    settings.mu, 0)
                eseq[chrom] = np.empty(meth_prob[chrom].shape)
                for i in range(meth_prob[chrom].shape[0]):
                    # pseq = 1 - pmeth
                    model.setSeqEnergy(1.0-meth_prob[chrom][i], settings.emax)
                    eseq[chrom][i] = model.getSeqEnergy()

        # Adjust the chemical potential if needed
        override = {}
        if use_zero_point_mu:
            avg_eseq = np.empty(len(chroms))
            for i,chrom in enumerate(chroms):
                avg_eseq[i] = np.median(eseq[chrom])
            avg_eseq = np.mean(avg_eseq)
            print(f"Using zero point mu: {avg_eseq}")
            override["mu"] = avg_eseq
            
        # Prepare the dataset object
        dataset = SimDataset.create(chroms=chroms, nmol=nmol, nsim=nsim,
                                    nbp=nbp, settings=settings, eseq=eseq,
                                    out_dir=out_dir, dataset_name=dataset_name)
        
        # Generate random seeds
        def seed_generator(parent_seed):
            ss = np.random.SeedSequence(parent_seed)
            while True: yield ss.spawn(1)[0].generate_state(1)[0]
        if seed is None:
            rng = np.random.default_rng()
            seed = rng.bit_generator.seed_seq.entropy
        seed_gen = seed_generator(seed)

        # Generate the parameter list
        def params_generator(settings):
            for chrom in chroms:
                molidxs = np.arange(nmol[chrom])[mols[chrom]]
                for molidx in molidxs:
                    idx = int(molidx)
                    # pseq = 1 - pmeth
                    pseq = np.asarray(1.0-meth_prob[chrom][molidx,:],
                                      dtype=np.float64).tobytes()
                    for run in range(nsim):
                        sim_file = dataset.sim_file(chrom, idx, run)
                        yield SimParams(chrom=chrom, mol=idx, run=run,
                                        settings=settings, seed=next(seed_gen),
                                        seq_prob=pseq, out_type=out_type,
                                        out_file=sim_file, override=override)
        param_gen = params_generator(settings)
        
        # Progress bar
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeRemainingColumn(),
            disable=not self.verbose)
        
        with progress:
            main_task = progress.add_task("[cyan]Running simulations ...",
                                          total=total_sim)
            if self.nworker > 1:
                ctx = mp.get_context(self.mp_context)
                with ctx.Pool(processes=self.nworker) as pool:
                    iterator = pool.imap_unordered(SimManager._run_job,
                                                   param_gen)
                    while True:
                        try:
                            result = next(iterator)
                            if result[3]:
                                pass
                            else:
                                chrom, mol, run, _, err_msg = result
                                progress.console.print(
                                    "[red]Error[/red] Simulation for chrom "
                                    f"{chrom}, molecule {mol}, run {run} "
                                    f"failed: {err_msg}")
                            progress.update(main_task, advance=1)
                        except StopIteration:
                            break
            else: # nworker = 1
                for param in param_gen:
                    SimManager._run_job(param)
                    progress.update(main_task, advance=1)

        # Save the dataset to file
        dataset.save()
        
        return dataset
        
    # Run a single simulation
    @staticmethod
    def _run_job(p : SimParams):
        def get_param(name):
            if name in p.override and name in dir(p.settings):
                return p.override[name]
            elif name in dir(p.settings):
                return getattr(p.settings, name)
            else:
                return None
        nucbp = get_param("nucbp")
        llink = get_param("llink")
        mu = get_param("mu")
        nsweep = get_param("nsweep")
        start_temp = get_param("start_temp")
        end_temp = get_param("end_temp")
        print_freq = get_param("print_freq")
        emax = get_param("emax")
        cool_option = SimSettings._cool_map[get_param("cool_option")]
        try: 
            # Create the output directory
            sim_dir = Path(p.out_file).parents[0]
            sim_dir.mkdir(exist_ok=True, parents=True)
            # Initialize the methylation energy landscape
            pseq_list = np.frombuffer(p.seq_prob, dtype=np.float64).tolist()
            nbp = len(pseq_list)
            # Create the cpp backend Monte Carlo simulation model
            model = NucPosModel(nucbp, nbp, llink, mu, p.seed)
            model.setSeqEnergy(pseq_list, emax)
            # For tracking all simulation data
            model.addTracker(
                sim.createDump(print_freq, str(p.out_file), p.out_type))
            # Run the simulation
            model.run(nsweep, start_temp, end_temp, cool_option)
            return (p.chrom, p.mol, p.run, True)
        except Exception as e:
            return (p.chrom, p.mol, p.run, False, str(e))
