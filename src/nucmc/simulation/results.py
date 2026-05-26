# results.py

import numpy as np
import h5py
import re
from functools import lru_cache
from itertools import islice
from threading import Semaphore
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, Self, Dict, Any, Callable, List
from collections import Counter
from collections.abc import Iterator, Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from ..containers import DataFrameMap, FixedKeyMap, DataclassPublicProxy
from .config import SimSettings
from .. import utils
from .. import h5_utils

# Limit the concurrent open files to stay under OS limits
MAX_OPEN_FILES = Semaphore(500)

@utils.add_frozen_properties
@dataclass(frozen=True, slots=True, init=False)
class SimData:
    """
    Container for results and parameters of a single simulation run.

    This class provides a read-only, memory-efficient representation of 
    Monte Carlo simulation data, including physical parameters (chemical 
    potential, fiber length) and recorded observables (energy, positions).

    .. note::
       Direct instantiation is disabled to ensure data integrity. Use the 
       factory method :meth:`SimData.load` to create instances from HDF5 files.
    """
    
    nucbp : int
    """The number of base pairs occupied by a single nucleosome."""
     
    nbp : int
    """The total number of base pairs in the chromatin fiber."""
    
    llink : int
    """The DNA linker length between nucleosomes."""

    mu : float
    """The chemical potential (energy gained by adding a nucleosome)."""

    seed : int
    """The seed used for initializing the random number generator."""

    _frozen_time : np.ndarray = field(
        metadata={"doc": "np.ndarray: The simulation time points."})

    _frozen_energy : np.ndarray = field(
        metadata={"doc": "np.ndarray: The system energy recorded at each time "
                  "point."})

    _frozen_position : Tuple[np.ndarray,...] = field(
        metadata={"doc": "tuple of np.ndarray: Nucleosome positions for each "
                  "time frame."})

    _frozen_temp : np.ndarray = field(
        metadata={"doc": "np.ndarray: The system temperature recorded at each "
                  "time point."})

    # Map from time values to frame index
    _time_to_idx : dict[int,int]
    _obs_map : dict[str,Any]
    _obs_list = ("energy", "position", "temp")

    def __init__(self, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use SimData.load() to instantiate this class")

        # Bulk assignment of fields
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is None:
                val = kwargs.get(f.name.removeprefix("_frozen_"))
            object.__setattr__(self, f.name, val)
            
        # Build time index lookup
        time_map = {t: i for i,t in enumerate(self.time)}        
        object.__setattr__(self, "_time_to_idx", time_map)

        # Build observable lookup
        obs_map = {"energy": self.energy, "position": self.position,
                   "temp": self.temp}
        object.__setattr__(self, "_obs_map", obs_map)
        
    def time_index(self, t: int) -> int:
        """
        Retrieve the internal frame index for a given time point.

        Parameters
        ----------
        t : int
            The simulation time point to look up.

        Returns
        -------
        int
            The integer index corresponding to time `t` in the data arrays.

        Raises
        ------
        KeyError
            If the specified time point does not exist in the dataset.
        """
        return self._time_to_idx[t]

    def at(self, t: int, obs : str):
        """
        Retrieve an observable value at a specific time point.

        Parameters
        ----------
        t : int
            The time point of interest.
        obs : str
            The observable of interest.

        Returns
        -------
        float or np.ndarray
            The observable value at time `t`.

        Raises
        ------
        ValueError
            If obs is not a valid observable.
        """
        if obs not in self._obs_map.keys():
            raise ValueError(f"'{obs}' is not a valid observable.")
        value = self._obs_map[obs][self._time_to_idx[t]]
        return value.item() if np.ndim(value) == 0 else value
    
    @classmethod
    def load(cls, sim_file : str | Path) -> Self:
        """
        Load simulation data from a persistent HDF5 file.

        Parameters
        ----------
        sim_file : str or Path
            The HDF5 file containing simulation results.

        Returns
        -------
        SimData
            A new instance populated with the loaded parameters and data.

        Raises
        ------
        OSError
            If the file cannot be opened or does not follow the SimData schema.
        """
        with h5py.File(sim_file, "r") as f:
            # Load model parameters
            g = f["params"]
            nucbp = g.attrs["nucbp"]
            nbp = g.attrs["nbp"]
            llink = g.attrs["llink"]
            mu = g.attrs["mu"]
            seed = g.attrs["seed"]
            
            # Load data
            g = f["data"]
            def load_data(g, name):
                # Unpack nucleosome position from a flatten array
                if name == "position":
                    flat = g["position_flat"][()]
                    offset = g["position_offset"][()]
                    data = [flat[offset[i]:offset[i+1]]
                            for i in range(len(offset)-1)]
                    return tuple(np.asarray(d, dtype=flat.dtype) for d in data)
                elif name in g:
                    return g[name][()]
                return np.nan
            time = load_data(g, "time")
            energy = load_data(g, "energy")
            position = load_data(g, "position")
            temp = load_data(g, "temp")
            return cls._create(nucbp=nucbp, nbp=nbp, llink=llink, mu=mu,
                               seed=seed, time=time, energy=energy,
                               position=position, temp=temp)

    # Extract the data for a specific obesrvable for a single time point
    @classmethod
    def extract(cls, sim_file : str | Path,
                time : int,
                obs : str) -> np.ndarray:
        """
        Extract a specific observable from a file without loading the full
        dataset.

        This method is optimized for batch processing and utilizes an internal 
        LRU cache to speed up repeated access to the same file.

        Parameters
        ----------
        sim_file : str or Path
            The simulation HDF5 file.
        time : int
            The target simulation time point.
        obs : str
            The name of the observable to extract (e.g., 'energy', 'position').

        Returns
        -------
        np.ndarray
            The extracted data for the specified observable and time. Returns 
            ``np.nan`` (or an array of nan for 'position') if extraction fails.

        Raises
        ------
        ValueError
            If obs is not a valid observable name
        """
        if obs not in cls._obs_list:
            raise ValueError(f"'{obs}' is not a valid observable.")
        sim_file = str(Path(sim_file).resolve())
        try:
            return cls._cached_extract(sim_file, time, obs)
        except Exception:
            return np.array([np.nan]) if obs == "position" else np.nan

    @staticmethod
    @lru_cache(maxsize=1024)    
    def _cached_extract(sim_file :
                        str, time :
                        int, obs : str) -> np.ndarray:
        with MAX_OPEN_FILES:
            with h5py.File(sim_file, "r") as f:
                tdata = f["data/time"][()]
                matches = np.where(tdata == time)[0]
                if len(matches) == 0:
                    raise KeyError(f"Cannot find data for time {time}.")
                idx = matches[0]
                if obs == "position":
                    flat = f["data/position_flat"]
                    offset = f["data/position_offset"][()]
                    start = offset[idx]
                    end = offset[idx+1] if idx+1 < len(offset) \
                        else flat.shape[0]
                    return np.array(flat[start:end], copy=True)
                else:
                    return f[f"data/{obs}"][idx].item()
    
    @classmethod
    def _create(cls, **kwargs) -> Self:
        nucbp = kwargs.get("nucbp")
        nbp = kwargs.get("nbp")        
        llink = kwargs.get("llink")
        seed = kwargs.get("seed")
        time = kwargs.get("time")
        energy = kwargs.get("energy")
        position = kwargs.get("position")
        temp = kwargs.get("temp")        
        # Some validation
        if nucbp <= 0 or nbp <= 0 or llink <= 0 or seed <= 0:
            raise ValueError("nucbp, nbp, llink, and seed must be positive.")
        if not (len(time) == len(energy) == len(temp) == len(position)):
            raise ValueError("time, energy, position, and temp must have the",
                             "same length.")
        return cls(**kwargs, _internal=True)

    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [attr for attr in self.__slots__ if
                        not attr.startswith('_')]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith('_'):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str
    
class SimFileMapper:
    def __init__(self, root_dir : str,
                 total_mols : int,
                 max_mols_per_dir : int = 100,
                 max_runs_per_mol : int = 100):
        self.root_dir = Path(root_dir)
        self.max_mols_per_dir = max_mols_per_dir
        self.max_runs_per_mol = max_runs_per_mol
        self.mol_pad = max(1, int(np.ceil(np.log10(total_mols))))
        self.run_pad = max(1, int(np.ceil(np.log10(max_runs_per_mol))))

    def params_to_file(self, chrom : str, molidx : int, run : int) -> Path:
        # Compute range
        start = int((molidx // self.max_mols_per_dir) * self.max_mols_per_dir)
        end = int(start + self.max_mols_per_dir - 1)
        # Get name at each level
        z = self.mol_pad
        r = self.run_pad
        shard_name = f"mol_{int(start):0{z}d}-{int(end):0{z}d}"
        mol_name = f"mol_{int(molidx):0{z}d}"
        run_name = f"run_{int(run):0{r}d}"
        sim_h5 = f"{chrom}_{mol_name}_{run_name}.h5"
        return self.root_dir/chrom/shard_name/mol_name/sim_h5

    def file_to_params(self, sim_file : str | Path):
        sim_file = Path(sim_file)
        chrom = sim_file.parts[-4]
        filename = sim_file.name
        match = re.search(r"mol_(\d+)_run_(\d+)\.h5$", filename)
        if match:
            molidx = int(match.group(1))
            run = int(match.group(2))        
            return chrom, molidx, run
        raise ValueError(f"Could not parse parameters from file: {sim_file}") 
    
@dataclass(slots=True, init=False)
class SimDataset:
    """
    A collection of simulation results and analyses across multiple
    chromosomes.

    This class acts as a central hub for managing simulation metadata, 
    performing parallel data extraction, and storing analysis results. It 
    uses a lazy accessor for retrieving raw simulation data from disk.

    .. note::
       Direct instantiation is disabled. Use the factory methods 
       :meth:`SimDataset.create` or :meth:`SimDataset.load` to initialize 
       this class.

    Examples
    --------
    Accessing raw simulation data using tuple-based indexing:

    >>> dataset = SimDataset.load("results/dataset.h5")
    >>> sim_data = dataset.raw["chr1", 0, 9]  # [chrom, molecule, run]
    """
    
    class SimDataAccessor:
        """
        Lazy accessor for simulation data.

        Enables access using:

        >>> dataset.raw[chrom, mol, run]
        """
        def __init__(self, parent):
            self._parent = parent
            
        def __getitem__(self, keys):
            chrom, mol, run = keys
            sim_file = self._parent.sim_file(chrom, mol, run)
            if sim_file.exists():
                return self._load_sim_data(str(sim_file))
            else:
                raise ValueError("Cannot find the simulation for chrom "
                                 f"{chrom}, molecule {mol}, run {run}")

        @staticmethod
        @lru_cache(maxsize=1024)
        def _load_sim_data(sim_file : str):
            return SimData.load(sim_file)
    
    _chroms : Iterable[str]
    _nmol : Mapping[str,int]
    _nsim : int
    _nbp : Mapping[str,int]
    _settings : SimSettings
    _eseq : Mapping[str,np.ndarray]
    _view_eseq : Mapping[str,np.ndarray]
    _raw_dir : Path
    _dataset_file : Path
    _file_map : SimFileMapper
    _analysis : Mapping[str, DataFrameMap]
    _global_analysis : DataFrameMap
    _raw_accessor : SimDataAccessor

    def __init__(self, *args : Any, **kwargs : Any):
        raise TypeError("Use SimDataset.load() or create() to instantiate "
                        "this class.")

    @classmethod
    def _create(cls, *,
                chroms : Iterable[str],
                nmol : Mapping[str,int],
                nsim : int,
                nbp : Mapping[str,int],
                settings : SimSettings,
                raw_dir : Path,
                dataset_file : Path,                
                file_map : SimFileMapper,
                eseq : Mapping[str, np.ndarray] | None):
        # Some validations
        if nsim <= 0:
            raise ValueError("Number of simulations must be positive")
        # Check the keys in nmol match up with chroms
        if set(nmol.keys()) != set(chroms):
            raise ValueError("Chromosomes in nmol do not match with chroms")
        # Check the number of molcules in each chromosome is valid
        for chrom in chroms:
            if nmol[chrom] <= 0:
                raise ValueError(f"Chromosome {chrom} must have at least one "
                                 "molecule")
        
        obj = cls.__new__(cls)
        obj._chroms = tuple(chroms)
        obj._nmol = dict(nmol)
        obj._nsim = int(nsim)
        obj._nbp = dict(nbp)
        obj._settings = settings
        obj._raw_dir = raw_dir
        obj._dataset_file = dataset_file
        obj._file_map = file_map
        obj._analysis = {chrom : DataFrameMap() for chrom in obj._chroms}
        obj._global_analysis = DataFrameMap()        
        obj._raw_accessor = cls.SimDataAccessor(obj)
        obj._eseq = eseq
        if obj._eseq is None:
            obj._view_eseq = {chrom:np.nan for chrom in chroms}
        else:
            obj._view_eseq = {chrom:obj._eseq[chrom] for chrom in chroms}
            for chrom in chroms:
                obj._view_eseq[chrom].flags.writeable = False
        return obj
        
    @classmethod
    def create(cls, *,
               chroms : str | Iterable[str],
               nmol : int | Mapping[str,int],
               nsim : int,
               nbp : int | Mapping[str,int],
               settings : SimSettings,
               out_dir : str | Path,
               dataset_name : str = "results",
               eseq : np.ndarray | Mapping[str,np.ndarray] | None = None) \
               -> Self:
        """
        Create a new simulation dataset with the specified parameters.

        This factory method initializes the dataset structure, sets up the 
        internal mapping for molecules per chromosome, and defines the 
        total number of simulations per molecule.

        Parameters
        ----------
        chroms : str or iterable of str
            The chromosome identifier(s) to include in the dataset.
        nmol : int or dict of {str: int}
            The number of molecules per chromosome. If an integer is provided, 
            the same count is applied to all chromosomes. If a mapping is 
            provided, it must specify the count for each chromosome name.
        nsim : int
            The number of simulation runs to perform for each molecule.
        nbp : int or dict of {str: int}
            The number of base-pair for each chromsome segment.
        settings : SimSettings
            The physical constants and Monte Carlo protocol (e.g., mu, temp,
            sweeps) that apply to all simulations in the dataset.
        out_dir : str or Path
            The directory where the simulation analysis and raw data will be
            stored. This directory must not exist beforehand.
        dataset_name : str, default "results"
            Name of the dataset HDF5 file storing the metadata and analysis of
            the simulation results. The file directory of this HDF5 file is
            `out_dir/{dataset_name}.h5` (e.g,. `out_dir/results.h5`).
        eseq : np.ndarray or None, default None
            The sequence-specific energy landscape derived from the
            methylation data.
        
        Returns
        -------
        SimDataset
            A newly initialized instance of the simulation dataset.

        Raises
        ------
        FileExistsError
            If `out_dir` already exists.
        ValueError
            If the chromosome keys in `nmol` do not match `chroms`.
        """
        chroms = utils.normalize_chroms(chroms)
        if isinstance(nmol, int):
            nmol = {chrom:nmol for chrom in chroms}
        elif not isinstance(nmol, Mapping):
            raise ValueError("nmol must either be an int or a mapping")
        elif set(nmol.keys()) != set(chroms):
            raise ValueError("nmol must have the same set of chromosome "
                             "identifiers as chroms")
        
        # Find the maximum number of molecules
        max_nmol = 0
        for chrom in chroms:
            if max_nmol < nmol[chrom]:
                max_nmol = nmol[chrom]

        # Check validity of the output directory
        out_dir = Path(out_dir).resolve()
        if out_dir.exists():
            raise FileExistsError(f"The directory '{out_dir}' already exists.")
        else:
            raw_dir = out_dir/"raw_data"
            raw_dir.mkdir(exist_ok=True, parents=True)
        dataset_file = out_dir/(dataset_name+".h5")
        file_map = SimFileMapper(raw_dir, max_nmol)
        return cls._create(chroms=chroms, nmol=nmol, nsim=nsim, nbp=nbp,
                           settings=settings, eseq=eseq, raw_dir=raw_dir,
                           dataset_file=dataset_file, file_map=file_map)
    
    @classmethod
    def load(cls,
             dataset_file : str | Path,
             new_raw_dir : str | Path | None = None) -> Self:
        """
        Load a simulation dataset and its analysis from an HDF5 file.
        
        Parameters
        ----------
        dataset_file : str or Path
            The HDF5 file containing the dataset metadata and analysis results.
        new_raw_dir : str or Path, optional
            An updated path to the `raw_data` directory. Use this if the 
            simulation files have been moved since the dataset was saved. 
            The default is None.
        
        Returns
        -------
        SimData
            The loaded simulation dataset object with all analysis maps 
            populated.

        Raises
        ------
        FileNotFoundError
            If the HDF5 file cannot be opened or if the raw data directory 
            cannot be located at the expected path.
        """
        dataset_file = Path(dataset_file).resolve()
        with h5py.File(dataset_file, "r") as h5stream:
            # Load metadata associated with raw simulation data
            gmeta = h5stream["metadata"]
            if new_raw_dir is None:
                if "raw_path" in gmeta.attrs: # For old versions 
                    raw_dir = Path(gmeta.attrs["raw_path"]).resolve()
                else:
                    raw_dir = Path(gmeta.attrs["raw_dir"]).resolve()
                if not raw_dir.exists():
                    raise FileNotFoundError("The raw simulation data directory "
                                            "cannot be found.")
            else:
                new_raw_dir = Path(new_raw_dir).resolve()
                if not new_raw_dir.exists():
                    raise FileNotFoundError("The raw simulation data directory "
                                            "cannot be found.")
                raw_dir = new_raw_dir
            nsim = gmeta.attrs["nsim"]
            chroms = list(gmeta["chroms"].asstr()[()])
            nmol_arr = gmeta["nmol"][()]
            nmol = {chrom:nmol_arr[i] for i,chrom in enumerate(chroms)}
            file_map = SimFileMapper(raw_dir, max(nmol_arr))
            nbp_arr = gmeta["nbp"][()]
            nbp = {chrom:nbp_arr[i] for i,chrom in enumerate(chroms)}
            # Load simulation settings
            gset = gmeta["settings"]
            settings = SimSettings(**dict(gset.attrs.items()))
            # Load methylation energy (if stored)
            if "eseq" in gmeta:
                geseq = gmeta["eseq"]
                eseq = {}
                for chrom in chroms:
                    eseq[chrom] = geseq[chrom][()]
            else:
                eseq = None
            obj = cls._create(chroms=chroms, nmol=nmol, nsim=nsim, nbp=nbp,
                              settings=settings, eseq=eseq, raw_dir=raw_dir,
                              file_map=file_map, dataset_file=dataset_file)
            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                gchrom = gana[chrom]
                for name in gchrom:
                    obj._analysis[chrom][name] = h5_utils.load_df(name, gchrom)
            gana = h5stream["global_analysis"]
            for name in gana:
                obj._global_analysis[name] = h5_utils.load_df(name, gana)
            return obj

    def save(self, dataset_file : str | Path | None = None):
        """
        Save the dataset metadata and analysis results to an HDF5 file.

        Serialize the current state of all chromosome-specific and global
        analysis :class:`DataFrameMap` objects into a persistent HDF5 format.
        
        Parameters
        ----------
        dataset_file : str | Path, optional
            The output HDF5 file for the dataset. If the file already exists,
            analysis groups will be overwritten. If None, data will be written to
            to the original dataset file from loading or creation.
        
        Raises
        ------
        OSError
            If the file cannot be written to disk.
        """        
        dt = h5py.string_dtype(encoding="utf-8")
        if dataset_file is None:
            dataset_file = self._dataset_file
            
        with h5py.File(dataset_file, "a") as h5stream:
            # Ensure that the raw file directory is up-to-date
            if "metadata" in h5stream:
                gmeta = h5stream["metadata"]
                gmeta.attrs["raw_dir"] = str(self._raw_dir)
            else: # if "metadata" not in h5stream:
                # Save metadata asssociated with raw simulation data
                gmeta = h5stream.create_group("metadata")
                gmeta.attrs["raw_dir"] = str(self._raw_dir)
                gmeta.attrs["nsim"] = self._nsim        
                gmeta.create_dataset("chroms", data=self._chroms, dtype=dt)
                nmol_arr = np.asarray([self._nmol[chrom]
                                       for chrom in self._chroms])
                gmeta.create_dataset("nmol", data=nmol_arr)
                nbp_arr = np.asarray([self._nbp[chrom]
                                      for chrom in self._chroms])
                gmeta.create_dataset("nbp", data=nbp_arr)
                gset = gmeta.create_group("settings")
                for field in fields(self._settings):
                    gset.attrs[field.name] = getattr(self._settings,
                                                     field.name)
                if self._eseq is not None:
                    geseq = gmeta.create_group("eseq")
                    for chrom in self._chroms:
                        geseq.create_dataset(chrom, data=self._eseq[chrom],
                                              compression="gzip")
            # Save any analysis data
            if "analysis" in h5stream: del h5stream["analysis"]
            gana = h5stream.create_group("analysis")
            for chrom, data in self._analysis.items():
                gchrom = gana.create_group(chrom)
                for name, df in data.items():
                    h5_utils.save_df(name, df, gchrom)
            if "global_analysis" in h5stream: del h5stream["global_analysis"]
            gana = h5stream.create_group("global_analysis")
            for name, df in self._global_analysis.items():
                h5_utils.save_df(name, df, gana)
            
    def iter_runs(self, chroms : str | Iterable[str] | None = None) \
                  -> Iterator[Tuple[str,int,int]]:
        """
        Yield simulation identifiers across specified chromosomes and
        molecules.

        This generator traverses the dataset raw data hierarchy, filtering by
        chromosome and verifying the existence of simulation files before
        yielding.

        Parameters
        ----------
        chroms : str or iterable of str, optional
            The chromosome(s) to iterate over. If None, all available
            chromosomes in the dataset are processed.

        Yields
        ------
        chrom : str
            The chromosome identifier.
        mol : int
            The molecule index.
        run : int
            The simulation run index.

        Notes
        -----
        Only triplets (chrom, mol, run) that have an existing simulation file on
        disk according to the internal file map are yielded.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=self._chroms)
        for chrom in chroms:
            if chrom not in self._chroms: continue
            for mol in range(self._nmol[chrom]):
                for run in range(self._nsim):
                    sim_file = self._file_map.params_to_file(chrom, mol, run)
                    if sim_file.exists(): yield (chrom, mol, run)

    def extract(self, *,
                time : int,
                obs : str,
                nworker : int = 1,
                batch_size : int = 1000,
                chroms : str | Iterable[str] | None = None,
                agg_func : Callable[...,Any] | None = None,
                **agg_kwargs : Any):
        """
        Extract and optionally aggregate simulation observables across the
        dataset.

        Use :class:`ThreadPoolExecutor` to extract data from simulation HDF5
        files in parallel. Results are organized by chromosome and can be 
        processed via a custom aggregation function as they complete.

        Parameters
        ----------
        time : int
            The specific time point to extract from the simulation.
        obs : str
            The name of the observable to extract (i.e., 'position', 'energy',
            or 'temp').
        nworker : int, default 1
            The maximum number of threads to use for parallel extraction.
        batch_size : int, default 1000
            The number of simulation files to queue in a single processing
            batch to manage memory and thread overhead.
        chroms : str or iterable of str, optional
            Specific chromosome(s) to process. If None (default), all 
            chromosomes in the dataset are included.
        agg_func : callable, optional
            A function used to aggregate results once all runs for a specific 
            molecule are finished. If provided, it is called as:
            ``agg_func(chrom, mol, raw_data, **agg_kwargs)``.
        **agg_kwargs : Any
            Additional keyword arguments passed directly to `agg_func`.
        
        Returns
        -------
        dict
            A dictionary keyed by chromosome name. If `agg_func` is None,
            values are :class:`numpy.ndarray` (or lists for 'position'). If
            `agg_func` is provided, values are the output of the aggregation
            function.

        Notes
        -----
        When `obs` is not 'position' and no `agg_func` is provided, the data 
        for each chromosome is automatically converted into a NumPy array with 
        shape ``(n_molecules, n_simulations, ...)``.
        """

        # Check that no extra arguments are provided when agg_func is not used
        if agg_func is None and agg_kwargs:
            unknown_args = ", ".join(agg_kwargs.keys())
            raise TypeError("extract() got unexpected keyword arguments: "
                            f"{unknown_args}. These can only be used when an "
                            "'agg_func' is provided.")
        
        chroms = utils.normalize_chroms(chroms, default_chroms=self._chroms)
        
        results = {}
        for chrom in chroms:
            results[chrom] = [[None for _ in range(self._nsim)] 
                              for _ in range(self._nmol[chrom])]

        # Get the expected number of simulations per molecule
        finished_counts = Counter()
        expected_counts = Counter()
        for sim_id in self.iter_runs(chroms):
            chrom, mol, _ = sim_id
            expected_counts[(chrom,mol)] += 1

        if nworker > 1:
            with ThreadPoolExecutor(max_workers=nworker) as executor:        
                while True:
                    # Grab a chunk of work
                    batch = list(islice(self.iter_runs(chroms), batch_size))
                    if not batch: break
                    # Submit only this batch
                    tasks : Dict[Future,Tuple] = {
                        executor.submit(SimData.extract,
                                        self._file_map.params_to_file(*sim_id),
                                        time, obs): sim_id for sim_id in batch}
                    # Process this batch as it completes
                    for future in as_completed(tasks):
                        chrom, mol, run = tasks[future]
                        try:
                            data = future.result()
                            results[chrom][mol][run] = data
                            key = (chrom, mol)
                            finished_counts[key] += 1
                            # Aggregate results from different runs if needed
                            if agg_func and \
                               finished_counts[key] == expected_counts[key]:
                                raw_data = results[chrom][mol]
                                if obs != "position":
                                    raw_data = np.asarray(raw_data)
                                results[chrom][mol] = \
                                    agg_func(chrom, mol, raw_data,
                                             **agg_kwargs)
                        except Exception as e:                
                            print(f"Fail to extract '{obs}' from chrom "
                                  f"{chrom}, molecule {mol}, run {run}: {e}")
                    # Clean up batch
                    tasks.clear()
        else: # nworker = 1
            for sim_id in self.iter_runs(chroms):
                chrom, mol, run = sim_id
                sim_file = self._file_map.params_to_file(chrom, mol, run)
                results[chrom][mol][run] = SimData.extract(sim_file, time, obs)
                key = (chrom, mol)
                finished_counts[key] += 1
                # Aggregate results from different runs if needed
                if agg_func and finished_counts[key] == self._nsim:
                    raw_data = results[chrom][mol]
                    if obs != "position":
                        raw_data = np.asarray(raw_data)
                    results[chrom][mol] = agg_func(chrom, mol, raw_data,
                                                   **agg_kwargs)
                
        # Tidy up results and return a numpy array for each chromosome data
        if agg_func is None and obs != "position":
            for chrom in chroms:
                results[chrom] = np.asarray(results[chrom])                
        return results
    
    def sim_file(self, chrom : str, mol : int, run : int) -> Path:
        """
        Retrieve the file directory for a specific simulation run.
        
        Parameters
        ----------
        chrom : str
            The identifier of the simulated chromosome.
        mol : int
            The index of the simulated molecule.
        run : int
            The index of the simulation run.

        Returns
        -------
        Path
            The directory of the HDF5 file for the specified simulation run.

        Notes
        -----
        This method returns the expected file directory based on the dataset
        mapping structure, regardless of whether the file actually exists on
        disk.
        """        
        return self._file_map.params_to_file(chrom, mol, run)

    @property
    def settings(self) -> Mapping[str,Any]:
        """
        A read-only, live view of the simulation settings and parameters.
        
        Provide zero-copy access to the underlying simulation settings object.
        To access a specific parameter, say the DNA linker length `llink`, one
        can do the following:

        >>> dataset = SimDataset.load("results/dataset.h5")
        >>> dataset.settings['llink']

        Returns
        -------
        Mapping[str,Any]
            An immutable mapping of parameter names and their values.
        """
        return DataclassPublicProxy(self._settings)
    
    @property
    def chroms(self) -> Iterable[str]:
        """
        List the identifiers of all chromosomes in the dataset.

        Returns
        -------
        Iterable of str
            The chromosome names [e.g., ('chr1', 'chr2')].
        """
        return tuple(self._chroms)

    @property
    def nmol(self) -> Mapping[str,int]:
        """
        The number of molecules simulated for each chromosome.

        Returns
        -------
        types.MappingProxyType
            A read-only mapping of chromosome names to their molecule 
            counts. This view is immutable; neither keys nor counts 
            can be modified.
        """
        return MappingProxyType(self._nmol)

    @property
    def nsim(self) -> int:
        """
        The total number of simulation runs performed per molecule.

        Returns
        -------
        int
            The count of independent simulation trajectories for each 
            molecule-chromosome pair.
        """
        return self._nsim

    @property
    def nbp(self) -> Mapping[str,int]:
        """
        The number of base pairs simulated for each chromosome.

        Returns
        -------
        types.MappingProxyType
            A read-only mapping of chromosome names to their length in base
            pairs. This view is immutable; neither keys nor counts 
            can be modified.
        """        
        return MappingProxyType(self._nbp)
    
    @property
    def eseq(self) -> Mapping[str, np.ndarray]:
        """
        The seqeunce-specific energy landscape derived from the methylation
        data for each molecule in each chromosome.

        Returns
        -------
        types.MappingProxyType
            A read-only mapping of chromosome names to the energy landscapes of
            the methylation data for those chromosome molecules. This view is
            immutable; neither keys nor counts can be modified.
        """        
        return MappingProxyType(self._view_eseq)
    
    @property
    def raw(self) -> SimDataAccessor:
        """
        Provide lazy access to the raw simulation data files.

        This property returns an accessor object that supports tuple-based 
        indexing to load individual :class:`SimData` objects on demand.

        Returns
        -------
        SimDataAccessor
            A coordinate-based accessor. Use ``dataset.raw[chrom, mol, run]`` 
        """
        return self._raw_accessor

    @property
    def analysis(self) -> Mapping[str, DataFrameMap]:
        """
        Access the analysis results for each simulated chromosome.

        Returns
        -------
        FixedKeyMap
            A mapping of chromosome names to :class:`DataFrameMap` objects. 
            The chromosome keys are fixed, but the data within each 
            map remains mutable.
        """
        return FixedKeyMap(self._analysis)

    @property
    def global_analysis(self) -> DataFrameMap:
        """
        Access analysis results aggregated across the entire dataset.

        Returns
        -------
        DataFrameMap
            A mutable map containing global analysis results.
        """
        return self._global_analysis

    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [attr for attr in self.__slots__ if
                        not attr.startswith('_')]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith('_'):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str
