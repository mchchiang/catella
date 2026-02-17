# sim_data.py

import numpy as np
import inspect
import h5py
import re
from functools import lru_cache
from itertools import islice
from threading import Semaphore
from dataclasses import dataclass, field, InitVar
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, Self, Dict, Any, Callable, List
from collections import Counter
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from .util import DataFrameMap, FixedKeyMap
from . import util

# Limit the concurrent open files to stay under OS limits
MAX_OPEN_FILES = Semaphore(500)

@util.copy_array_properties
@dataclass(frozen=True, slots=True, init=False)
class SimData:
    """
    Container for a single simulation result.

    A ``SimData`` object stores:

    - simulation parameters
    - raw data arrays

    Parameters
    ----------
    nucbp : int
        The number of base pairs occupied by a nucleosome.
    nbp : int
        The number of base pairs of the chromatin fiber.
    llink : int
        The DNA linker length between nucleosomes.
    mu : float
        The ``chemical potential'' - the energy gained by adding a nucleosome
        to the fiber (a positive mu favours nucleosome binding).
    seed : int
        The seed for initializing the random generator needed for running the
        Monte Carlo simulation.

    Notes
    -----
    Some notes.
    """
    
    nucbp : int
    nbp : int
    llink : int
    mu : float
    seed : int
    _time : np.ndarray
    _energy : np.ndarray
    _position : Tuple[np.ndarray,...]
    _temp : np.ndarray

    # Internal fields
    _time_to_idx : dict[int,int] = field(repr=False)

    def __init__(self, *args : Any, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use SimData.load() to instantiate this class")
        sig = inspect.signature(self._create)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for key,val in bound.arguments.items():
            object.__setattr__(self, key, val)
    
        # Build lookup: time -> index
        time_map = {int(t): i for i,t in enumerate(self.time)}
        object.__setattr__(self, "_time_to_idx", time_map)

    def idx(self, t: int) -> int:
        """
        Return the frame index for a specific time point.

        Parameters
        ----------
        t : int
            Time point of interest.

        Returns
        -------
        int
            Frame index.
        """        
        return self._time_to_idx[t]

    def energy_at(self, t: int):
        """
        Return the system's energy at a specific time point.

        Parameters
        ----------
        t : int
            Time point of interest for the system's energy.

        Returns
        -------
        float
            The system's energy at time t.
        """        
        return None if self._energy is None else self._energy[self.idx(t)]

    def position_at(self, t: int):
        """
        Return the nucleosome positions at a specific time point.

        Parameters
        ----------
        t : int
            Time point of interest for the positions.

        Returns
        -------
        Tuple
            Tuple of nucleosome positions (left-aligned) at time t.
        """
        return None if self._position is None else \
            tuple(self._position[self.idx(t)])
    
    def save(self, path : str | Path):
        """
        Save simulation data to an HDF5 file.

        Parameters
        ----------
        path : str | pathlib.Path
            Output file path.

        Raises
        ------
        OSError
            If file cannot be written.
        """
        with h5py.File(path, "w") as f:
            # Store model parameters
            g = f.create_group("params")
            g.attrs["nucbp"] = self.nucbp
            g.attrs["nbp"] = self.nbp
            g.attrs["llink"] = self.llink
            g.attrs["mu"] = self.mu
            g.attrs["seed"] = self.seed
            
            # Store data
            g = f.create_group("data")
            def save_data(g, name, arr):
                if arr is not None:
                    g.create_dataset(name, arr, compression="gzip")
            save_data(g, "energy", self.energy)
            save_data(g, "temp", self.temp)
            save_data(g, "time", self.time)

    @classmethod
    def load(cls, path : str | Path) -> Self:
        """
        Load simulation data from an HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            Simulation data file path.

        Returns
        -------
        SimData
            Loaded simulation object.

        Raises
        ------
        OSError
            If file cannot be open.
        """        
        with h5py.File(path, "r") as f:
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
                return None
            time = load_data(g, "time")
            energy = load_data(g, "energy")
            position = load_data(g, "position")
            temp = load_data(g, "temp")
            return cls._create(nucbp, nbp, llink, mu, seed, time, energy,
                               position, temp)

    # Extract the data for a specific obesrvable for a single time point
    @classmethod
    def extract(cls, path : str | Path, time : int, obs : str) -> np.ndarray:
        norm_path = str(Path(path).resolve())
        try:
            return cls._cached_extract(norm_path, time, obs)
        except Exception:
            return np.array([np.nan]) if obs == "position" else np.nan

    @staticmethod
    @lru_cache(maxsize=1024)    
    def _cached_extract(path : str, time : int, obs : str) -> np.ndarray:
        with MAX_OPEN_FILES:
            with h5py.File(path, "r") as f:
                tdata = f["data/time"][()]
                idx = np.argmin(np.abs(tdata-time))
                if obs == "position":
                    flat = f["data/position_flat"]
                    offset = f["data/position_offset"][()]
                    start = offset[idx]
                    end = offset[idx+1] if idx+1 < len(offset) \
                        else flat.shape[0]
                    return flat[start:end]
                else:
                    return f[f"data/{obs}"][idx]
    
    @classmethod
    def _create(cls, nucbp, nbp, llink, mu, seed, _time, _energy, _position,
                _temp) -> Self:
        # Some validation
        if nucbp <= 0 or nbp <= 0 or llink <= 0 or seed <= 0:
            raise ValueError("nucbp, nbp, llink and seeed must be positive")
        if not (len(_time) == len(_energy) == len(_temp)):
            raise ValueError("time, energy, and temp must have the",
                             "same length")
        if not all(isinstance(p, np.ndarray) for p in _position):
            raise TypeError("position must be a tuple of np.ndarray")        
        return cls(nucbp, nbp, llink, mu, seed, _time, _energy, _position,
                   _temp, _internal=True)

    
class SimPathMapper:
    def __init__(self, root_dir : str, total_mols : int,
                 max_mols_per_dir : int = 100):
        self.root_path = Path(root_dir)
        self.max_mols_per_dir = max_mols_per_dir
        self.zpad = int(np.ceil(np.log10(total_mols)))

    def params_to_path(self, chrom : str, molidx : int, run : int) -> Path:
        # Compute range
        start = int((molidx // self.max_mols_per_dir) * self.max_mols_per_dir)
        end = int(start + self.max_mols_per_dir - 1)
        # Get name at each level
        z = self.zpad
        shard_name = f"mol_{int(start):0{z}d}-{int(end):0{z}d}"
        mol_name = f"mol_{int(molidx):0{z}d}"
        run_name = f"run_{int(run):02d}"
        sim_h5 = f"{chrom}_{mol_name}_{run_name}.h5"
        return self.root_path/chrom/shard_name/mol_name/sim_h5

    def path_to_params(self, path : str | Path):
        path = Path(path)
        chrom = path.parts[-4]
        filename = path.name
        match = re.search(r"mol_(\d+)_run_(\d+)\.h5$", filename)
        if match:
            molidx = int(match.group(1))
            run = int(match.group(2))        
            return chrom, molidx, run
        raise ValueError(f"Could not parse parameters from path: {path}") 
    
@dataclass(slots=True, init=False)
class SimDataset:
    """
    Filesystem-backed collection of simulation results.

    Provides lazy access to stored simulations via
    tuple-based indexing:

    >>> dataset = SimDataset("results/dataset.h5")
    >>> sim = dataset.sim["chr1", 0, 10]

    Attributes
    ----------
    path : pathlib.Path
        Base directory containing simulation files.
    """
    
    class SimDataAccessor:
        """
        Lazy accessor for simulation data.

        Enables access using:

        >>> dataset.sim[chrom, mol, run]
        """
        def __init__(self, parent):
            self._parent = parent
            
        def __getitem__(self, keys):
            chrom, mol, run = keys
            path = self._parent.get_sim_path(chrom, mol, run)
            if path.exists():
                return self._load_sim_data(path)
            else:
                raise ValueError("Cannot find the simulation for chrom "
                                 f"{chrom}, molecule {mol}, run {run}")

        @staticmethod
        @lru_cache(maxsize=1024)
        def _load_sim_data(path : str | Path):
            return SimData.load(path)
    
    _chroms : Iterable[str]
    _nmol : Mapping[str,int]
    _nsim : int
    _raw_path : str | Path    
    _path_map : SimPathMapper
    _analysis : Mapping[str, DataFrameMap]
    _global_analysis : DataFrameMap    
    _raw_accessor : SimDataAccessor

    def __init__(self, *args : Any, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use SimDataset.load() or .create_from_sim() ",
                            "to instantiate this class")
        sig = inspect.signature(self._create)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        for key,val in bound.arguments.items():
            setattr(self, key, val)

        # Create the dicts for analysis
        self._analysis = {chrom : DataFrameMap() for chrom in self._chroms}
        self._global_analysis = DataFrameMap()
        
        self._raw_accessor = self.SimDataAccessor(self)

    @classmethod
    def _create(cls, _chroms, _nmol, _nsim, _raw_path, _path_map):
        # Some validation
        if _nsim <= 0:
            raise ValueError("Number of simulations must be positive")
        # Check the keys in nmol match up with chroms
        if set(_chroms) != set(_nmol.keys()):
            raise ValueError("Chromosomes in nmol do not match with chroms")
        # Check the number of molcules in each chromosome is valid
        for chrom in _chroms:
            if _nmol[chrom] <= 0:
                raise ValueError(f"Chromosome {chrom} must have at least one "
                                 "molecule")
        return cls(_chroms, _nmol, _nsim, _raw_path, _path_map, _internal=True)
        
    @classmethod
    def create_from_sim(cls, chroms : Iterable[str],
                        nmol : Mapping[str,int],
                        nsim : int,
                        out_path : str | Path):
        # Get the path mapper
        # Find the maximum number of molecules
        max_nmol = 0
        for chrom in chroms:
            if max_nmol < nmol[chrom]:
                max_nmol = nmol[chrom]
        raw_path = Path(out_path)/"raw_data"
        raw_path.mkdir(exist_ok=True, parents=True)
        path_map = SimPathMapper(raw_path, max_nmol)
        return cls._create(chroms, nmol, nsim, raw_path, path_map)

    @classmethod
    def load(cls, path : str | Path):
        with h5py.File(path, "r") as h5stream:
            # Load metadata associated with raw simulation data
            gmeta = h5stream["metadata"]
            raw_path = Path(gmeta.attrs["raw_path"])
            nsim = gmeta.attrs["nsim"]
            chroms = [c.decode() for c in gmeta["chroms"][:]]
            nmol_arr = gmeta["nmol"][:]
            nmol = {chrom:nmol_arr[i] for i,chrom in enumerate(chroms)}
            path_map = SimPathMapper(raw_path, max(nmol_arr))            
            obj = cls._create(chroms, nmol, nsim, raw_path, path_map)
            
            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                gchrom = gana[chrom]
                for name in gchrom:
                    obj._analysis[chrom][name] = util.load_df(name, gchrom)
            gana = h5stream["global_analysis"]
            for name in gana:
                obj._global_analysis[name] = util.load_df(name, gana)
            return obj

    def save(self, path : str | Path):
        dt = h5py.string_dtype(encoding="utf-8")
        with h5py.File(path, "a") as h5stream:
            # Save metadata asssociated with raw simulation data
            if not "metadata" in h5stream:
                gmeta = h5stream.create_group("metadata")
                gmeta.attrs["raw_path"] = str(self._raw_path)
                gmeta.attrs["nsim"] = self._nsim        
                gmeta.create_dataset("chroms", data=self._chroms, dtype=dt)
                nmol_arr = np.asarray([self._nmol[chrom]
                                       for chrom in self._chroms])
                gmeta.create_dataset("nmol", data=nmol_arr)

            # Save any analysis data
            if "analysis" in h5stream: del h5stream["analysis"]
            gana = h5stream.create_group("analysis")
            for chrom, data in self._analysis.items():
                gchrom = gana.create_group(chrom)
                for name, df in data.items():
                    util.save_df(name, df, gchrom)
            if "global_analysis" in h5stream: del h5stream["global_analysis"]
            gana = h5stream.create_group("global_analysis")
            for name, df in self._global_analysis.items():
                util.save_df(name, df, gana)
            
    def generate_sim_paths(self,
                           chroms : str | Iterable[str] | None = None):
        chroms = util.normalize_chroms(chroms, default_chroms=self._chroms)
        for chrom in chroms:
            if chrom not in self._chroms: continue
            for mol in range(self._nmol[chrom]):
                for run in range(self._nsim):
                    path = self._path_map.params_to_path(chrom, mol, run)
                    if path.exists():
                        yield path, (chrom, mol, run)

    def extract(self,
                time : int,
                obs : str,
                nworker : int = 100,
                batch_size : int = 5000,
                chroms : str | Iterable[str] | None = None,
                agg_func : Callable[...,Any] | None = None,
                **agg_kwargs : Any):
        chroms = util.normalize_chroms(chroms, default_chroms=self._chroms)
        results = {}
        finished_counts = Counter()
        for chrom in chroms:
            results[chrom] = [[None for _ in range(self._nsim)] 
                              for _ in range(self._nmol[chrom])]
        path_gen = self.generate_sim_paths(chroms)
        with ThreadPoolExecutor(max_workers=nworker) as executor:        
            while True:
                # Grab a chunk of work
                batch = list(islice(path_gen, batch_size))
                if not batch: break
                # Submit only this batch
                tasks : Dict[Future,Tuple] = {
                    executor.submit(SimData.extract, path, time, obs): sim_id 
                    for path, sim_id in batch}
                # Process this batch as it completes
                for future in as_completed(tasks):
                    chrom, mol, run = tasks[future]
                    try:
                        data = future.result()
                        results[chrom][mol][run] = data
                        key = (chrom, mol)
                        finished_counts[key] += 1
                        # Aggregate results from different runs if needed
                        if agg_func and finished_counts[key] == self._nsim:
                            raw_runs = results[chrom][mol]
                            if obs != "position":
                                raw_runs = np.asarray(raw_runs)
                            results[chrom][mol] = \
                                agg_func(chrom, mol, raw_runs, **agg_kwargs)
                    except Exception as e:                   
                        print(f"Fail to extract '{obs}' from chrom {chrom}, "
                              f"molecule {mol}, run {run}: {e}")
                # Clean up batch
                tasks.clear()
        # Tidy up results and return a numpy array for each chromosome data
        if agg_func is None and obs != "position":
            for chrom in chroms:
                results[chrom] = np.asarray(results[chrom])
        return results

    def get_sim_path(self,
                     chrom : str,
                     mol : int,
                     run : int) -> Path:
        return self._path_map.params_to_path(chrom, mol, run)

    @property
    def chroms(self):
        # Return an immutable view
        return tuple(self._chroms)

    @property
    def nmol(self):
        return MappingProxyType(self._nmol)

    @property
    def nsim(self):
        return self._nsim
    
    @property
    def raw(self):
        return self._raw_accessor

    @property
    def analysis(self) -> Mapping[str, DataFrameMap]:
        return FixedKeyMap(self._analysis)

    @property
    def global_analysis(self) -> DataFrameMap:
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
