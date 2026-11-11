# results.py

import os
import shutil
import weakref
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from threading import Semaphore
from types import MappingProxyType
from typing import Any, Self

import h5py
import numpy as np

from catella import h5_utils, utils
from catella.containers import DataclassPublicProxy, DataFrameMap, FixedKeyMap
from catella.h5_array import H5Array
from catella.simulation._io import (
    SimFileMapper,
    _read_entries,
    _read_mcmc_file,
    _read_metadata,
    _read_tm_file,
    _replace_group,
    _tm_sample_fields,
    _write_entries,
    _write_metadata,
)
from catella.simulation._parallel import _threaded_map
from catella.simulation.config import SimSettings

# Limit the concurrent open files to stay under OS limits
MAX_OPEN_FILES = Semaphore(500)


@utils.add_frozen_properties
@dataclass(frozen=True, slots=True, init=False)
class SimMCMCData:
    """
    Container for results and parameters of a single MCMC simulation run.

    This class provides a read-only, memory-efficient representation of
    Monte Carlo simulation data, including physical parameters (chemical
    potential, fiber length) and recorded observables (energy, positions).

    .. note::
       Direct instantiation is disabled to ensure data integrity. Use the
       factory method :meth:`SimMCMCData.load` to create instances from
       HDF5 files.
    """

    nucbp: int
    """The number of base pairs occupied by a single nucleosome."""

    nbp: int
    """The total number of base pairs in the chromatin fiber."""

    llink: int
    """The DNA linker length between nucleosomes."""

    elink: float
    """The energy scale of the linker repulsion potential."""

    mu: float
    """The chemical potential (energy gained by adding a nucleosome)."""

    seed: int
    """The seed used for initializing the random number generator."""

    _frozen_time: np.ndarray = field(
        metadata={"doc": "np.ndarray: The simulation time points."}
    )

    _frozen_energy: np.ndarray = field(
        metadata={
            "doc": "np.ndarray: The system energy recorded at each time point."
        }
    )

    _frozen_position: tuple[np.ndarray, ...] = field(
        metadata={
            "doc": "tuple of np.ndarray: Nucleosome positions for each "
            "time frame."
        }
    )

    _frozen_temp: np.ndarray = field(
        metadata={
            "doc": "np.ndarray: The system temperature recorded at each "
            "time point."
        }
    )

    # Map from time values to frame index
    _time_to_idx: dict[int, int]
    _obs_map: dict[str, Any]
    _obs_list = ("energy", "position", "temp")

    def __init__(self, **kwargs: Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("use SimMCMCData.load() to instantiate this class")

        # Bulk assignment of fields
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is None:
                val = kwargs.get(f.name.removeprefix("_frozen_"))
            object.__setattr__(self, f.name, val)

        # Build time index lookup
        time_map = {t: i for i, t in enumerate(self.time)}
        object.__setattr__(self, "_time_to_idx", time_map)

        # Build observable lookup
        obs_map = {
            "energy": self.energy,
            "position": self.position,
            "temp": self.temp,
        }
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

    def at(self, t: int, obs: str):
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
        if obs not in self._obs_map:
            raise ValueError(f"'{obs}' is not a valid observable")
        value = self._obs_map[obs][self._time_to_idx[t]]
        return value.item() if np.ndim(value) == 0 else value

    @classmethod
    def load(cls, sim_file: str | Path) -> Self:
        """
        Load simulation data from a persistent HDF5 file.

        Parameters
        ----------
        sim_file : str or Path
            The HDF5 file containing simulation results.

        Returns
        -------
        SimMCMCData
            A new instance populated with the loaded parameters and data.

        Raises
        ------
        OSError
            If the file cannot be opened or does not follow the
            SimMCMCData schema.
        """
        return cls._create(**_read_mcmc_file(sim_file))

    # Extract the data for a specific obesrvable for a single time point
    @classmethod
    def extract(cls, sim_file: str | Path, time: int, obs: str) -> np.ndarray:
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
            raise ValueError(f"'{obs}' is not a valid observable")
        sim_file = str(Path(sim_file).resolve())
        try:
            return cls._cached_extract(sim_file, time, obs)
        except Exception:  # noqa: BLE001
            return np.array([np.nan]) if obs == "position" else np.nan

    @staticmethod
    @lru_cache(maxsize=1024)
    def _cached_extract(sim_file: str, time: int, obs: str) -> np.ndarray:
        with MAX_OPEN_FILES, h5py.File(sim_file, "r") as f:
            tdata = f["data/time"][()]
            matches = np.where(tdata == time)[0]
            if len(matches) == 0:
                raise KeyError(f"cannot find data for time {time}")
            idx = matches[0]
            if obs == "position":
                flat = f["data/position_flat"]
                offset = f["data/position_offset"][()]
                start = offset[idx]
                end = (
                    offset[idx + 1] if idx + 1 < len(offset) else flat.shape[0]
                )
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
        if nucbp <= 0 or nbp <= 0 or seed <= 0:
            raise ValueError("nucbp, nbp, and seed must be positive")
        if llink < 0:
            raise ValueError(f"llink must be non-negative, got {llink}")
        if not (len(time) == len(energy) == len(temp) == len(position)):
            raise ValueError(
                "time, energy, position, and temp must have the "
                "same length"
            )
        return cls(**kwargs, _internal=True)

    def __repr__(self):
        # Get all public attributes by filtering out private attributes
        # (those starting with '_')
        public_attrs = [
            attr for attr in self.__slots__ if not attr.startswith("_")
        ]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith("_"):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str


@utils.add_frozen_properties
@dataclass(frozen=True, slots=True, init=False)
class SimTMData:
    """
    Container for one exact sample from the transfer-matrix solver.

    Unlike SimMCMCData, this holds no trajectory: a single TM run is
    one independent, exact sample drawn via stochastic traceback, not
    a sequence of Monte Carlo sweeps. There is no time or temperature
    axis.

    .. note::
       Direct instantiation is disabled. Access samples via
       :attr:`SimDataset.raw`.
    """

    nucbp: int
    """The number of base pairs occupied by a single nucleosome."""

    nbp: int
    """The total number of base pairs in the chromatin fiber."""

    llink: int
    """The DNA linker length between nucleosomes."""

    elink: float
    """The energy scale of the linker repulsion potential."""

    mu: float
    """The chemical potential (energy gained by adding a nucleosome)."""

    seed: int
    """The seed used to draw this exact sample."""

    _frozen_position: np.ndarray = field(
        metadata={"doc": "np.ndarray: The sampled nucleosome start "
                  "positions."}
    )

    _frozen_energy: float = field(
        metadata={
            "doc": "float: The exact Hamiltonian energy of this sample."
        }
    )

    _frozen_logz: float = field(
        metadata={
            "doc": "float: The log partition function for this molecule."
        }
    )

    def __init__(self, **kwargs: Any):
        if not kwargs.pop("_internal", False):
            raise TypeError(
                "SimTMData cannot be instantiated directly; access samples "
                "via SimDataset.raw"
            )

        for f in fields(self):
            val = kwargs.get(f.name)
            if val is None:
                val = kwargs.get(f.name.removeprefix("_frozen_"))
            object.__setattr__(self, f.name, val)

    @classmethod
    def _create(cls, **kwargs) -> Self:
        nucbp = kwargs.get("nucbp")
        nbp = kwargs.get("nbp")
        llink = kwargs.get("llink")
        seed = kwargs.get("seed")
        if nucbp <= 0 or nbp <= 0 or seed <= 0:
            raise ValueError("nucbp, nbp, and seed must be positive")
        if llink < 0:
            raise ValueError(f"llink must be non-negative, got {llink}")
        return cls(**kwargs, _internal=True)

    def __repr__(self):
        public_attrs = [
            attr for attr in self.__slots__ if not attr.startswith("_")
        ]
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith("_"):
                public_attrs.append(attr)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str


def _read_tm_sample(combined_file: str | Path, run: int) -> SimTMData:
    """
    Load one sample by index from a combined per-molecule TM file.

    Parameters
    ----------
    combined_file : str or Path
        The combined file written by :func:`_write_tm_samples`.
    run : int
        The sample index within the combined file.

    Returns
    -------
    SimTMData
        The requested sample.
    """
    arrays = _read_tm_file(combined_file)
    return SimTMData._create(**_tm_sample_fields(arrays, run))


@dataclass(slots=True, init=False, weakref_slot=True)
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
            if self._parent._method == "tm":
                combined_file = self._parent._file_map.params_to_tm_file(
                    chrom, mol
                )
                if not combined_file.exists():
                    raise ValueError(
                        "cannot find the simulation for chrom "
                        f"{chrom}, molecule {mol}, run {run}"
                    )
                arrays = self._load_tm_combined(str(combined_file))
                return self._slice_tm_sample(arrays, run)
            sim_file = self._parent.mcmc_file(chrom, mol, run)
            if sim_file.exists():
                return self._load_sim_data(str(sim_file))
            else:
                raise ValueError(
                    "cannot find the simulation for chrom "
                    f"{chrom}, molecule {mol}, run {run}"
                )

        @staticmethod
        @lru_cache(maxsize=1024)
        def _load_sim_data(sim_file: str):
            return SimMCMCData.load(sim_file)

        @staticmethod
        @lru_cache(maxsize=1024)
        def _load_tm_combined(combined_file: str):
            return _read_tm_file(combined_file)

        @staticmethod
        def _slice_tm_sample(arrays, run: int) -> "SimTMData":
            return SimTMData._create(**_tm_sample_fields(arrays, run))

    _chroms: Iterable[str]
    _nmol: Mapping[str, int]
    _nsim: int
    _nbp: Mapping[str, int]
    _settings: SimSettings
    _method: str
    _eseq: Mapping[str, np.ndarray]
    _view_eseq: Mapping[str, np.ndarray]
    _out_type: Any
    _seed: int
    _seed_table: Mapping[str, np.ndarray]
    _raw_dir: Path
    _dataset_file: Path
    _file_map: SimFileMapper
    _analysis: Mapping[str, DataFrameMap]
    _global_analysis: DataFrameMap
    _raw_accessor: SimDataAccessor
    _tmp_dir: str | None
    _finalizer: Any | None

    def __init__(self, *args: Any, **kwargs: Any):
        raise TypeError(
            "use SimDataset.load() or create() to instantiate this class"
        )

    @classmethod
    def _create(
        cls,
        *,
        chroms: Iterable[str],
        nmol: Mapping[str, int],
        nsim: int,
        nbp: Mapping[str, int],
        settings: SimSettings,
        raw_dir: Path,
        dataset_file: Path,
        file_map: SimFileMapper,
        eseq: Mapping[str, np.ndarray] | None,
        method: str = "mcmc",
        out_type: Any | None = None,
        seed: int | None = None,
        seed_table: Mapping[str, np.ndarray] | None = None,
    ):
        # Some validations
        if method not in ("mcmc", "tm"):
            raise ValueError("method must be 'mcmc' or 'tm'")
        if nsim < 0:
            raise ValueError("number of simulations cannot be negative")
        if method == "mcmc" and nsim == 0:
            raise ValueError(
                "number of simulations must be positive for method="
                "'mcmc'; only 'tm' datasets can start with nsim=0 "
                "(exact quantities do not require any samples)"
            )
        # Check the keys in nmol match up with chroms
        if set(nmol.keys()) != set(chroms):
            raise ValueError("chromosomes in nmol do not match with chroms")
        # Check the number of molcules in each chromosome is valid
        for chrom in chroms:
            if nmol[chrom] <= 0:
                raise ValueError(
                    f"chromosome {chrom} must have at least one molecule"
                )

        obj = cls.__new__(cls)
        obj._chroms = tuple(chroms)
        obj._nmol = dict(nmol)
        obj._nsim = int(nsim)
        obj._nbp = dict(nbp)
        obj._settings = settings
        obj._method = method
        obj._raw_dir = raw_dir
        obj._dataset_file = dataset_file
        obj._file_map = file_map
        obj._analysis = {chrom: DataFrameMap() for chrom in obj._chroms}
        obj._global_analysis = DataFrameMap()
        obj._raw_accessor = cls.SimDataAccessor(obj)
        obj._eseq = eseq
        if obj._eseq is None:
            obj._view_eseq = {chrom: np.nan for chrom in chroms}
        else:
            obj._view_eseq = {chrom: obj._eseq[chrom] for chrom in chroms}
            for chrom in chroms:
                obj._view_eseq[chrom].flags.writeable = False
        obj._out_type = out_type
        obj._seed = seed
        obj._seed_table = seed_table
        obj._tmp_dir = None
        obj._finalizer = None
        return obj

    @classmethod
    def create(
        cls,
        *,
        chroms: str | Iterable[str],
        nmol: int | Mapping[str, int],
        nsim: int,
        nbp: int | Mapping[str, int],
        settings: SimSettings,
        out_dir: str | Path,
        dataset_name: str = "results",
        eseq: np.ndarray | Mapping[str, np.ndarray] | None = None,
        method: str = "mcmc",
        out_type: Any | None = None,
        seed: int | None = None,
        seed_table: Mapping[str, np.ndarray] | None = None,
    ) -> Self:
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
            The number of simulation runs (or, for `method='tm'`, exact
            samples) per molecule. May be 0 for `method='tm'`, since the
            exact occupancy/mean-N/variance quantities require no
            samples at all; `method='mcmc'` requires a positive value.
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
        method : {"mcmc", "tm"}, default "mcmc"
            Which engine generated (or will generate) this dataset's raw
            samples.
        out_type : Dump.OutputType, optional
            The format or scope of data recorded for each simulation run.
        seed : int, optional
            The batch-level master seed the dataset's simulations were
            generated from.
        seed_table : Mapping[str, np.ndarray], optional
            The per-`(chrom, mol, run)` seed used for each simulation, one
            array per chromosome with shape `(nmol[chrom], nsim)`. Enables
            exact reproduction of any individual run later, even ones
            whose raw output file is missing or corrupted.

        Returns
        -------
        SimDataset
            A newly initialized instance of the simulation dataset.

        Raises
        ------
        FileExistsError
            If `out_dir` already exists.
        TypeError
            If `nmol` is neither an int nor a mapping.
        ValueError
            If the chromosome keys in `nmol` do not match `chroms`.
        """
        chroms = utils.normalize_chroms(chroms)
        if isinstance(nmol, int):
            nmol = {chrom: nmol for chrom in chroms}
        elif not isinstance(nmol, Mapping):
            raise TypeError("nmol must either be an int or a mapping")
        elif set(nmol.keys()) != set(chroms):
            raise ValueError(
                "nmol must have the same set of chromosome "
                "identifiers as chroms"
            )

        # Find the maximum number of molecules
        max_nmol = 0
        for chrom in chroms:
            max_nmol = max(max_nmol, nmol[chrom])

        # Check validity of the output directory
        out_dir = Path(out_dir).resolve()
        if out_dir.exists():
            raise FileExistsError(f"the directory '{out_dir}' already exists")
        else:
            raw_dir = out_dir / "raw_data"
            raw_dir.mkdir(exist_ok=True, parents=True)
        dataset_file = out_dir / (dataset_name + ".h5")
        file_map = SimFileMapper(raw_dir, max_nmol)
        return cls._create(
            chroms=chroms,
            nmol=nmol,
            nsim=nsim,
            nbp=nbp,
            settings=settings,
            eseq=eseq,
            raw_dir=raw_dir,
            dataset_file=dataset_file,
            file_map=file_map,
            method=method,
            out_type=out_type,
            seed=seed,
            seed_table=seed_table,
        )

    @classmethod
    def load(
        cls, dataset_file: str | Path, new_raw_dir: str | Path | None = None
    ) -> Self:
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
        SimMCMCData
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
            meta = _read_metadata(h5stream)
            stored_raw_dir = meta.pop("raw_dir")
            raw_dir = Path(
                stored_raw_dir if new_raw_dir is None else new_raw_dir
            ).resolve()
            if not raw_dir.exists():
                raise FileNotFoundError(
                    "the raw simulation data directory cannot be found"
                )
            file_map = SimFileMapper(raw_dir, max(meta["nmol"].values()))
            obj = cls._create(
                **meta,
                raw_dir=raw_dir,
                file_map=file_map,
                dataset_file=dataset_file,
            )
            gana = h5stream["analysis"]
            for chrom in gana:
                entries = _read_entries(gana[chrom], dataset_file)
                for name, entry in entries.items():
                    obj._analysis[chrom][name] = entry
            entries = _read_entries(h5stream["global_analysis"], dataset_file)
            for name, entry in entries.items():
                obj._global_analysis[name] = entry
            return obj

    def save(
        self,
        dataset_file: str | Path | None = None,
        *,
        overwrite: bool = False,
    ):
        """
        Save the dataset metadata and analysis results to an HDF5 file.

        Serialize the current state of all chromosome-specific and global
        analysis :class:`DataFrameMap` objects into a persistent HDF5 format.

        Parameters
        ----------
        dataset_file : str | Path, optional
            The output HDF5 file for the dataset. If the file already exists,
            analysis groups will be overwritten. If None, data will be written
            to the original dataset file from loading or creation.
        overwrite : bool, default False
            If True, allow saving to the same file that backs an
            existing `H5Array` analysis entry, by writing to a temporary
            sibling file and atomically renaming it into place.

        Raises
        ------
        OSError
            If the file cannot be written to disk.
        ValueError
            If `dataset_file` is the same file that backs an `H5Array`
            analysis entry already held by this dataset, and `overwrite`
            is False.
        """
        if dataset_file is None:
            dataset_file = self._dataset_file
        dest_path = Path(dataset_file)
        colliding = self._entries_backed_by(dest_path)
        if colliding:
            if not overwrite:
                raise ValueError(
                    "cannot save to the same file that backs an "
                    "existing H5Array analysis entry; save to a "
                    "different path, or pass overwrite=True to safely "
                    "replace it in place"
                )
            self._save_replacing(dest_path, colliding)
            return

        with h5py.File(dataset_file, "a") as h5stream:
            _write_metadata(
                h5stream,
                raw_dir=self._raw_dir,
                nsim=self._nsim,
                method=self._method,
                chroms=self._chroms,
                nmol=self._nmol,
                nbp=self._nbp,
                settings=self._settings,
                eseq=self._eseq,
                out_type=self._out_type,
                seed=self._seed,
                seed_table=self._seed_table,
            )
            gana = _replace_group(h5stream, "analysis")
            for chrom, data in self._analysis.items():
                _write_entries(gana.create_group(chrom), data)
            gglobal = _replace_group(h5stream, "global_analysis")
            _write_entries(gglobal, self._global_analysis)

    def _entries_backed_by(self, path):
        """Return `(data, name, dataset_path)` of H5Array entries in `path`."""
        dest = str(Path(path).resolve())
        return [
            (data, name, entry.dataset_path)
            for data in [*self._analysis.values(), self._global_analysis]
            for name, entry in data.items()
            if isinstance(entry, H5Array)
            and str(Path(entry.path).resolve()) == dest
        ]

    def _save_replacing(self, dest_path, colliding):
        """Save to a sibling file, then atomically replace `dest_path`."""
        tmp_path = dest_path.with_name(dest_path.name + f".tmp{os.getpid()}")
        closed = False
        try:
            self.save(tmp_path)
            # Windows cannot rename over an open file, so close entries first
            for data, name, _ in colliding:
                data[name].close()
            closed = True
            os.replace(tmp_path, dest_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise
        finally:
            if closed:
                for data, name, dataset_path in colliding:
                    data[name] = H5Array.load_from(dest_path, dataset_path)

    @staticmethod
    def _cleanup_tmp_dir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def resolve_tmp_dir(self):
        """
        Return this dataset's scratch directory for temporary HDF5
        files, creating and caching one under the system default
        temporary directory the first time it is needed. The directory
        is removed once the dataset is closed or garbage-collected; see
        `close`.

        Returns
        -------
        str
            Path to the resolved scratch directory.
        """
        if self._tmp_dir is None:
            self._tmp_dir = h5_utils.fresh_tmp_dir()
            self._finalizer = weakref.finalize(
                self, SimDataset._cleanup_tmp_dir, self._tmp_dir
            )
        return self._tmp_dir

    def close(self):
        """
        Delete this dataset's scratch directory (all temporary HDF5
        files created for it, e.g. by `SimAnalysis.compute_occup`), if
        one was created. No-op for datasets with no scratch directory.
        """
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def iter_runs(
        self,
        chroms: str | Iterable[str] | None = None,
        mols: Iterable[int] | None = None,
    ) -> Iterator[tuple[str, int, int]]:
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
        mols : iterable of int, optional
            Molecule indices to restrict iteration to, applied to every
            chromosome in `chroms`. If None (default), all molecules of
            each chromosome are iterated.

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
        Only triplets (chrom, mol, run) that have an existing simulation
        file on disk according to the internal file map are yielded.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=self._chroms)
        for chrom in chroms:
            if chrom not in self._chroms:
                continue
            mol_range = range(self._nmol[chrom]) if mols is None else mols
            for mol in mol_range:
                for run in range(self._nsim):
                    sim_file = self._file_map.params_to_mcmc_file(
                        chrom, mol, run
                    )
                    if sim_file.exists():
                        yield (chrom, mol, run)

    def extract(
        self,
        *,
        time: int,
        obs: str,
        nworker: int = 1,
        batch_size: int = 1000,
        chroms: str | Iterable[str] | None = None,
        mols: Iterable[int] | None = None,
        agg_func: Callable[..., Any] | None = None,
        **agg_kwargs: Any,
    ):
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
        mols : iterable of int, optional
            Molecule indices to restrict extraction to, applied to every
            chromosome in `chroms`. If None (default), all molecules of
            each chromosome are processed. Use this to extract in
            molecule batches and bound peak memory when `agg_func`
            condenses each molecule's data.
        agg_func : callable, optional
            A function used to aggregate results once all runs for a specific
            molecule are finished. If provided, it is called as:
            ``agg_func(chrom, mol, raw_data, **agg_kwargs)``, where
            `raw_data` holds only the runs with a simulation file on disk.
        **agg_kwargs : Any
            Additional keyword arguments passed directly to `agg_func`.

        Returns
        -------
        dict
            A dictionary keyed by chromosome name. If `agg_func` is None,
            values are :class:`numpy.ndarray` (or lists for 'position'). If
            `agg_func` is provided, values are the output of the aggregation
            function, except for molecules with no simulation files on
            disk, which are `None`.

        Notes
        -----
        When `obs` is not 'position' and no `agg_func` is provided, the data
        for each chromosome is automatically converted into a NumPy array with
        shape ``(n_molecules, n_simulations, ...)``.
        """

        # Check that no extra arguments are provided when agg_func is not used
        if agg_func is None and agg_kwargs:
            unknown_args = ", ".join(agg_kwargs.keys())
            raise TypeError(
                "extract() got unexpected keyword arguments: "
                f"{unknown_args}; these can only be used when an "
                "'agg_func' is provided"
            )

        chroms = utils.normalize_chroms(chroms, default_chroms=self._chroms)

        results = {}
        for chrom in chroms:
            results[chrom] = [
                [None for _ in range(self._nsim)]
                for _ in range(self._nmol[chrom])
            ]

        # Get the expected number of simulations per molecule
        finished_counts = Counter()
        expected_counts = Counter()
        for sim_id in self.iter_runs(chroms, mols):
            chrom, mol, _ = sim_id
            expected_counts[(chrom, mol)] += 1

        def store(chrom, mol, run, data):
            # Aggregate the runs on disk once all of them are extracted
            results[chrom][mol][run] = data
            key = (chrom, mol)
            finished_counts[key] += 1
            if agg_func and finished_counts[key] == expected_counts[key]:
                raw_data = [x for x in results[chrom][mol] if x is not None]
                if obs != "position":
                    raw_data = np.asarray(raw_data)
                results[chrom][mol] = agg_func(
                    chrom, mol, raw_data, **agg_kwargs
                )

        def extract_one(sim_id):
            sim_file = self._file_map.params_to_mcmc_file(*sim_id)
            return SimMCMCData.extract(sim_file, time, obs)

        extracted = _threaded_map(
            extract_one,
            self.iter_runs(chroms, mols),
            nworker=nworker,
            batch_size=batch_size,
        )
        for (chrom, mol, run), data in extracted:
            store(chrom, mol, run, data)

        # Mark molecules with no simulation files on disk as missing
        if agg_func is not None:
            for chrom in chroms:
                mol_range = range(self._nmol[chrom]) if mols is None else mols
                for mol in mol_range:
                    if expected_counts[(chrom, mol)] == 0:
                        results[chrom][mol] = None

        # Tidy up results and return a numpy array for each chromosome data
        if agg_func is None and obs != "position":
            for chrom in chroms:
                results[chrom] = np.asarray(results[chrom])
        return results

    def _classify_run(self, chrom: str, mol: int, run: int) -> str | None:
        """
        Classify the on-disk status of a single simulation run.

        Parameters
        ----------
        chrom : str
            The chromosome identifier.
        mol : int
            The molecule index.
        run : int
            The simulation run index.

        Returns
        -------
        str or None
            One of 'missing' (no file on disk, or -- for `method='tm'`
            -- fewer samples stored than `run` requires), 'corrupted'
            (the file exists but cannot be read), 'truncated' (MCMC
            only: the file is readable but its trajectory is shorter
            than `nsweep // print_freq + 1` expects), or None if the
            run is complete.
        """
        if self._method == "tm":
            combined_file = self._file_map.params_to_tm_file(chrom, mol)
            if not combined_file.exists():
                return "missing"
            try:
                with MAX_OPEN_FILES, h5py.File(combined_file, "r") as f:
                    n_samples = len(f["data"]["energy"])
            except Exception:  # noqa: BLE001
                return "corrupted"
            if run >= n_samples:
                return "missing"
            return None
        sim_file = self._file_map.params_to_mcmc_file(chrom, mol, run)
        if not sim_file.exists():
            return "missing"
        try:
            with MAX_OPEN_FILES:
                data = SimMCMCData.load(sim_file)
        except Exception:  # noqa: BLE001
            return "corrupted"
        expected = self._settings.nsweep // self._settings.print_freq + 1
        if len(data.time) < expected:
            return "truncated"
        return None

    def find_incomplete_runs(
        self,
        *,
        chroms: str | Iterable[str] | None = None,
        nworker: int = 1,
        batch_size: int = 1000,
    ) -> dict[str, list[tuple[str, int, int]]]:
        """
        Find simulation runs that are missing, corrupted, or truncated.

        Scans every `(chrom, mol, run)` combination expected for this
        dataset -- not just ones with an existing raw file -- and checks
        whether each one completed successfully.

        Parameters
        ----------
        chroms : str or iterable of str, optional
            Specific chromosome(s) to check. If None (default), all
            chromosomes in the dataset are included.
        nworker : int, default 1
            The maximum number of threads to use for parallel scanning.
        batch_size : int, default 1000
            The number of simulation files to queue in a single processing
            batch to manage memory and thread overhead.

        Returns
        -------
        dict
            A dictionary with keys 'missing', 'corrupted', and 'truncated',
            each mapping to a list of `(chrom, mol, run)` triplets. Runs
            that completed successfully are not included.

        Notes
        -----
        This opens every expected file that exists on disk to verify it,
        so cost scales with the size of the dataset. Use `nworker > 1` to
        speed up scanning of large datasets.
        """
        chroms = utils.normalize_chroms(chroms, default_chroms=self._chroms)

        def all_triplets():
            for chrom in chroms:
                for mol in range(self._nmol[chrom]):
                    for run in range(self._nsim):
                        yield (chrom, mol, run)

        incomplete = {"missing": [], "corrupted": [], "truncated": []}
        statuses = _threaded_map(
            lambda sim_id: self._classify_run(*sim_id),
            all_triplets(),
            nworker=nworker,
            batch_size=batch_size,
        )
        for sim_id, status in statuses:
            if status is not None:
                incomplete[status].append(sim_id)
        return incomplete

    def mcmc_file(self, chrom: str, mol: int, run: int) -> Path:
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
        return self._file_map.params_to_mcmc_file(chrom, mol, run)

    def tm_file(self, chrom: str, mol: int) -> Path:
        """
        Retrieve the file directory for a molecule's combined
        transfer-matrix samples.

        Parameters
        ----------
        chrom : str
            The identifier of the simulated chromosome.
        mol : int
            The index of the simulated molecule.

        Returns
        -------
        Path
            The directory of the combined HDF5 file holding all of this
            molecule's exact samples, regardless of whether the file
            actually exists on disk.
        """
        return self._file_map.params_to_tm_file(chrom, mol)

    @property
    def method(self) -> str:
        """
        Which engine generated (or will generate) this dataset's raw
        samples.

        Returns
        -------
        str
            Either 'mcmc' or 'tm'.
        """
        return self._method

    @property
    def settings(self) -> Mapping[str, Any]:
        """
        A read-only, live view of the simulation settings and parameters.

        Provide zero-copy access to the underlying simulation settings object.
        To access a specific parameter, say the DNA linker length `llink`, one
        can do the following:

        >>> dataset = SimDataset.load("results/dataset.h5")
        >>> dataset.settings['llink']

        This is an integer, or a mapping of chromosome to value if `llink`
        was set per chromosome.

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
    def nmol(self) -> Mapping[str, int]:
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
        The number of simulation runs per molecule (`method='mcmc'`) or
        exact samples per molecule (`method='tm'`; may be 0).

        Returns
        -------
        int
            The count of independent simulation trajectories, or exact
            samples, for each molecule-chromosome pair.
        """
        return self._nsim

    @property
    def nbp(self) -> Mapping[str, int]:
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
    def has_eseq(self) -> bool:
        """
        Whether the dataset stores a per-molecule energy landscape.

        Returns
        -------
        bool
            True if the dataset was created with `store_eseq=True`,
            False otherwise (in which case `eseq` holds NaN
            placeholders, not real energy landscapes).
        """
        return self._eseq is not None

    @property
    def out_type(self) -> Any | None:
        """
        The format or scope of data recorded for each simulation run.

        Returns
        -------
        Dump.OutputType or None
            The output type used when the dataset's simulations were run,
            or None if the dataset predates this being persisted.
        """
        return self._out_type

    @property
    def seed(self) -> int | None:
        """
        The batch-level master seed the dataset's simulations were
        generated from.

        Returns
        -------
        int or None
            The master seed, or None if the dataset predates this being
            persisted. See `seed_table` for the actual per-run seed used
            by each simulation.
        """
        return self._seed

    @property
    def seed_table(self) -> Mapping[str, np.ndarray] | None:
        """
        The per-`(chrom, mol, run)` seed used for each simulation.

        Returns
        -------
        types.MappingProxyType
            A read-only mapping of chromosome names to an array of shape
            `(nmol[chrom], nsim)` holding the seed used for each molecule
            and run, or None if the dataset predates this being
            persisted. Enables exact reproduction of any individual run,
            even ones whose raw output file is missing or corrupted.
        """
        if self._seed_table is None:
            return None
        return MappingProxyType(self._seed_table)

    @property
    def raw(self) -> SimDataAccessor:
        """
        Provide lazy access to the raw simulation data files.

        This property returns an accessor object that supports tuple-based
        indexing to load individual :class:`SimMCMCData` (`method='mcmc'`)
        or :class:`SimTMData` (`method='tm'`) objects on demand.

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
        public_attrs = [
            attr for attr in self.__slots__ if not attr.startswith("_")
        ]
        # Add public properties explicitly (we check for them in __dict__)
        for attr, value in self.__class__.__dict__.items():
            if isinstance(value, property) and not attr.startswith("_"):
                public_attrs.append(attr)
        # Create a string of the public attribute names (not values)
        repr_str = f"{self.__class__.__name__}({', '.join(public_attrs)})"
        return repr_str
