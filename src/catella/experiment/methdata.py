# methdata.py

import os
import shutil
import tempfile
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Self

import h5py
import numpy as np
import pandas as pd

from catella import h5_utils, utils
from catella.containers import DataFrameMap, FixedKeyMap
from catella.experiment._dropout import (
    _apply_keep_mask,
    _dropout_fracs,
    _label_coverage,
    _resolve_sources,
)
from catella.experiment._ingest import (
    _ingest_into,
    _ingest_sources,
    _load_refseqs,
    _merge_staging,
    _methylatable_positions,
    _normalize_mtase,
    _read_chromsize,
)
from catella.h5_array import H5Array

# Molecule-count threshold above which MethPrintExperiment.to_dense()
# warns when materializing a pd.DataFrame (as opposed to streaming to
# an H5Array).
_DENSE_WARN_ROWS = 5000


@utils.add_frozen_properties
@dataclass(frozen=True, slots=True, init=False, eq=False)
class MethPrintData:
    """
    Container for MethPrint experimental data for a single chromosome.

    This class stores methylation signal data for test samples and optional
    control samples (unmethylated and fully methylated). Data are stored
    as Pandas DataFrames and retrieved via read-only properties that
    provide defensive copies. An optional reference nucleotide sequence
    for the chromosome may also be stored, via `refseq`.

    .. note::
       This class is intended for internal use within a
       :class:`MethPrintExperiment`. Use the experiment's loading
       mechanisms rather than instantiating this class directly.
    """

    chrom: str
    """The chromosome identifier for this data block."""

    nbp: int
    """The total number of base pairs in the chromatin fiber."""

    _frozen_refseq: str | None = field(
        metadata={
            "doc": "str: The reference nucleotide sequence for this "
            "chromosome, or None if not provided."
        }
    )

    _frozen_test_mol_id: np.ndarray = field(
        metadata={
            "doc": "np.ndarray: Array of molecule identifiers for the "
            "test samples."
        }
    )

    _frozen_test_data: pd.DataFrame = field(
        metadata={
            "doc": "pd.DataFrame: Raw methylation quality scores for "
            "test samples."
        }
    )

    _frozen_unmeth_mol_id: np.ndarray | None = field(
        metadata={
            "doc": "np.ndarray: Array of molecule identifiers for "
            "unmethylated controls."
        }
    )

    _frozen_unmeth_data: pd.DataFrame | None = field(
        metadata={
            "doc": "pd.DataFrame: Raw methylation quality scores for "
            "unmethylated controls."
        }
    )

    _frozen_meth_mol_id: np.ndarray | None = field(
        metadata={
            "doc": "np.ndarray: Array of molecule identifiers for "
            "methylated controls."
        }
    )

    _frozen_meth_data: pd.DataFrame | None = field(
        metadata={
            "doc": "pd.DataFrame: Raw methylation quality scores for "
            "methylated controls."
        }
    )

    def __init__(self, **kwargs: Any):
        if not kwargs.pop("_internal", False):
            raise TypeError(
                "use MethPrintData._load() to instantiate this class"
            )
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is None:
                val = kwargs.get(f.name.removeprefix("_frozen_"))
            object.__setattr__(self, f.name, val)

    def _save(self, group):
        dt = h5py.string_dtype(encoding="utf-8")  # For storing strings
        gchrom = group.create_group(self.chrom)
        gmeta = gchrom.create_group("metadata")
        gmeta.attrs["chrom"] = self.chrom
        gmeta.attrs["nbp"] = self.nbp
        if self._frozen_refseq is not None:
            gmeta.create_dataset("refseq", data=self._frozen_refseq, dtype=dt)
        gdata = gchrom.create_group("data")

        def save_data(name, mol_id, df, gdata):
            if mol_id is not None and df is not None:
                gdata.create_dataset(name + "_mol_id", data=mol_id, dtype=dt)
                h5_utils.save_df(name + "_data", df, gdata)

        save_data(
            "test", self._frozen_test_mol_id, self._frozen_test_data, gdata
        )
        save_data(
            "unmeth",
            self._frozen_unmeth_mol_id,
            self._frozen_unmeth_data,
            gdata,
        )
        save_data(
            "meth", self._frozen_meth_mol_id, self._frozen_meth_data, gdata
        )

    @classmethod
    def _load(cls, group, chrom) -> Self:
        if chrom not in group:
            raise ValueError(f"cannot find data for the chrom {chrom}")
        gchrom = group[chrom]
        gmeta = gchrom["metadata"]
        chrom = gmeta.attrs["chrom"]
        nbp = int(gmeta.attrs["nbp"])
        refseq = gmeta["refseq"].asstr()[()] if "refseq" in gmeta else None

        def load_data(name, gdata):
            id_name = name + "_mol_id"
            data_name = name + "_data"
            if id_name in gdata and data_name in gdata:
                mol_id = gdata[id_name].asstr()[()]
                df = h5_utils.load_df(data_name, gdata)
                return mol_id, df
            return None, None

        gdata = gchrom["data"]
        test_mol_id, test_data = load_data("test", gdata)
        unmeth_mol_id, unmeth_data = load_data("unmeth", gdata)
        meth_mol_id, meth_data = load_data("meth", gdata)
        return cls._create(
            chrom=chrom,
            nbp=nbp,
            refseq=refseq,
            test_mol_id=test_mol_id,
            test_data=test_data,
            unmeth_mol_id=unmeth_mol_id,
            unmeth_data=unmeth_data,
            meth_mol_id=meth_mol_id,
            meth_data=meth_data,
        )

    @classmethod
    def _create(cls, **kwargs) -> Self:
        # Some validation
        if kwargs.get("nbp") <= 0:
            raise ValueError("nbp must be positive")
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


class LazyRawDataMap(ABCMapping):
    """
    Load per-chromosome MethPrintData from an HDF5 file on demand.

    Chromosome data is read from disk the first time it is accessed and
    kept in a small least-recently-used cache. Accessing a different
    chromosome once the cache is full evicts the least-recently-used
    entry, bounding how many chromosomes' raw data can be resident in
    memory at once, while repeated access to the same chromosome avoids
    re-reading it from disk.
    """

    def __init__(self, path, chroms, *, max_cached=1):
        self._path = str(path)
        self._chroms = tuple(chroms)
        self._max_cached = max_cached
        self._cache = OrderedDict()

    def __getitem__(self, chrom):
        if chrom not in self._chroms:
            raise KeyError(chrom)
        if chrom in self._cache:
            self._cache.move_to_end(chrom)
            return self._cache[chrom]
        with h5py.File(self._path, "r") as h5stream:
            data = MethPrintData._load(h5stream["raw_data"], chrom)
        self._cache[chrom] = data
        if len(self._cache) > self._max_cached:
            self._cache.popitem(last=False)
        return data

    def __iter__(self):
        return iter(self._chroms)

    def __len__(self):
        return len(self._chroms)


# Valid values for MethPrintExperiment.filter_dropout()'s
# 'unmapped_strand' and 'method' arguments.
_VALID_UNMAPPED_STRAND = {"union", "drop", "+", "-"}
_VALID_FILTER_DROPOUT_METHOD = {"separate", "aggregate"}


@dataclass(slots=True, init=False, weakref_slot=True)
class MethPrintExperiment:
    """
    Manager for MethPrint experimental data and analysis results.

    This class provides tools to load raw methylation signals from sequencing
    files, manage control datasets (unmethylated and methylated, which are
    optional), and store the results of normalization and downstream analyses.

    .. note::
       Direct instantiation is disabled. Use :meth:`load_raw` to process new
       sequencing data or :meth:`load` to open an existing HDF5 dataset.
    """

    _raw_data: Mapping[str, MethPrintData]
    _analysis: Mapping[str, DataFrameMap]
    _global_analysis: DataFrameMap
    _finalizer: Any | None
    _tmp_dir: str | None
    _exp_file: str | None
    _mtase: tuple[str, ...] | None
    _wrap: bool
    _ignore_strand: bool

    def __init__(self, **kwargs: Any):
        if not kwargs.pop("_internal", False):
            raise TypeError(
                "use MethPrintExperiment.load() or .load_raw() "
                "to instantiate this class"
            )

        # Bulk assignment of fields
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is not None:
                setattr(self, f.name, val)

        # Create the dicts for analysis
        self._analysis = {chrom: DataFrameMap() for chrom in self._raw_data}
        self._global_analysis = DataFrameMap()

        self._finalizer = kwargs.get("_finalizer")

        # Scratch directory shared by this experiment's own staging
        # file and any later smooth()/meth_prob() calls that don't
        # specify their own tmp_dir (set by load_raw() when it
        # generates one, or lazily by resolve_tmp_dir()).
        self._tmp_dir = kwargs.get("_tmp_dir")

        # File this experiment was loaded from, or last saved to (None
        # if never saved); used as save()'s default destination.
        self._exp_file = kwargs.get("_exp_file")

        # Methyltransferase(s) used, or None; set at load_raw() time.
        self._mtase = kwargs.get("_mtase")

        # Whether raw positions were folded around the fiber center;
        # set at load_raw() time.
        self._wrap = kwargs.get("_wrap", False)

        # Whether strand was ignored when resolving methylatable
        # positions (both strands' motif context OR'd together); set
        # at load_raw() time.
        self._ignore_strand = kwargs.get("_ignore_strand", False)

    @staticmethod
    def _cleanup_tmp_dir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def resolve_tmp_dir(self):
        """
        Return this experiment's scratch directory for temporary HDF5
        files, creating and caching one under the system default
        temporary directory the first time it's needed. The directory
        is removed once the experiment is closed or garbage-collected;
        see `close`.

        Returns
        -------
        str
            Path to the resolved scratch directory.
        """
        if self._tmp_dir is None:
            self._tmp_dir = h5_utils.fresh_tmp_dir()
            self._finalizer = weakref.finalize(
                self, MethPrintExperiment._cleanup_tmp_dir, self._tmp_dir
            )
        return self._tmp_dir

    def close(self):
        """
        Delete this experiment's scratch directory (all temporary HDF5
        files created for it, e.g. by `load_raw`, `smooth`, or
        `meth_prob`), if one was created. No-op for experiments with no
        scratch directory (e.g. from `.load()` with no later
        `smooth`/`meth_prob` calls).
        """
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    @classmethod
    def load_raw(
        cls,
        chromsize: str | Path,
        test_file: str | Path,
        unmeth_file: str | Path | None = None,
        meth_file: str | Path | None = None,
        fasta_file: str | Path | None = None,
        mtase: str | list[str] | None = None,
        chroms: list[str] | None = None,
        wrap: bool = False,
        ignore_strand: bool = False,
        colidx: list | None = None,
        max_nmol: int | None = None,
        seed: int | None = None,
        chunk_size: int = 1000000,
        tmp_dir: str | Path | None = None,
        max_cached_chroms: int = 1,
        nworker: int = 1,
    ) -> Self:
        """
        Create an experiment by processing raw sequencing data files.

        Reads chromosome sizes and experimental data (typically modkit
        output), drops canonical calls (`mod_code` of '-'), re-orients
        data if requested, and splits signals by chromosome into a
        staging HDF5 file that the returned experiment lazily loads
        (see `load`). If more than one of `test_file`/`unmeth_file`/
        `meth_file` is given, each is read independently (see
        `nworker`) and merged afterward.

        Parameters
        ----------
        chromsize : str or pathlib.Path
            Tab-separated file of chromosome names and lengths (bp).
        test_file : str or pathlib.Path
            Raw experimental (test) data file.
        unmeth_file : str or pathlib.Path, optional
            Unmethylated control data file.
        meth_file : str or pathlib.Path, optional
            Fully methylated control data file.
        fasta_file : str or pathlib.Path, optional
            Multi-FASTA file of per-chromosome reference sequences
            (record id matching `chromsize`); stored on
            `MethPrintData.refseq`.
        mtase : str or list of str, optional
            Methyltransferase(s) used to generate the test data: 'A'
            (any-context adenine, e.g., EcoGII), 'CG' (CpG, e.g.,
            M.SssI), 'GC' (GpC, e.g., M.CviPI). One label, a list of
            labels, or None (default) if unspecified.
        chroms : list of str, optional
            Restrict processing to these chromosomes. If None
            (default), all chromosomes in `chromsize` are processed.
        wrap : bool, default False
            If True, calculates positions relative to the fiber center
            (useful for circular or symmetrical fibers).
        ignore_strand : bool, default False
            If True, ignore each row's recorded `strand` for `mtase`
            context filtering, and in `filter_dropout`,
            `summarize_dropout`, `dropout_fractions`. Independent of
            `norm_by_strand`.
        colidx : list of int, optional
            Explicit column indices if the input file has no header
            matching Modkit's column names, in canonical field order:
            `read_id`, `ref_position`, `chrom`, `ref_strand`,
            `mod_qual`, `mod_code`.
        max_nmol : int or None
            Maximum number of molecules to extract for each chromosome.
        seed : int or None
            Random number generator seed for downsampling when
            `max_nmol` is given; independent per data file when more
            than one is given.
        chunk_size : int, default 1000000
            Approximate number of rows read (and held in memory) per
            streamed chunk.
        tmp_dir : str or Path, optional
            Directory for the scratch staging file backing the
            returned experiment's raw data (removed once the
            experiment is closed or garbage-collected; see `close`).
            A fresh `catella_<timestamp>_<hex>` subfolder is created
            under it (or under the system default temporary directory)
            and reused by later `smooth`/`meth_prob` calls.
        max_cached_chroms : int, default 1
            Maximum number of chromosomes' raw data kept in memory at
            once by the returned experiment (forwarded to `load`).
        nworker : int, default 1
            Maximum number of threads used to read test/unmeth/meth
            files concurrently. Only relevant when more than one is
            given; never changes the resulting data.

        Returns
        -------
        MethPrintExperiment
            A newly initialized experiment object containing the processed
            data.

        Raises
        ------
        ValueError
            If chromosome names are not unique, if control datasets
            contain chromosomes not found in the test dataset, if
            `chroms` contains a chromosome not found in `chromsize`,
            if a sequence in `fasta_file` does not match its
            chromosome's length in `chromsize`, or if `mtase`
            contains an unknown or duplicate label.
        """
        mtase = _normalize_mtase(mtase)
        full_sizes, sizes = _read_chromsize(chromsize, wrap, chroms)
        refseq_by_chrom = _load_refseqs(fasta_file, full_sizes)

        tmp_dir = h5_utils.fresh_tmp_dir(base_dir=tmp_dir)
        fd, tmp_file = tempfile.mkstemp(suffix=".h5", dir=tmp_dir)
        os.close(fd)

        files = [("test", test_file)]
        if unmeth_file is not None:
            files.append(("unmeth", unmeth_file))
        if meth_file is not None:
            files.append(("meth", meth_file))
        ingest_kwargs = {
            "sep": "\t",
            "colidx": colidx,
            "chunk_size": chunk_size,
            "block_size": 64 * 1024 * 1024,
            "sizes": sizes,
            "full_sizes": full_sizes,
            "wrap": wrap,
            "refseq_by_chrom": refseq_by_chrom,
            "mtase": mtase,
            "ignore_strand": ignore_strand,
            "max_nmol": max_nmol,
            "tmp_dir": tmp_dir,
        }

        try:
            with h5py.File(tmp_file, "w") as h5stream:
                graw = h5stream.create_group("raw_data")
                h5stream.create_group("analysis")
                h5stream.create_group("global_analysis")
                if mtase is not None:
                    h5stream.attrs["mtase"] = ",".join(mtase)
                h5stream.attrs["wrap"] = bool(wrap)
                h5stream.attrs["ignore_strand"] = bool(ignore_strand)

                if len(files) == 1:
                    rng = (
                        None
                        if max_nmol is None
                        else np.random.default_rng(seed)
                    )
                    _ingest_into(graw, *files[0], rng=rng, **ingest_kwargs)
                else:
                    ingested = _ingest_sources(
                        files, nworker=nworker, seed=seed, **ingest_kwargs
                    )
                    _merge_staging(graw, ingested)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise

        exp = cls.load(tmp_file, max_cached_chroms=max_cached_chroms)
        # tmp_file is an internal staging file, not a real save target
        exp._exp_file = None
        exp._tmp_dir = tmp_dir
        exp._finalizer = weakref.finalize(
            exp, MethPrintExperiment._cleanup_tmp_dir, tmp_dir
        )
        return exp

    def save(
        self, exp_file: str | Path | None = None, *, overwrite: bool = False
    ):
        """
        Save the experiment data and analysis to an HDF5 file.

        Parameters
        ----------
        exp_file : str or pathlib.Path, optional
            The output file path. Raw data is skipped only if this is
            the file the experiment is already backed by. If None,
            saves to the file this experiment was loaded from or last
            saved to.
        overwrite : bool, default False
            If True, allow saving to the same file that backs an
            existing `H5Array` analysis entry, by writing to a temporary
            sibling file and atomically renaming it into place.

        Raises
        ------
        ValueError
            If `exp_file` is None and this experiment has no file to
            default to (e.g. fresh from `load_raw()` and never saved),
            or if `exp_file` is the same file that backs an `H5Array`
            analysis entry already held by this experiment and
            `overwrite` is False.
        OSError
            If the file cannot be written to disk.
        """
        if exp_file is None:
            exp_file = self._exp_file
            if exp_file is None:
                raise ValueError(
                    "no exp_file given and this experiment has not "
                    "been saved before; pass 'exp_file' explicitly"
                )

        dest_path = Path(exp_file)
        dest = str(dest_path.resolve())
        collision = any(
            isinstance(entry, H5Array)
            and str(Path(entry.path).resolve()) == dest
            for data in list(self._analysis.values()) + [self._global_analysis]
            for entry in data.values()
        )
        if collision:
            if not overwrite:
                raise ValueError(
                    "cannot save to the same file that backs an "
                    "existing H5Array analysis entry; save to a "
                    "different path, or pass overwrite=True to safely "
                    "replace it in place"
                )
            colliding = [
                (data, name, entry.dataset_path)
                for data in list(self._analysis.values())
                + [self._global_analysis]
                for name, entry in data.items()
                if isinstance(entry, H5Array)
                and str(Path(entry.path).resolve()) == dest
            ]
            tmp_path = dest_path.with_name(
                dest_path.name + f".tmp{os.getpid()}"
            )
            closed = False
            try:
                self.save(tmp_path)
                # Release the OS-level handles on dest before
                # replacing it: unlike POSIX, Windows refuses to
                # rename over a file that is still open.
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
            self._exp_file = dest
            return

        with h5py.File(exp_file, "a") as h5stream:
            # Skip only if re-saving to the same file we're already
            # backed by; other destinations always get a fresh copy.
            same_file = (
                self._exp_file is not None
                and str(Path(self._exp_file).resolve()) == dest
            )
            if not same_file or "raw_data" not in h5stream:
                if "raw_data" in h5stream:
                    del h5stream["raw_data"]
                graw = h5stream.create_group("raw_data")
                for chrom, rdata in self._raw_data.items():
                    rdata._save(graw)
                if self._mtase is not None:
                    h5stream.attrs["mtase"] = ",".join(self._mtase)
                h5stream.attrs["wrap"] = self._wrap
                h5stream.attrs["ignore_strand"] = self._ignore_strand

            # Save any analysis data
            if "analysis" in h5stream:
                del h5stream["analysis"]
            gana = h5stream.create_group("analysis")
            for chrom, data in self._analysis.items():
                gchrom = gana.create_group(chrom)
                for name, entry in data.items():
                    if isinstance(entry, H5Array):
                        entry.save_to(gchrom, name)
                    else:
                        h5_utils.save_df(name, entry, gchrom)
            if "global_analysis" in h5stream:
                del h5stream["global_analysis"]
            gana = h5stream.create_group("global_analysis")
            for name, entry in self._global_analysis.items():
                if isinstance(entry, H5Array):
                    entry.save_to(gana, name)
                else:
                    h5_utils.save_df(name, entry, gana)
        self._exp_file = dest

    @classmethod
    def load(
        cls,
        exp_file: str | Path,
        chroms: list[str] | None = None,
        max_cached_chroms: int = 1,
    ) -> Self:
        """
        Load an experiment from a persistent HDF5 file.

        Raw per-chromosome data is loaded lazily: `MethPrintData` for a
        chromosome is only read from disk when accessed via `raw`, and
        is cached for at most `max_cached_chroms` chromosomes at a time
        (least-recently-used eviction), so not all chromosomes need to
        be resident in memory simultaneously.

        Parameters
        ----------
        exp_file : str or pathlib.Path
            HDF5 file containing the experiment.
        chroms : list of str, optional
            Restrict the loaded experiment to these chromosomes. If
            None (default), all chromosomes in the file are exposed.
        max_cached_chroms : int, default 1
            Maximum number of chromosomes' raw data kept in memory at
            once. Accessing more distinct chromosomes than this evicts
            the least-recently-used one.

        Returns
        -------
        MethPrintExperiment
            The loaded experiment object with all data and analysis maps.

        Raises
        ------
        ValueError
            If `chroms` contains a chromosome not found in the file.
        """
        with h5py.File(exp_file, "r") as h5stream:
            # Set up lazy, memory-bounded access to the raw data
            available_chroms = list(h5stream["raw_data"].keys())
            if chroms is not None:
                unknown = set(chroms) - set(available_chroms)
                if unknown:
                    raise ValueError(
                        f"chromosomes not found in {exp_file}: "
                        f"{sorted(unknown)}"
                    )
                selected = list(chroms)
            else:
                selected = available_chroms
            raw_data = LazyRawDataMap(
                exp_file, selected, max_cached=max_cached_chroms
            )
            mtase = (
                tuple(h5stream.attrs["mtase"].split(","))
                if "mtase" in h5stream.attrs
                else None
            )
            wrap = bool(h5stream.attrs.get("wrap", False))
            ignore_strand = bool(h5stream.attrs.get("ignore_strand", False))
            obj = cls._create(
                _raw_data=raw_data,
                _exp_file=str(Path(exp_file).resolve()),
                _mtase=mtase,
                _wrap=wrap,
                _ignore_strand=ignore_strand,
            )

            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                if chrom not in selected:
                    continue
                gchrom = gana[chrom]
                for name in gchrom:
                    if isinstance(gchrom[name], h5py.Dataset):
                        obj._analysis[chrom][name] = H5Array.load_from(
                            exp_file, f"analysis/{chrom}/{name}"
                        )
                    else:
                        obj._analysis[chrom][name] = h5_utils.load_df(
                            name, gchrom
                        )
            gana = h5stream["global_analysis"]
            for name in gana:
                if isinstance(gana[name], h5py.Dataset):
                    obj._global_analysis[name] = H5Array.load_from(
                        exp_file, f"global_analysis/{name}"
                    )
                else:
                    obj._global_analysis[name] = h5_utils.load_df(name, gana)
            return obj

    @classmethod
    def _create(cls, **kwargs) -> Self:
        return cls(**kwargs, _internal=True)

    @property
    def chroms(self) -> tuple[str, ...]:
        """
        Get the identifiers of all chromosomes in the experiment.

        Returns
        -------
        tuple of str
            A sorted tuple of chromosome names.
        """
        return tuple(self._raw_data.keys())

    @property
    def mtase(self) -> tuple[str, ...] | None:
        """
        Get the methyltransferase(s) used for this experiment.

        Returns
        -------
        tuple of str or None
            One or more of 'A', 'CG', 'GC', or None if unspecified.
        """
        return self._mtase

    @property
    def wrap(self) -> bool:
        """
        Get whether raw positions were folded around the fiber center.

        Returns
        -------
        bool
            True if `load_raw` was called with `wrap=True`.
        """
        return self._wrap

    @property
    def ignore_strand(self) -> bool:
        """
        Get whether strand was ignored for methylatable positions.

        Returns
        -------
        bool
            True if `load_raw` was called with `ignore_strand=True`.
        """
        return self._ignore_strand

    @property
    def raw(self) -> Mapping[str, MethPrintData]:
        """
        Provide read-only access to the raw experimental data.

        Returns
        -------
        MappingProxyType
            A frozen mapping where keys are chromosome names and values
            are :class:`MethPrintData` objects.
        """
        return MappingProxyType(self._raw_data)

    # Allow iterating over the raw data by chromosome
    def __iter__(self):
        return iter(self._raw_data)

    def __len__(self):
        return len(self._raw_data)

    @property
    def analysis(self) -> Mapping[str, DataFrameMap]:
        """
        Access the analysis results for each chromosome.

        Returns
        -------
        FixedKeyMap
            A mapping where chromosome keys are fixed, but the analysis
            DataFrames remain mutable.
        """
        return FixedKeyMap(self._analysis)

    @property
    def global_analysis(self) -> DataFrameMap:
        """
        Access analysis results aggregated across the entire experiment.

        Returns
        -------
        DataFrameMap
            A mutable map containing global analysis results.
        """
        return self._global_analysis

    def to_dense(
        self,
        chrom,
        which="test",
        *,
        mols=None,
        as_h5array=True,
        dtype=np.float64,
        batch_size=20000,
        mask_name=None,
    ):
        """
        Convert a raw long-format methylation table to a dense
        representation.

        Parameters
        ----------
        chrom : str
            Chromosome identifier.
        which : {"test", "meth", "unmeth"}, default "test"
            Which raw table to convert.
        mols : int or sequence of int, optional
            Molecule index/indices (values of `mol_index`) to include.
            If None (default), every molecule is included.
        as_h5array : bool, default True
            Whether to stream the result to a disk-backed `H5Array`
            (True) or materialize it as an in-memory `pd.DataFrame`
            (False). Independent of `mols` -- e.g. `mols=[3, 7]` with
            `as_h5array=True` (the default) returns a 2-row `H5Array`.
        dtype : data-type, default np.float64
            Numeric dtype of the returned dense data. Must be a
            floating dtype, since missing (mol_index, pos) combinations
            are represented as NaN.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per
            batch. Only used when `as_h5array` is True.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `filter_dropout(mask_name=mask_name)` call for this
            source are returned as all-NaN rows. Looked up as
            `analysis[chrom][f"{which}_{mask_name}"]`.

        Returns
        -------
        H5Array or pd.DataFrame
            `H5Array` if `as_h5array` is True; otherwise a
            `pd.DataFrame`, indexed by the selected `mol_index` values.

        Raises
        ------
        ValueError
            If no data of the requested kind exists for `chrom`, or if
            `dtype` is not a floating dtype.

        Warns
        -----
        UserWarning
            If `as_h5array` is False and more than `_DENSE_WARN_ROWS`
            molecules would be materialized in memory.

        Notes
        -----
        Scratch files backing a resulting `H5Array` are written to this
        experiment's scratch directory (see `resolve_tmp_dir`), shared
        with any other scratch files from the same experiment (e.g.
        from `load_raw`, `smooth`, or `meth_prob`).
        """
        if not np.issubdtype(np.dtype(dtype), np.floating):
            raise ValueError(
                "'dtype' must be a floating dtype to "
                f"represent missing values as NaN, got "
                f"{dtype}"
            )

        raw = self.raw[chrom]
        df = getattr(raw, f"{which}_data")
        mol_id = getattr(raw, f"{which}_mol_id")
        if df is None or mol_id is None:
            raise ValueError(
                f"no '{which}' data available for chrom '{chrom}'"
            )
        nbp = raw.nbp
        all_pos = pd.Index(range(nbp))

        keep = None
        if mask_name is not None:
            keep = self._analysis[chrom][f"{which}_{mask_name}"][
                "keep"
            ].to_numpy()

        mol_ids = (
            np.arange(len(mol_id)) if mols is None else np.atleast_1d(mols)
        )
        if mols is not None:
            df = df[df["mol_index"].isin(mol_ids)]

        if not as_h5array:
            if len(mol_ids) > _DENSE_WARN_ROWS:
                warnings.warn(
                    f"materializing {len(mol_ids)} molecules as a "
                    f"pd.DataFrame ({len(mol_ids)}x{nbp}); this may "
                    "use significant memory; pass as_h5array=True to "
                    "stream to disk instead",
                    stacklevel=2,
                )
            piv = df.pivot(
                index="mol_index", columns="pos", values="mod_qual"
            ).reindex(index=mol_ids, columns=all_pos)
            piv = piv.astype(dtype)
            if keep is not None:
                piv = _apply_keep_mask(piv, keep[mol_ids])
            return piv

        df = df.sort_values("mol_index", kind="stable")
        mol_index = df["mol_index"].to_numpy()
        nmol_out = len(mol_ids)
        out = H5Array.create(
            (nmol_out, nbp),
            dtype=dtype,
            index=(mol_ids if mols is not None else None),
            dir=self.resolve_tmp_dir(),
        )
        for start in range(0, nmol_out, batch_size):
            stop = min(start + batch_size, nmol_out)
            chunk_ids = mol_ids[start:stop]
            if mols is None:
                # mol_ids == arange(nmol_out): contiguous, sorted --
                # take the fast searchsorted range slice.
                lo, hi = np.searchsorted(mol_index, [start, stop])
                batch_df = df.iloc[lo:hi]
            else:
                # chunk_ids may be an arbitrary (unsorted,
                # non-contiguous) subset -- select by membership.
                batch_df = df[df["mol_index"].isin(chunk_ids)]
            piv = batch_df.pivot(
                index="mol_index", columns="pos", values="mod_qual"
            ).reindex(index=chunk_ids, columns=all_pos)
            arr = piv.to_numpy()
            if keep is not None:
                arr = _apply_keep_mask(arr, keep[chunk_ids])
            out.write_batch(start, stop, arr)
        return out

    def _resolve_mtase_subset(self, mtase):
        """
        Validate and resolve an `mtase` label subset.

        Parameters
        ----------
        mtase : list of str or None
            Subset of `self.mtase` to use. None resolves to all of
            `self.mtase`.

        Returns
        -------
        Tuple[str, ...]

        Raises
        ------
        ValueError
            If `self.mtase` is unset, or `mtase` contains a label not
            in `self.mtase`.
        """
        if self._mtase is None:
            raise ValueError(
                "this experiment has no 'mtase' set (specify it via "
                "load_raw()); required to determine methylatable "
                "positions"
            )
        if mtase is None:
            return self._mtase
        labels = _normalize_mtase(mtase)
        unknown = set(labels) - set(self._mtase)
        if unknown:
            raise ValueError(
                "'mtase' contains label(s) not in this experiment's "
                f"mtase {list(self._mtase)}: {sorted(unknown)}"
            )
        return labels

    def _resolve_chroms_subset(self, chroms):
        """
        Validate and resolve a `chroms` subset.

        Parameters
        ----------
        chroms : list of str or None
            Subset of `self.chroms` to use. None resolves to all of
            `self.chroms`.

        Returns
        -------
        Tuple[str, ...]

        Raises
        ------
        ValueError
            If `chroms` contains a chromosome not in `self.chroms`.
        """
        if chroms is None:
            return self.chroms
        unknown = set(chroms) - set(self.chroms)
        if unknown:
            raise ValueError(
                "'chroms' contains chromosome(s) not in this "
                f"experiment's chroms {list(self.chroms)}: "
                f"{sorted(unknown)}"
            )
        return tuple(chroms)

    def filter_dropout(
        self,
        *,
        which: str | None = None,
        mtase: list | None = None,
        chroms: list | None = None,
        thres_min: float = 0.0,
        thres_max: float = 1.0,
        unmapped_strand: str = "union",
        method: str = "separate",
        mask_name: str = "dropout_mask",
    ) -> None:
        """
        Flag molecules with poor coverage at methylatable positions.

        Computes, per molecule, the fraction of methylatable positions
        (per `mtase` label, on that molecule's own strand) with no
        signal, and stores a keep/drop mask in `analysis`. Raw data is
        never modified -- pass `mask_name` to `to_dense`/
        `MethPrintAnalysis.smooth`/`.meth_prob` to apply it downstream.
        If this experiment was loaded with `ignore_strand=True`,
        `unmapped_strand` governs coverage for every molecule.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to evaluate. None evaluates every source present
            for each chromosome.
        mtase : list of str, optional
            Subset of `self.mtase` labels to evaluate. None uses all
            of them.
        chroms : list of str, optional
            Subset of `self.chroms` to evaluate. None (default)
            evaluates every chromosome.
        thres_min : float, default 0.0
            Min allowed no-signal fraction (per label, or of the
            pooled total under `method="aggregate"`) to be kept.
        thres_max : float, default 1.0
            Max allowed no-signal fraction (per label, or of the
            pooled total under `method="aggregate"`) to be kept.
        unmapped_strand : {"union", "drop", "+", "-"}, default "union"
            How to evaluate unmapped ('.') strand molecules: union of
            '+'/'-' position sets, always dropout, or treat as that
            strand.
        method : {"separate", "aggregate"}, default "separate"
            How multiple `mtase` labels combine into `keep`.
            "separate": must clear `thres_min`/`thres_max` per label.
            "aggregate":
            site counts pooled across labels into one fraction first.
            Irrelevant for a single label.
        mask_name : str, default "dropout_mask"
            Key for the mask in `analysis[chrom]`, as
            `f"{source}_{mask_name}"` -- a single-column DataFrame
            with a boolean `keep` column, indexed by `mol_index`.

        Raises
        ------
        ValueError
            If `mtase` is unset on this experiment, `mtase` contains a
            label not in `self.mtase`, `chroms` contains a chromosome
            not in `self.chroms`, `thres_min` or `thres_max` is not
            in [0, 1], `thres_min` is greater than `thres_max`,
            `unmapped_strand`/`method` is invalid, a requested source
            is missing for some chromosome, or `refseq` is missing.
        """
        labels = self._resolve_mtase_subset(mtase)
        chroms = self._resolve_chroms_subset(chroms)
        if not (0.0 <= thres_min <= 1.0):
            raise ValueError(
                f"'thres_min' must be in [0, 1], got {thres_min}"
            )
        if not (0.0 <= thres_max <= 1.0):
            raise ValueError(
                f"'thres_max' must be in [0, 1], got {thres_max}"
            )
        if thres_min > thres_max:
            raise ValueError(
                "'thres_min' must be <= 'thres_max', got "
                f"thres_min={thres_min}, thres_max={thres_max}"
            )
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}"
            )
        if method not in _VALID_FILTER_DROPOUT_METHOD:
            raise ValueError(
                "'method' must be one of "
                f"{sorted(_VALID_FILTER_DROPOUT_METHOD)}, got "
                f"{method!r}"
            )

        for chrom in chroms:
            raw = self.raw[chrom]
            if raw.refseq is None:
                raise ValueError(
                    f"no refseq available for chrom '{chrom}'; "
                    "required to determine methylatable positions"
                )
            sources = _resolve_sources(raw, which)

            for src in sources:
                df = getattr(raw, f"{src}_data")
                mol_id = getattr(raw, f"{src}_mol_id")
                if df is None or mol_id is None:
                    raise ValueError(
                        f"no '{src}' data available for chrom '{chrom}'"
                    )
                cov = _label_coverage(
                    df,
                    mol_id,
                    raw.refseq,
                    labels,
                    unmapped_strand,
                    ignore_strand=self._ignore_strand,
                )
                dropout_frac, combined_frac = _dropout_fracs(cov, labels)

                if method == "separate":
                    keep = (
                        (dropout_frac >= thres_min)
                        & (dropout_frac <= thres_max)
                    ).all(axis=0)
                else:
                    keep = (combined_frac >= thres_min) & (
                        combined_frac <= thres_max
                    )

                # Stored as int, not bool: bool isn't a numeric dtype
                # to h5_utils' save/load, so it would round-trip
                # through the string block as literal "True"/"False"
                # text -- and casting that back to bool makes every
                # non-empty string truthy, silently breaking the mask.
                self._analysis[chrom][f"{src}_{mask_name}"] = pd.DataFrame(
                    {"keep": keep.astype(np.int8)}
                )

    def summarize_dropout(
        self,
        *,
        which: str | None = None,
        mtase: list | None = None,
        chroms: list | None = None,
        unmapped_strand: str = "union",
    ) -> pd.DataFrame:
        """
        Return a per-label dropout fraction summary (QC check).

        Same coverage computation as `filter_dropout`, reporting the
        per-molecule dropout fraction distribution instead of
        filtering. Nothing is stored in `analysis`. `n_sites_plus`/
        `n_sites_minus` always report reference-level per-strand
        counts, regardless of `unmapped_strand` or `ignore_strand`.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to summarize. None summarizes every source
            present for each chromosome.
        mtase : list of str, optional
            Subset of `self.mtase` labels to summarize. None uses all.
        chroms : list of str, optional
            Subset of `self.chroms` to summarize. None (default)
            summarizes every chromosome.
        unmapped_strand : {"union", "drop", "+", "-"}, default "union"
            How to evaluate unmapped ('.') strand molecules.

        Returns
        -------
        pd.DataFrame
            Columns `chrom`, `source`, `label`, `n_sites_plus`,
            `n_sites_minus`, `p0`/`p25`/`p50`/`p75`/`p100` (dropout
            fraction percentiles) -- one row per (chrom, source,
            label), plus a `label="aggregate"` row per (chrom,
            source) pooling all labels (as in
            `filter_dropout(method="aggregate")`), if more than one
            label is evaluated.

        Raises
        ------
        ValueError
            If `mtase` is unset, contains an unknown label, `chroms`
            contains a chromosome not in `self.chroms`,
            `unmapped_strand` is invalid, a source is missing for
            some chromosome, or `refseq` is missing.
        """
        labels = self._resolve_mtase_subset(mtase)
        chroms = self._resolve_chroms_subset(chroms)
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}"
            )

        percentiles = [0, 25, 50, 75, 100]
        pct_cols = [f"p{p}" for p in percentiles]
        rows = []
        for chrom in chroms:
            raw = self.raw[chrom]
            if raw.refseq is None:
                raise ValueError(
                    f"no refseq available for chrom '{chrom}'; "
                    "required to determine methylatable positions"
                )
            sources = _resolve_sources(raw, which)

            for src in sources:
                df = getattr(raw, f"{src}_data")
                mol_id = getattr(raw, f"{src}_mol_id")
                if df is None or mol_id is None:
                    raise ValueError(
                        f"no '{src}' data available for chrom '{chrom}'"
                    )
                cov = _label_coverage(
                    df,
                    mol_id,
                    raw.refseq,
                    labels,
                    unmapped_strand,
                    ignore_strand=self._ignore_strand,
                )
                frac, pooled = _dropout_fracs(cov, labels)

                for i, label in enumerate(labels):
                    plus_n = int(
                        _methylatable_positions(raw.refseq, label, "+").sum()
                    )
                    minus_n = int(
                        _methylatable_positions(raw.refseq, label, "-").sum()
                    )
                    row = {
                        "chrom": chrom,
                        "source": src,
                        "label": label,
                        "n_sites_plus": plus_n,
                        "n_sites_minus": minus_n,
                    }
                    row.update(
                        zip(pct_cols, np.percentile(frac[i], percentiles))
                    )
                    rows.append(row)

                if len(labels) > 1:
                    row = {
                        "chrom": chrom,
                        "source": src,
                        "label": "aggregate",
                        "n_sites_plus": sum(
                            int(
                                _methylatable_positions(
                                    raw.refseq, label, "+"
                                ).sum()
                            )
                            for label in labels
                        ),
                        "n_sites_minus": sum(
                            int(
                                _methylatable_positions(
                                    raw.refseq, label, "-"
                                ).sum()
                            )
                            for label in labels
                        ),
                    }
                    row.update(
                        zip(pct_cols, np.percentile(pooled, percentiles))
                    )
                    rows.append(row)

        return pd.DataFrame(rows)

    def dropout_fractions(
        self,
        *,
        which: str | None = None,
        mtase: list | None = None,
        chroms: list | None = None,
        unmapped_strand: str = "union",
    ) -> dict:
        """
        Return each molecule's dropout fraction (QC check).

        If this experiment was loaded with `ignore_strand=True`,
        `unmapped_strand` governs coverage for every molecule.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to evaluate. None evaluates every source
            present for each chromosome.
        mtase : list of str, optional
            Subset of `self.mtase` labels to evaluate. None uses
            all of them.
        chroms : list of str, optional
            Subset of `self.chroms` to evaluate. None (default)
            evaluates every chromosome. Restricting this avoids the
            cost (time and memory) of computing coverage for
            chromosomes not needed by the caller, e.g. a single
            chromosome about to be plotted.
        unmapped_strand : {"union", "drop", "+", "-"}, default
            "union"
            How to evaluate unmapped ('.') strand molecules.

        Returns
        -------
        dict of (str, str, str) to np.ndarray
            Maps `(chrom, source, label)` to that group's
            per-molecule dropout fraction array. Includes a
            `label="aggregate"` entry per (chrom, source) pooling
            all labels, if more than one label is evaluated.

        Raises
        ------
        ValueError
            If `mtase` is unset on this experiment, `mtase` contains
            a label not in `self.mtase`, `chroms` contains a
            chromosome not in `self.chroms`, `unmapped_strand` is
            invalid, a requested source is missing for some
            chromosome, or `refseq` is missing.
        """
        labels = self._resolve_mtase_subset(mtase)
        chroms = self._resolve_chroms_subset(chroms)
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}"
            )

        result = {}
        for chrom in chroms:
            raw = self.raw[chrom]
            if raw.refseq is None:
                raise ValueError(
                    f"no refseq available for chrom '{chrom}'; "
                    "required to determine methylatable positions"
                )
            sources = _resolve_sources(raw, which)

            for src in sources:
                df = getattr(raw, f"{src}_data")
                mol_id = getattr(raw, f"{src}_mol_id")
                if df is None or mol_id is None:
                    raise ValueError(
                        f"no '{src}' data available for chrom '{chrom}'"
                    )
                cov = _label_coverage(
                    df,
                    mol_id,
                    raw.refseq,
                    labels,
                    unmapped_strand,
                    ignore_strand=self._ignore_strand,
                )
                frac, pooled = _dropout_fracs(cov, labels)

                for i, label in enumerate(labels):
                    result[(chrom, src, label)] = frac[i]
                if len(labels) > 1:
                    result[(chrom, src, "aggregate")] = pooled

        return result

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
