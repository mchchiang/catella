# methdata.py

import os
import tempfile
import weakref
from collections import OrderedDict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, List, Mapping, Dict, Self, Any
from ..containers import DataFrameMap, FixedKeyMap
from ..h5_array import H5Array
from .. import utils
from .. import h5_utils
import numpy as np
import pandas as pd
import h5py
import pyarrow as pa
import pyarrow.csv as pyarrow_csv

# Column names as they appear in raw Modkit output, and their internal
# canonical equivalents (positionally parallel).
_MODKIT_COLNAMES = ["read_id", "ref_position", "chrom", "ref_strand",
                    "mod_qual", "mod_code"]
_CANONICAL_NAMES = ["mol_id", "upos", "chrom", "strand", "mod_qual",
                    "mod_code"]
_CANONICAL_PA_TYPES = {
    "mol_id": pa.string(), "upos": pa.int64(), "chrom": pa.string(),
    "strand": pa.string(), "mod_qual": pa.float64(), "mod_code": pa.string(),
}
# Fixed schema for the raw per-chromosome long-format tables, used by
# the streaming AppendableDF writer in load_raw().
_RAW_COLUMN_SPEC = {"mol_index": "numeric", "pos": "numeric",
                    "strand": "string", "mod_qual": "numeric",
                    "mod_code": "string"}


def _resolve_modkit_schema(data_file, sep, colidx):
    """
    Determine how to stream-read a raw Modkit-style file with pyarrow.

    Parameters
    ----------
    data_file : str or pathlib.Path
        The raw data file to be parsed.
    sep : str
        Field delimiter used in the file.
    colidx : list of int or None
        Explicit column indices, positionally parallel to
        `_CANONICAL_NAMES` (i.e. `colidx[0]` is the file column index
        holding `read_id`/`mol_id`, `colidx[1]` holds `ref_position`/
        `upos`, etc). If None, the file is assumed to have a header row
        with Modkit's own column names.

    Returns
    -------
    header_names : list of str or None
        Value for `pyarrow.csv.ReadOptions.column_names`. None means
        the file has its own header row for pyarrow to consume.
    name_for : dict of str to str
        Canonical field name -> the column name pyarrow will parse it
        as.
    """
    if colidx is None:
        header_df = pd.read_csv(data_file, sep=sep, nrows=0)
        raw_by_modkit = {str(c).lstrip("#").strip(): c
                        for c in header_df.columns}
        name_for = {canon: raw_by_modkit[modkit]
                   for modkit, canon in zip(_MODKIT_COLNAMES,
                                            _CANONICAL_NAMES)
                   if modkit in raw_by_modkit}
        return None, name_for

    with open(data_file, "r") as fh:
        first_line = fh.readline()
    ncols = len(first_line.rstrip("\n").split(sep))
    header_names = [f"_col{i}" for i in range(ncols)]
    for canon, idx in zip(_CANONICAL_NAMES, colidx):
        header_names[idx] = canon
    name_for = {canon: canon for canon in _CANONICAL_NAMES}
    return header_names, name_for


def _iter_csv_chunks(data_file, header_names, name_for, sep,
                     needed_canonical, chunk_size, block_size):
    """
    Stream a raw data file in row-chunks with canonical column names.

    Parameters
    ----------
    data_file : str or pathlib.Path
        The raw data file to be parsed.
    header_names : list of str or None
        As returned by `_resolve_modkit_schema`.
    name_for : dict of str to str
        As returned by `_resolve_modkit_schema`.
    sep : str
        Field delimiter used in the file.
    needed_canonical : list of str
        Which canonical columns to read (a subset of
        `_CANONICAL_NAMES`); others are skipped at parse time.
    chunk_size : int
        Approximate minimum number of rows accumulated before a chunk
        is yielded (the last chunk may be smaller).
    block_size : int
        Byte size of the underlying pyarrow read buffer per batch.

    Yields
    ------
    pd.DataFrame
        Successive row-chunks with columns renamed to their canonical
        names.
    """
    include = [name_for[c] for c in needed_canonical]
    types = {name_for[c]: _CANONICAL_PA_TYPES[c] for c in needed_canonical}
    read_opts = pyarrow_csv.ReadOptions(
        block_size=block_size, column_names=header_names,
        autogenerate_column_names=False)
    parse_opts = pyarrow_csv.ParseOptions(delimiter=sep)
    convert_opts = pyarrow_csv.ConvertOptions(
        include_columns=include, column_types=types)
    reader = pyarrow_csv.open_csv(
        str(data_file), read_options=read_opts, parse_options=parse_opts,
        convert_options=convert_opts)
    rename = {name_for[c]: c for c in needed_canonical}

    buf, buf_rows = [], 0
    for batch in reader:
        buf.append(batch)
        buf_rows += batch.num_rows
        if buf_rows >= chunk_size:
            yield pa.Table.from_batches(buf).to_pandas().rename(
                columns=rename)
            buf, buf_rows = [], 0
    if buf:
        yield pa.Table.from_batches(buf).to_pandas().rename(columns=rename)


def _scan_accepted_mols(data_file, header_names, name_for, sep,
                        chunk_size, block_size, max_nmol, rng):
    """
    Determine, per chromosome, which molecules are kept and their
    `mol_index`.

    Streams only the `chrom`+`mol_id` columns (much cheaper than full
    rows) to build each chromosome's distinct molecule set in
    file-encounter order, then reproduces the legacy single-shot
    downsampling exactly: chromosomes are visited in sorted order,
    `rng.choice(mols, max_nmol, replace=False)` is called only when a
    chromosome has more than `max_nmol` molecules, and `mol_index` is
    the rank of each accepted `mol_id` in ascending sorted order
    (equivalent to `pd.factorize` on an already mol_id-sorted frame,
    which is what the legacy single-shot code produced).

    Parameters
    ----------
    data_file : str or pathlib.Path
        The raw data file to be parsed.
    header_names : list of str or None
        As returned by `_resolve_modkit_schema`.
    name_for : dict of str to str
        As returned by `_resolve_modkit_schema`.
    sep : str
        Field delimiter used in the file.
    chunk_size : int
        Approximate rows accumulated per streamed chunk.
    block_size : int
        Byte size of the underlying pyarrow read buffer per batch.
    max_nmol : int or None
        Maximum number of molecules to keep per chromosome.
    rng : np.random.Generator or None
        Random number generator for downsampling; required if
        `max_nmol` is not None.

    Returns
    -------
    dict of str to dict
        `{chrom: {"mol_id": np.ndarray (sorted, accepted mol_ids),
        "index": dict mapping mol_id -> mol_index}}`, covering every
        chromosome found in the file (regardless of whether it is
        present in the chromosome-size table).
    """
    mol_order = {}
    for batch_df in _iter_csv_chunks(
            data_file, header_names, name_for, sep, ["chrom", "mol_id"],
            chunk_size, block_size):
        dedup = batch_df.drop_duplicates(["chrom", "mol_id"])
        for chrom, mol_id in zip(dedup["chrom"].to_numpy(),
                                 dedup["mol_id"].to_numpy()):
            seen = mol_order.setdefault(chrom, {})
            if mol_id not in seen:
                seen[mol_id] = None

    result = {}
    for chrom in sorted(mol_order.keys()):
        mols = np.array(list(mol_order[chrom].keys()), dtype=object)
        if max_nmol is not None and len(mols) > max_nmol:
            mols = rng.choice(mols, max_nmol, replace=False)
        sorted_mols = np.sort(mols)
        result[chrom] = {"mol_id": sorted_mols,
                         "index": {m: i for i, m in enumerate(sorted_mols)}}
    return result


def _stream_rows_to_staging(data_file, header_names, name_for, sep,
                            chunk_size, block_size, full_sizes, wrap,
                            mol_maps, appenders):
    """
    Stream full rows, transform them, and append to per-chromosome
    disk-backed appenders.

    Parameters
    ----------
    data_file : str or pathlib.Path
        The raw data file to be parsed.
    header_names : list of str or None
        As returned by `_resolve_modkit_schema`.
    name_for : dict of str to str
        As returned by `_resolve_modkit_schema`.
    sep : str
        Field delimiter used in the file.
    chunk_size : int
        Approximate rows accumulated per streamed chunk.
    block_size : int
        Byte size of the underlying pyarrow read buffer per batch.
    full_sizes : dict of str to int
        Chromosome name -> full (unhalved) length (bp), used for the
        `wrap` position transform.
    wrap : bool
        If True, positions are folded around the fiber center.
    mol_maps : dict of str to dict
        As returned by `_scan_accepted_mols`, already filtered down to
        the chromosomes actually being processed.
    appenders : dict of str to h5_utils.AppendableDF
        One appender per chromosome in `mol_maps`, already created.
    """
    needed = ["mol_id", "upos", "chrom", "strand", "mod_qual", "mod_code"]
    for batch_df in _iter_csv_chunks(
            data_file, header_names, name_for, sep, needed, chunk_size,
            block_size):
        for chrom, group in batch_df.groupby("chrom", sort=False):
            mapping = mol_maps.get(chrom)
            if mapping is None:
                continue
            mol_index = group["mol_id"].map(mapping["index"])
            keep = mol_index.notna()
            if not keep.any():
                continue
            group = group.loc[keep]
            mol_index = mol_index[keep].astype(np.int64).to_numpy()

            upos = group["upos"].to_numpy()
            if wrap:
                length = full_sizes[chrom]
                pos = np.where(upos > (length-1)/2,
                              length - upos - 1, upos)
            else:
                pos = upos

            out = pd.DataFrame({
                "mol_index": mol_index,
                "pos": pos.astype(np.int64),
                "strand": group["strand"].to_numpy(),
                "mod_qual": group["mod_qual"].astype(np.float64).to_numpy(),
                "mod_code": group["mod_code"].to_numpy(),
            })
            appenders[chrom].append(out)


@utils.add_frozen_properties
@dataclass(frozen=True, slots=True, init=False)
class MethPrintData:
    """
    Container for MethPrint experimental data for a single chromosome.

    This class stores methylation signal data for test samples and optional 
    control samples (unmethylated and fully methylated). Data are stored 
    as Pandas DataFrames and retrieved via read-only properties that 
    provide defensive copies.

    .. note::
       This class is intended for internal use within a 
       :class:`MethPrintExperiment`. Use the experiment's loading 
       mechanisms rather than instantiating this class directly.
    """
    
    chrom : str
    """The chromosome identifier for this data block."""
    
    nbp : int
    """The total number of base pairs in the chromatin fiber."""
    
    _frozen_test_mol_id : np.ndarray = field(
        metadata={"doc": "np.ndarray: Array of molecule identifiers for the "
                  "test samples."})
    
    _frozen_test_data : pd.DataFrame = field(
        metadata={"doc": "pd.DataFrame: Raw methylation quality scores for "
                  "test samples."})
    
    _frozen_unmeth_mol_id : np.ndarray | None = field(
        metadata={"doc": "np.ndarray: Array of molecule identifiers for "
                  "unmethylated controls."})
    
    _frozen_unmeth_data : pd.DataFrame | None = field(
        metadata={"doc": "pd.DataFrame: Raw methylation quality scores for "
                  "unmethylated controls."})
    
    _frozen_meth_mol_id : np.ndarray | None = field(
        metadata={"doc": "np.ndarray: Array of molecule identifiers for "
                  "methylated controls."})
    
    _frozen_meth_data : pd.DataFrame | None = field(
        metadata={"doc": "pd.DataFrame: Raw methylation quality scores for "
                  "methylated controls."})

    def __init__(self, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use MethPrintData._load() to instantiate ",
                            "this class.")
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is None:
                val = kwargs.get(f.name.removeprefix("_frozen_"))
            object.__setattr__(self, f.name, val)
    
    def _save(self, group):
        dt = h5py.string_dtype(encoding="utf-8") # For storing strings        
        gchrom = group.create_group(self.chrom)
        gmeta = gchrom.create_group("metadata")
        gmeta.attrs["chrom"] = self.chrom
        gmeta.attrs["nbp"] = self.nbp            
        gdata = gchrom.create_group("data")
        def save_data(name, mol_id, df, gdata):
            if mol_id is not None and df is not None:
                gdata.create_dataset(name+"_mol_id", data=mol_id, dtype=dt)
                h5_utils.save_df(name+"_data", df, gdata)
        save_data("test", self._frozen_test_mol_id,
                  self._frozen_test_data, gdata)
        save_data("unmeth", self._frozen_unmeth_mol_id,
                  self._frozen_unmeth_data, gdata)
        save_data("meth", self._frozen_meth_mol_id,
                  self._frozen_meth_data, gdata)        
        
    @classmethod
    def _load(cls, group, chrom) -> Self:
        if chrom not in group:
            raise ValueError(f"Cannot find data for the chrom {chrom}.")
        gchrom = group[chrom]
        gmeta = gchrom["metadata"]
        chrom = gmeta.attrs["chrom"]
        nbp = int(gmeta.attrs["nbp"])
        def load_data(name, gdata):
            id_name = name+"_mol_id"
            data_name = name+"_data"
            if id_name in gdata and data_name in gdata:
                mol_id = gdata[id_name].asstr()[()]
                df = h5_utils.load_df(data_name, gdata)
                return mol_id, df
            return None, None
        gdata = gchrom["data"]
        test_mol_id, test_data = load_data("test", gdata)        
        unmeth_mol_id, unmeth_data = load_data("unmeth", gdata)
        meth_mol_id, meth_data = load_data("meth", gdata)
        return cls._create(chrom=chrom, nbp=nbp, test_mol_id=test_mol_id,
                           test_data=test_data, unmeth_mol_id=unmeth_mol_id,
                           unmeth_data=unmeth_data, meth_mol_id=meth_mol_id,
                           meth_data=meth_data)
    
    @classmethod
    def _create(cls, **kwargs) -> Self:
        # Some validation
        if kwargs.get("nbp") <= 0:
            raise ValueError("nbp must be positive")
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
    
    _raw_data : Mapping[str,MethPrintData]
    _analysis : Mapping[str,DataFrameMap]
    _global_analysis : DataFrameMap
    _scratch_path : str | None
    _finalizer : Any | None
    _tmp_dir : str | None

    def __init__(self, **kwargs : Any):
        if not kwargs.pop("_internal", False):
            raise TypeError("Use MethPrintExperiment.load() or .load_raw()",
                            "to instantiate this class")

        # Bulk assignment of fields
        for f in fields(self):
            val = kwargs.get(f.name)
            if val is not None:
                setattr(self, f.name, val)

        # Create the dicts for analysis
        self._analysis = {chrom : DataFrameMap()
                          for chrom in self._raw_data.keys()}
        self._global_analysis = DataFrameMap()

        # Scratch staging file (set by load_raw() when it owns one)
        self._scratch_path = kwargs.get("_scratch_path")
        self._finalizer = kwargs.get("_finalizer")

        # Scratch directory shared by this experiment's own staging
        # file and any later smooth()/meth_prob() calls that don't
        # specify their own tmp_dir (set by load_raw() when it
        # generates one, or lazily by resolve_tmp_dir()).
        self._tmp_dir = kwargs.get("_tmp_dir")

    @staticmethod
    def _cleanup_scratch(path):
        Path(path).unlink(missing_ok=True)

    def resolve_tmp_dir(self):
        """
        Return this experiment's scratch directory for temporary HDF5
        files, creating and caching one under the system default
        temporary directory the first time it's needed.

        Returns
        -------
        str
            Path to the resolved scratch directory.
        """
        if self._tmp_dir is None:
            self._tmp_dir = h5_utils.fresh_tmp_dir()
        return self._tmp_dir

    def close(self):
        """
        Delete the scratch staging file backing this experiment's raw
        data, if `load_raw` created one. No-op for experiments obtained
        via `.load()` or `_create`.
        """
        if self._finalizer is not None:
            self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
    
        
    @classmethod
    def load_raw(cls,
                 chromsize : str | Path,
                 test_file : str | Path,
                 unmeth_file : str | Path | None = None,
                 meth_file : str | Path | None = None,
                 chroms : List[str] | None = None,
                 wrap : bool = False,
                 colidx : List | None = None,
                 max_nmol : int | None = None,
                 seed : int | None = None,
                 chunk_size : int = 1000000,
                 tmp_dir : str | Path | None = None,
                 max_cached_chroms : int = 1) -> Self:
        """
        Create an experiment by processing raw sequencing data files.

        This method reads chromosome sizes and experimental data (typically
        modkit CSV output), re-orients the data if requested, and splits
        the signals by chromosome. Each file is streamed in two passes
        (molecule discovery, then row transfer) so the full file is
        never held in memory at once; raw data ends up in a staging
        HDF5 file and the returned experiment lazily loads it (see
        `load`), so not all chromosomes need to be resident in memory
        either.

        Parameters
        ----------
        chromsize : str or pathlib.Path
            Path to a tab-separated file containing chromosome names and
            their lengths (bp).
        test_file : str or pathlib.Path
            Path to the raw experimental (test) data file.
        unmeth_file : str or pathlib.Path, optional
            Path to the unmethylated control data file.
        meth_file : str or pathlib.Path, optional
            Path to the fully methylated control data file.
        chroms : list of str, optional
            Restrict processing to these chromosomes. If None
            (default), all chromosomes in `chromsize` are processed.
        wrap : bool, default False
            If True, calculates positions relative to the fiber center
            (useful for circular or symmetrical fibers).
        colidx : list of int, optional
            Explicit column indices if the input file has no header
            matching Modkit's column names. Positionally parallel to
            the canonical field order: `colidx[0]` is the file column
            index holding `read_id`, `colidx[1]` holds `ref_position`,
            `colidx[2]` holds `chrom`, `colidx[3]` holds `ref_strand`,
            `colidx[4]` holds `mod_qual`, `colidx[5]` holds `mod_code`.
        max_nmol : int or None
            Maximum number of molecules to extract for each chromosome.
        seed : int or None
            The seed for the random number generator selecting the molecules
            if `max_nmol` is specified.
        chunk_size : int, default 1000000
            Approximate number of rows read (and held in memory) per
            streamed chunk.
        tmp_dir : str or Path, optional
            Directory used for the scratch staging file backing the
            returned experiment's raw data (removed once the experiment
            is closed or garbage-collected; see `close`). A fresh
            `nucmc_<timestamp>_<hex>` subfolder is created for it —
            under this directory if given, otherwise under the system
            default temporary directory — and cached on the returned
            experiment so that later `smooth`/`meth_prob` calls on it
            reuse the same subfolder by default.
        max_cached_chroms : int, default 1
            Maximum number of chromosomes' raw data kept in memory at
            once by the returned experiment (forwarded to `load`).

        Returns
        -------
        MethPrintExperiment
            A newly initialized experiment object containing the processed
            data.

        Raises
        ------
        ValueError
            If chromosome names are not unique, if control datasets
            contain chromosomes not found in the test dataset, or if
            `chroms` contains a chromosome not found in `chromsize`.
        """

        # Read chromosome sizes
        size_cols = ["chrom", "length"]
        with open(chromsize, "r") as reader:
            first_line = reader.readline().strip()
        if first_line == "\t".join(size_cols):
            df_size = pd.read_csv(chromsize, sep="\t")
        else:
            df_size = pd.read_csv(chromsize, sep="\t", names=size_cols,
                                  header=None)
        if "chrom" not in df_size.columns:
            raise ValueError("Column 'chrom' containing the chromosome "
                             "identifier is missing.")
        if "length" not in df_size.columns:
            raise ValueError("Column 'length' containing the chromosome "
                             "length is missing.")
        if not df_size["chrom"].is_unique:
            raise ValueError("Chromosomes must be unique in chromsize file")

        # Full (unhalved) chromosome lengths, needed for the wrap fold
        # transform itself; `sizes` (below) is the *storage* length
        # (halved under wrap, since positions are folded into that
        # range) used for `nbp`.
        full_sizes = dict(zip(df_size["chrom"], df_size["length"].astype(int)))
        if wrap:
            sizes = dict(zip(df_size["chrom"],
                             (df_size["length"]/2).astype(int)))
        else:
            sizes = full_sizes

        if chroms is not None:
            unknown = set(chroms) - set(df_size["chrom"])
            if unknown:
                raise ValueError(
                    f"Chromosomes not found in chromsize: {sorted(unknown)}")
            full_sizes = {c: v for c, v in full_sizes.items()
                         if c in chroms}
            sizes = {c: v for c, v in sizes.items() if c in chroms}

        # Random generator for downsampling
        rng = None if max_nmol is None else np.random.default_rng(seed)

        sep = "\t"
        block_size = 64 * 1024 * 1024

        tmp_dir = h5_utils.fresh_tmp_dir(base_dir=tmp_dir)
        fd, staging_path = tempfile.mkstemp(suffix=".h5", dir=tmp_dir)
        os.close(fd)

        dt = h5py.string_dtype(encoding="utf-8")

        def ensure_chrom_data_group(graw, chrom, nbp):
            gchrom = graw.require_group(chrom)
            if "metadata" not in gchrom:
                gmeta = gchrom.create_group("metadata")
                gmeta.attrs["chrom"] = chrom
                gmeta.attrs["nbp"] = int(nbp)
            return gchrom.require_group("data")

        try:
            with h5py.File(staging_path, "w") as h5stream:
                graw = h5stream.create_group("raw_data")
                h5stream.create_group("analysis")
                h5stream.create_group("global_analysis")

                files = [("test", test_file)]
                if unmeth_file is not None:
                    files.append(("unmeth", unmeth_file))
                if meth_file is not None:
                    files.append(("meth", meth_file))

                test_chroms = None
                for name, data_file in files:
                    print(f"Reading {data_file} ...")
                    header_names, name_for = _resolve_modkit_schema(
                        data_file, sep, colidx)
                    mol_maps = _scan_accepted_mols(
                        data_file, header_names, name_for, sep, chunk_size,
                        block_size, max_nmol, rng)
                    # Chromosomes not covered by the chromsize table are
                    # silently dropped, matching the legacy inner merge
                    mol_maps = {c: m for c, m in mol_maps.items()
                               if c in sizes}

                    if name == "test":
                        test_chroms = set(mol_maps.keys())
                    elif set(mol_maps.keys()) != test_chroms:
                        raise ValueError(
                            "Different number of chromosomes in test and "
                            f"{name} datasets")

                    appenders = {}
                    for chrom, m in mol_maps.items():
                        gdata = ensure_chrom_data_group(
                            graw, chrom, sizes[chrom])
                        gdata.create_dataset(f"{name}_mol_id",
                                             data=m["mol_id"], dtype=dt)
                        appenders[chrom] = h5_utils.AppendableDF(
                            gdata, f"{name}_data",
                            columns=list(_RAW_COLUMN_SPEC),
                            dtypes=_RAW_COLUMN_SPEC)

                    _stream_rows_to_staging(
                        data_file, header_names, name_for, sep, chunk_size,
                        block_size, full_sizes, wrap, mol_maps, appenders)
        except Exception:
            Path(staging_path).unlink(missing_ok=True)
            raise

        exp = cls.load(staging_path, max_cached_chroms=max_cached_chroms)
        exp._scratch_path = staging_path
        exp._finalizer = weakref.finalize(
            exp, MethPrintExperiment._cleanup_scratch, staging_path)
        exp._tmp_dir = tmp_dir
        return exp
                
    def save(self, path: str | Path):
        """
        Save the experiment data and analysis to an HDF5 file.

        Parameters
        ----------
        path : str or pathlib.Path
            The output file path. Raw data is only written if it does not 
            already exist in the file.

        Raises
        ------
        OSError
            If the file cannot be written to disk.
        ValueError
            If `path` is the same file that backs an `H5Array` analysis
            entry already held by this experiment.
        """
        dest = str(Path(path).resolve())
        for data in list(self._analysis.values()) + [self._global_analysis]:
            for entry in data.values():
                if isinstance(entry, H5Array) and \
                        str(Path(entry.path).resolve()) == dest:
                    raise ValueError(
                        "Cannot save to the same file that backs an "
                        "existing H5Array analysis entry; save to a "
                        "different path.")

        with h5py.File(path, "a") as h5stream:
            # Save the raw data - write once if raw_data does not exist
            if not "raw_data" in h5stream:
                graw = h5stream.create_group("raw_data")
                for chrom, rdata in self._raw_data.items():
                    rdata._save(graw)

            # Save any analysis data
            if "analysis" in h5stream: del h5stream["analysis"]
            gana = h5stream.create_group("analysis")
            for chrom, data in self._analysis.items():
                gchrom = gana.create_group(chrom)
                for name, entry in data.items():
                    if isinstance(entry, H5Array):
                        entry.save_to(gchrom, name)
                    else:
                        h5_utils.save_df(name, entry, gchrom)
            if "global_analysis" in h5stream: del h5stream["global_analysis"]
            gana = h5stream.create_group("global_analysis")
            for name, entry in self._global_analysis.items():
                if isinstance(entry, H5Array):
                    entry.save_to(gana, name)
                else:
                    h5_utils.save_df(name, entry, gana)
                
    @classmethod
    def load(cls, path: str | Path,
             chroms: List[str] | None = None,
             max_cached_chroms: int = 1) -> Self:
        """
        Load an experiment from a persistent HDF5 file.

        Raw per-chromosome data is loaded lazily: `MethPrintData` for a
        chromosome is only read from disk when accessed via `raw`, and
        is cached for at most `max_cached_chroms` chromosomes at a time
        (least-recently-used eviction), so not all chromosomes need to
        be resident in memory simultaneously.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the HDF5 file containing the experiment.
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
        with h5py.File(path, "r") as h5stream:
            # Set up lazy, memory-bounded access to the raw data
            available_chroms = list(h5stream["raw_data"].keys())
            if chroms is not None:
                unknown = set(chroms) - set(available_chroms)
                if unknown:
                    raise ValueError(
                        f"Chromosomes not found in {path}: "
                        f"{sorted(unknown)}")
                selected = list(chroms)
            else:
                selected = available_chroms
            raw_data = LazyRawDataMap(path, selected,
                                      max_cached=max_cached_chroms)
            obj = cls._create(_raw_data=raw_data)

            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                if chrom not in selected:
                    continue
                gchrom = gana[chrom]
                for name in gchrom:
                    if isinstance(gchrom[name], h5py.Dataset):
                        obj._analysis[chrom][name] = H5Array.load_from(
                            path, f"analysis/{chrom}/{name}")
                    else:
                        obj._analysis[chrom][name] = h5_utils.load_df(
                            name, gchrom)
            gana = h5stream["global_analysis"]
            for name in gana:
                if isinstance(gana[name], h5py.Dataset):
                    obj._global_analysis[name] = H5Array.load_from(
                        path, f"global_analysis/{name}")
                else:
                    obj._global_analysis[name] = h5_utils.load_df(name, gana)
            return obj
                
    @classmethod
    def _create(cls, **kwargs) -> Self:
        return cls(**kwargs, _internal=True)

    @property
    def chroms(self) -> Tuple[str,...]:
        """
        Get the identifiers of all chromosomes in the experiment.

        Returns
        -------
        tuple of str
            A sorted tuple of chromosome names.
        """
        return tuple(self._raw_data.keys())

    @property
    def raw(self) -> Mapping[str,MethPrintData]:
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
