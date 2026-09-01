# methdata.py

import os
import shutil
import tempfile
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Tuple, List, Mapping, Dict, Self, Any
from catella.containers import DataFrameMap, FixedKeyMap
from catella.h5_array import H5Array
from catella import utils
from catella import h5_utils
import numpy as np
import pandas as pd
import h5py
import pyarrow as pa
import pyarrow.csv as pyarrow_csv
import pyfaidx

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

# Molecule-count threshold above which MethPrintExperiment.to_dense()
# warns when materializing a pd.DataFrame (as opposed to streaming to
# an H5Array).
_DENSE_WARN_ROWS = 5000

# Recognized methyltransferase labels: 'A' (any-context adenine,
# e.g., EcoGII), 'CG' (CpG, e.g., M.SssI), 'GC' (GpC, e.g., M.CviPI).
_VALID_MTASE = {"A", "CG", "GC"}


def _normalize_mtase(mtase):
    """
    Validate and normalize the `mtase` argument.

    Parameters
    ----------
    mtase : str, sequence of str, or None
        One or more methyltransferase labels.

    Returns
    -------
    tuple of str or None
        Deduplicated labels in the order given, or None if `mtase`
        is None.

    Raises
    ------
    ValueError
        If a label is not one of `_VALID_MTASE`, or a label repeats.
    """
    if mtase is None:
        return None
    values = (mtase,) if isinstance(mtase, str) else tuple(mtase)
    unknown = sorted(set(values) - _VALID_MTASE)
    if unknown:
        raise ValueError(
            f"Unknown mtase value(s) {unknown}; must be one or more "
            f"of {sorted(_VALID_MTASE)}.")
    if len(set(values)) != len(values):
        raise ValueError("Duplicate mtase values given.")
    return values


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


def _parse_fasta(fasta_file, wanted_chroms):
    """
    Read reference sequences for a set of chromosomes from a FASTA file.

    Parameters
    ----------
    fasta_file : str or pathlib.Path
        Multi-FASTA file, one record per chromosome.
    wanted_chroms : set of str
        Record ids to read. Records not in this set are not loaded
        into memory.

    Returns
    -------
    dict of str to str
        Chromosome id -> sequence, for every id in `wanted_chroms`
        found in the file. Ids in `wanted_chroms` absent from the
        file are simply not present in the returned dict.
    """
    sequences = {}
    fasta = pyfaidx.Fasta(str(fasta_file), as_raw=True, build_index=True)
    try:
        for chrom in wanted_chroms:
            if chrom in fasta:
                sequences[chrom] = str(fasta[chrom][:])
    finally:
        fasta.close()
    return sequences


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
    
    chrom : str
    """The chromosome identifier for this data block."""
    
    nbp : int
    """The total number of base pairs in the chromatin fiber."""

    _frozen_refseq : str | None = field(
        metadata={"doc": "str: The reference nucleotide sequence for this "
                  "chromosome, or None if not provided."})

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
        if self._frozen_refseq is not None:
            gmeta.create_dataset("refseq", data=self._frozen_refseq,
                                 dtype=dt)
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
        refseq = gmeta["refseq"].asstr()[()] if "refseq" in gmeta else None
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
        return cls._create(chrom=chrom, nbp=nbp, refseq=refseq,
                           test_mol_id=test_mol_id, test_data=test_data,
                           unmeth_mol_id=unmeth_mol_id,
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


# Valid values for MethPrintExperiment.filter_dropout()'s
# 'unmapped_strand' and 'method' arguments.
_VALID_UNMAPPED_STRAND = {"union", "drop", "+", "-"}
_VALID_FILTER_DROPOUT_METHOD = {"separate", "aggregate"}


def _methylatable_positions(refseq, mtase_label, strand):
    """
    Boolean mask of positions in `refseq` methylatable by
    `mtase_label`, for a molecule on the given `strand`.

    Parameters
    ----------
    refseq : str
        Reference nucleotide sequence.
    mtase_label : {"A", "CG", "GC"}
        Methyltransferase context.
    strand : {"+", "-"}
        Strand of the molecule. Callers resolve unmapped ('.')
        strand separately (there is no single correct answer here).

    Returns
    -------
    np.ndarray of bool
        Length `len(refseq)`.
    """
    arr = np.frombuffer(refseq.upper().encode(), dtype="S1")
    mask = np.zeros(len(arr), dtype=bool)
    if mtase_label == "A":
        base = b"A" if strand == "+" else b"T"
        mask |= (arr == base)
        return mask
    b1, b2 = mtase_label[0].encode(), mtase_label[1].encode()
    dinuc = (arr[:-1] == b1) & (arr[1:] == b2)
    # + strand: the C sits at offset 0 of a 'CG' match, offset 1 of a
    # 'GC' match (and the opposite offset on - strand).
    c_at_start = (mtase_label == "CG") == (strand == "+")
    if c_at_start:
        mask[:-1] |= dinuc
    else:
        mask[1:] |= dinuc
    return mask


def _strand_of_mol(df, nmol):
    """
    One strand label per molecule, looked up by `mol_index`.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format table with `mol_index`/`strand` columns.
    nmol : int
        Total molecule count; a `mol_index` absent from `df` (no rows
        at all) defaults to '.'.

    Returns
    -------
    np.ndarray of str
        Length `nmol`, each one of '+', '-', '.'.
    """
    labels = np.full(nmol, ".", dtype="<U1")
    first = df.drop_duplicates("mol_index")[["mol_index", "strand"]]
    idx = first["mol_index"].to_numpy().astype(np.int64)
    labels[idx] = first["strand"].to_numpy()
    return labels


def _resolve_sources(raw, which):
    """
    Source name(s) to evaluate for dropout, on one chromosome.

    Parameters
    ----------
    raw : MethPrintData
        Raw data block for one chromosome.
    which : {"test", "meth", "unmeth"} or None
        Single source, or None for every source present.

    Returns
    -------
    list of str
    """
    if which is not None:
        return [which]
    return [src for src in ("test", "meth", "unmeth")
            if getattr(raw, f"{src}_data") is not None]


def _label_coverage(df, mol_id, refseq, labels, unmapped_strand):
    """
    Per-molecule covered/total methylatable-site counts, per label.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format table with `mol_index`/`pos`/`strand` columns.
    mol_id : array-like
        This source's molecule identifiers (only its length is used).
    refseq : str
        Reference sequence for this chromosome.
    labels : sequence of str
        `mtase` labels to evaluate, each in {"A", "CG", "GC"}.
    unmapped_strand : {"union", "drop", "+", "-"}
        How to evaluate unmapped ('.') strand molecules.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Maps each label to `(covered, n_total)`, each length
        `len(mol_id)`: `covered` counts rows measured at that label's
        methylatable positions (on each molecule's own strand);
        `n_total` is that label's methylatable-site count for the
        molecule's strand.
    """
    nmol = len(mol_id)
    strand = _strand_of_mol(df, nmol)
    pos = df["pos"].to_numpy().astype(np.int64)
    mol_index = df["mol_index"].to_numpy().astype(np.int64)
    row_strand = strand[mol_index]
    is_plus = row_strand == "+"
    is_minus = row_strand == "-"
    is_dot = row_strand == "."

    result = {}
    for label in labels:
        plus_mask = _methylatable_positions(refseq, label, "+")
        minus_mask = _methylatable_positions(refseq, label, "-")
        union_mask = plus_mask | minus_mask

        in_plus = plus_mask[pos]
        in_minus = minus_mask[pos]
        row_covered = np.zeros(len(pos), dtype=bool)
        row_covered[is_plus] = in_plus[is_plus]
        row_covered[is_minus] = in_minus[is_minus]
        if unmapped_strand == "union":
            in_union = union_mask[pos]
            row_covered[is_dot] = in_union[is_dot]
        elif unmapped_strand == "+":
            row_covered[is_dot] = in_plus[is_dot]
        elif unmapped_strand == "-":
            row_covered[is_dot] = in_minus[is_dot]
        # "drop": row_covered[is_dot] stays False

        covered = np.bincount(mol_index[row_covered], minlength=nmol)

        total = np.empty(nmol, dtype=np.int64)
        total[strand == "+"] = plus_mask.sum()
        total[strand == "-"] = minus_mask.sum()
        if unmapped_strand == "+":
            total[strand == "."] = plus_mask.sum()
        elif unmapped_strand == "-":
            total[strand == "."] = minus_mask.sum()
        else:
            total[strand == "."] = union_mask.sum()

        result[label] = (covered, total)
    return result


def _apply_keep_mask(data, keep, batch_size=20000):
    """
    Set rows for `keep=False` molecules entirely to NaN.

    Parameters
    ----------
    data : pd.DataFrame, np.ndarray, or H5Array
        Dense data, one row per molecule.
    keep : array-like of bool
        Aligned with `data`'s rows.
    batch_size : int, default 20000
        Rows processed per batch when `data` is an `H5Array`
        (rewritten on disk in place, batch by batch).

    Returns
    -------
    pd.DataFrame, np.ndarray, or H5Array
        `data` with masked-out rows set to NaN. `H5Array` is rewritten
        on disk in place; `pd.DataFrame`/`np.ndarray` is copied (a
        `to_numpy()` result may be a read-only view).
    """
    keep = np.asarray(keep, dtype=bool)
    if isinstance(data, pd.DataFrame):
        data = data.copy()
        data.loc[~keep, :] = np.nan
        return data
    if isinstance(data, H5Array):
        for start in range(0, data.shape[0], batch_size):
            stop = min(start + batch_size, data.shape[0])
            batch = data[start:stop, :]
            batch[~keep[start:stop], :] = np.nan
            data.write_batch(start, stop, batch)
        return data
    data = np.array(data, copy=True)
    data[~keep, :] = np.nan
    return data


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
    _finalizer : Any | None
    _tmp_dir : str | None
    _exp_file : str | None
    _mtase : Tuple[str, ...] | None

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
                self, MethPrintExperiment._cleanup_tmp_dir, self._tmp_dir)
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
    def load_raw(cls,
                 chromsize : str | Path,
                 test_file : str | Path,
                 unmeth_file : str | Path | None = None,
                 meth_file : str | Path | None = None,
                 fasta_file : str | Path | None = None,
                 mtase : str | List[str] | None = None,
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
            Tab-separated file containing chromosome names and
            their lengths (bp).
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
            `catella_<timestamp>_<hex>` subfolder is created for it —
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
            contain chromosomes not found in the test dataset, if
            `chroms` contains a chromosome not found in `chromsize`,
            if a sequence in `fasta_file` does not match its
            chromosome's length in `chromsize`, or if `mtase`
            contains an unknown or duplicate label.
        """
        mtase = _normalize_mtase(mtase)

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

        refseq_by_chrom = {}
        if fasta_file is not None:
            refseq_by_chrom = _parse_fasta(fasta_file, set(full_sizes))
            for chrom, seq in refseq_by_chrom.items():
                if len(seq) != full_sizes[chrom]:
                    raise ValueError(
                        f"Reference sequence length for chromosome "
                        f"{chrom!r} ({len(seq)}) does not match its "
                        f"chromsize length ({full_sizes[chrom]}).")

        # Random generator for downsampling
        rng = None if max_nmol is None else np.random.default_rng(seed)

        sep = "\t"
        block_size = 64 * 1024 * 1024

        tmp_dir = h5_utils.fresh_tmp_dir(base_dir=tmp_dir)
        fd, tmp_file = tempfile.mkstemp(suffix=".h5", dir=tmp_dir)
        os.close(fd)

        dt = h5py.string_dtype(encoding="utf-8")

        def ensure_chrom_data_group(graw, chrom, nbp):
            gchrom = graw.require_group(chrom)
            if "metadata" not in gchrom:
                gmeta = gchrom.create_group("metadata")
                gmeta.attrs["chrom"] = chrom
                gmeta.attrs["nbp"] = int(nbp)
                seq = refseq_by_chrom.get(chrom)
                if seq is not None:
                    gmeta.create_dataset("refseq", data=seq, dtype=dt)
            return gchrom.require_group("data")

        try:
            with h5py.File(tmp_file, "w") as h5stream:
                graw = h5stream.create_group("raw_data")
                h5stream.create_group("analysis")
                h5stream.create_group("global_analysis")
                if mtase is not None:
                    h5stream.attrs["mtase"] = ",".join(mtase)

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
            Path(tmp_file).unlink(missing_ok=True)
            raise

        exp = cls.load(tmp_file, max_cached_chroms=max_cached_chroms)
        # tmp_file is an internal staging file, not a real save target
        exp._exp_file = None
        exp._tmp_dir = tmp_dir
        exp._finalizer = weakref.finalize(
            exp, MethPrintExperiment._cleanup_tmp_dir, tmp_dir)
        return exp
                
    def save(self, exp_file: str | Path | None = None, *,
            overwrite : bool = False):
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
                    "No exp_file given and this experiment has not "
                    "been saved before; pass 'exp_file' explicitly.")

        dest_path = Path(exp_file)
        dest = str(dest_path.resolve())
        collision = any(
            isinstance(entry, H5Array)
            and str(Path(entry.path).resolve()) == dest
            for data in list(self._analysis.values()) + [self._global_analysis]
            for entry in data.values())
        if collision:
            if not overwrite:
                raise ValueError(
                    "Cannot save to the same file that backs an "
                    "existing H5Array analysis entry; save to a "
                    "different path, or pass overwrite=True to safely "
                    "replace it in place.")
            tmp_path = dest_path.with_name(
                dest_path.name + f".tmp{os.getpid()}")
            self.save(tmp_path)
            os.replace(tmp_path, dest_path)
            self._exp_file = dest
            return

        with h5py.File(exp_file, "a") as h5stream:
            # Skip only if re-saving to the same file we're already
            # backed by; other destinations always get a fresh copy.
            same_file = (self._exp_file is not None and
                        str(Path(self._exp_file).resolve()) == dest)
            if not same_file or "raw_data" not in h5stream:
                if "raw_data" in h5stream:
                    del h5stream["raw_data"]
                graw = h5stream.create_group("raw_data")
                for chrom, rdata in self._raw_data.items():
                    rdata._save(graw)
                if self._mtase is not None:
                    h5stream.attrs["mtase"] = ",".join(self._mtase)

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
        self._exp_file = dest
                
    @classmethod
    def load(cls, exp_file: str | Path,
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
                        f"Chromosomes not found in {exp_file}: "
                        f"{sorted(unknown)}")
                selected = list(chroms)
            else:
                selected = available_chroms
            raw_data = LazyRawDataMap(exp_file, selected,
                                      max_cached=max_cached_chroms)
            mtase = tuple(h5stream.attrs["mtase"].split(",")) \
                if "mtase" in h5stream.attrs else None
            obj = cls._create(_raw_data=raw_data,
                              _exp_file=str(Path(exp_file).resolve()),
                              _mtase=mtase)

            # Load any analysis data
            gana = h5stream["analysis"]
            for chrom in gana:
                if chrom not in selected:
                    continue
                gchrom = gana[chrom]
                for name in gchrom:
                    if isinstance(gchrom[name], h5py.Dataset):
                        obj._analysis[chrom][name] = H5Array.load_from(
                            exp_file, f"analysis/{chrom}/{name}")
                    else:
                        obj._analysis[chrom][name] = h5_utils.load_df(
                            name, gchrom)
            gana = h5stream["global_analysis"]
            for name in gana:
                if isinstance(gana[name], h5py.Dataset):
                    obj._global_analysis[name] = H5Array.load_from(
                        exp_file, f"global_analysis/{name}")
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
    def mtase(self) -> Tuple[str,...] | None:
        """
        Get the methyltransferase(s) used for this experiment.

        Returns
        -------
        tuple of str or None
            One or more of 'A', 'CG', 'GC', or None if unspecified.
        """
        return self._mtase

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

    def to_dense(self, chrom, which="test", *, mols=None,
                as_h5array=True, dtype=np.float64, batch_size=20000,
                mask_name=None):
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
            raise ValueError("'dtype' must be a floating dtype to "
                             f"represent missing values as NaN, got "
                             f"{dtype}")

        raw = self.raw[chrom]
        df = getattr(raw, f"{which}_data")
        mol_id = getattr(raw, f"{which}_mol_id")
        if df is None or mol_id is None:
            raise ValueError(
                f"No '{which}' data available for chrom '{chrom}'")
        nbp = raw.nbp
        all_pos = pd.Index(range(nbp))

        keep = None
        if mask_name is not None:
            keep = self._analysis[chrom][
                f"{which}_{mask_name}"]["keep"].to_numpy()

        mol_ids = np.arange(len(mol_id)) if mols is None \
            else np.atleast_1d(mols)
        if mols is not None:
            df = df[df["mol_index"].isin(mol_ids)]

        if not as_h5array:
            if len(mol_ids) > _DENSE_WARN_ROWS:
                warnings.warn(
                    f"Materializing {len(mol_ids)} molecules as a "
                    f"pd.DataFrame ({len(mol_ids)}x{nbp}); this may "
                    "use significant memory. Pass as_h5array=True to "
                    "stream to disk instead.", stacklevel=2)
            piv = df.pivot(index="mol_index", columns="pos",
                           values="mod_qual").reindex(index=mol_ids,
                                                      columns=all_pos)
            piv = piv.astype(dtype)
            if keep is not None:
                piv = _apply_keep_mask(piv, keep[mol_ids])
            return piv

        df = df.sort_values("mol_index", kind="stable")
        mol_index = df["mol_index"].to_numpy()
        nmol_out = len(mol_ids)
        out = H5Array.create((nmol_out, nbp), dtype=dtype,
                             index=(mol_ids if mols is not None
                                   else None),
                             dir=self.resolve_tmp_dir())
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
                index="mol_index", columns="pos",
                values="mod_qual").reindex(index=chunk_ids,
                                           columns=all_pos)
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
                "This experiment has no 'mtase' set (specify it via "
                "load_raw()); required to determine methylatable "
                "positions.")
        if mtase is None:
            return self._mtase
        labels = _normalize_mtase(mtase)
        unknown = set(labels) - set(self._mtase)
        if unknown:
            raise ValueError(
                "'mtase' contains label(s) not in this experiment's "
                f"mtase {list(self._mtase)}: {sorted(unknown)}.")
        return labels

    def filter_dropout(self, *, which: str | None = None,
                       mtase: list | None = None,
                       threshold: float = 0.2,
                       unmapped_strand: str = "union",
                       method: str = "separate",
                       mask_name: str = "dropout_mask") -> None:
        """
        Flag molecules with poor coverage at methylatable positions.

        Computes, per molecule, the fraction of methylatable positions
        (per `mtase` label, on that molecule's own strand) with no
        signal, and stores a keep/drop mask in `analysis`. Raw data is
        never modified -- pass `mask_name` to `to_dense`/
        `MethPrintAnalysis.smooth`/`.meth_prob` to apply it downstream.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to evaluate. None evaluates every source present
            for each chromosome.
        mtase : list of str, optional
            Subset of `self.mtase` labels to evaluate. None uses all
            of them.
        threshold : float, default 0.2
            Max allowed no-signal fraction (per label, or of the
            pooled total under `method="aggregate"`) to be kept.
        unmapped_strand : {"union", "drop", "+", "-"}, default "union"
            How to evaluate unmapped ('.') strand molecules: union of
            '+'/'-' position sets, always dropout, or treat as that
            strand.
        method : {"separate", "aggregate"}, default "separate"
            How multiple `mtase` labels combine into `keep`.
            "separate": must clear `threshold` per label. "aggregate":
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
            label not in `self.mtase`, `threshold` is not in [0, 1],
            `unmapped_strand`/`method` is invalid, a requested source
            is missing for some chromosome, or `refseq` is missing.
        """
        labels = self._resolve_mtase_subset(mtase)
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(
                f"'threshold' must be in [0, 1], got {threshold}.")
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}.")
        if method not in _VALID_FILTER_DROPOUT_METHOD:
            raise ValueError(
                "'method' must be one of "
                f"{sorted(_VALID_FILTER_DROPOUT_METHOD)}, got "
                f"{method!r}.")

        for chrom in self.chroms:
            raw = self.raw[chrom]
            if raw.refseq is None:
                raise ValueError(
                    f"No refseq available for chrom '{chrom}'; "
                    "required to determine methylatable positions.")
            sources = _resolve_sources(raw, which)

            for src in sources:
                df = getattr(raw, f"{src}_data")
                mol_id = getattr(raw, f"{src}_mol_id")
                if df is None or mol_id is None:
                    raise ValueError(
                        f"No '{src}' data available for chrom '{chrom}'")
                nmol = len(mol_id)
                cov = _label_coverage(
                    df, mol_id, raw.refseq, labels, unmapped_strand)
                covered = np.stack([cov[label][0] for label in labels])
                n_total = np.stack([cov[label][1] for label in labels])

                dropout_frac = np.where(
                    n_total > 0,
                    1.0 - covered / np.maximum(n_total, 1), 0.0)

                if method == "separate":
                    keep = (dropout_frac <= threshold).all(axis=0)
                else:
                    covered_sum = covered.sum(axis=0)
                    n_total_sum = n_total.sum(axis=0)
                    combined_frac = np.where(
                        n_total_sum > 0,
                        1.0 - covered_sum / np.maximum(n_total_sum, 1),
                        0.0)
                    keep = combined_frac <= threshold

                # Stored as int, not bool: bool isn't a numeric dtype
                # to h5_utils' save/load, so it would round-trip
                # through the string block as literal "True"/"False"
                # text -- and casting that back to bool makes every
                # non-empty string truthy, silently breaking the mask.
                self._analysis[chrom][f"{src}_{mask_name}"] = \
                    pd.DataFrame({"keep": keep.astype(np.int8)})

    def summarize_dropout(self, *, which: str | None = None,
                          mtase: list | None = None,
                          unmapped_strand: str = "union") -> pd.DataFrame:
        """
        Return a per-label dropout fraction summary (QC check).

        Same coverage computation as `filter_dropout`, reporting the
        per-molecule dropout fraction distribution instead of
        filtering. Nothing is stored in `analysis`.

        Parameters
        ----------
        which : {"test", "meth", "unmeth"} or None, default None
            Source(s) to summarize. None summarizes every source
            present for each chromosome.
        mtase : list of str, optional
            Subset of `self.mtase` labels to summarize. None uses all.
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
            If `mtase` is unset, contains an unknown label,
            `unmapped_strand` is invalid, a source is missing for
            some chromosome, or `refseq` is missing.
        """
        labels = self._resolve_mtase_subset(mtase)
        if unmapped_strand not in _VALID_UNMAPPED_STRAND:
            raise ValueError(
                "'unmapped_strand' must be one of "
                f"{sorted(_VALID_UNMAPPED_STRAND)}, got "
                f"{unmapped_strand!r}.")

        percentiles = [0, 25, 50, 75, 100]
        pct_cols = [f"p{p}" for p in percentiles]
        rows = []
        for chrom in self.chroms:
            raw = self.raw[chrom]
            if raw.refseq is None:
                raise ValueError(
                    f"No refseq available for chrom '{chrom}'; "
                    "required to determine methylatable positions.")
            sources = _resolve_sources(raw, which)

            for src in sources:
                df = getattr(raw, f"{src}_data")
                mol_id = getattr(raw, f"{src}_mol_id")
                if df is None or mol_id is None:
                    raise ValueError(
                        f"No '{src}' data available for chrom '{chrom}'")
                cov = _label_coverage(
                    df, mol_id, raw.refseq, labels, unmapped_strand)

                covered_list, total_list = [], []
                for label in labels:
                    covered, total = cov[label]
                    covered_list.append(covered)
                    total_list.append(total)
                    frac = np.where(
                        total > 0, 1.0 - covered / np.maximum(total, 1),
                        0.0)
                    plus_n = int(_methylatable_positions(
                        raw.refseq, label, "+").sum())
                    minus_n = int(_methylatable_positions(
                        raw.refseq, label, "-").sum())
                    row = {"chrom": chrom, "source": src, "label": label,
                           "n_sites_plus": plus_n,
                           "n_sites_minus": minus_n}
                    row.update(zip(pct_cols,
                                   np.percentile(frac, percentiles)))
                    rows.append(row)

                if len(labels) > 1:
                    covered_sum = np.sum(covered_list, axis=0)
                    total_sum = np.sum(total_list, axis=0)
                    frac = np.where(
                        total_sum > 0,
                        1.0 - covered_sum / np.maximum(total_sum, 1), 0.0)
                    row = {"chrom": chrom, "source": src,
                           "label": "aggregate",
                           "n_sites_plus": sum(int(_methylatable_positions(
                               raw.refseq, label, "+").sum())
                               for label in labels),
                           "n_sites_minus": sum(int(_methylatable_positions(
                               raw.refseq, label, "-").sum())
                               for label in labels)}
                    row.update(zip(pct_cols,
                                   np.percentile(frac, percentiles)))
                    rows.append(row)

        return pd.DataFrame(rows)

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
