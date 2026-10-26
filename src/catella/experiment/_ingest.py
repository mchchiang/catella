# _ingest.py

import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pyarrow_csv
import pyfaidx

from catella import h5_utils

# Column names as they appear in raw Modkit output, and their internal
# canonical equivalents (positionally parallel).
_MODKIT_COLNAMES = [
    "read_id",
    "ref_position",
    "chrom",
    "ref_strand",
    "mod_qual",
    "mod_code",
]
_CANONICAL_NAMES = [
    "mol_id",
    "upos",
    "chrom",
    "strand",
    "mod_qual",
    "mod_code",
]
_CANONICAL_PA_TYPES = {
    "mol_id": pa.string(),
    "upos": pa.int64(),
    "chrom": pa.string(),
    "strand": pa.string(),
    "mod_qual": pa.float64(),
    "mod_code": pa.string(),
}
# Fixed schema for the raw per-chromosome long-format tables, used by
# the streaming AppendableDF writer in load_raw().
_RAW_COLUMN_SPEC = {
    "mol_index": "numeric",
    "pos": "numeric",
    "strand": "string",
    "mod_qual": "numeric",
    "mod_code": "string",
}

# Recognized methyltransferase labels: 'A' (any-context adenine,
# e.g., EcoGII), 'CG' (CpG, e.g., M.SssI), 'GC' (GpC, e.g., M.CviPI).
_VALID_MTASE = {"A", "CG", "GC"}

# File extension -> pyarrow compression codec, for _decompress_once.
_COMPRESSION_EXTS = {
    ".gz": "gzip",
    ".bz2": "bz2",
    ".xz": "lzma",
    ".zst": "zstd",
}


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
            f"of {sorted(_VALID_MTASE)}."
        )
    if len(set(values)) != len(values):
        raise ValueError("Duplicate mtase values given.")
    return values


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
        mask |= arr == base
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


def _decompress_once(data_file, tmp_dir):
    """
    Inflate a compressed raw data file once to a local scratch copy, so
    later streamed passes over it do not each re-decompress it.

    Parameters
    ----------
    data_file : str or pathlib.Path
        The raw data file to be parsed. Only recognized compressed
        extensions (see `_COMPRESSION_EXTS`) trigger decompression;
        any other extension is left alone.
    tmp_dir : str or pathlib.Path
        Directory to create the scratch copy in.

    Returns
    -------
    str or None
        Path of the scratch copy, to be read from (in place of
        `data_file`) and removed once done with it. None if
        `data_file` was not a recognized compressed extension, so no
        copy was made and `data_file` should be read directly.
    """
    codec = _COMPRESSION_EXTS.get(Path(data_file).suffix)
    if codec is None:
        return None

    fd, scratch_path = tempfile.mkstemp(suffix=".tsv", dir=tmp_dir)
    os.close(fd)
    with (
        pa.input_stream(str(data_file), compression=codec) as src,
        open(scratch_path, "wb") as dst,
    ):
        while True:
            chunk = src.read(64 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
    return scratch_path


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
        raw_by_modkit = {
            str(c).lstrip("#").strip(): c for c in header_df.columns
        }
        name_for = {
            canon: raw_by_modkit[modkit]
            for modkit, canon in zip(_MODKIT_COLNAMES, _CANONICAL_NAMES)
            if modkit in raw_by_modkit
        }
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


def _iter_csv_chunks(
    data_file,
    header_names,
    name_for,
    sep,
    needed_canonical,
    chunk_size,
    block_size,
):
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
        block_size=block_size,
        column_names=header_names,
        autogenerate_column_names=False,
    )
    parse_opts = pyarrow_csv.ParseOptions(delimiter=sep)
    convert_opts = pyarrow_csv.ConvertOptions(
        include_columns=include, column_types=types
    )
    reader = pyarrow_csv.open_csv(
        str(data_file),
        read_options=read_opts,
        parse_options=parse_opts,
        convert_options=convert_opts,
    )
    rename = {name_for[c]: c for c in needed_canonical}

    buf, buf_rows = [], 0
    for batch in reader:
        buf.append(batch)
        buf_rows += batch.num_rows
        if buf_rows >= chunk_size:
            yield pa.Table.from_batches(buf).to_pandas().rename(columns=rename)
            buf, buf_rows = [], 0
    if buf:
        yield pa.Table.from_batches(buf).to_pandas().rename(columns=rename)


def _scan_accepted_mols(
    data_file,
    header_names,
    name_for,
    sep,
    chunk_size,
    block_size,
    max_nmol,
    rng,
):
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
        data_file,
        header_names,
        name_for,
        sep,
        ["chrom", "mol_id"],
        chunk_size,
        block_size,
    ):
        dedup = batch_df.drop_duplicates(["chrom", "mol_id"])
        for chrom, mol_id in zip(
            dedup["chrom"].to_numpy(), dedup["mol_id"].to_numpy()
        ):
            seen = mol_order.setdefault(chrom, {})
            if mol_id not in seen:
                seen[mol_id] = None

    result = {}
    for chrom in sorted(mol_order.keys()):
        mols = np.array(list(mol_order[chrom].keys()), dtype=object)
        if max_nmol is not None and len(mols) > max_nmol:
            mols = rng.choice(mols, max_nmol, replace=False)
        sorted_mols = np.sort(mols)
        result[chrom] = {
            "mol_id": sorted_mols,
            "index": {m: i for i, m in enumerate(sorted_mols)},
        }
    return result


def _stream_rows_to_staging(
    data_file,
    header_names,
    name_for,
    sep,
    chunk_size,
    block_size,
    full_sizes,
    wrap,
    mol_maps,
    appenders,
    refseq_by_chrom=None,
    mtase=None,
    ignore_strand=False,
):
    """
    Stream full rows, transform them, and append to per-chromosome
    disk-backed appenders.

    Rows with a canonical call (`mod_code` of '-', i.e. no methylation
    signal) are dropped, since the emission model only considers sites
    with an actual modification call.

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
    refseq_by_chrom : dict of str to str, optional
        Chromosome name -> reference sequence, as returned by
        `_parse_fasta`. Together with `mtase`, used to drop rows whose
        `strand` is inconsistent with the reference base at their
        (unfolded) `upos` -- e.g. a '+'-strand row at a position whose
        reference base cannot carry a real target base for any given
        `mtase` label on '+'. Chromosomes absent from this mapping (or
        when it is None) are not filtered.
    mtase : tuple of str, optional
        Methyltransferase label(s) (as normalized by
        `_normalize_mtase`) to validate rows against. If None, no
        strand-consistency filtering is applied.
    ignore_strand : bool, default False
        If True, a row is kept if its `upos` is methylatable by any
        `mtase` label on either strand, instead of only the row's own
        recorded `strand`.
    """
    valid_by_chrom = {}
    if mtase is not None and refseq_by_chrom is not None:
        for chrom, seq in refseq_by_chrom.items():
            plus = np.zeros(len(seq), dtype=bool)
            minus = np.zeros(len(seq), dtype=bool)
            for label in mtase:
                plus |= _methylatable_positions(seq, label, "+")
                minus |= _methylatable_positions(seq, label, "-")
            valid_by_chrom[chrom] = (plus, minus)

    needed = ["mol_id", "upos", "chrom", "strand", "mod_qual", "mod_code"]
    for batch_df in _iter_csv_chunks(
        data_file, header_names, name_for, sep, needed, chunk_size, block_size
    ):
        for chrom, group in batch_df.groupby("chrom", sort=False):
            mapping = mol_maps.get(chrom)
            if mapping is None:
                continue
            mol_index = group["mol_id"].map(mapping["index"])
            keep = mol_index.notna().to_numpy() & (
                group["mod_code"].to_numpy() != "-"
            )

            masks = valid_by_chrom.get(chrom)
            if masks is not None:
                plus, minus = masks
                upos_all = group["upos"].to_numpy()
                if ignore_strand:
                    consistent = (plus | minus)[upos_all]
                else:
                    strand_all = group["strand"].to_numpy()
                    # Unmapped strand ('.') is not validated -- kept.
                    consistent = np.where(
                        strand_all == "+",
                        plus[upos_all],
                        np.where(strand_all == "-", minus[upos_all], True),
                    )
                keep = keep & consistent

            if not keep.any():
                continue
            group = group.loc[keep]
            mol_index = mol_index[keep].astype(np.int64).to_numpy()

            upos = group["upos"].to_numpy()
            if wrap:
                length = full_sizes[chrom]
                pos = np.where(
                    upos > (length - 1) / 2, length - upos - 1, upos
                )
            else:
                pos = upos

            out = pd.DataFrame(
                {
                    "mol_index": mol_index,
                    "pos": pos.astype(np.int64),
                    "strand": group["strand"].to_numpy(),
                    "mod_qual": group["mod_qual"]
                    .astype(np.float64)
                    .to_numpy(),
                    "mod_code": group["mod_code"].to_numpy(),
                }
            )
            appenders[chrom].append(out)


def _ensure_chrom_data_group(graw, chrom, nbp, refseq_by_chrom):
    """
    Create a chromosome's `metadata` and `data` groups under `graw`.

    Each chromosome is created at most once per private staging file,
    so no idempotency check is needed here (unlike a shared file).

    Returns
    -------
    h5py.Group
        The new `data` group.
    """
    gchrom = graw.create_group(chrom)
    gmeta = gchrom.create_group("metadata")
    gmeta.attrs["chrom"] = chrom
    gmeta.attrs["nbp"] = int(nbp)
    seq = refseq_by_chrom.get(chrom)
    if seq is not None:
        dt = h5py.string_dtype(encoding="utf-8")
        gmeta.create_dataset("refseq", data=seq, dtype=dt)
    return gchrom.create_group("data")


def _ingest_into(
    graw,
    name,
    data_file,
    *,
    sep,
    colidx,
    chunk_size,
    block_size,
    sizes,
    full_sizes,
    wrap,
    refseq_by_chrom,
    mtase,
    ignore_strand,
    max_nmol,
    rng,
    tmp_dir,
):
    """
    Resolve, scan, and stream one raw data file into the `raw_data`
    group of a staging HDF5 file.

    Parameters
    ----------
    graw : h5py.Group
        The `raw_data` group to write into.
    name : str
        Source name: `"test"`, `"unmeth"`, or `"meth"`.
    data_file : str or pathlib.Path
        The raw data file to be parsed.
    rng : numpy.random.Generator or None
        Random number generator for downsampling; see
        `MethPrintExperiment.load_raw`'s `seed` parameter.
    tmp_dir : str or pathlib.Path
        Directory for any decompression scratch copy.
    Other parameters are as in `MethPrintExperiment.load_raw`.

    Returns
    -------
    set of str
        Chromosomes found in the file, already filtered to `sizes`.
    """
    print(f"Reading {data_file} ...")
    scratch_path = _decompress_once(data_file, tmp_dir)
    read_path = scratch_path or str(data_file)
    try:
        header_names, name_for = _resolve_modkit_schema(read_path, sep, colidx)
        mol_maps = _scan_accepted_mols(
            read_path,
            header_names,
            name_for,
            sep,
            chunk_size,
            block_size,
            max_nmol,
            rng,
        )
        # Chromosomes not covered by the chromsize table are silently
        # dropped, matching the legacy inner merge
        mol_maps = {c: m for c, m in mol_maps.items() if c in sizes}

        dt = h5py.string_dtype(encoding="utf-8")
        appenders = {}
        for chrom, m in mol_maps.items():
            gdata = _ensure_chrom_data_group(
                graw, chrom, sizes[chrom], refseq_by_chrom
            )
            gdata.create_dataset(f"{name}_mol_id", data=m["mol_id"], dtype=dt)
            appenders[chrom] = h5_utils.AppendableDF(
                gdata,
                f"{name}_data",
                columns=list(_RAW_COLUMN_SPEC),
                dtypes=_RAW_COLUMN_SPEC,
            )

        _stream_rows_to_staging(
            read_path,
            header_names,
            name_for,
            sep,
            chunk_size,
            block_size,
            full_sizes,
            wrap,
            mol_maps,
            appenders,
            refseq_by_chrom=refseq_by_chrom,
            mtase=mtase,
            ignore_strand=ignore_strand,
        )
    finally:
        if scratch_path is not None:
            os.remove(scratch_path)
    return set(mol_maps.keys())


def _ingest_file(name, data_file, *, tmp_dir, **kwargs):
    """
    Ingest one raw data file into its own private staging HDF5 file,
    independent of any other source file.

    Used when more than one of `test_file`/`unmeth_file`/`meth_file`
    is given to `MethPrintExperiment.load_raw`, so each file's read
    can proceed with zero shared mutable state, whether called
    sequentially or concurrently from a thread pool.

    Parameters
    ----------
    name : str
        Source name: `"test"`, `"unmeth"`, or `"meth"`.
    data_file : str or pathlib.Path
        The raw data file to be parsed.
    tmp_dir : str or pathlib.Path
        Directory for the private staging file and any decompression
        scratch copy.
    **kwargs
        Forwarded to `_ingest_into`.

    Returns
    -------
    name : str
        Echoes the `name` argument.
    chroms : set of str
        Chromosomes found in the file, already filtered to `sizes`.
    private_path : str
        Path to the private staging file, to be merged and removed by
        the caller.
    """
    fd, private_path = tempfile.mkstemp(suffix=f".{name}.h5", dir=tmp_dir)
    os.close(fd)
    with h5py.File(private_path, "w") as h5stream:
        chroms = _ingest_into(
            h5stream.create_group("raw_data"),
            name,
            data_file,
            tmp_dir=tmp_dir,
            **kwargs,
        )
    return name, chroms, private_path


def _ingest_sources(files, *, nworker, seed, **kwargs):
    """
    Ingest several raw data files into private staging files, each
    with an independent random number generator.

    Parameters
    ----------
    files : list of (str, str or pathlib.Path)
        `(name, data_file)` per source, test first.
    nworker : int
        Maximum number of threads reading files concurrently.
    seed : int or None
        Seed spawning one independent generator per file, used only
        if `kwargs["max_nmol"]` is not None.
    **kwargs
        Forwarded to `_ingest_file`.

    Returns
    -------
    list of (str, set of str, str)
        `_ingest_file`'s result per source, in the order of `files`.
    """
    if kwargs["max_nmol"] is None:
        rngs = [None] * len(files)
    else:
        children = np.random.SeedSequence(seed).spawn(len(files))
        rngs = [np.random.default_rng(child) for child in children]
    tasks = [(name, f, rng) for (name, f), rng in zip(files, rngs)]

    def ingest(task):
        name, data_file, rng = task
        return _ingest_file(name, data_file, rng=rng, **kwargs)

    if nworker > 1:
        workers = min(nworker, len(tasks))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(ingest, tasks))
    return [ingest(task) for task in tasks]


def _merge_staging(graw, ingested):
    """
    Copy every source's private staging data into `graw`, then remove
    the private files.

    Parameters
    ----------
    graw : h5py.Group
        The `raw_data` group of the shared staging file.
    ingested : list of (str, set of str, str)
        From `_ingest_sources`, test first.

    Raises
    ------
    ValueError
        If a control file covers different chromosomes than the test
        file.
    """
    test_chroms = ingested[0][1]
    for name, chroms_found, _ in ingested[1:]:
        if chroms_found != test_chroms:
            raise ValueError(
                f"Different number of chromosomes in test and {name} datasets"
            )

    handles = {name: h5py.File(path, "r") for name, _, path in ingested}
    try:
        for chrom in sorted(test_chroms):
            gchrom = graw.create_group(chrom)
            handles["test"].copy(
                f"raw_data/{chrom}/metadata", gchrom, name="metadata"
            )
            gdata = gchrom.create_group("data")
            for name, _, _ in ingested:
                src = handles[name][f"raw_data/{chrom}/data"]
                src.copy(f"{name}_mol_id", gdata, name=f"{name}_mol_id")
                src.copy(f"{name}_data", gdata, name=f"{name}_data")
    finally:
        for h in handles.values():
            h.close()
    for _, _, path in ingested:
        os.remove(path)


def _read_chromsize(chromsize, wrap, chroms):
    """
    Read and validate a chromosome-size table.

    Parameters
    ----------
    chromsize : str or pathlib.Path
        Tab-separated file of chromosome names and lengths (bp), with
        or without a `chrom`/`length` header row.
    wrap : bool
        If True, the storage lengths are halved (positions are folded
        around the fiber center).
    chroms : list of str or None
        Chromosomes to keep; None keeps all of them.

    Returns
    -------
    full_sizes : dict of str to int
        Chromosome -> full length (bp).
    sizes : dict of str to int
        Chromosome -> storage length (bp), halved if `wrap`.

    Raises
    ------
    ValueError
        If a column is missing, chromosome names are not unique, or
        `chroms` has a chromosome not in the table.
    """
    size_cols = ["chrom", "length"]
    with open(chromsize, "r") as reader:
        first_line = reader.readline().strip()
    if first_line == "\t".join(size_cols):
        df_size = pd.read_csv(chromsize, sep="\t")
    else:
        df_size = pd.read_csv(
            chromsize, sep="\t", names=size_cols, header=None
        )
    if "chrom" not in df_size.columns:
        raise ValueError(
            "Column 'chrom' containing the chromosome identifier is missing."
        )
    if "length" not in df_size.columns:
        raise ValueError(
            "Column 'length' containing the chromosome length is missing."
        )
    if not df_size["chrom"].is_unique:
        raise ValueError("Chromosomes must be unique in chromsize file")

    full_sizes = dict(zip(df_size["chrom"], df_size["length"].astype(int)))
    if wrap:
        sizes = dict(
            zip(df_size["chrom"], (df_size["length"] / 2).astype(int))
        )
    else:
        sizes = full_sizes

    if chroms is not None:
        unknown = set(chroms) - set(df_size["chrom"])
        if unknown:
            raise ValueError(
                f"Chromosomes not found in chromsize: {sorted(unknown)}"
            )
        full_sizes = {c: v for c, v in full_sizes.items() if c in chroms}
        sizes = {c: v for c, v in sizes.items() if c in chroms}
    return full_sizes, sizes


def _load_refseqs(fasta_file, full_sizes):
    """
    Reference sequence per chromosome of `full_sizes`, checked against
    its length.

    Parameters
    ----------
    fasta_file : str or pathlib.Path or None
        Multi-FASTA file; None returns an empty dict.
    full_sizes : dict of str to int
        Chromosome -> full length (bp).

    Returns
    -------
    dict of str to str
        Chromosome -> sequence, for the chromosomes found in the file.

    Raises
    ------
    ValueError
        If a sequence length does not match `full_sizes`.
    """
    if fasta_file is None:
        return {}
    refseq_by_chrom = _parse_fasta(fasta_file, set(full_sizes))
    for chrom, seq in refseq_by_chrom.items():
        if len(seq) != full_sizes[chrom]:
            raise ValueError(
                f"Reference sequence length for chromosome "
                f"{chrom!r} ({len(seq)}) does not match its "
                f"chromsize length ({full_sizes[chrom]})."
            )
    return refseq_by_chrom
