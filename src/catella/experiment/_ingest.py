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
from catella._regions import Region, _RegionMap
from catella._warnings import _warn

# Column names as they appear in raw Modkit output, and their internal
# canonical equivalents (positionally parallel).
_MODKIT_COLNAMES = [
    "read_id",
    "ref_position",
    "chrom",
    "ref_strand",
    "mod_qual",
    "mod_code",
    "alignment_start",
    "alignment_end",
    "flag",
    "forward_read_position",
]
_CANONICAL_NAMES = [
    "mol_id",
    "upos",
    "chrom",
    "strand",
    "mod_qual",
    "mod_code",
    "align_start",
    "align_end",
    "flag",
    "read_pos",
]
# Canonical columns a file may lack; the read extent then falls back
# to the first and last row of each molecule.
_OPTIONAL_NAMES = ("align_start", "align_end")
# Columns `colidx` can map (6 required, or 8 with the alignment
# columns); later ones are read only from a Modkit header.
_COLIDX_NAMES = _CANONICAL_NAMES[:8]
# Optional per-molecule datasets stored as `{src}_{key}` next to
# `{src}_mol_id` (read extents and internal gaps)
_PER_MOL_KEYS = ("mol_start", "mol_end", "gaps", "gap_offset")
# Genomic coordinates stored as attrs of each region's metadata group
_COORD_KEYS = ("ref_chrom", "ref_start", "ref_end", "ref_strand")
# SAM flags of secondary and supplementary alignments
_NON_PRIMARY_FLAGS = 256 | 2048
_CANONICAL_PA_TYPES = {
    "mol_id": pa.string(),
    "upos": pa.int64(),
    "chrom": pa.string(),
    "strand": pa.string(),
    "mod_qual": pa.float64(),
    "mod_code": pa.string(),
    "align_start": pa.int64(),
    "align_end": pa.int64(),
    "flag": pa.int64(),
    "read_pos": pa.int64(),
}
# Fixed schema for the raw per-region long-format tables, used by
# the streaming AppendableDF writer in load_raw().
_RAW_COLUMN_SPEC = {
    "mol_index": "integer",
    "pos": "integer",
    "strand": "S1",
    "mod_qual": "numeric",
    "mod_code": "S8",
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
            f"unknown mtase value(s) {unknown}; must be one or more "
            f"of {sorted(_VALID_MTASE)}"
        )
    if len(set(values)) != len(values):
        raise ValueError("duplicate mtase values given")
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


# Modkit mod codes (letter and ChEBI) of the mark each mtase deposits
_MTASE_MOD_CODES = {
    "A": ("a", "28871"),
    "CG": ("m", "27551"),
    "GC": ("m", "27551"),
}


def _region_labels(mtase, refseq):
    """
    Accepted mod codes and methylatable positions of each label.

    Parameters
    ----------
    mtase : tuple of str
        Normalized methyltransferase labels.
    refseq : str or None
        Reference sequence of the region; None skips the base check.

    Returns
    -------
    list of (tuple of str, (np.ndarray, np.ndarray) or None)
        Per label, its mod codes and its methylatable positions on '+'
        and '-' (None without `refseq`).
    """
    labels = []
    for label in mtase:
        masks = None
        if refseq is not None:
            masks = (
                _methylatable_positions(refseq, label, "+"),
                _methylatable_positions(refseq, label, "-"),
            )
        labels.append((_MTASE_MOD_CODES[label], masks))
    return labels


def _site_consistent(masks, upos, strand, ignore_strand):
    """Bool mask of rows methylatable at their position and strand."""
    plus, minus = masks
    if ignore_strand:
        return (plus | minus)[upos]
    # Unmapped strand ('.') is not validated -- kept.
    return np.where(
        strand == "+", plus[upos], np.where(strand == "-", minus[upos], True)
    )


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
        `upos`, etc). Six entries map the required columns; eight also
        map `alignment_start`/`alignment_end`. If None, the file is
        assumed to have a header row with Modkit's own column names.

    Returns
    -------
    header_names : list of str or None
        Value for `pyarrow.csv.ReadOptions.column_names`. None means
        the file has its own header row for pyarrow to consume.
    name_for : dict of str to str
        Canonical field name -> the column name pyarrow will parse it
        as. Optional columns absent from the file are left out; `flag`
        and `read_pos` are only mapped from a header.

    Raises
    ------
    ValueError
        If `colidx` has neither 6 nor 8 entries.
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

    n_required = len(_COLIDX_NAMES) - len(_OPTIONAL_NAMES)
    if len(colidx) not in (n_required, len(_COLIDX_NAMES)):
        raise ValueError(
            f"colidx must have {n_required} or {len(_COLIDX_NAMES)} "
            f"entries, got {len(colidx)}"
        )
    with open(data_file, "r") as fh:
        first_line = fh.readline()
    ncols = len(first_line.rstrip("\n").split(sep))
    header_names = [f"_col{i}" for i in range(ncols)]
    for canon, idx in zip(_COLIDX_NAMES, colidx):
        header_names[idx] = canon
    name_for = {canon: canon for canon in _COLIDX_NAMES[: len(colidx)]}
    return header_names, name_for


def _ensure_fasta_index(fasta_file):
    """
    Build the `.fai` index of a FASTA file if it is missing or stale.

    The index is written under a unique temporary name in the same
    directory and then moved into place, so processes reading the
    same FASTA file concurrently never see a partially written index.
    A fresh index is never replaced, since Windows refuses to open a
    file while another process replaces it.

    Parameters
    ----------
    fasta_file : str or pathlib.Path
        FASTA file; its index is `<fasta_file>.fai`.

    Raises
    ------
    PermissionError
        If a stale index cannot be replaced and no fresh index exists.
    """
    fasta_file = os.path.abspath(fasta_file)
    fai = fasta_file + ".fai"
    if _fresh_index(fasta_file, fai):
        return
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(fai) + ".",
        suffix=".tmp",
        dir=os.path.dirname(fai),
    )
    os.close(fd)
    os.remove(tmp)
    try:
        pyfaidx.Faidx(fasta_file, indexname=tmp).close()
        _install_index(fasta_file, tmp, fai)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _install_index(fasta_file, tmp, fai):
    """
    Move a newly built index into place unless a fresh one exists.

    Parameters
    ----------
    fasta_file : str
        FASTA file the index belongs to.
    tmp : str
        The newly built index.
    fai : str
        The index path.

    Raises
    ------
    PermissionError
        If a stale index cannot be replaced and no fresh index exists.
    """
    if not os.path.exists(fai):
        try:
            # Unlike os.replace, os.rename never overwrites on Windows
            os.rename(tmp, fai)
        except FileExistsError:
            pass  # Another process installed its index first
        return
    if _fresh_index(fasta_file, fai):
        return
    try:
        os.replace(tmp, fai)
    except PermissionError:
        if not _fresh_index(fasta_file, fai):
            raise


def _fresh_index(fasta_file, fai):
    """Return True if `fai` exists and is not older than `fasta_file`."""
    return os.path.exists(fai) and (
        os.path.getmtime(fai) >= os.path.getmtime(fasta_file)
    )


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
    _ensure_fasta_index(fasta_file)
    fasta = pyfaidx.Fasta(str(fasta_file), as_raw=True, build_index=True)
    try:
        for chrom in wanted_chroms:
            if chrom in fasta:
                sequences[chrom] = str(fasta[chrom][:])
    finally:
        fasta.close()
    return sequences


def _load_region_refseqs(fasta_file, regions, chrom_sizes):
    """
    Reference sequence of each region, sliced from its chromosome.

    Parameters
    ----------
    fasta_file : str or pathlib.Path or None
        Multi-FASTA file; None returns an empty dict.
    regions : list of Region
        Regions given as intervals.
    chrom_sizes : dict of str to int
        Chromosome -> length (bp), checked against the FASTA.

    Returns
    -------
    dict of str to str
        Region name -> sequence, for regions whose chromosome is in
        the file.

    Raises
    ------
    ValueError
        If a chromosome's sequence length does not match
        `chrom_sizes`.
    """
    if fasta_file is None:
        return {}
    _ensure_fasta_index(fasta_file)
    fasta = pyfaidx.Fasta(str(fasta_file), as_raw=True, build_index=True)
    sequences = {}
    try:
        for r in regions:
            if r.chrom not in fasta:
                continue
            length = len(fasta[r.chrom])
            if length != chrom_sizes[r.chrom]:
                raise ValueError(
                    f"reference sequence length for chromosome "
                    f"{r.chrom!r} ({length}) does not match its "
                    f"chromsize length ({chrom_sizes[r.chrom]})"
                )
            sequences[r.name] = str(fasta[r.chrom][r.start : r.end])
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


def _is_primary(df):
    """Bool mask of rows from primary alignments, per the `flag` column."""
    return (df["flag"].to_numpy() & _NON_PRIMARY_FLAGS) == 0


def _scan_accepted_mols(
    data_file,
    header_names,
    name_for,
    sep,
    chunk_size,
    block_size,
    max_nmol,
    rng,
    region_map,
    balance_strand=False,
    chrom_sizes=None,
    extents=True,
    min_overlap=0,
):
    """
    Determine, per region, which molecules are kept, their
    `mol_index`, and their read extents.

    A molecule belongs to a region if it has a row in it. Regions are
    sampled in sorted order, `rng.choice(mols, max_nmol,
    replace=False)` only when a region has more than `max_nmol`, and
    `mol_index` is the rank of each accepted `mol_id`. Extents use all
    of a molecule's rows on the chromosome, then are clipped and
    shifted to each region.

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
        Maximum number of molecules to keep per region.
    rng : np.random.Generator or None
        Random number generator for downsampling; required if
        `max_nmol` is not None.
    region_map : _RegionMap
        The regions to keep. Rows outside every region are skipped
        before sampling, so they neither warn nor consume `rng` draws.
    balance_strand : bool, default False
        If True, keep an equal number of '+' and '-' strand molecules
        per region (see `_balanced_mols`); requires `max_nmol`.
    chrom_sizes : dict of str to int, optional
        Reference chromosome -> length (bp); rows at or beyond it do
        not count toward extents. Chromosomes absent from it (or all,
        if None) are only bounded below by 0.
    extents : bool, default True
        Whether to compute read extents (False with `wrap=True`).
    min_overlap : int, default 0
        Molecules whose extent overlaps a region by fewer bp are
        dropped from it before sampling; requires `extents`. Regions
        left without molecules are omitted.

    Returns
    -------
    dict of str to dict
        `{region: {"mol_id", "index", "start", "end"}}`: the sorted
        accepted mol_ids, mol_id -> mol_index, and the read extent
        `[start, end)` of each molecule in region coordinates (int64;
        `(0, 0)` without an in-bounds row), or None for `start`/`end`
        if not `extents`. Covers every region with a row in the file.
    """
    has_align = all(c in name_for for c in _OPTIONAL_NAMES)
    cols = ["chrom", "mol_id", "upos"]
    if balance_strand:
        cols.append("strand")
    if "flag" in name_for:
        cols.append("flag")
    if extents and has_align:
        cols += list(_OPTIONAL_NAMES)
    mol_order = {}
    extent_of = {}
    for batch_df in _iter_csv_chunks(
        data_file,
        header_names,
        name_for,
        sep,
        cols,
        chunk_size,
        block_size,
    ):
        if "flag" in batch_df:
            batch_df = batch_df[_is_primary(batch_df)]
        batch_df = batch_df[batch_df["chrom"].isin(region_map.chroms)]
        if extents:
            _merge_chunk_extents(extent_of, batch_df, chrom_sizes, has_align)
        dedup = _split_rows(region_map, batch_df).drop_duplicates(
            ["chrom", "mol_id"]
        )
        strands = (
            dedup["strand"].to_numpy()
            if balance_strand
            else [None] * len(dedup)
        )
        for region, mol_id, strand in zip(
            dedup["chrom"].to_numpy(), dedup["mol_id"].to_numpy(), strands
        ):
            seen = mol_order.setdefault(region, {})
            if mol_id not in seen:
                seen[mol_id] = strand

    result = {}
    for region in sorted(mol_order.keys()):
        candidates = mol_order[region]
        if min_overlap:
            reg = region_map.regions[region]
            ids = list(candidates)
            start, end = _extent_arrays(
                extent_of.get(reg.chrom, {}), ids, reg
            )
            candidates = {
                m: candidates[m]
                for m, ok in zip(ids, end - start >= min_overlap)
                if ok
            }
            if not candidates:
                continue
        if balance_strand:
            mols = _balanced_mols(region, candidates, max_nmol, rng)
        else:
            mols = np.array(list(candidates.keys()), dtype=object)
            if max_nmol is not None and len(mols) > max_nmol:
                mols = rng.choice(mols, max_nmol, replace=False)
        sorted_mols = np.sort(mols)
        start = end = None
        if extents:
            reg = region_map.regions[region]
            start, end = _extent_arrays(
                extent_of.get(reg.chrom, {}), sorted_mols, reg
            )
        result[region] = {
            "mol_id": sorted_mols,
            "index": {m: i for i, m in enumerate(sorted_mols)},
            "start": start,
            "end": end,
        }
    return result


def _split_rows(region_map, df):
    """
    Rows of `df` copied into each region holding them (see
    `_RegionMap.split`), chromosome by chromosome.
    """
    parts = [
        region_map.split(group, chrom)
        for chrom, group in df.groupby("chrom", sort=False)
    ]
    return pd.concat(parts) if parts else df.iloc[:0]


def _merge_chunk_extents(extent_of, batch_df, chrom_sizes, has_align):
    """
    Widen per-molecule extents in place with one chunk of rows.

    Parameters
    ----------
    extent_of : dict of str to dict
        Reference chromosome -> mol_id -> `(start, end)`, updated in
        place.
    batch_df : pd.DataFrame
        Rows with `chrom`, `mol_id`, `upos` and, if `has_align`,
        `align_start`/`align_end` columns.
    chrom_sizes : dict of str to int or None
        Chromosome -> length (bp); rows with `upos` outside
        `[0, length)` are ignored. Chromosomes without a length are
        only bounded below by 0.
    has_align : bool
        Whether to use the alignment columns instead of `upos`.
    """
    upos = batch_df["upos"].to_numpy()
    length = (
        batch_df["chrom"].map(chrom_sizes).fillna(np.inf).to_numpy()
        if chrom_sizes
        else np.inf
    )
    rows = batch_df[(upos >= 0) & (upos < length)]
    if rows.empty:
        return
    if has_align:
        lo, hi = rows["align_start"], rows["align_end"]
    else:
        lo, hi = rows["upos"], rows["upos"] + 1
    agg = (
        pd.DataFrame(
            {
                "chrom": rows["chrom"],
                "mol_id": rows["mol_id"],
                "lo": lo,
                "hi": hi,
            }
        )
        .groupby(["chrom", "mol_id"], sort=False)
        .agg(lo=("lo", "min"), hi=("hi", "max"))
    )
    for (chrom, mol_id), lo_, hi_ in zip(
        agg.index, agg["lo"].to_numpy(), agg["hi"].to_numpy()
    ):
        by_mol = extent_of.setdefault(chrom, {})
        cur = by_mol.get(mol_id)
        if cur is not None:
            lo_, hi_ = min(cur[0], lo_), max(cur[1], hi_)
        by_mol[mol_id] = (int(lo_), int(hi_))


def _extent_arrays(extent_of_chrom, mol_ids, region):
    """
    Read extents of `mol_ids` in the coordinates of `region`.

    Parameters
    ----------
    extent_of_chrom : dict of str to (int, int)
        mol_id -> `(start, end)` on the region's chromosome, from
        `_merge_chunk_extents`.
    mol_ids : sequence of str
        Molecules, in `mol_index` order.
    region : Region
        The region; extents are shifted by its start and clipped to
        its length.

    Returns
    -------
    start, end : np.ndarray
        int64; `(0, 0)` for a molecule without an in-bounds row.
    """
    missing = (region.end, region.start)
    start = np.array(
        [extent_of_chrom.get(m, missing)[0] for m in mol_ids],
        dtype=np.int64,
    )
    end = np.array(
        [extent_of_chrom.get(m, missing)[1] for m in mol_ids],
        dtype=np.int64,
    )
    return _clip_extents(
        start - region.start, end - region.start, region.end - region.start
    )


def _balanced_mols(region, strand_of, max_nmol, rng):
    """
    Draw an equal number of '+' and '-' strand molecules.

    Each strand contributes `k = min(n_plus, n_minus, max_nmol // 2)`
    molecules, drawn without replacement ('+' first, then '-').
    Molecules with any other strand (e.g., '.') are dropped.

    Parameters
    ----------
    region : str
        Region name, used only in the warning message.
    strand_of : dict of str to str
        Molecule id -> strand, in file-encounter order.
    max_nmol : int
        Maximum total number of molecules; at least 2.
    rng : np.random.Generator
        Random number generator for the draws.

    Returns
    -------
    np.ndarray
        Accepted molecule ids (object dtype, unsorted).

    Warns
    -----
    UserWarning
        If either strand has fewer than `max_nmol // 2` molecules.
    """
    pools = {
        s: np.array([m for m, t in strand_of.items() if t == s], dtype=object)
        for s in ("+", "-")
    }
    half = max_nmol // 2
    k = min(len(pools["+"]), len(pools["-"]), half)
    if k < half:
        short = "+" if len(pools["+"]) < len(pools["-"]) else "-"
        _warn(
            f"only {len(pools[short])} '{short}' strand molecules on "
            f"{region}; kept {k} per strand"
        )
    return np.concatenate(
        [rng.choice(pools[s], k, replace=False) for s in ("+", "-")]
    )


def _stream_rows_to_staging(
    data_file,
    header_names,
    name_for,
    sep,
    chunk_size,
    block_size,
    wrap,
    mol_maps,
    appenders,
    region_map,
    src_maps,
    chrom_sizes=None,
    refseq_by_region=None,
    mtase=None,
    ignore_strand=False,
    min_gap=None,
):
    """
    Stream full rows, transform them, and append to per-region
    disk-backed appenders, inferring internal gaps along the way.

    Rows with a canonical call (`mod_code` of '-', i.e. no methylation
    signal) are dropped, since the emission model only considers sites
    with an actual modification call. With `mtase`, a row is kept only
    if some label accepts both its `mod_code` and its position. Calls
    of different mod codes at one read and position are summed into
    one row (see `_merge_codes`).

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
    wrap : bool
        If True, positions are folded around the region center.
    mol_maps : dict of str to dict
        As returned by `_scan_accepted_mols`, already filtered down to
        the regions actually being processed.
    appenders : dict of str to h5_utils.AppendableDF
        One appender per region in `mol_maps`, already created.
    region_map : _RegionMap
        Assigns rows to regions; region lengths are used for `wrap`.
    src_maps : dict of str to dict
        Per reference chromosome, the accepted molecules of all its
        regions (see `_source_maps`); gaps are tracked in this index.
    chrom_sizes : dict of str to int, optional
        Reference chromosome -> length (bp), for the bounds check;
        without a length, only negative `upos` are dropped.
    refseq_by_region : dict of str to str, optional
        Region name -> reference sequence, as returned by
        `_load_refseqs`. Together with `mtase`, used to drop rows whose
        `strand` is inconsistent with the reference base at their
        (unfolded) `upos` -- e.g. a '+'-strand row at a position whose
        reference base cannot carry a real target base for any given
        `mtase` label on '+'. Regions absent from this mapping (or
        when it is None) are not filtered.
    mtase : tuple of str, optional
        Methyltransferase label(s) (as normalized by
        `_normalize_mtase`) to validate rows against (mod code and
        position, see `_MTASE_MOD_CODES`). If None, no row is dropped
        for its mod code or position.
    ignore_strand : bool, default False
        If True, a row is kept if its `upos` is methylatable by any
        `mtase` label on either strand, instead of only the row's own
        recorded `strand`.
    min_gap : int, optional
        Minimum net deletion (bp) recorded as an internal gap (see
        `_GapTracker`). None records no gaps.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray) or None
        Reference chromosome -> `(gaps, offset)` from
        `_GapTracker.result`, in chromosome coordinates and the
        `src_maps` index, inferred from all rows before the `mod_code`
        and strand filters. None if `wrap`, if `min_gap` is None, or if
        the file has no `read_pos` column.

    Warns
    -----
    UserWarning
        If rows have `upos` outside the chromosome (e.g., -1 for
        unaligned bases), or come from secondary or supplementary
        alignments (per a `flag` column); those rows are dropped. Also
        if some molecules' rows are not ordered along the reference;
        those molecules get no gaps. Also if calls at positions
        methylatable by `mtase` are dropped for their mod code.

    Raises
    ------
    ValueError
        If `mtase` is given and the file has calls of accepted
        molecules, but none with a mod code of `mtase`. Also if a read
        has two calls of the same mod code at one position.
    """
    n_out = {}
    trackers = None
    if not wrap and min_gap is not None and "read_pos" in name_for:
        trackers = {
            chrom: _GapTracker(len(m["mol_id"]), min_gap)
            for chrom, m in src_maps.items()
        }
    labels_by_region = {}
    if mtase is not None:
        refseqs = refseq_by_region or {}
        labels_by_region = {
            region: _region_labels(mtase, refseqs.get(region))
            for region in mol_maps
        }
    stats = _CodeStats()

    needed = ["mol_id", "upos", "chrom", "strand", "mod_qual", "mod_code"]
    if "flag" in name_for:
        needed.append("flag")
    if trackers is not None:
        needed.append("read_pos")
    n_non_primary = 0
    for batch_df in _whole_bases(
        _iter_csv_chunks(
            data_file,
            header_names,
            name_for,
            sep,
            needed,
            chunk_size,
            block_size,
        )
    ):
        if "flag" in batch_df:
            primary = _is_primary(batch_df)
            n_non_primary += int((~primary).sum())
            batch_df = batch_df[primary]
        for chrom, group in batch_df.groupby("chrom", sort=False):
            if chrom not in src_maps:
                continue
            upos_all = group["upos"].to_numpy()
            length = (chrom_sizes or {}).get(chrom)
            in_bounds = upos_all >= 0
            if length is not None:
                in_bounds &= upos_all < length
            if not in_bounds.all():
                n_out[chrom] = n_out.get(chrom, 0) + int((~in_bounds).sum())
                group = group.loc[in_bounds]
            if trackers is not None:
                src_index = group["mol_id"].map(src_maps[chrom]["index"])
                mapped = src_index.notna().to_numpy()
                trackers[chrom].update(
                    src_index.to_numpy()[mapped].astype(np.int64),
                    group["upos"].to_numpy()[mapped].astype(np.int64),
                    group["read_pos"].to_numpy()[mapped].astype(np.int64),
                )
            split = region_map.split(group, chrom)
            for region, rows in split.groupby("chrom", sort=False):
                if region not in mol_maps:
                    continue
                reg = region_map.regions[region]
                out = _stage_region_rows(
                    rows,
                    mol_maps[region]["index"],
                    labels_by_region.get(region),
                    ignore_strand,
                    reg.end - reg.start if wrap else None,
                    stats,
                )
                if out is not None:
                    appenders[region].append(out)

    stats.report(data_file, mtase)
    if n_non_primary:
        _warn(
            f"dropped {n_non_primary} row(s) of secondary or "
            f"supplementary alignments in {data_file}"
        )
    for chrom, n in n_out.items():
        length = (chrom_sizes or {}).get(chrom)
        bound = f"[0, {length})" if length is not None else "[0, ...)"
        _warn(
            f"dropped {n} row(s) on chrom '{chrom}' with ref_position "
            f"outside {bound}"
        )
    if trackers is None:
        return None
    gaps = {}
    n_unordered = 0
    for chrom, tracker in trackers.items():
        length = (chrom_sizes or {}).get(chrom, np.iinfo(np.int64).max)
        flat, offset, n_bad = tracker.result(length)
        gaps[chrom] = (flat, offset)
        n_unordered += n_bad
    if n_unordered:
        _warn(
            f"{n_unordered} molecule(s) in {data_file} have rows not "
            "ordered along the reference; no internal gaps are inferred "
            "for them (sort the file by read and reference position, or "
            "pass gaps_bed)"
        )
    return gaps


def _whole_bases(chunks):
    """
    Re-chunk rows so the rows of one read at one position (e.g., its
    calls of several mod codes) are never split across chunks.

    Parameters
    ----------
    chunks : iterable of pd.DataFrame
        Row-chunks with `mol_id` and `upos` columns.

    Yields
    ------
    pd.DataFrame
        The same rows in the same order; each chunk's trailing rows of
        its last read and position are moved to the next chunk.
    """
    carry = None
    for df in chunks:
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        if len(df) == 0:
            continue
        mol_id = df["mol_id"].to_numpy()
        upos = df["upos"].to_numpy()
        last = (mol_id == mol_id[-1]) & (upos == upos[-1])
        head = 0 if last.all() else len(df) - int(np.argmin(last[::-1]))
        carry = df.iloc[head:]
        if head:
            yield df.iloc[:head]
    if carry is not None and len(carry):
        yield carry


class _CodeStats:
    """
    Counts of the calls of one file, for its mod code checks.

    Attributes
    ----------
    nseen : int
        Calls (not '-') of accepted molecules, counted with `mtase`.
    nknown : int
        Of those, calls whose mod code some label accepts.
    codes : set of str
        Mod codes of the calls seen.
    dropped : dict of str to int
        Per mod code, calls at a position methylatable by some label
        but dropped for their mod code.
    nrepeat : int
        Kept calls repeating the mod code of another call at the same
        read and position.
    """

    def __init__(self):
        self.nseen = 0
        self.nknown = 0
        self.codes = set()
        self.dropped = {}
        self.nrepeat = 0

    def update(self, code, seen, known, dropped):
        """Add the counts of one batch of rows (bool masks)."""
        self.nseen += int(seen.sum())
        self.nknown += int(known.sum())
        self.codes.update(np.unique(code[seen]).tolist())
        names, counts = np.unique(code[dropped], return_counts=True)
        for name, n in zip(names.tolist(), counts.tolist()):
            self.dropped[name] = self.dropped.get(name, 0) + n

    def report(self, data_file, mtase):
        """Raise on repeats or no known mod code; warn about drops."""
        if self.nrepeat:
            raise ValueError(
                f"{self.nrepeat} call(s) in {data_file} repeat a mod_code "
                "at the same read and position"
            )
        if mtase is None:
            return
        if self.nseen and not self.nknown:
            raise ValueError(
                f"no call in {data_file} has a mod_code of mtase "
                f"{list(mtase)}; found mod_code(s) {sorted(self.codes)}"
            )
        if self.dropped:
            n = sum(self.dropped.values())
            _warn(
                f"dropped {n} call(s) with mod_code(s) "
                f"{sorted(self.dropped)} not used by mtase {list(mtase)} "
                f"in {data_file}"
            )


def _stage_region_rows(
    rows, index, labels, ignore_strand, fold_length, stats=None
):
    """
    Filter and transform the rows of one region for staging.

    Parameters
    ----------
    rows : pd.DataFrame
        Rows of the region, in region coordinates.
    index : dict of str to int
        mol_id -> mol_index of the region's accepted molecules.
    labels : list or None
        Per `mtase` label, its mod codes and methylatable positions
        (see `_region_labels`); a row is kept if some label accepts
        both. None keeps every call.
    ignore_strand : bool
        Accept a row methylatable on either strand.
    fold_length : int or None
        Unfolded length, if positions are folded (`wrap`).
    stats : _CodeStats, optional
        Updated with the mod code counts (if `labels` is given) and the
        repeated mod codes.

    Returns
    -------
    pd.DataFrame or None
        The staged rows (`_RAW_COLUMN_SPEC` columns), one per molecule
        and position (see `_merge_codes`), or None if none are kept.
    """
    mol_index = rows["mol_id"].map(index)
    code = rows["mod_code"].to_numpy().astype(str)
    keep = mol_index.notna().to_numpy() & (code != "-")
    if labels is not None:
        upos = rows["upos"].to_numpy()
        strand = rows["strand"].to_numpy()
        known = np.zeros(len(rows), dtype=bool)
        site = np.zeros(len(rows), dtype=bool)
        both = np.zeros(len(rows), dtype=bool)
        for codes, masks in labels:
            label_code = np.isin(code, codes)
            label_site = (
                np.ones(len(rows), dtype=bool)
                if masks is None
                else _site_consistent(masks, upos, strand, ignore_strand)
            )
            known |= label_code
            site |= label_site
            both |= label_code & label_site
        if stats is not None:
            stats.update(code, keep, keep & known, keep & site & ~both)
        keep &= both
    if not keep.any():
        return None
    rows = rows.loc[keep]
    out, nrepeat = _merge_codes(
        pd.DataFrame(
            {
                "mol_index": mol_index[keep].astype(np.int64).to_numpy(),
                "pos": rows["upos"].to_numpy().astype(np.int64),
                "strand": rows["strand"].to_numpy(),
                "mod_qual": rows["mod_qual"].astype(np.float64).to_numpy(),
                "mod_code": rows["mod_code"].to_numpy(),
            }
        )
    )
    if stats is not None:
        stats.nrepeat += nrepeat
    if fold_length is not None:
        upos = out["pos"].to_numpy()
        out["pos"] = np.where(
            upos > (fold_length - 1) / 2, fold_length - upos - 1, upos
        )
    return out


def _merge_codes(df):
    """
    Merge the calls of one read at one position into one row.

    Parameters
    ----------
    df : pd.DataFrame
        Staged rows, with unfolded positions in `pos`.

    Returns
    -------
    df : pd.DataFrame
        One row per `mol_index` and `pos`: `mod_qual` summed over the
        mod codes (at most 1), the other columns from the first row.
    nrepeat : int
        Rows repeating the mod code of an earlier row at the same
        `mol_index` and `pos`.
    """
    key = ["mol_index", "pos"]
    first = ~df.duplicated(key).to_numpy()
    if first.all():
        return df, 0
    nrepeat = int(df.duplicated([*key, "mod_code"]).sum())
    total = df.groupby(key, sort=False)["mod_qual"].transform("sum")
    out = df.loc[first].copy()
    out["mod_qual"] = np.minimum(total.to_numpy()[first], 1.0)
    return out.reset_index(drop=True), nrepeat


class _GapTracker:
    """
    Infer internal gaps (deletions) of each molecule of one region
    from its rows, streamed in chunks.

    Between two consecutive rows of a molecule, the reference step
    minus the read step is the net number of deleted bases; if it is
    at least `min_gap`, the open interval between the two rows is a
    gap. An insertion between the same rows offsets the deletion, so
    the count is a lower bound. A molecule whose rows are not ordered
    along the reference (in either direction) is flagged as unordered
    and gets no gaps.

    Parameters
    ----------
    nmol : int
        Number of molecules.
    min_gap : int
        Minimum net deletion (bp) to record a gap.
    """

    def __init__(self, nmol, min_gap):
        self.min_gap = min_gap
        self.last_ref = np.full(nmol, -1, dtype=np.int64)
        self.last_read = np.zeros(nmol, dtype=np.int64)
        self.sign = np.zeros(nmol, dtype=np.int8)
        self.unordered = np.zeros(nmol, dtype=bool)
        self.found = []

    def update(self, idx, ref, read):
        """
        Add one chunk of rows, in file order.

        Parameters
        ----------
        idx : np.ndarray
            int64, molecule index of each row.
        ref : np.ndarray
            int64, reference position of each row (non-negative).
        read : np.ndarray
            int64, position of each row on the forward read.
        """
        if not len(idx):
            return
        order = np.argsort(idx, kind="stable")
        idx, ref, read = idx[order], ref[order], read[order]
        first = np.ones(len(idx), dtype=bool)
        first[1:] = idx[1:] != idx[:-1]
        prev_ref = np.roll(ref, 1)
        prev_read = np.roll(read, 1)
        prev_ref[first] = self.last_ref[idx[first]]
        prev_read[first] = self.last_read[idx[first]]
        has_prev = prev_ref >= 0

        dref = np.where(has_prev, ref - prev_ref, 0)
        up = np.zeros(len(self.sign), dtype=bool)
        down = np.zeros(len(self.sign), dtype=bool)
        np.logical_or.at(up, idx, dref > 0)
        np.logical_or.at(down, idx, dref < 0)
        self.unordered |= (up & down) | (up & (self.sign < 0))
        self.unordered |= down & (self.sign > 0)
        self.sign[(self.sign == 0) & up] = 1
        self.sign[(self.sign == 0) & down] = -1

        net = np.abs(dref) - np.abs(read - prev_read)
        hit = has_prev & (net >= self.min_gap)
        if hit.any():
            lo = np.minimum(ref[hit], prev_ref[hit]) + 1
            hi = np.maximum(ref[hit], prev_ref[hit])
            self.found.append(np.stack([idx[hit], lo, hi], axis=1))

        last = np.ones(len(idx), dtype=bool)
        last[:-1] = idx[:-1] != idx[1:]
        self.last_ref[idx[last]] = ref[last]
        self.last_read[idx[last]] = read[last]

    def result(self, nbp):
        """
        Gaps per molecule, sorted, clipped to `[0, nbp)`.

        Parameters
        ----------
        nbp : int
            Region length (bp).

        Returns
        -------
        gaps : np.ndarray
            int64, (n_gap, 2), half-open `[start, end)` of each gap,
            grouped by molecule.
        offset : np.ndarray
            int64, length nmol + 1; the gaps of molecule `i` are
            `gaps[offset[i]:offset[i + 1]]`.
        n_unordered : int
            Number of molecules skipped as unordered.
        """
        nmol = len(self.sign)
        rows = (
            np.concatenate(self.found)
            if self.found
            else np.empty((0, 3), dtype=np.int64)
        )
        rows = rows[~self.unordered[rows[:, 0]]]
        rows[:, 1:] = np.clip(rows[:, 1:], 0, nbp)
        rows = rows[rows[:, 1] < rows[:, 2]]
        rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
        offset = np.zeros(nmol + 1, dtype=np.int64)
        offset[1:] = np.cumsum(np.bincount(rows[:, 0], minlength=nmol))
        n_unordered = int(self.unordered.sum())
        return rows[:, 1:].copy(), offset, n_unordered


def _read_bed_gaps(bed_files, mol_maps, chrom_sizes, min_gap, chunk_size):
    """
    Internal gaps of the accepted molecules from BED12 files.

    Gaps are the stretches between consecutive blocks of a line (as
    written by `bedtools bamtobed -bed12 -splitD`), at least `min_gap`
    long. Lines are matched to molecules by chromosome and read name.

    Parameters
    ----------
    bed_files : sequence of str or pathlib.Path
        BED12 files, without a header.
    mol_maps : dict of str to dict
        Chromosome -> accepted molecules (see `_source_maps`).
    chrom_sizes : dict of str to int or None
        Chromosome -> length (bp), to clip gaps to; chromosomes
        without a length are clipped below by 0 only.
    min_gap : int
        Minimum gap length (bp).
    chunk_size : int
        Lines read per chunk.

    Returns
    -------
    listed : dict of str to np.ndarray
        Region -> bool per molecule, True if it has exactly one line.
    rows : dict of str to np.ndarray
        Region -> int64, (n_gap, 3), `(mol_index, start, end)`
        of each gap of the listed molecules.
    n_multi : int
        Molecules with more than one line (e.g., supplementary
        alignments); they count as unlisted.
    """
    n_lines = {
        r: np.zeros(len(m["mol_id"]), np.int64) for r, m in mol_maps.items()
    }
    found = {r: [] for r in mol_maps}
    cols = [0, 1, 3, 9, 10, 11]
    names = ["chrom", "start", "name", "count", "sizes", "starts"]
    for bed in bed_files:
        reader = pd.read_csv(
            bed,
            sep="\t",
            header=None,
            comment="#",
            usecols=cols,
            names=names,
            dtype={"chrom": str, "name": str, "sizes": str, "starts": str},
            chunksize=chunk_size,
        )
        for chunk in reader:
            for chrom, group in chunk.groupby("chrom", sort=False):
                mapping = mol_maps.get(chrom)
                if mapping is None:
                    continue
                idx = group["name"].map(mapping["index"])
                group = group[idx.notna()]
                idx = idx[idx.notna()].astype(np.int64).to_numpy()
                np.add.at(n_lines[chrom], idx, 1)
                for i, line in zip(idx, group.itertuples(index=False)):
                    if line.count < 2:
                        continue
                    sizes = np.array(line.sizes.rstrip(",").split(","), int)
                    starts = np.array(line.starts.rstrip(",").split(","), int)
                    lo = line.start + starts[:-1] + sizes[:-1]
                    hi = line.start + starts[1:]
                    keep = hi - lo >= min_gap
                    for a, b in zip(lo[keep], hi[keep]):
                        found[chrom].append((i, a, b))

    listed, rows, n_multi = {}, {}, 0
    for region, counts in n_lines.items():
        listed[region] = counts == 1
        n_multi += int((counts > 1).sum())
        arr = np.array(found[region], dtype=np.int64).reshape(-1, 3)
        arr = arr[listed[region][arr[:, 0]]]
        length = (chrom_sizes or {}).get(region, np.iinfo(np.int64).max)
        arr[:, 1:] = np.clip(arr[:, 1:], 0, length)
        rows[region] = arr[arr[:, 1] < arr[:, 2]]
    return listed, rows, n_multi


def _merge_gaps(inferred, listed, bed_rows, nmol):
    """
    Combine inferred gaps with BED gaps, which replace them for the
    molecules listed in the BED files.

    Parameters
    ----------
    inferred : (np.ndarray, np.ndarray) or None
        `(gaps, offset)` from `_GapTracker.result`, or None.
    listed : np.ndarray
        bool per molecule, from `_read_bed_gaps`.
    bed_rows : np.ndarray
        int64, (n_gap, 3), from `_read_bed_gaps`.
    nmol : int
        Number of molecules.

    Returns
    -------
    gaps : np.ndarray
        int64, (n_gap, 2), grouped by molecule and sorted.
    offset : np.ndarray
        int64, length nmol + 1.
    """
    parts = [bed_rows]
    if inferred is not None:
        gaps, offset = inferred
        mol = np.repeat(np.arange(nmol), np.diff(offset))
        keep = ~listed[mol]
        parts.append(np.column_stack([mol[keep], gaps[keep]]))
    rows = np.concatenate(parts).astype(np.int64)
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    offset = np.zeros(nmol + 1, dtype=np.int64)
    offset[1:] = np.cumsum(np.bincount(rows[:, 0], minlength=nmol))
    return rows[:, 1:].copy(), offset


def _source_maps(mol_maps, region_map):
    """
    Per reference chromosome, the accepted molecules of all its regions.

    Parameters
    ----------
    mol_maps : dict of str to dict
        From `_scan_accepted_mols`, by region.
    region_map : _RegionMap
        The regions.

    Returns
    -------
    dict of str to dict
        Chromosome -> `{"mol_id": sorted ids, "index": id -> index}`.
    """
    ids_of = {}
    for region, m in mol_maps.items():
        chrom = region_map.regions[region].chrom
        ids_of.setdefault(chrom, set()).update(m["mol_id"])
    out = {}
    for chrom, ids in ids_of.items():
        mol_id = np.array(sorted(ids), dtype=object)
        out[chrom] = {
            "mol_id": mol_id,
            "index": {m: i for i, m in enumerate(mol_id)},
        }
    return out


def _gaps_to_regions(src_gaps, src_maps, mol_maps, region_map):
    """
    Per-chromosome gaps clipped and shifted to each region.

    Parameters
    ----------
    src_gaps : dict of str to (np.ndarray, np.ndarray)
        Chromosome -> `(gaps, offset)` in chromosome coordinates and
        the `src_maps` index.
    src_maps : dict of str to dict
        From `_source_maps`.
    mol_maps : dict of str to dict
        From `_scan_accepted_mols`, by region.
    region_map : _RegionMap
        The regions.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Region -> `(gaps, offset)` in region coordinates and the
        region's `mol_index`; gaps outside the region are dropped.
    """
    out = {}
    for region, m in mol_maps.items():
        reg = region_map.regions[region]
        flat, offset = src_gaps[reg.chrom]
        index = src_maps[reg.chrom]["index"]
        idx = np.array([index[x] for x in m["mol_id"]], dtype=np.int64)
        counts = offset[idx + 1] - offset[idx]
        mol = np.repeat(np.arange(len(idx)), counts)
        within = np.arange(counts.sum()) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        gaps = flat[np.repeat(offset[idx], counts) + within] - reg.start
        gaps = np.clip(gaps, 0, reg.end - reg.start).reshape(-1, 2)
        keep = gaps[:, 0] < gaps[:, 1]
        gaps, mol = gaps[keep], mol[keep]
        new_offset = np.zeros(len(idx) + 1, dtype=np.int64)
        new_offset[1:] = np.cumsum(np.bincount(mol, minlength=len(idx)))
        out[region] = (gaps.astype(np.int64), new_offset)
    return out


def _apply_bed_gaps(
    inferred, bed_files, mol_maps, chrom_sizes, min_gap, chunk_size, data_file
):
    """
    Replace inferred gaps with BED12 gaps for the molecules listed.

    Parameters
    ----------
    inferred : dict of str to (np.ndarray, np.ndarray) or None
        Gaps from `_stream_rows_to_staging`.
    bed_files : sequence of str or pathlib.Path
        BED12 files (see `_read_bed_gaps`).
    mol_maps, chrom_sizes, min_gap, chunk_size
        As in `_read_bed_gaps`.
    data_file : str or pathlib.Path
        The raw data file, for the warning.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Chromosome -> `(gaps, offset)`.

    Warns
    -----
    UserWarning
        If molecules are not listed exactly once in `bed_files`.
    """
    listed, rows, n_multi = _read_bed_gaps(
        bed_files, mol_maps, chrom_sizes, min_gap, chunk_size
    )
    merged = {}
    n_missing = 0
    for region, m in mol_maps.items():
        nmol = len(m["mol_id"])
        own = None if inferred is None else inferred[region]
        merged[region] = _merge_gaps(own, listed[region], rows[region], nmol)
        n_missing += int((~listed[region]).sum())
    if n_missing:
        fallback = (
            "inferred from rows" if inferred is not None else "not recorded"
        )
        _warn(
            f"{n_missing} molecule(s) of {data_file} are not listed "
            f"exactly once in gaps_bed ({n_multi} listed more than "
            f"once); their gaps are {fallback}"
        )
    return merged


def _clip_extents(start, end, nbp):
    """
    Clip extents to `[0, nbp]`, with empty ones set to `(0, 0)`.

    Parameters
    ----------
    start, end : np.ndarray
        int64 half-open extent per molecule.
    nbp : int
        Region length (bp).

    Returns
    -------
    start, end : np.ndarray
        int64, clipped copies.
    """
    start = np.clip(start, 0, nbp)
    end = np.clip(end, 0, nbp)
    empty = start >= end
    start[empty] = 0
    end[empty] = 0
    return start, end


def _write_coords(gmeta, coords):
    """
    Store a region's genomic coordinates as attrs of its metadata group.

    Parameters
    ----------
    gmeta : h5py.Group
        The region's `metadata` group.
    coords : Region or None
        The region's reference interval and strand; None writes
        nothing.
    """
    if coords is None:
        return
    gmeta.attrs["ref_chrom"] = coords.chrom
    gmeta.attrs["ref_start"] = int(coords.start)
    gmeta.attrs["ref_end"] = int(coords.end)
    gmeta.attrs["ref_strand"] = coords.strand


def _read_coords(gmeta, region):
    """
    A region's genomic coordinates from its metadata group.

    Parameters
    ----------
    gmeta : h5py.Group
        The region's `metadata` group.
    region : str
        The region name.

    Returns
    -------
    Region or None
        None for a file written before coordinates were stored.
    """
    attrs = gmeta.attrs
    if not all(key in attrs for key in _COORD_KEYS):
        return None
    return Region(
        str(attrs["ref_chrom"]),
        int(attrs["ref_start"]),
        int(attrs["ref_end"]),
        str(region),
        str(attrs["ref_strand"]),
    )


def _ensure_region_data_group(graw, region, nbp, refseq_by_region, coords):
    """
    Create a region's `metadata` and `data` groups under `graw`.

    Each region is created at most once per private staging file, so
    no idempotency check is needed here (unlike a shared file). The
    region name is stored in the `chrom` attribute (on-disk key), and
    its genomic coordinates in `ref_*` attributes (see `_write_coords`).

    Returns
    -------
    h5py.Group
        The new `data` group.
    """
    gregion = graw.create_group(region)
    gmeta = gregion.create_group("metadata")
    gmeta.attrs["chrom"] = region
    gmeta.attrs["nbp"] = int(nbp)
    _write_coords(gmeta, coords)
    seq = refseq_by_region.get(region)
    if seq is not None:
        dt = h5py.string_dtype(encoding="utf-8")
        gmeta.create_dataset("refseq", data=seq, dtype=dt)
    return gregion.create_group("data")


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
    coords,
    wrap,
    refseq_by_region,
    mtase,
    ignore_strand,
    max_nmol,
    balance_strand,
    rng,
    tmp_dir,
    min_gap=None,
    gaps_bed=None,
    chrom_sizes=None,
    min_overlap=0,
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
    sizes : dict of str to int
        Region name -> stored length (bp); halved if `wrap`.
    coords : dict of str to Region
        Region name -> reference interval, used to assign rows and
        stored with each region (see `_write_coords`).
    chrom_sizes : dict of str to int, optional
        Reference chromosome -> length (bp), bounding rows; None
        bounds them below by 0 only.
    Other parameters are as in `MethPrintExperiment.load_raw`.

    Returns
    -------
    set of str
        Regions found in the file, already filtered to `sizes`.
    """
    region_map = _RegionMap([coords[r] for r in sizes])
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
            region_map,
            balance_strand=balance_strand,
            chrom_sizes=chrom_sizes,
            extents=not wrap,
            min_overlap=min_overlap,
        )
        src_maps = _source_maps(mol_maps, region_map)

        dt = h5py.string_dtype(encoding="utf-8")
        appenders = {}
        gdatas = {}
        for region, m in mol_maps.items():
            gdata = _ensure_region_data_group(
                graw,
                region,
                sizes[region],
                refseq_by_region,
                coords[region],
            )
            gdata.create_dataset(f"{name}_mol_id", data=m["mol_id"], dtype=dt)
            if m["start"] is not None:
                gdata.create_dataset(f"{name}_mol_start", data=m["start"])
                gdata.create_dataset(f"{name}_mol_end", data=m["end"])
            appenders[region] = h5_utils.AppendableDF(
                gdata,
                f"{name}_data",
                columns=list(_RAW_COLUMN_SPEC),
                dtypes=_RAW_COLUMN_SPEC,
            )
            gdatas[region] = gdata

        gaps = _stream_rows_to_staging(
            read_path,
            header_names,
            name_for,
            sep,
            chunk_size,
            block_size,
            wrap,
            mol_maps,
            appenders,
            region_map,
            src_maps,
            chrom_sizes=chrom_sizes,
            refseq_by_region=refseq_by_region,
            mtase=mtase,
            ignore_strand=ignore_strand,
            min_gap=min_gap,
        )
        if gaps_bed and not wrap:
            gaps = _apply_bed_gaps(
                gaps,
                gaps_bed,
                src_maps,
                chrom_sizes,
                min_gap,
                chunk_size,
                data_file,
            )
        if gaps is not None:
            gaps = _gaps_to_regions(gaps, src_maps, mol_maps, region_map)
        for region, (flat, offset) in (gaps or {}).items():
            gdatas[region].create_dataset(f"{name}_gaps", data=flat)
            gdatas[region].create_dataset(f"{name}_gap_offset", data=offset)
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
    regions : set of str
        Regions found in the file, already filtered to `sizes`.
    private_path : str
        Path to the private staging file, to be merged and removed by
        the caller.
    """
    fd, private_path = tempfile.mkstemp(suffix=f".{name}.h5", dir=tmp_dir)
    os.close(fd)
    with h5py.File(private_path, "w") as h5stream:
        regions = _ingest_into(
            h5stream.create_group("raw_data"),
            name,
            data_file,
            tmp_dir=tmp_dir,
            **kwargs,
        )
    return name, regions, private_path


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


def _keep_regions(found, requested):
    """
    Regions with molecules in every given file; the others are dropped.

    Parameters
    ----------
    found : list of set of str
        Regions found in each file.
    requested : iterable of str
        Regions asked for.

    Returns
    -------
    set of str
        The regions to keep.

    Raises
    ------
    ValueError
        If no region is kept.

    Warns
    -----
    UserWarning
        If some requested regions are dropped.
    """
    keep = set(requested).intersection(*found)
    dropped = sorted(set(requested) - keep)
    if not keep:
        raise ValueError("no region has molecules in every given file")
    if dropped:
        names = ", ".join(repr(r) for r in dropped[:5])
        if len(dropped) > 5:
            names += f", ... (+{len(dropped) - 5} more)"
        _warn(
            f"dropped {len(dropped)} region(s) without molecules in every "
            f"given file: {names}"
        )
    return keep


def _merge_staging(graw, ingested, regions=None):
    """
    Copy every source's private staging data into `graw`, then remove
    the private files.

    Parameters
    ----------
    graw : h5py.Group
        The `raw_data` group of the shared staging file.
    ingested : list of (str, set of str, str)
        From `_ingest_sources`, test first.
    regions : set of str, optional
        Regions to copy (see `_keep_regions`). If None, every source
        must cover the same regions.

    Raises
    ------
    ValueError
        If `regions` is None and a control file covers different
        regions than the test file.
    """
    if regions is None:
        regions = ingested[0][1]
        for name, regions_found, _ in ingested[1:]:
            if regions_found != regions:
                raise ValueError(
                    f"different number of regions in test and {name} "
                    "datasets"
                )

    handles = {name: h5py.File(path, "r") for name, _, path in ingested}
    try:
        for region in sorted(regions):
            gregion = graw.create_group(region)
            handles["test"].copy(
                f"raw_data/{region}/metadata", gregion, name="metadata"
            )
            gdata = gregion.create_group("data")
            for name, _, _ in ingested:
                src = handles[name][f"raw_data/{region}/data"]
                src.copy(f"{name}_mol_id", gdata, name=f"{name}_mol_id")
                src.copy(f"{name}_data", gdata, name=f"{name}_data")
                for key in _PER_MOL_KEYS:
                    if f"{name}_{key}" in src:
                        src.copy(f"{name}_{key}", gdata, name=f"{name}_{key}")
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
    # Read names as text so numeric names (e.g., "1") match the data
    if first_line == "\t".join(size_cols):
        df_size = pd.read_csv(chromsize, sep="\t", dtype={"chrom": str})
    else:
        df_size = pd.read_csv(
            chromsize,
            sep="\t",
            names=size_cols,
            header=None,
            dtype={"chrom": str},
        )
    if "chrom" not in df_size.columns:
        raise ValueError(
            "column 'chrom' containing the chromosome identifier is missing"
        )
    if "length" not in df_size.columns:
        raise ValueError(
            "column 'length' containing the chromosome length is missing"
        )
    if not df_size["chrom"].is_unique:
        raise ValueError("chromosomes must be unique in chromsize file")

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
                f"chromosomes not found in chromsize: {sorted(unknown)}"
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
                f"reference sequence length for chromosome "
                f"{chrom!r} ({len(seq)}) does not match its "
                f"chromsize length ({full_sizes[chrom]})"
            )
    return refseq_by_chrom
