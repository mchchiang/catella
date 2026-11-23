# _reads.py

from typing import NamedTuple

import numpy as np

from catella._batches import _row_batches
from catella.experiment._contexts import NONE
from catella.experiment._dropout import _strand_of_mol

_UNMAPPED_STRAND_MSG = (
    "cannot do normalization by strand with unmapped strands '.'"
)


def _mol_strand(exp, region, which):
    """Per-molecule strand labels ('+'/'-'/'.') of `which`."""
    raw = exp.raw[region]
    return _strand_of_mol(
        getattr(raw, f"{which}_data"), len(getattr(raw, f"{which}_mol_id"))
    )


def _check_mapped_strands(raw, has_controls):
    """
    Raise ValueError if the test data, or the controls when
    `has_controls`, have rows with an unmapped strand ('.').
    """
    sources = ("test", "meth", "unmeth") if has_controls else ("test",)
    for src in sources:
        if (getattr(raw, f"{src}_data")["strand"] == ".").any():
            raise ValueError(_UNMAPPED_STRAND_MSG)


def _read_sites(
    batch, assay_rows, ctx_ok, span="full", extent=None, gap=None
):
    """
    Confidence scores over the sites each molecule reads, with a
    missing call at a read site treated as unmethylated.

    Parameters
    ----------
    batch : np.ndarray
        float64, (n_mol, L), `mod_qual` per position; nan where the
        molecule has no row.
    assay_rows : np.ndarray
        bool, (n_mol, L), positions readable on each molecule's strand
        (see `_strand_assay_masks`).
    ctx_ok : np.ndarray
        bool, length L, positions with a context.
    span : {"full", "calls", "read"}, default "full"
        Extent of each molecule: the whole fiber, from its first to
        its last row, or its read extent `extent`.
    extent : (np.ndarray, np.ndarray), optional
        int64 half-open `(start, end)` per molecule. Required if
        `span` is "read", ignored otherwise.
    gap : np.ndarray, optional
        bool, (n_mol, L), positions inside each molecule's internal
        gaps; sites there without a call are not read, under every
        `span`.

    Returns
    -------
    q : np.ndarray
        float64, (n_mol, L). The row's `mod_qual` where present, 0 at
        read sites without a row, nan elsewhere.
    read : np.ndarray
        bool, (n_mol, L), the sites each molecule reads.
    """
    called = ~np.isnan(batch) & ctx_ok[None, :]
    read = assay_rows & ctx_ok[None, :]
    if span == "calls":
        L = batch.shape[1]
        first = np.argmax(called, axis=1)
        last = L - 1 - np.argmax(called[:, ::-1], axis=1)
        idx = np.arange(L)[None, :]
        in_span = (idx >= first[:, None]) & (idx <= last[:, None])
        read = read & in_span & called.any(axis=1)[:, None]
    elif span == "read":
        idx = np.arange(batch.shape[1])[None, :]
        start, end = extent
        in_span = (idx >= start[:, None]) & (idx < end[:, None])
        read = read & in_span
        called = called & in_span
    if gap is not None:
        read = read & ~gap
    read = read | called
    q = np.where(read, np.where(called, batch, 0.0), np.nan)
    return q, read


class _ReadBatch(NamedTuple):
    """One batch of read-site data from `_iter_read_sites`."""

    start: int
    stop: int
    keep: np.ndarray
    labels: np.ndarray
    q: np.ndarray
    read: np.ndarray
    called: np.ndarray
    blank: np.ndarray
    frac: dict


def _accumulate_call_frac(stats, b, assay, ctx, channels):
    """
    Add one batch's per-channel call and read-site counts to `stats`,
    a dict of channel code -> [n_called, n_read], in place.
    """
    w = _read_weights(b, assay)
    for c in channels:
        sel = ctx == c
        st = stats.setdefault(c, [0.0, 0.0])
        st[0] += b.called[:, sel].sum()
        st[1] += w[:, sel].sum()


def _read_weights(b, assay, strand=None):
    """
    Per-site read weights of a batch, for one strand or pooled.

    A blank molecule (kept, unknown strand, no call) reads the sites
    of strand `s` with weight `b.frac[s]`, its expected membership.

    Parameters
    ----------
    b : _ReadBatch
        From `_iter_read_sites`.
    assay : dict of str to np.ndarray
        From `_strand_assay_masks`.
    strand : {"+", "-"}, optional
        Restrict to the molecules of this strand. None pools both.

    Returns
    -------
    np.ndarray
        float64, (stop - start, L).
    """
    if strand is None:
        w = b.read.astype(np.float64)
        fill = b.frac["+"] * assay["+"] + b.frac["-"] * assay["-"]
    else:
        w = (b.read & (b.labels == strand)[:, None]).astype(np.float64)
        fill = b.frac[strand] * assay[strand]
    w[b.blank] = fill
    return w


def _gap_mask(gaps, offset, start, stop, nbp):
    """
    Positions inside the internal gaps of molecules `start:stop`.

    Parameters
    ----------
    gaps : np.ndarray
        int64, (n_gap, 2), half-open gaps grouped by molecule.
    offset : np.ndarray
        int64, length nmol + 1; the gaps of molecule `i` are
        `gaps[offset[i]:offset[i + 1]]`.
    start, stop : int
        Molecule range.
    nbp : int
        Number of positions.

    Returns
    -------
    np.ndarray
        bool, (stop - start, nbp).
    """
    lo, hi = offset[start], offset[stop]
    mol = np.repeat(np.arange(stop - start), np.diff(offset[start : stop + 1]))
    edges = np.zeros((stop - start, nbp + 1), dtype=np.int64)
    np.add.at(edges, (mol, gaps[lo:hi, 0]), 1)
    np.add.at(edges, (mol, gaps[lo:hi, 1]), -1)
    return np.cumsum(edges, axis=1)[:, :nbp] > 0


def _dense_batches(df, nmol, nbp, batch_size):
    """
    Stream a raw long-format table as dense batches of molecules.

    Parameters
    ----------
    df : pd.DataFrame
        Raw table with `mol_index`/`pos`/`mod_qual` columns. Rows with
        `pos` outside `[0, nbp)` are ignored.
    nmol : int
        Number of molecules.
    nbp : int
        Number of positions.
    batch_size : int
        Number of molecules per batch.

    Yields
    ------
    start, stop : int
        The row range of the batch.
    batch : np.ndarray
        float64, (stop - start, nbp), `mod_qual` per position; nan where
        the molecule has no row.

    Raises
    ------
    ValueError
        If a batch has several rows for the same molecule and position.
    """
    df = df.sort_values("mol_index", kind="stable")
    mol_index = df["mol_index"].to_numpy(dtype=np.int64)
    pos = df["pos"].to_numpy(dtype=np.int64)
    qual = df["mod_qual"].to_numpy(dtype=np.float64)
    in_range = (pos >= 0) & (pos < nbp)
    mol_index, pos, qual = mol_index[in_range], pos[in_range], qual[in_range]
    for start, stop in _row_batches(0, nmol, batch_size):
        lo, hi = np.searchsorted(mol_index, [start, stop])
        rows, cols = mol_index[lo:hi] - start, pos[lo:hi]
        seen = np.zeros((stop - start, nbp), dtype=bool)
        seen[rows, cols] = True
        if seen.sum() != hi - lo:
            raise ValueError("raw table has duplicate (mol_index, pos) rows")
        batch = np.full((stop - start, nbp), np.nan)
        batch[rows, cols] = qual[lo:hi]
        yield start, stop, batch


def _iter_read_sites(
    exp,
    region,
    which,
    ctx,
    assay,
    *,
    batch_size,
    mask_name,
    span="full",
    q_thres=None,
):
    """
    Stream one raw source in batches as read-site confidence scores
    (see `_read_sites`), with dropped molecules reading no site.

    A kept molecule with no call has no strand label, so the sites it
    reads are unknown; it is flagged as blank, and `_read_weights`
    spreads it over the strands by their fractions among the kept
    molecules with a strand.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    region : str
        Region identifier.
    which : {"test", "meth", "unmeth"}
        Which raw table to read.
    ctx : np.ndarray
        int8, length L, context code per position.
    assay : dict of str to np.ndarray
        From `_strand_assay_masks`.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        If given, molecules flagged as dropout by
        `filter_dropout(mask_name=mask_name)` are not kept.
    span : {"full", "calls", "read"}, default "full"
        Forwarded to `_read_sites`; "read" uses the read extents
        stored at ingest (`{which}_mol_start`/`{which}_mol_end`).
        Internal gaps stored at ingest (`{which}_gaps`), if any, are
        not read under every span.
    q_thres : float, optional
        If given, each read site's score in `q` becomes 1 if it is at
        least `q_thres` and 0 otherwise. `read` and `called` are not
        affected.

    Yields
    ------
    _ReadBatch
        `start`/`stop`, the row range of the batch; `keep`, bool per
        molecule; `labels`, strand label per molecule; `q`/`read`, as
        returned by `_read_sites` (all nan / False on rows not kept);
        `called`, bool (n_mol, L), read sites with a call;
        `blank`, bool per molecule, kept molecules with neither a
        strand nor a call (only with `span="full"`); and `frac`, the
        fraction of kept stranded molecules on each strand ("+"/"-").

    Raises
    ------
    ValueError
        If `span` is "read" and `which` has no stored read extents.
    """
    raw = exp.raw[region]
    extent = None
    if span == "read":
        extent = (
            getattr(raw, f"{which}_mol_start"),
            getattr(raw, f"{which}_mol_end"),
        )
        if extent[0] is None or extent[1] is None:
            raise ValueError(
                f"span='read' requires read extents, but '{which}' on "
                f"region '{region}' has none; reload the raw data with "
                "load_raw and wrap=False"
            )
    keep_all = exp.is_kept(region, which=which, mask_name=mask_name)
    labels_all = _mol_strand(exp, region, which)
    n_strand = {s: (keep_all & (labels_all == s)).sum() for s in "+-"}
    total = n_strand["+"] + n_strand["-"]
    frac = {s: n_strand[s] / total if total else 0.5 for s in "+-"}
    code = np.where(labels_all == "+", 0, np.where(labels_all == "-", 1, 2))
    stacked = np.stack([assay["+"], assay["-"], assay["."]])
    ctx_ok = ctx != NONE
    gaps = getattr(raw, f"{which}_gaps")
    gap_offset = getattr(raw, f"{which}_gap_offset")
    dense = _dense_batches(
        getattr(raw, f"{which}_data"), len(keep_all), raw.nbp, batch_size
    )
    for start, stop, batch in dense:
        ext = None
        if extent is not None:
            ext = (extent[0][start:stop], extent[1][start:stop])
        gap = None
        if gaps is not None and gap_offset is not None:
            gap = _gap_mask(gaps, gap_offset, start, stop, raw.nbp)
        q, read = _read_sites(
            batch, stacked[code[start:stop]], ctx_ok, span, ext, gap
        )
        if q_thres is not None:
            q = np.where(np.isnan(q), np.nan, (q >= q_thres).astype(float))
        keep = keep_all[start:stop]
        labels = labels_all[start:stop]
        q[~keep] = np.nan
        read[~keep] = False
        called = read & ~np.isnan(batch)
        blank = keep & (labels == ".") & ~read.any(axis=1)
        if span != "full":
            blank[:] = False
        yield _ReadBatch(
            start, stop, keep, labels, q, read, called, blank, frac
        )
