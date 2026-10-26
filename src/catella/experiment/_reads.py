# _reads.py

from typing import NamedTuple

import numpy as np

from catella.experiment._contexts import NONE
from catella.experiment._dropout import _strand_of_mol

_UNMAPPED_STRAND_MSG = (
    "Cannot do normalization by strand with unmapped strands '.'."
)


def _lookup_mask(exp, chrom, source, mask_name):
    """Fetch a filter_dropout() 'keep' array, or raise KeyError."""
    key = f"{source}_{mask_name}"
    if key not in exp.analysis[chrom]:
        raise KeyError(
            f"'{key}' not found in exp.analysis['{chrom}']. Run "
            "filter_dropout() for this source first, matching "
            "mask_name."
        )
    return exp.analysis[chrom][key]["keep"].to_numpy()


def _mol_keep(exp, chrom, which, mask_name):
    """Per-molecule keep flags of `which`; all True without a mask."""
    nmol = len(getattr(exp.raw[chrom], f"{which}_mol_id"))
    if mask_name is None:
        return np.ones(nmol, dtype=bool)
    return np.asarray(_lookup_mask(exp, chrom, which, mask_name), dtype=bool)


def _mol_strand(exp, chrom, which):
    """Per-molecule strand labels ('+'/'-'/'.') of `which`."""
    raw = exp.raw[chrom]
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


def _read_sites(batch, assay_rows, ctx_ok, span="full"):
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
    span : {"full", "calls"}, default "full"
        Extent of each molecule: the whole fiber, or from its first to
        its last row.

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


def _iter_read_sites(
    exp, chrom, which, ctx, assay, *, batch_size, mask_name, span="full"
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
    chrom : str
        Chromosome identifier.
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
    span : {"full", "calls"}, default "full"
        Forwarded to `_read_sites`.

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
    """
    keep_all = _mol_keep(exp, chrom, which, mask_name)
    labels_all = _mol_strand(exp, chrom, which)
    n_strand = {s: (keep_all & (labels_all == s)).sum() for s in "+-"}
    total = n_strand["+"] + n_strand["-"]
    frac = {s: n_strand[s] / total if total else 0.5 for s in "+-"}
    code = np.where(labels_all == "+", 0, np.where(labels_all == "-", 1, 2))
    stacked = np.stack([assay["+"], assay["-"], assay["."]])
    ctx_ok = ctx != NONE
    arr = exp.to_dense(
        chrom,
        which=which,
        as_h5array=True,
        batch_size=batch_size,
        mask_name=mask_name,
    )
    try:
        n_total = arr.shape[0]
        for start in range(0, n_total, batch_size):
            stop = min(start + batch_size, n_total)
            batch = arr[start:stop, :]
            q, read = _read_sites(
                batch, stacked[code[start:stop]], ctx_ok, span
            )
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
    finally:
        arr.close()
