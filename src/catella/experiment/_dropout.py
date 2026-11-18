# _dropout.py

import numpy as np
import pandas as pd

from catella.experiment._contexts import (
    _MTASE_CHANNELS,
    CONTEXT_NAMES,
    _merge_contexts,
    _reference_contexts,
)
from catella.experiment._ingest import _methylatable_positions
from catella.h5_array import H5Array


def _strand_of_mol(df, nmol):
    """
    One strand per molecule, looked up by `mol_index`.

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
    strand = np.full(nmol, ".", dtype="<U1")
    first = df.drop_duplicates("mol_index")[["mol_index", "strand"]]
    idx = first["mol_index"].to_numpy().astype(np.int64)
    strand[idx] = first["strand"].to_numpy()
    return strand


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
    return [
        src
        for src in ("test", "meth", "unmeth")
        if getattr(raw, f"{src}_data") is not None
    ]


def _channel_site_masks(refseq, channels):
    """
    Per-strand site masks for each context channel.

    Parameters
    ----------
    refseq : str
        Reference sequence for this chromosome.
    channels : iterable of int
        Channel codes, as from `_resolve_channels`; merged channels
        (GC, CG) pool the sites of their parts.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Maps each channel name to `(plus_mask, minus_mask)`, bool,
        length `len(refseq)`.
    """
    channels = tuple(channels)
    ctx = _merge_contexts(_reference_contexts(refseq), channels)
    strand_sites = {
        s: np.logical_or.reduce(
            [_methylatable_positions(refseq, m, s) for m in _MTASE_CHANNELS]
        )
        for s in ("+", "-")
    }
    return {
        CONTEXT_NAMES[c]: (
            strand_sites["+"] & (ctx == c),
            strand_sites["-"] & (ctx == c),
        )
        for c in channels
    }


def _fold_site_masks(site_masks, nbp):
    """
    Fold per-strand site masks onto the stored positions of a wrapped
    fibre.

    A wrapped molecule reads both halves, folded onto [0, nbp), so a
    folded position is a site if either half has one there (as in
    `_strand_assay_masks`).

    Parameters
    ----------
    site_masks : dict of str to (np.ndarray, np.ndarray)
        Masks over the full reference, as from `_channel_site_masks`.
    nbp : int
        Stored (halved) length.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Same keys, masks of length `nbp`.
    """
    return {
        name: tuple(m[:nbp] | m[len(m) - nbp :][::-1] for m in pair)
        for name, pair in site_masks.items()
    }


def _covered_rows(pos, row_strand, plus_mask, minus_mask, unmapped_strand):
    """
    Whether each raw row falls on a site of its molecule's strand.

    Parameters
    ----------
    pos : np.ndarray
        int64, position of each row.
    row_strand : np.ndarray of str
        Strand of each row's molecule, one of '+', '-', '.'.
    plus_mask, minus_mask : np.ndarray
        bool, sites on the '+' and '-' strand.
    unmapped_strand : {"union", "drop", "+", "-"}
        Sites used for unmapped ('.') rows; "drop" covers none.

    Returns
    -------
    np.ndarray
        bool, one per row.
    """
    in_plus = plus_mask[pos]
    in_minus = minus_mask[pos]
    covered = np.zeros(len(pos), dtype=bool)
    is_plus = row_strand == "+"
    is_minus = row_strand == "-"
    is_dot = row_strand == "."
    covered[is_plus] = in_plus[is_plus]
    covered[is_minus] = in_minus[is_minus]
    if unmapped_strand == "union":
        covered[is_dot] = (in_plus | in_minus)[is_dot]
    elif unmapped_strand == "+":
        covered[is_dot] = in_plus[is_dot]
    elif unmapped_strand == "-":
        covered[is_dot] = in_minus[is_dot]
    return covered


def _site_coverage(df, nmol, site_masks, unmapped_strand, ignore_strand=False):
    """
    Per-molecule covered/total site counts, per site set.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format table with `mol_index`/`pos`/`strand` columns.
    nmol : int
        Number of molecules in this source.
    site_masks : dict of str to (np.ndarray, np.ndarray)
        Maps a name (an mtase or a channel) to its
        `(plus_mask, minus_mask)` site masks.
    unmapped_strand : {"union", "drop", "+", "-"}
        How to evaluate unmapped ('.') strand molecules.
    ignore_strand : bool, default False
        If True, treat every row and molecule as unmapped-strand, so
        `unmapped_strand` governs coverage for all of them.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Maps each name to `(covered, n_total)`, each length `nmol`:
        `covered` counts rows on that set's sites (on each molecule's
        own strand); `n_total` is that set's site count for the
        molecule's strand.
    """
    strand = _strand_of_mol(df, nmol)
    pos = df["pos"].to_numpy().astype(np.int64)
    mol_index = df["mol_index"].to_numpy().astype(np.int64)
    if ignore_strand:
        strand = np.full(nmol, ".", dtype="<U1")
        row_strand = np.full(len(pos), ".", dtype="<U1")
    else:
        row_strand = strand[mol_index]

    result = {}
    for name, (plus_mask, minus_mask) in site_masks.items():
        rows = _covered_rows(
            pos, row_strand, plus_mask, minus_mask, unmapped_strand
        )
        covered = np.bincount(mol_index[rows], minlength=nmol)

        total = np.empty(nmol, dtype=np.int64)
        total[strand == "+"] = plus_mask.sum()
        total[strand == "-"] = minus_mask.sum()
        if unmapped_strand == "+":
            total[strand == "."] = plus_mask.sum()
        elif unmapped_strand == "-":
            total[strand == "."] = minus_mask.sum()
        else:
            total[strand == "."] = (plus_mask | minus_mask).sum()

        result[name] = (covered, total)
    return result


def _dropout_fracs(cov, names):
    """
    Per-site-set and pooled dropout fractions from coverage counts.

    Parameters
    ----------
    cov : dict of str to (np.ndarray, np.ndarray)
        Output of `_site_coverage`.
    names : sequence of str
        Site sets to evaluate, in the order to stack `frac`.

    Returns
    -------
    frac : np.ndarray, shape (len(names), n_mol)
        Dropout fraction per site set.
    pooled : np.ndarray, shape (n_mol,)
        Dropout fraction with site counts pooled across sets (as in
        `filter_dropout(method="aggregate")`).
    """
    covered = np.stack([cov[name][0] for name in names])
    n_total = np.stack([cov[name][1] for name in names])
    frac = np.where(n_total > 0, 1.0 - covered / np.maximum(n_total, 1), 0.0)

    covered_sum = covered.sum(axis=0)
    n_total_sum = n_total.sum(axis=0)
    pooled = np.where(
        n_total_sum > 0, 1.0 - covered_sum / np.maximum(n_total_sum, 1), 0.0
    )
    return frac, pooled


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
