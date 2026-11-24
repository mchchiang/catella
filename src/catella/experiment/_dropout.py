# _dropout.py

import numpy as np
import pandas as pd

from catella._batches import _row_batches
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
    Source name(s) to evaluate for dropout, on one region.

    Parameters
    ----------
    raw : MethPrintData
        Raw data block for one region.
    which : {"test", "meth", "unmeth"} or None
        Single source, or None for every source present.

    Returns
    -------
    list of str
    """
    if which is not None:
        return [which]
    return [src for src in ("test", "meth", "unmeth") if src in raw.sources]


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


def _site_coverage(
    df,
    nmol,
    site_masks,
    unmapped_strand,
    ignore_strand=False,
    span="full",
    extent=None,
    gaps=None,
):
    """
    Per-molecule covered/total site counts, per site set.

    Each molecule is measured over `[lo, hi)`: the whole region
    (`span="full"`), its first to last call on a site of any set
    (`"calls"`), or its read extent (`"read"`). Sites inside its
    internal gaps are not counted, unless they have a call, as in
    `_read_sites`.

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
    span : {"full", "calls", "read"}, default "full"
        Stretch of each molecule over which sites are counted.
    extent : (np.ndarray, np.ndarray), optional
        int64 half-open `(start, end)` per molecule; required if
        `span` is "read".
    gaps : (np.ndarray, np.ndarray), optional
        `(gaps, offset)` of the molecules' internal gaps (see
        `_gap_mask`), non-overlapping within a molecule.

    Returns
    -------
    dict of str to (np.ndarray, np.ndarray)
        Maps each name to `(covered, n_total)`, each length `nmol`:
        `covered` counts rows on that set's sites within the span (on
        each molecule's own strand); `n_total` is that set's site
        count for the molecule's strand within the span, without gap
        sites lacking a call.
    """
    strand = _strand_of_mol(df, nmol)
    pos = df["pos"].to_numpy().astype(np.int64)
    mol_index = df["mol_index"].to_numpy().astype(np.int64)
    if ignore_strand:
        strand = np.full(nmol, ".", dtype="<U1")
        row_strand = np.full(len(pos), ".", dtype="<U1")
    else:
        row_strand = strand[mol_index]

    L = len(next(iter(site_masks.values()))[0]) if site_masks else 0
    lo, hi = _span_bounds(pos, mol_index, nmol, L, site_masks, span, extent)
    in_span = (pos >= lo[mol_index]) & (pos < hi[mol_index])
    in_gap = _rows_in_gaps(pos, mol_index, gaps, L)

    result = {}
    for name, (plus_mask, minus_mask) in site_masks.items():
        rows = in_span & _covered_rows(
            pos, row_strand, plus_mask, minus_mask, unmapped_strand
        )
        covered = np.bincount(mol_index[rows], minlength=nmol)

        if unmapped_strand == "+":
            dot_mask = plus_mask
        elif unmapped_strand == "-":
            dot_mask = minus_mask
        else:
            dot_mask = plus_mask | minus_mask
        cum = np.stack(
            [
                np.concatenate([[0], np.cumsum(m)])
                for m in (plus_mask, minus_mask, dot_mask)
            ]
        )
        which = np.where(strand == "+", 0, np.where(strand == "-", 1, 2))
        total = cum[which, hi] - cum[which, lo]
        if gaps is not None:
            total -= _gap_sites(cum, which, lo, hi, gaps)
            total += np.bincount(mol_index[rows & in_gap], minlength=nmol)

        result[name] = (covered, total)
    return result


def _span_bounds(pos, mol_index, nmol, L, site_masks, span, extent):
    """
    Per-molecule half-open bounds `[lo, hi)` of `span` (see
    `_site_coverage`); `(0, 0)` for a molecule with nothing in it.
    """
    if span == "full":
        return np.zeros(nmol, dtype=np.int64), np.full(nmol, L, np.int64)
    if span == "read":
        start, end = extent
        return start.astype(np.int64), end.astype(np.int64)
    on_site = np.zeros(len(pos), dtype=bool)
    for plus_mask, minus_mask in site_masks.values():
        on_site |= (plus_mask | minus_mask)[pos]
    lo = np.full(nmol, L, dtype=np.int64)
    hi = np.zeros(nmol, dtype=np.int64)
    np.minimum.at(lo, mol_index[on_site], pos[on_site])
    np.maximum.at(hi, mol_index[on_site], pos[on_site] + 1)
    empty = lo >= hi
    lo[empty] = 0
    hi[empty] = 0
    return lo, hi


def _rows_in_gaps(pos, mol_index, gaps, L):
    """Whether each row lies inside an internal gap of its molecule."""
    if gaps is None or len(gaps[0]) == 0:
        return np.zeros(len(pos), dtype=bool)
    flat, offset = gaps
    gmol = np.repeat(np.arange(len(offset) - 1), np.diff(offset))
    gkey = gmol * (L + 1) + flat[:, 0]
    order = np.argsort(gkey, kind="stable")
    gkey, gmol, gend = gkey[order], gmol[order], flat[order, 1]
    j = np.searchsorted(gkey, mol_index * (L + 1) + pos, side="right") - 1
    valid = j >= 0
    j = np.maximum(j, 0)
    return valid & (gmol[j] == mol_index) & (pos < gend[j])


def _gap_sites(cum, which, lo, hi, gaps):
    """
    Per-molecule number of sites inside its gaps and within `[lo, hi)`,
    from cumulative site counts `cum` (rows: '+', '-', '.') picked by
    `which`.
    """
    flat, offset = gaps
    nmol = len(lo)
    gmol = np.repeat(np.arange(nmol), np.diff(offset))
    a = np.maximum(flat[:, 0], lo[gmol])
    b = np.minimum(flat[:, 1], hi[gmol])
    b = np.maximum(a, b)
    counts = cum[which[gmol], b] - cum[which[gmol], a]
    return np.bincount(gmol, weights=counts, minlength=nmol).astype(np.int64)


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
        Dropout fraction per site set; NaN without sites to measure.
    pooled : np.ndarray, shape (n_mol,)
        Dropout fraction with site counts pooled across sets (as in
        `filter_dropout(method="aggregate")`); NaN without sites in any
        set.
    """
    covered = np.stack([cov[name][0] for name in names])
    n_total = np.stack([cov[name][1] for name in names])
    frac = np.where(
        n_total > 0, 1.0 - covered / np.maximum(n_total, 1), np.nan
    )

    covered_sum = covered.sum(axis=0)
    n_total_sum = n_total.sum(axis=0)
    pooled = np.where(
        n_total_sum > 0,
        1.0 - covered_sum / np.maximum(n_total_sum, 1),
        np.nan,
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
        for start, stop in _row_batches(0, data.shape[0], batch_size):
            batch = data[start:stop, :]
            batch[~keep[start:stop], :] = np.nan
            data.write_batch(start, stop, batch)
        return data
    data = np.array(data, copy=True)
    data[~keep, :] = np.nan
    return data
