# _dropout.py

import numpy as np
import pandas as pd

from catella.experiment._ingest import _methylatable_positions
from catella.h5_array import H5Array


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
    return [
        src
        for src in ("test", "meth", "unmeth")
        if getattr(raw, f"{src}_data") is not None
    ]


def _label_coverage(
    df, mol_id, refseq, labels, unmapped_strand, ignore_strand=False
):
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
    ignore_strand : bool, default False
        If True, treat every row and molecule as unmapped-strand, so
        `unmapped_strand` governs coverage for all of them.

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
    if ignore_strand:
        strand = np.full(nmol, ".", dtype="<U1")
        row_strand = np.full(len(pos), ".", dtype="<U1")
    else:
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


def _dropout_fracs(cov, labels):
    """
    Per-label and pooled dropout fractions from coverage counts.

    Parameters
    ----------
    cov : dict of str to (np.ndarray, np.ndarray)
        Output of `_label_coverage`.
    labels : sequence of str
        Labels to evaluate, in the order to stack `frac`.

    Returns
    -------
    frac : np.ndarray, shape (len(labels), n_mol)
        Per-label dropout fraction.
    pooled : np.ndarray, shape (n_mol,)
        Dropout fraction with site counts pooled across labels (as
        in `filter_dropout(method="aggregate")`).
    """
    covered = np.stack([cov[label][0] for label in labels])
    n_total = np.stack([cov[label][1] for label in labels])
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
