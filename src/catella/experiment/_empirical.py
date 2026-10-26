# _empirical.py

import numpy as np
import pandas as pd

from catella.experiment._dropout import _apply_keep_mask
from catella.h5_array import H5Array


def _fill_smoothed_nans(res, nan_method, fill_edge):
    """
    Fill the nan values of a smoothed batch.

    Parameters
    ----------
    res : pd.DataFrame
        Smoothed values, one row per position and one column per
        molecule.
    nan_method : {"mean", "interpolate", "none"}
        How to fill interior nan values (gaps with valid data on both
        sides): molecule mean, linear interpolation, or left unfilled.
    fill_edge : float or "mean"
        Value for the remaining (leading/trailing edge) nan values, or
        "mean" for the molecule mean.

    Returns
    -------
    pd.DataFrame
        `res` with nan values filled.
    """
    # interpolate() with limit_area="inside" fills only interior nans,
    # leaving edge nans for the fill below.
    if nan_method == "interpolate":
        res = res.interpolate(method="linear", limit_area="inside")
    elif nan_method == "mean":
        interior_interp = res.interpolate(method="linear", limit_area="inside")
        interior_mask = res.isna() & interior_interp.notna()
        res = res.where(~interior_mask, res.fillna(res.mean()))

    if isinstance(fill_edge, str) and fill_edge == "mean":
        return res.fillna(res.mean())
    return res.fillna(fill_edge)


def _smooth_source(
    df, nbp, nmol, keep, *, lnuc, nan_method, fill_edge, batch_size, tmp_dir
):
    """
    Left-aligned rolling mean of one raw source, streamed in batches of
    molecules to a disk-backed array.

    Parameters
    ----------
    df : pd.DataFrame
        Raw long-format table with `mol_index`/`pos`/`mod_qual` columns.
    nbp : int
        Number of positions.
    nmol : int
        Number of molecules.
    keep : np.ndarray or None
        bool, length `nmol`; molecules with False are set to all nan.
        None keeps every molecule.
    lnuc : int
        Window size (bp); position i summarizes `[i, i + lnuc)`.
    nan_method, fill_edge
        Forwarded to `_fill_smoothed_nans`.
    batch_size : int
        Number of molecules processed per batch.
    tmp_dir : str or pathlib.Path
        Directory for the scratch file backing the result.

    Returns
    -------
    H5Array
        float64, (nmol, nbp).

    Raises
    ------
    ValueError
        If `fill_edge` is a literal float outside the data range.
    """
    if not isinstance(fill_edge, str) and not np.isnan(fill_edge):
        vmin, vmax = df["mod_qual"].min(), df["mod_qual"].max()
        if not (vmin <= fill_edge <= vmax):
            raise ValueError(
                f"'fill_edge'={fill_edge} is outside the data "
                f"range [{vmin}, {vmax}]."
            )

    all_pos = pd.Index(range(nbp))
    df = df.sort_values("mol_index", kind="stable")
    mol_index = df["mol_index"].to_numpy()
    out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)

    for start in range(0, nmol, batch_size):
        stop = min(start + batch_size, nmol)
        lo, hi = np.searchsorted(mol_index, [start, stop])
        df_piv = (
            df.iloc[lo:hi]
            .pivot(index="pos", columns="mol_index", values="mod_qual")
            .reindex(all_pos)
        )
        res = df_piv.rolling(window=lnuc, min_periods=1).mean()
        res = _fill_smoothed_nans(
            res.shift(-(lnuc - 1)), nan_method, fill_edge
        )
        out_batch = res.to_numpy().T
        if keep is not None:
            out_batch = _apply_keep_mask(out_batch, keep[start:stop])
        out.write_batch(start, stop, out_batch)
    return out
