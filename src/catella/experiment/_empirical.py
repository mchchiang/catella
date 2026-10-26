# _empirical.py

import numpy as np
import pandas as pd

from catella.experiment._dropout import _apply_keep_mask
from catella.experiment._reads import _lookup_mask
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


def _masked_smoothed(exp, chrom, src, smoothed_name, mask_name, batch_size):
    """Smoothed array of `src`, with dropout molecules set to nan."""
    arr = exp.analysis[chrom][f"{src}_{smoothed_name}"]
    if mask_name is None:
        return arr
    keep = _lookup_mask(exp, chrom, src, mask_name)
    return _apply_keep_mask(arr, keep, batch_size)


def _control_means(arrs, labels, norm_by_strand, batch_size):
    """
    Per-position mean of the meth and unmeth controls, per stratum.

    Parameters
    ----------
    arrs : dict of str to H5Array
        Smoothed arrays, with "meth" and "unmeth" keys.
    labels : dict of str to np.ndarray
        Strand label per molecule, per source. Only read if
        `norm_by_strand` is True.
    norm_by_strand : bool
        If True, average each strand separately.
    batch_size : int
        Number of molecules processed per batch.

    Returns
    -------
    dict of str or None to (np.ndarray, np.ndarray)
        Stratum ("+"/"-", or None when pooled) -> `(meth_avg,
        unmeth_avg)`, each float64 of length L.
    """
    strata = ("+", "-") if norm_by_strand else (None,)

    def mean(src, s):
        mask = None if s is None else labels[src] == s
        return arrs[src]._streamed_reduce("mean", 0, batch_size, row_mask=mask)

    return {s: (mean("meth", s), mean("unmeth", s)) for s in strata}


def _normalize_to_controls(batch, labels, means, eps):
    """
    Rescale a batch so that the unmeth mean maps to 0 and the meth mean
    to 1.

    Parameters
    ----------
    batch : np.ndarray
        float64, (n_mol, L).
    labels : np.ndarray or None
        Strand label per molecule of `batch`. Only read when `means` is
        per strand.
    means : dict or None
        From `_control_means`. None returns `batch` unchanged.
    eps : float
        Floor on the magnitude of `meth_avg - unmeth_avg`.

    Returns
    -------
    np.ndarray
        float64, (n_mol, L).
    """
    if means is None:
        return batch

    def norm(x, meth_avg, unmeth_avg):
        denom = meth_avg - unmeth_avg
        denom = np.where(np.abs(denom) < eps, eps, denom)
        return (x - unmeth_avg) / denom

    if None in means:
        return norm(batch, *means[None])
    out = np.empty_like(batch)
    for s, (meth_avg, unmeth_avg) in means.items():
        sel = labels == s
        out[sel] = norm(batch[sel], meth_avg, unmeth_avg)
    return out


def _percentile_sample(
    arr, labels, means, *, sample_size, batch_size, end_idx, rng, eps
):
    """
    Random subsample of the molecules of `arr`, normalized by
    `_normalize_to_controls`, drawn proportionally within each batch.

    Parameters
    ----------
    arr : H5Array or np.ndarray
        float64, (n_mol, L).
    labels : np.ndarray or None
        Strand label per molecule of `arr`.
    means : dict or None
        From `_control_means`.
    sample_size : int
        Approximate number of molecules to draw; all of them if `arr`
        has fewer.
    batch_size : int
        Number of molecules read per batch.
    end_idx : int or None
        Positions from `end_idx` on are dropped from the sample.
    rng : np.random.Generator
        Random number generator for the draw.
    eps : float
        Forwarded to `_normalize_to_controls`.

    Returns
    -------
    np.ndarray
        float64, (n_drawn, L'), with L' set by `end_idx`.
    """
    n_total = arr.shape[0]
    sample_frac = min(sample_size, n_total) / n_total
    chunks = []
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch_labels = None if labels is None else labels[start:stop]
        batch = _normalize_to_controls(
            arr[start:stop, :], batch_labels, means, eps
        )
        n_take = min(stop - start, max(1, round((stop - start) * sample_frac)))
        rows = rng.choice(stop - start, size=n_take, replace=False)
        chunks.append(batch[rows, :end_idx])
    return np.concatenate(chunks, axis=0)


def _clip_bounds(arrs, labels, means, *, clip_low, clip_high, **sample_kw):
    """
    Lower and upper percentile bounds for probability scaling.

    The bounds come from the normalized controls when available, so
    they are comparable across test samples sharing the same controls,
    and from the test signal itself otherwise.

    Parameters
    ----------
    arrs : dict of str to H5Array or np.ndarray
        Smoothed arrays per source; "test", plus "meth" and "unmeth" if
        `means` is given.
    labels : dict of str to np.ndarray or None
        Strand label per molecule, per source.
    means : dict or None
        From `_control_means`; None if there are no controls.
    clip_low, clip_high : float
        Percentiles in [0, 100].
    **sample_kw
        Forwarded to `_percentile_sample`.

    Returns
    -------
    vmin, vmax : float
    """
    if means is None:
        sample = _percentile_sample(arrs["test"], None, None, **sample_kw)
        return (
            np.nanpercentile(sample, clip_low),
            np.nanpercentile(sample, clip_high),
        )
    unmeth = _percentile_sample(
        arrs["unmeth"], labels["unmeth"], means, **sample_kw
    )
    meth = _percentile_sample(arrs["meth"], labels["meth"], means, **sample_kw)
    vmin = np.nanpercentile(unmeth, clip_low)
    return vmin, np.nanpercentile(meth, clip_high)


def _write_empirical_prob(
    arr,
    labels,
    means,
    vmin,
    vmax,
    *,
    end_idx,
    fill_edge,
    batch_size,
    tmp_dir,
    eps,
):
    """
    Scale the normalized test signal to [0, 1] between `vmin` and `vmax`,
    streamed in batches to a disk-backed array.

    Interior gaps are filled with 0.5, positions from `end_idx` on with
    `fill_edge`, and all-nan (dropped) molecules stay all nan.

    Parameters
    ----------
    arr : H5Array
        float64, (n_mol, L), smoothed test signal.
    labels : np.ndarray or None
        Strand label per molecule of `arr`.
    means : dict or None
        From `_control_means`.
    vmin, vmax : float
        From `_clip_bounds`.
    end_idx : int or None
        Start of the trailing positions without a full window.
    fill_edge : float
        Probability for the trailing positions.
    batch_size : int
        Number of molecules processed per batch.
    tmp_dir : str or pathlib.Path
        Directory for the scratch file backing the result.
    eps : float
        Floor on `vmax - vmin`, also forwarded to
        `_normalize_to_controls`.

    Returns
    -------
    H5Array
        float64, (n_mol, L).
    """
    nmol, nbp = arr.shape
    denom = np.maximum(vmax - vmin, eps)
    out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
    for start in range(0, nmol, batch_size):
        stop = min(start + batch_size, nmol)
        batch_labels = None if labels is None else labels[start:stop]
        batch = _normalize_to_controls(
            arr[start:stop, :], batch_labels, means, eps
        )
        masked = np.isnan(batch).all(axis=1)
        prob = np.clip((batch - vmin) / denom, 0.0, 1.0)
        interior = prob[:, :end_idx]
        prob[:, :end_idx] = np.where(np.isnan(interior), 0.5, interior)
        if end_idx is not None:
            prob[:, end_idx:] = fill_edge
        prob[masked, :] = np.nan
        out.write_batch(start, stop, prob)
    return out
