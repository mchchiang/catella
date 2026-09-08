# preprocessing.py

import warnings
import numpy as np
import pandas as pd
from typing import List
from catella.experiment.methdata import (
    MethPrintExperiment, _apply_keep_mask, _strand_of_mol)
from catella.h5_array import H5Array
from catella import utils
import matplotlib.pyplot as plt


def _lookup_mask(exp, chrom, source, mask_name):
    """Fetch a filter_dropout() 'keep' array, or raise KeyError."""
    key = f"{source}_{mask_name}"
    if key not in exp.analysis[chrom]:
        raise KeyError(
            f"'{key}' not found in exp.analysis['{chrom}']. Run "
            "filter_dropout() for this source first, matching "
            "mask_name.")
    return exp.analysis[chrom][key]["keep"].to_numpy()


def _patch_fill_edge(arr, fill_edge, vmin, vmax, keep, batch_size):
    """
    Refill only the still-NaN edge positions of an H5Array last
    smoothed with fill_edge=NaN, in place, streamed in batches. `keep`
    (or None) excludes dropout rows from filling.
    """
    if not isinstance(fill_edge, str) and not np.isnan(fill_edge):
        if not (vmin <= fill_edge <= vmax):
            raise ValueError(
                f"'fill_edge'={fill_edge} is outside the data range "
                f"[{vmin}, {vmax}].")
    nrow = arr.shape[0]
    for start in range(0, nrow, batch_size):
        stop = min(start + batch_size, nrow)
        batch = arr[start:stop, :]
        nanmask = np.isnan(batch)
        if keep is not None:
            nanmask &= keep[start:stop, None]
        if not nanmask.any():
            continue
        if isinstance(fill_edge, str):  # "mean"
            fill = np.nanmean(batch, axis=1)
            batch = np.where(nanmask, fill[:, None], batch)
        else:
            batch = np.where(nanmask, fill_edge, batch)
        arr.write_batch(start, stop, batch)


NONE, M6A, GCH, HCG, GCG = 0, 1, 2, 3, 4
CONTEXT_NAMES = {NONE: "none", M6A: "M6A", GCH: "GCH", HCG: "HCG",
                 GCG: "GCG"}

# Channels eta is estimated/applied per, and the reverse of
# CONTEXT_NAMES restricted to them, for parsing user eta overrides.
_ETA_CHANNELS = (M6A, GCH, HCG, GCG)
_CHANNEL_NAME_TO_CODE = {CONTEXT_NAMES[c]: c for c in _ETA_CHANNELS}

# Warn (not raise) if more than this fraction of wrap-mirrored position
# pairs disagree on context when folding ctx in model_prob.
_WRAP_CTX_DISAGREE_WARN_FRAC = 0.02

_UNMAPPED_STRAND_MSG = ("Cannot do normalization by strand with "
                        "unmapped strands '.'.")


def _reference_contexts(seq):
    """
    Classify every reference position into a footprinting context.

    Strand logic:
      m6A     : A on forward, or T on forward (= A on reverse)
      fwd CpG : C with G at i+1        rev CpG : G with C at i-1
      fwd GpC : C with G at i-1        rev GpC : G with C at i+1
    A position matching both CpG and GpC rules is GCG.

    Parameters
    ----------
    seq : str or bytes
        The reference nucleotide sequence.

    Returns
    -------
    np.ndarray
        int8, length `len(seq)`, one of NONE/M6A/GCH/HCG/GCG per
        position.
    """
    s = np.frombuffer(
        seq.upper().encode() if isinstance(seq, str) else seq.upper(),
        dtype="S1")
    L = len(s)
    ctx = np.zeros(L, dtype=np.int8)

    prev = np.concatenate([[b"N"], s[:-1]])
    nxt = np.concatenate([s[1:], [b"N"]])

    ctx[(s == b"A") | (s == b"T")] = M6A

    fwd_C = s == b"C"
    rev_C = s == b"G"
    is_cg = (fwd_C & (nxt == b"G")) | (rev_C & (prev == b"C"))
    is_gc = (fwd_C & (prev == b"G")) | (rev_C & (nxt == b"C"))

    ctx[is_gc & ~is_cg] = GCH
    ctx[is_cg & ~is_gc] = HCG
    ctx[is_cg & is_gc] = GCG
    return ctx


def _context_prior_rate(k, n_mol, ctx, nu=10.0, eps=1e-4):
    """
    Per-position call rate, shrunk toward its context group's mean.

    `nu` is the pseudo-count strength borrowed from the context
    mean: larger shrinks harder at low depth. Used on the meth control,
    the unmeth control, or the test set, so it is not controls-specific.

    Parameters
    ----------
    k : np.ndarray
        float64, length L, per-position count of methylated calls,
        summed across every molecule in the sample. Build this with
        `_streamed_call_count` so the full sample never needs to be
        held in memory at once.
    n_mol : int
        Molecule count in the sample.
    ctx : np.ndarray
        int8, length L, context code per position.
    nu : float, default 10.0
        Pseudo-count strength. 0 disables shrinkage.
    eps : float, default 1e-4
        Clip bound keeping rates away from exactly 0 or 1.

    Returns
    -------
    np.ndarray
        float64, length L, in [eps, 1 - eps].
    """
    theta = np.full(len(ctx), np.nan)
    for code in (M6A, GCH, HCG, GCG):
        sel = ctx == code
        if not sel.any() or n_mol == 0:
            continue
        p0 = k[sel].sum() / (n_mol * sel.sum())
        theta[sel] = (k[sel] + nu * p0) / (n_mol + nu)
    return np.clip(theta, eps, 1 - eps)


def _streamed_call_count(exp, chrom, which, ctx, batch_size, mask_name,
                         row_masks=None):
    """
    Per-position summed methylation-calling confidence and molecule
    count for one raw source, streamed in batches so peak memory is
    O(batch_size x L) rather than O(n_mol x L).

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    which : {"test", "meth", "unmeth"}
        Which raw table to read, forwarded to `exp.to_dense`.
    ctx : np.ndarray
        int8, length L, context code per position.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to `exp.to_dense`.
    row_masks : dict of str to np.ndarray, optional
        Named boolean row masks (e.g. strand labels), each length
        `n_mol`, aligned to the source's `to_dense` row order. If
        given, counts are accumulated separately per key from a
        single streamed pass, and the return value is keyed
        accordingly.

    Returns
    -------
    k : np.ndarray
        float64, length L, per-position sum of `mod_qual` confidence
        scores across every molecule in the sample (nan/no-call
        positions excluded). Only if `row_masks` is None.
    n_mol : int
        Molecule count in the sample. Only if `row_masks` is None.
    counts : dict of str to (k, n_mol)
        One `(k, n_mol)` pair per `row_masks` key. Only if
        `row_masks` is given.
    """
    arr = exp.to_dense(chrom, which=which, as_h5array=True,
                       batch_size=batch_size, mask_name=mask_name)
    n_total, L = arr.shape
    ctx_ok = ctx != NONE

    if row_masks is None:
        k = np.zeros(L, dtype=np.float64)
        n_mol = 0
        for start in range(0, n_total, batch_size):
            stop = min(start + batch_size, n_total)
            batch = arr[start:stop, :]
            # mask_name marks dropped molecules as all-nan rows (see
            # _apply_keep_mask); exclude them so they don't inflate
            # n_mol.
            kept = ~np.isnan(batch).all(axis=1)
            q = np.where(ctx_ok[None, :], batch[kept], np.nan)
            k += np.nansum(q, axis=0)
            n_mol += int(kept.sum())
        return k, n_mol

    acc = {key: {"k": np.zeros(L, dtype=np.float64), "n_mol": 0}
          for key in row_masks}
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = arr[start:stop, :]
        kept = ~np.isnan(batch).all(axis=1)
        for key, mask in row_masks.items():
            sel = kept & mask[start:stop]
            q = np.where(ctx_ok[None, :], batch[sel], np.nan)
            acc[key]["k"] += np.nansum(q, axis=0)
            acc[key]["n_mol"] += int(sel.sum())
    return {key: (acc[key]["k"], acc[key]["n_mol"]) for key in row_masks}


def _leak_interpolated_rate(theta_acc, fpr, rho_leak=0.1, eps=1e-4):
    """
    theta_prot interpolated between the caller FPR (perfect protection)
    and full accessibility.

    The unmethylated control measures only the caller's false-positive
    rate, a lower bound on theta_prot: real nucleosomes breathe at the
    entry and exit points, so true theta_prot sits above the FPR.
    `rho_leak` in [0, 1] is the one global parameter capturing that
    (0 means perfect protection).

    Parameters
    ----------
    theta_acc : np.ndarray
        float64, length L, accessible-state call rate.
    fpr : np.ndarray
        float64, length L, false-positive rate from the unmeth control.
    rho_leak : float, default 0.1
        Leak fraction toward `theta_acc`.
    eps : float, default 1e-4
        Clip bound keeping rates away from exactly 0 or 1.

    Returns
    -------
    np.ndarray
        float64, length L, in [eps, 1 - eps].
    """
    theta_prot = fpr + rho_leak * (theta_acc - fpr)
    return np.clip(theta_prot, eps, 1 - eps)


def _informative_mask(theta_acc, fpr, ctx, min_gap=0.05):
    """Positions the assay can discriminate; elsewhere contributes zero."""
    return ((ctx != NONE) & np.isfinite(theta_acc)
            & ((theta_acc - fpr) > min_gap))


def _calibrate_from_controls(meth_k, meth_n, unmeth_k, unmeth_n, ctx,
                             nu=10.0, rho_leak=0.1, min_gap=0.05):
    """
    Estimate theta_prot/theta_acc/informative from meth/unmeth controls.

    Parameters
    ----------
    meth_k, meth_n : np.ndarray, int
        From `_streamed_call_count` on the fully-methylated control.
    unmeth_k, unmeth_n : np.ndarray, int
        From `_streamed_call_count` on the unmethylated control.
    ctx : np.ndarray
        int8, length L, context code per position.
    nu : float, default 10.0
        Forwarded to `_context_prior_rate`.
    rho_leak : float, default 0.1
        Forwarded to `_leak_interpolated_rate`.
    min_gap : float, default 0.05
        Forwarded to `_informative_mask`.

    Returns
    -------
    tuple of np.ndarray
        `(theta_prot, theta_acc, informative)`, each length L.
    """
    theta_acc = _context_prior_rate(meth_k, meth_n, ctx, nu=nu)
    fpr = _context_prior_rate(unmeth_k, unmeth_n, ctx, nu=nu)
    theta_prot = _leak_interpolated_rate(theta_acc, fpr, rho_leak=rho_leak)
    informative = _informative_mask(theta_acc, fpr, ctx, min_gap=min_gap)
    return theta_prot, theta_acc, informative


def _streamed_window_count(exp, chrom, ctx, l_nuc, n_min, batch_size,
                           mask_name, row_masks=None):
    """
    Per-window, per-read summed methylation-calling confidence and
    trial counts for the test sample, streamed in batches so peak
    memory is O(batch_size x L) rather than O(n_mol x L).

    Windows are non-overlapping. A window's trial count (its count of
    context-eligible sites) never varies by read, since coverage is
    assumed complete, so windows below `n_min` are dropped up front,
    before any read data is touched.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    ctx : np.ndarray
        int8, length L, context code per position.
    l_nuc : int
        Nucleosome footprint size (bp), used as the window size.
    n_min : int
        Minimum context-eligible sites (summed over all contexts) a
        window needs to be kept.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to `exp.to_dense`.
    row_masks : dict of str to np.ndarray, optional
        Named boolean row masks (e.g. strand labels), each length
        `n_mol`, aligned to the test source's `to_dense` row order.
        If given, `K`/`N`/`codes` are accumulated separately per key
        from a single streamed pass, and the return value is keyed
        accordingly. Window geometry (`n_min`-eligibility) is shared
        across keys, since it depends only on `ctx`.

    Returns
    -------
    K, N : np.ndarray
        float64, (n_ctx, n_mol * n_kept_windows). `K` is the per-window,
        per-read sum of `mod_qual` confidence scores (nan/no-call
        positions excluded). Only if `row_masks` is None.
    codes : list of int
        Context codes present in `ctx`, in the same order as `K`/`N`'s
        first axis. Only if `row_masks` is None.
    counts : dict of str to (K, N, codes)
        One `(K, N, codes)` triple per `row_masks` key. Only if
        `row_masks` is given.

    Raises
    ------
    ValueError
        If `ctx` has no assayable contexts, or too few windows have
        `n_min` context-eligible sites (per key, when `row_masks` is
        given).
    """
    arr = exp.to_dense(chrom, which="test", as_h5array=True,
                       batch_size=batch_size, mask_name=mask_name)
    n_total, L = arr.shape
    starts = np.arange(0, L - l_nuc + 1, l_nuc)
    codes = [c for c in (M6A, GCH, HCG, GCG) if (ctx == c).any()]
    if not codes:
        raise ValueError("no assayable contexts in ctx")

    n_win = {c: np.array([(ctx[s:s + l_nuc] == c).sum() for s in starts])
             for c in codes}
    total_win = sum(n_win[c] for c in codes)
    keep = total_win >= n_min
    if not keep.any():
        raise ValueError("too few windows with >= n_min context-eligible "
                         "sites")
    kept_starts = starts[keep]

    ctx_ok = ctx != NONE
    keys = list(row_masks) if row_masks is not None else [None]
    K_batches = {key: {c: [] for c in codes} for key in keys}
    n_mol = {key: 0 for key in keys}

    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = arr[start:stop, :]
        # mask_name marks dropped molecules as all-nan rows (see
        # _apply_keep_mask); exclude them so they don't inflate n_mol.
        kept_rows = ~np.isnan(batch).all(axis=1)
        batch = batch[kept_rows]
        q_full = np.where(ctx_ok[None, :], batch, np.nan)
        for key in keys:
            if key is None:
                q = q_full
            else:
                strand_sel = row_masks[key][start:stop][kept_rows]
                q = q_full[strand_sel]
            n_mol[key] += q.shape[0]
            for c in codes:
                sel = ctx == c
                k_c = np.stack(
                    [np.nansum(q[:, s:s + l_nuc][:, sel[s:s + l_nuc]],
                              axis=1) for s in kept_starts], axis=1)
                K_batches[key][c].append(k_c)

    def _build(key):
        if n_mol[key] * keep.sum() < 10:
            suffix = "" if key is None else f" for strand {key!r}"
            raise ValueError("too few windows with >= n_min "
                             f"context-eligible sites{suffix}")
        K = np.array([np.concatenate(K_batches[key][c], axis=0).ravel()
                     for c in codes]).astype(np.float64)
        N = np.array([np.tile(n_win[c][keep], n_mol[key])
                     for c in codes]).astype(np.float64)
        return K, N, codes

    if row_masks is None:
        return _build(None)
    return {key: _build(key) for key in keys}


def _calibrate_from_data(K, N, ctx, codes, iters=200, eps=1e-3,
                         init_prot=0.05, init_acc=0.95, tol=1e-8,
                         min_gap=0.05):
    """
    Fit theta_prot/theta_acc per context with expectation-maximization,
    from precomputed per-window call counts, when no meth/unmeth
    controls are available.

    Fits a two-component binomial mixture (a shared latent protection
    state per window, pooled across reads), since without controls the
    two rates must be estimated from the test sample alone.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs), from `_streamed_window_count`.
    ctx : np.ndarray
        int8, length L, context code per position.
    codes : list of int
        Context codes present in `ctx`, in the same order as `K`/`N`'s
        first axis.
    iters : int, default 200
        Maximum number of expectation-maximization iterations.
    eps : float, default 1e-3
        Clip bound keeping rates away from exactly 0 or 1.
    init_prot : float, default 0.05
        Initial guess for theta_prot.
    init_acc : float, default 0.95
        Initial guess for theta_acc.
    tol : float, default 1e-8
        Relative log-likelihood convergence tolerance.
    min_gap : float, default 0.05
        Forwarded to the informative-mask computation.

    Returns
    -------
    tuple of np.ndarray
        `(theta_prot, theta_acc, informative)`, each length L. Rates are
        constant within a context, since position-specific efficiency is
        not resolvable without controls.
    """
    from scipy.stats import binom

    tp = np.full(len(codes), init_prot)
    ta = np.full(len(codes), init_acc)
    pi = 0.75
    prev = -np.inf

    for it in range(iters):
        lp = np.log(pi) + sum(binom.logpmf(K[c], N[c], tp[c])
                              for c in range(len(codes)))
        la = np.log1p(-pi) + sum(binom.logpmf(K[c], N[c], ta[c])
                                 for c in range(len(codes)))
        m = np.maximum(lp, la)
        ll = (m + np.log(np.exp(lp - m) + np.exp(la - m))).sum()
        r = 1.0 / (1.0 + np.exp(la - lp))          # P(protected | window)

        for c in range(len(codes)):
            dp, da = (r * N[c]).sum(), ((1 - r) * N[c]).sum()
            if dp > 0:
                tp[c] = np.clip((r * K[c]).sum() / dp, eps, 1 - eps)
            if da > 0:
                ta[c] = np.clip(((1 - r) * K[c]).sum() / da, eps, 1 - eps)
        pi = float(r.mean())

        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll

    # Label switching: "protected" must be the low-methylation component.
    if (np.average(tp, weights=N.sum(axis=1))
            > np.average(ta, weights=N.sum(axis=1))):
        tp, ta = ta, tp

    theta_prot = np.full(len(ctx), np.nan)
    theta_acc = np.full(len(ctx), np.nan)
    for c, code in enumerate(codes):
        sel = ctx == code
        theta_prot[sel], theta_acc[sel] = tp[c], ta[c]

    informative = ((ctx != NONE) & np.isfinite(theta_acc)
                   & ((theta_acc - theta_prot) > min_gap))
    return theta_prot, theta_acc, informative


def _per_base_log_odds(q, theta_prot, theta_acc, informative, pi0=0.5,
                       eta=1.0):
    """
    Per-position, per-read log-odds of "protected" versus "accessible".

    Computes lambda_x = log[(L_x*theta_P + 1-theta_P) /
    (L_x*theta_A + 1-theta_A)], with L_x = q/(1-q) * (1-pi0)/pi0, using
    the numerically stable form obtained by multiplying through by
    (1-q), which is well-defined at q = 0 or 1.

    Parameters
    ----------
    q : np.ndarray
        float64, (n_mol, L), methylation-calling confidence score in
        [0, 1] (`mod_qual`); nan where no call was emitted.
    theta_prot : np.ndarray
        float64, length L.
    theta_acc : np.ndarray
        float64, length L.
    informative : np.ndarray
        bool, length L.
    pi0 : float, default 0.5
        Prior probability that a site is methylated, used to convert `q`
        into the likelihood ratio `L_x`. 0.5 is the uninformative choice
        used when the base caller's training prior is unknown.
    eta : float or np.ndarray, default 1.0
        Multiplicative correction for inflated log-likelihood ratios
        from correlated nearby sites (e.g. palindromic CpG/GpC); 1.0
        leaves the log-odds unscaled. Either a scalar applied
        everywhere, or length-L, broadcasting a per-position (e.g.
        per-channel) correction.

    Returns
    -------
    np.ndarray
        float64, (n_mol, L), zero outside `informative` positions and
        wherever `q` is nan (no call emitted).

    Notes
    -----
    `theta_prot`/`theta_acc` are nan outside assayable contexts; the log
    ratios below are computed with warnings suppressed there, since
    `informative` always zeroes those positions regardless.
    """
    w = (1.0 - pi0) / pi0
    with np.errstate(invalid="ignore"):
        num = w * q * theta_prot + (1.0 - q) * (1.0 - theta_prot)
        den = w * q * theta_acc + (1.0 - q) * (1.0 - theta_acc)
        log_odds = eta * np.log(num / den)
    log_odds = np.where(np.isnan(log_odds), 0.0, log_odds)
    return np.where(informative, log_odds, 0.0)


def _window_sum_log_odds(log_odds, l_nuc, fill_edge=0.0):
    """
    Left-aligned rolling sum of `log_odds` over a window of `l_nuc`
    positions: position i summarizes `[i, i + l_nuc)`. The trailing
    `l_nuc - 1` positions, which have no full window, are filled with
    `fill_edge`.

    Parameters
    ----------
    log_odds : np.ndarray
        float64, (n_mol, L).
    l_nuc : int
        Window size.
    fill_edge : float, default 0.0
        Value used for the trailing positions.

    Returns
    -------
    np.ndarray
        float64, (n_mol, L).
    """
    n_mol, L = log_odds.shape
    out = np.full((n_mol, L), fill_edge)
    n_win = L - l_nuc + 1
    if n_win <= 0:
        return out
    cum = np.concatenate(
        [np.zeros((n_mol, 1)), np.cumsum(log_odds, axis=1)], axis=1)
    out[:, :n_win] = cum[:, l_nuc:] - cum[:, :-l_nuc]
    return out


def _resolve_eta_overrides(eta):
    """
    Validate and normalize a user-supplied `eta` into per-channel
    overrides.

    Parameters
    ----------
    eta : float, dict of str to float, or None
        `None` requests auto-estimation for every channel; a float
        pins every channel to that value; a dict pins only the named
        channels (keys from `CONTEXT_NAMES`, e.g. "M6A"), leaving any
        channel not mentioned to be auto-estimated.

    Returns
    -------
    overrides : dict of int to float
        Channel code -> pinned eta value, for every channel the
        caller specified. Empty if `eta` is None.

    Raises
    ------
    ValueError
        If `eta` is a dict with an unrecognized channel name.
    """
    if eta is None:
        return {}
    if isinstance(eta, dict):
        overrides = {}
        for name, val in eta.items():
            if name not in _CHANNEL_NAME_TO_CODE:
                raise ValueError(
                    f"Unrecognized eta channel {name!r}; expected one "
                    f"of {sorted(_CHANNEL_NAME_TO_CODE)}.")
            overrides[_CHANNEL_NAME_TO_CODE[name]] = float(val)
        return overrides
    return {c: float(eta) for c in _ETA_CHANNELS}


def _init_autocorr_stats(max_lag):
    """Zeroed per-channel accumulators for `_accumulate_autocorr_stats`."""
    return {c: {"n": 0.0, "s1": 0.0, "s2": 0.0,
               "npair": np.zeros(max_lag), "pprod": np.zeros(max_lag),
               "psumA": np.zeros(max_lag), "psumB": np.zeros(max_lag)}
           for c in _ETA_CHANNELS}


def _accumulate_autocorr_stats(stats, log_odds, called, ctx, max_lag,
                               channels=None):
    """
    Fold one batch's contribution into per-channel lag-k
    autocorrelation sums, in place.

    Accumulates raw (uncentered) sums rather than pre-centering by
    the channel mean, since that mean is only known once every batch
    (across every chromosome) has been seen; `_finalize_channel_eta`
    does the centering afterward.

    Parameters
    ----------
    stats : dict
        Per-channel accumulators, as returned by
        `_init_autocorr_stats`, updated in place.
    log_odds : np.ndarray
        float64, (n_mol, L), raw (`eta=1`) per-position log-odds from
        `_per_base_log_odds`.
    called : np.ndarray
        bool, (n_mol, L). True where a site was both informative and
        actually called (i.e. `informative[None, :] & ~np.isnan(q)`).
    ctx : np.ndarray
        int8, length L, context code per position.
    max_lag : int
        Maximum lag (bp) to accumulate pair sums for.
    channels : iterable of int, optional
        Channel codes to accumulate. Defaults to all of
        `_ETA_CHANNELS`.
    """
    if channels is None:
        channels = _ETA_CHANNELS
    L = log_odds.shape[1]
    for c in channels:
        mask_c = called & (ctx == c)[None, :]
        if not mask_c.any():
            continue
        lam = log_odds * mask_c
        st = stats[c]
        st["n"] += mask_c.sum()          # count of called sites
        st["s1"] += lam.sum()            # sum(lambda_x)
        st["s2"] += (lam * lam).sum()    # sum(lambda_x^2)
        for k in range(1, min(max_lag, L - 1) + 1):
            left, right = mask_c[:, :L - k], mask_c[:, k:]
            pair = left & right          # both x and x+k called
            if not pair.any():
                continue
            lo_left, lo_right = log_odds[:, :L - k], log_odds[:, k:]
            i = k - 1
            st["npair"][i] += pair.sum()  # count of pairs
            # sum(lambda_x * lambda_x+k) over pairs
            st["pprod"][i] += (lo_left * lo_right * pair).sum()
            # sum(lambda_x) over pairs
            st["psumA"][i] += (lo_left * pair).sum()
            # sum(lambda_x+k) over pairs
            st["psumB"][i] += (lo_right * pair).sum()


def _finalize_channel_eta(stats, max_lag, min_n=None):
    """
    Convert accumulated autocorrelation sums into a per-channel eta,
    the inverse of the variance-inflation factor from lag-k
    autocorrelation in the channel's log-odds sequence.

    Parameters
    ----------
    stats : dict
        Per-channel accumulators from `_accumulate_autocorr_stats`.
    max_lag : int
        Maximum lag (bp) summed over.
    min_n : int, optional
        Minimum called-site count `n` a channel needs before it is
        estimated; below this, `eta_c = 1.0` (no correction) since
        there isn't enough data for a stable estimate. Defaults to
        `max_lag`.

    Returns
    -------
    eta : dict of int to float
        Channel code -> estimated eta, for every channel in `stats`.
    rho : dict of int to np.ndarray
        Channel code -> lag-k autocorrelation `rho(k)` (length
        `max_lag`, for `k = 1..max_lag`), for every channel that had
        enough data to be estimated (a subset of `eta`'s keys;
        channels that fell back to `eta_c = 1.0` are omitted).
    """
    if min_n is None:
        min_n = max_lag
    k = np.arange(1, max_lag + 1)
    eta = {}
    rho = {}
    for c, st in stats.items():
        n = st["n"]
        if n <= min_n:
            eta[c] = 1.0
            continue
        bar = st["s1"] / n                    # global channel mean
        denom = st["s2"] - n * bar * bar       # sum((lambda_x - bar)^2)
        if denom <= 0:
            eta[c] = 1.0
            continue
        npair, pprod = st["npair"], st["pprod"]
        # sum((lambda_x - bar)(lambda_x+k - bar)) over valid pairs
        numerator = (pprod - bar * st["psumA"] - bar * st["psumB"]
                    + npair * bar * bar)
        rho_c = np.where(npair > 0, numerator / denom, 0.0)
        v = 1.0 + 2.0 * np.sum((1.0 - k / n) * rho_c)
        eta[c] = 1.0 / max(v, 1e-3)
        rho[c] = rho_c
    return eta, rho


def _accumulate_eta_source(exp, chrom, which, ctx, theta_prot, theta_acc,
                           informative, pi0, batch_size, mask_name, stats,
                           max_lag, channels, norm_by_strand=False,
                           strand_labels=None):
    """
    Stream one raw source and feed its raw (`eta=1`) per-read
    log-odds into `_accumulate_autocorr_stats`.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    which : {"test", "meth"}
        Which raw table to read: the methylated control when
        available (isolates the crosstalk artifact from genuine,
        occupancy-driven correlation), otherwise the test data
        itself as a fallback.
    ctx, theta_prot, theta_acc, informative : np.ndarray or dict
        From `_chrom_calibration`; plain arrays, or (if
        `norm_by_strand`) `{"+": ..., "-": ...}` dicts of such
        arrays.
    pi0 : float
        Forwarded to `_per_base_log_odds`.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to `exp.to_dense`.
    stats : dict
        Per-channel accumulators, updated in place. Both strands'
        contributions are folded into the same accumulator, so eta
        stays pooled across strands even when calibration is split.
    max_lag : int
        Forwarded to `_accumulate_autocorr_stats`.
    channels : iterable of int
        Channel codes to accumulate.
    norm_by_strand : bool, default False
        Whether `theta_prot`/`theta_acc`/`informative` are per-strand
        dicts requiring a per-strand log-odds computation.
    strand_labels : np.ndarray, optional
        Length `n_mol` strand label ('+'/'-') per molecule of
        `which`'s source, aligned to `to_dense`'s row order. Required
        if `norm_by_strand` is True.
    """
    arr = exp.to_dense(chrom, which=which, as_h5array=True,
                       batch_size=batch_size, mask_name=mask_name)
    n_total, L = arr.shape
    ctx_ok = ctx[None, :] != NONE
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = arr[start:stop, :]
        q = np.where(ctx_ok, batch, np.nan)
        if norm_by_strand:
            labels = strand_labels[start:stop]
            for s in ("+", "-"):
                sel = labels == s
                if not sel.any():
                    continue
                log_odds = _per_base_log_odds(
                    q[sel], theta_prot[s], theta_acc[s], informative[s],
                    pi0=pi0, eta=1.0)
                called = informative[s][None, :] & ~np.isnan(q[sel])
                _accumulate_autocorr_stats(stats, log_odds, called, ctx,
                                           max_lag, channels=channels)
        else:
            log_odds = _per_base_log_odds(q, theta_prot, theta_acc,
                                          informative, pi0=pi0, eta=1.0)
            called = informative[None, :] & ~np.isnan(q)
            _accumulate_autocorr_stats(stats, log_odds, called, ctx,
                                       max_lag, channels=channels)


def _chrom_calibration(exp, chrom, *, nu, rho_leak, min_gap, l_nuc, n_min,
                       iters, init_prot, init_acc, tol, batch_size,
                       mask_name, norm_by_strand=False):
    """
    Classify `chrom`'s reference into contexts and calibrate
    theta_prot/theta_acc/informative, from meth/unmeth controls when
    available or by expectation-maximization on the test sample
    otherwise.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    nu, rho_leak, min_gap, l_nuc, n_min, iters, init_prot, init_acc,
    tol : as in `model_prob`.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to the streamed counting helpers.
    norm_by_strand : bool, default False
        Whether to calibrate separately per strand. If True, the
        source(s) feeding calibration (meth/unmeth controls when
        available, otherwise the test data) are split by strand
        before counting, and `theta_prot`/`theta_acc`/`informative`
        are returned as `{"+": ..., "-": ...}` dicts instead of plain
        arrays.

    Returns
    -------
    ctx : np.ndarray
        int8, length L, context code per position.
    theta_prot, theta_acc : np.ndarray or dict of str to np.ndarray
        float64, length L, or (if `norm_by_strand`) a `{"+": ...,
        "-": ...}` dict of such arrays.
    informative : np.ndarray or dict of str to np.ndarray
        bool, length L, or (if `norm_by_strand`) a `{"+": ...,
        "-": ...}` dict of such arrays.
    has_controls : bool
        Whether meth/unmeth controls were used.

    Raises
    ------
    ValueError
        If `chrom` has no reference sequence, if the no-controls
        path cannot find enough windows with `n_min` context-eligible
        sites, or if `norm_by_strand` is True but molecules with
        unmapped strands ('.') exist.

    Warns
    -----
    UserWarning
        If `exp.wrap` is True and many wrap-mirrored position pairs
        disagree on context.
    """
    raw = exp.raw[chrom]
    if raw.refseq is None:
        raise ValueError(
            f"No reference sequence for chrom '{chrom}'; "
            "model_prob needs 'fasta_file' at load_raw.")
    full_ctx = _reference_contexts(raw.refseq)
    if exp.wrap:
        nbp = raw.nbp
        length = len(raw.refseq)
        lower = full_ctx[:nbp]
        upper = full_ctx[length - nbp:][::-1]
        agree = lower == upper
        ctx = np.where(agree, lower, NONE).astype(lower.dtype)
        frac_disagree = 1.0 - agree.mean()
        if frac_disagree > _WRAP_CTX_DISAGREE_WARN_FRAC:
            warnings.warn(
                f"refseq for chrom '{chrom}' disagrees on "
                f"context at {(~agree).sum()}/{nbp} "
                f"({frac_disagree:.1%}) wrap-mirrored position "
                "pairs (folded to NONE there); this may "
                "indicate the wrong chromsize length or a "
                "non-symmetric reference, rather than a small "
                "loop/junction region.", stacklevel=3)
    else:
        ctx = full_ctx
    has_controls = (raw.meth_data is not None
                    and raw.unmeth_data is not None)

    if norm_by_strand:
        if (raw.test_data["strand"] == ".").any():
            raise ValueError(_UNMAPPED_STRAND_MSG)
        if has_controls and ((raw.meth_data["strand"] == ".").any()
                             or (raw.unmeth_data["strand"] == ".").any()):
            raise ValueError(_UNMAPPED_STRAND_MSG)

    if has_controls:
        if norm_by_strand:
            meth_strand = _strand_of_mol(raw.meth_data,
                                         len(raw.meth_mol_id))
            unmeth_strand = _strand_of_mol(raw.unmeth_data,
                                           len(raw.unmeth_mol_id))
            meth_counts = _streamed_call_count(
                exp, chrom, "meth", ctx, batch_size, mask_name,
                row_masks={"+": meth_strand == "+",
                          "-": meth_strand == "-"})
            unmeth_counts = _streamed_call_count(
                exp, chrom, "unmeth", ctx, batch_size, mask_name,
                row_masks={"+": unmeth_strand == "+",
                          "-": unmeth_strand == "-"})
            theta_prot, theta_acc, informative = {}, {}, {}
            for s in ("+", "-"):
                theta_prot[s], theta_acc[s], informative[s] = \
                    _calibrate_from_controls(
                        *meth_counts[s], *unmeth_counts[s], ctx,
                        nu=nu, rho_leak=rho_leak, min_gap=min_gap)
        else:
            meth_k, meth_n = _streamed_call_count(
                exp, chrom, "meth", ctx, batch_size, mask_name)
            unmeth_k, unmeth_n = _streamed_call_count(
                exp, chrom, "unmeth", ctx, batch_size, mask_name)
            theta_prot, theta_acc, informative = _calibrate_from_controls(
                meth_k, meth_n, unmeth_k, unmeth_n, ctx,
                nu=nu, rho_leak=rho_leak, min_gap=min_gap)
    else:
        if norm_by_strand:
            test_strand = _strand_of_mol(raw.test_data,
                                         len(raw.test_mol_id))
            win_counts = _streamed_window_count(
                exp, chrom, ctx, l_nuc, n_min, batch_size, mask_name,
                row_masks={"+": test_strand == "+",
                          "-": test_strand == "-"})
            theta_prot, theta_acc, informative = {}, {}, {}
            for s in ("+", "-"):
                K, N, codes = win_counts[s]
                theta_prot[s], theta_acc[s], informative[s] = \
                    _calibrate_from_data(
                        K, N, ctx, codes, iters=iters, init_prot=init_prot,
                        init_acc=init_acc, tol=tol, min_gap=min_gap)
        else:
            K, N, codes = _streamed_window_count(
                exp, chrom, ctx, l_nuc, n_min, batch_size, mask_name)
            theta_prot, theta_acc, informative = _calibrate_from_data(
                K, N, ctx, codes, iters=iters, init_prot=init_prot,
                init_acc=init_acc, tol=tol, min_gap=min_gap)
    return ctx, theta_prot, theta_acc, informative, has_controls


class MethPrintAnalysis:
    """
    Normalization and smoothing suite for MethPrintExperiment data.

    Process raw methylation signals into probability scores [0, 1] using
    rolling average smoothing and molecule-wise percentile scaling. Support
    relative normalization against unmethylated and fully methylated controls.
    """
    
    _EPSILON = np.finfo(float).eps  # Smallest float to avoid DivByZero

    def smooth(self, *, binsize : int,
               exp : MethPrintExperiment,
               name : str = "smoothed",
               nan_method : str = "mean",
               fill_edge : float | str = np.nan,
               batch_size : int = 20000,
               mask_name : str | None = None):
        """
        Smooth methylation signals across an experiment using a rolling
        average.

        Iterate through all chromosomes to perform rolling average smoothing.
        Molecules are processed in batches and streamed to a disk-backed
        array so that peak memory scales with `batch_size` rather than the
        total number of molecules. Results are stored in-place within the
        experiment's analysis map as `H5Array` objects.

        Parameters
        ----------
        binsize : int
            Window size in base pairs for the rolling average smoothing.
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        name : str, default "smoothed"
            The suffix used to store the resulting array in `exp.analysis`.
            Results are stored as 'test_{name}', 'meth_{name}', etc.
        nan_method : {"mean", "interpolate"}, default "mean"
            How to fill interior nan values (gaps with valid data on both
            sides): molecule mean, or linear interpolation.
        fill_edge : float or "mean", default np.nan
            How to fill leading/trailing edge nan values, independent of
            `nan_method`. A literal float must lie within the data range.
            See `catella.mapping.CoordsTransform.fill_edge`.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `MethPrintExperiment.filter_dropout(mask_name=mask_name)`
            call are set to all-NaN in the smoothed output, per
            source. Looked up as `exp.analysis[chrom][f"{source}_
            {mask_name}"]`.

        Raises
        ------
        ValueError
            If `nan_method` is not "mean" or "interpolate".
            If `fill_edge` is a literal float outside the data range.
        KeyError
            If `mask_name` is given but no matching mask is found for
            some source/chromosome.

        Notes
        -----
        Scratch files backing the resulting `H5Array` objects are
        written to `exp`'s scratch directory (see
        `MethPrintExperiment.resolve_tmp_dir`), shared with any other
        scratch files from the same experiment (e.g. from `load_raw` or
        `meth_prob`).
        """

        if nan_method not in ("mean", "interpolate"):
            raise ValueError("'nan_method' must be 'mean' or 'interpolate', "
                             f"got {nan_method!r}.")

        exp.global_analysis[f"{name}_params"] = pd.DataFrame([{
            "binsize": binsize, "nan_method": nan_method,
            "fill_edge": fill_edge, "mask_name": mask_name or ""}])
        tmp_dir = exp.resolve_tmp_dir()
        
        # Some helper functions
        def smooth_df(df, nbp, nmol, keep):
            if not isinstance(fill_edge, str) and not np.isnan(fill_edge):
                vmin, vmax = df["mod_qual"].min(), df["mod_qual"].max()
                if not (vmin <= fill_edge <= vmax):
                    raise ValueError(
                        f"'fill_edge'={fill_edge} is outside the data "
                        f"range [{vmin}, {vmax}].")

            all_pos = pd.Index(range(0, nbp))
            df = df.sort_values("mol_index", kind="stable")
            mol_index = df["mol_index"].to_numpy()
            out = H5Array.create((nmol, nbp), dtype=np.float64,
                                 dir=tmp_dir)

            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                lo, hi = np.searchsorted(mol_index, [start, stop])
                batch_df = df.iloc[lo:hi]

                # Pivot and reindex to ensure all positions are represented
                df_piv = batch_df.pivot(
                    index="pos", columns="mol_index",
                    values="mod_qual").reindex(all_pos)

                # Rolling mean centered by shifting and store the results
                # in a left-aligned manner
                res = df_piv.rolling(
                    window=binsize, min_periods=1).mean().shift(
                        -(binsize-1))

                # Interior nans (bounded by valid data on both sides) are
                # filled per nan_method; interpolate() with
                # limit_area="inside" only fills those, leaving edge nans
                # untouched so they can be probed for below.
                interior_interp = res.interpolate(method="linear",
                                                   limit_area="inside")
                if nan_method == "interpolate":
                    res = interior_interp
                else:
                    interior_mask = res.isna() & interior_interp.notna()
                    res = res.where(~interior_mask, res.fillna(res.mean()))

                # Fill remaining edge nans independently of nan_method
                if isinstance(fill_edge, str) and fill_edge == "mean":
                    res = res.fillna(res.mean())
                else:
                    res = res.fillna(fill_edge)

                out_batch = res.to_numpy().T
                if keep is not None:
                    out_batch = _apply_keep_mask(out_batch, keep[start:stop])
                out.write_batch(start, stop, out_batch)
            return out

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            nbp = raw.nbp
            def src_keep(src):
                return _lookup_mask(exp, chrom, src, mask_name) \
                    if mask_name is not None else None
            exp.analysis[chrom][f"test_{name}"] = smooth_df(
                raw.test_data, nbp, len(raw.test_mol_id),
                src_keep("test"))
            if raw.meth_data is not None:
                exp.analysis[chrom][f"meth_{name}"] = smooth_df(
                    raw.meth_data, nbp, len(raw.meth_mol_id),
                    src_keep("meth"))
            if raw.unmeth_data is not None:
                exp.analysis[chrom][f"unmeth_{name}"] = smooth_df(
                    raw.unmeth_data, nbp, len(raw.unmeth_mol_id),
                    src_keep("unmeth"))
                
            
    def empirical_prob(self, *, exp : MethPrintExperiment,
                  binsize : int | None = None,
                  smoothed_name : str = "smoothed",
                  prob_name : str = "meth_prob",
                  clip_low : float = 0.1,
                  clip_high : float = 99.9,
                  norm_by_strand : bool = False,
                  resmooth : bool = False,
                  nan_method : str = "mean",
                  fill_edge : float | str = np.nan,
                  batch_size : int = 20000,
                  percentile_sample_size : int = 100000,
                  seed : int | None = None,
                  mask_name : str | None = None):

        """
        Convert the smoothed methylation signal into a methylation probability
        profile with values ranging between 0 and 1.

        Iterate through all chromosomes to perform optional control-based
        relative normalization and percentile-based probability scaling.
        Molecules are processed in batches: normalization means are
        accumulated in a streaming pass, percentile clip bounds are
        estimated from a random subsample of molecules, and the final
        probabilities are streamed to a disk-backed array. Peak memory
        scales with `batch_size` rather than the total number of
        molecules.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        binsize : int, optional
            Window size in base pairs used for the rolling average smoothing.
            If None, looked up from `exp.global_analysis[f"{smoothed_
            name}_params"]`, set by a previous `smooth` call under the
            same `smoothed_name`.
        smoothed_name : str, default "smoothed"
            The dictionary key suffix for the smoothed signals in
            `exp.analysis`.
        prob_name : str, default "meth_prob"
            The dictionary key used to store the resulting methylation
            probabilities in `exp.analysis`.
        clip_low : float, default 0.1
            Lower percentile bound for signal clipping. Values below this
            percentile are set to 0. If `exp` has meth/unmeth controls,
            this percentile is estimated from the normalized unmeth
            control source; otherwise it is estimated from the test
            signal itself.
        clip_high : float, default 99.9
            Upper percentile bound for signal clipping. Values above this
            percentile are set to 1. If `exp` has meth/unmeth controls,
            this percentile is estimated from the normalized meth
            control source; otherwise it is estimated from the test
            signal itself.
        norm_by_strand : bool, default False
            Whether to perform normalization separately based on strandedness.
        resmooth : bool, default False
            If True, always re-run `smooth`, even if `test_{smoothed_
            name}` already exists in `exp.analysis`.
        nan_method : {"mean", "interpolate"}, default "mean"
            Passed through to `smooth` if smoothing is triggered
            lazily, and must match the previous smoothing if
            `test_{smoothed_name}` already exists (see `resmooth`).
        fill_edge : float or "mean", default np.nan
            Passed through to `smooth` if smoothing is triggered
            lazily. If `test_{smoothed_name}` already exists and only
            `fill_edge` differs from the previous smoothing, the
            existing edges are patched in place instead of requiring
            `resmooth`.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        percentile_sample_size : int, default 100000
            Approximate number of molecules used to estimate `clip_low`/
            `clip_high` percentile bounds. If the relevant source (the
            unmeth/meth controls when present, otherwise the test
            signal) has fewer molecules than this, all of them are used
            (exact bounds).
        seed : int, optional
            Seed for the random number generator used for percentile
            subsampling.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `MethPrintExperiment.filter_dropout(mask_name=mask_name)`
            call are excluded (set to NaN) from the test signal and
            from control-based normalization statistics, per source --
            applied fresh on every call, regardless of whether this
            call's `mask_name` matches what `smooth` itself last used.
            Also passed through to `smooth` if smoothing is triggered
            lazily.

        Raises
        ------
        ValueError
            If `binsize` is not provided and no cached `binsize` exists.
            If `test_{smoothed_name}` already exists and an
            explicitly-given `binsize`/`nan_method` (or `fill_edge`
            together with a mismatched `binsize`/`nan_method`) doesn't
            match what was used to produce it.
            If `norm_by_strand` is True but molecules with unmapped strands
            ('.') exist.
        KeyError
            If `mask_name` is given but no matching mask is found for
            some source/chromosome.

        Notes
        -----
        The scratch file backing the resulting probability `H5Array`,
        and any scratch files from smoothing triggered lazily, are
        written to `exp`'s scratch directory (see
        `MethPrintExperiment.resolve_tmp_dir`), shared with any other
        scratch files from the same experiment.
        """

        # Some helper functions
        def strands(df):
            pve = df[df["strand"] == "+"]["mol_index"].unique()
            nve = df[df["strand"] == "-"]["mol_index"].unique()
            uni = df[df["strand"] == "."]["mol_index"].unique()
            return pve, nve, uni

        def strand_of_mol(df, nmol):
            # One strand label per molecule, looked up by mol_index
            labels = np.full(nmol, ".", dtype="<U1")
            first = df.drop_duplicates("mol_index")[["mol_index", "strand"]]
            idx = first["mol_index"].to_numpy().astype(np.int64)
            labels[idx] = first["strand"].to_numpy()
            return labels

        def streamed_nanmean(arr, row_mask=None):
            # Per-position mean over molecules (rows), streamed in
            # batches, delegating to H5Array's shared reduction logic
            return arr._streamed_reduce("mean", 0, batch_size,
                                        row_mask=row_mask)

        def apply_norm(test_batch, meth_avg, unmeth_avg):
            denom = meth_avg - unmeth_avg
            # Avoid division by zero
            denom = np.where(np.abs(denom) < self._EPSILON,
                             self._EPSILON, denom)
            return (test_batch - unmeth_avg) / denom

        tmp_dir = exp.resolve_tmp_dir()

        # Check for cached binsize, or perform (or re-verify) smoothing
        already_smoothed = (
            f"test_{smoothed_name}" in exp.analysis[exp.chroms[0]])
        cached_params = exp.global_analysis.get(f"{smoothed_name}_params")

        if resmooth or not already_smoothed:
            if binsize is None and cached_params is not None:
                binsize = int(cached_params["binsize"].iloc[0])
            if binsize is None:
                raise ValueError("'binsize' must be specified if data are not "
                                 "already smoothed.")
            self.smooth(binsize=binsize, exp=exp, name=smoothed_name,
                       nan_method=nan_method, fill_edge=fill_edge,
                       batch_size=batch_size, mask_name=mask_name)
        elif cached_params is not None:
            row = cached_params.iloc[0]
            mismatched = []
            if binsize is not None and binsize != int(row["binsize"]):
                mismatched.append("binsize")
            if nan_method != "mean" and nan_method != row["nan_method"]:
                mismatched.append("nan_method")
            # mask_name is *not* compared here: unlike binsize/
            # nan_method, it's reapplied fresh to the output on every
            # call via _apply_keep_mask below, regardless of what (if
            # any) mask_name smooth() itself used.

            is_default_fill = (isinstance(fill_edge, float)
                               and np.isnan(fill_edge))
            old_fill_edge = row["fill_edge"]
            old_is_nan = (isinstance(old_fill_edge, float)
                         and np.isnan(old_fill_edge))
            fill_edge_changed = (not is_default_fill
                                 and fill_edge != old_fill_edge)

            if fill_edge_changed and not mismatched and old_is_nan:
                # Only fill_edge differs, and the stored edges are
                # still NaN -- patch in place instead of a resmooth.
                # Excludes rows dropped by whatever mask_name smooth()
                # itself used (baked into storage as all-NaN rows),
                # not this call's mask_name (applied fresh below).
                orig_mask_name = row["mask_name"] or None
                for chrom in exp.chroms:
                    raw = exp.raw[chrom]
                    ana_map = exp.analysis[chrom]
                    for source, df in (("test", raw.test_data),
                                       ("meth", raw.meth_data),
                                       ("unmeth", raw.unmeth_data)):
                        key = f"{source}_{smoothed_name}"
                        if df is None or key not in ana_map:
                            continue
                        keep = _lookup_mask(
                            exp, chrom, source, orig_mask_name) \
                            if orig_mask_name is not None else None
                        _patch_fill_edge(
                            ana_map[key], fill_edge,
                            df["mod_qual"].min(), df["mod_qual"].max(),
                            keep, batch_size)
                exp.global_analysis[f"{smoothed_name}_params"] = \
                    pd.DataFrame([{
                        "binsize": int(row["binsize"]),
                        "nan_method": row["nan_method"],
                        "fill_edge": fill_edge,
                        "mask_name": row["mask_name"]}])
            else:
                if fill_edge_changed:
                    mismatched.append("fill_edge")
                if mismatched:
                    raise ValueError(
                        f"'{smoothed_name}' was already smoothed with "
                        f"different {', '.join(mismatched)}; pass "
                        "resmooth=True to recompute, or use a "
                        "different 'smoothed_name'.")

        # binsize may still be unset if smoothing was skipped and the
        # caller didn't pass it; downstream code needs the real value.
        if binsize is None and cached_params is not None:
            binsize = int(cached_params["binsize"].iloc[0])

        rng = np.random.default_rng(seed)

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            ana = exp.analysis[chrom]
            test_arr = ana[f"test_{smoothed_name}"]
            if mask_name is not None:
                test_arr = _apply_keep_mask(
                    test_arr, _lookup_mask(exp, chrom, "test", mask_name),
                    batch_size)
            nmol, nbp = test_arr.shape

            has_controls = (raw.meth_data is not None and
                            raw.unmeth_data is not None)
            test_strand = None
            meth_avg = unmeth_avg = None
            meth_avg_pos = meth_avg_neg = None
            unmeth_avg_pos = unmeth_avg_neg = None

            if has_controls:
                meth_arr = ana[f"meth_{smoothed_name}"]
                unmeth_arr = ana[f"unmeth_{smoothed_name}"]
                if mask_name is not None:
                    meth_arr = _apply_keep_mask(
                        meth_arr,
                        _lookup_mask(exp, chrom, "meth", mask_name),
                        batch_size)
                    unmeth_arr = _apply_keep_mask(
                        unmeth_arr,
                        _lookup_mask(exp, chrom, "unmeth", mask_name),
                        batch_size)
                if norm_by_strand:
                    tp, tn, tu = strands(raw.test_data)
                    mp, mn, mu = strands(raw.meth_data)
                    up, un, uu = strands(raw.unmeth_data)
                    if len(tu) > 0 or len(mu) > 0 or len(uu) > 0:
                        raise ValueError("Cannot do normalization by strand "
                                         "with unmapped strands '.'.")
                    test_strand = strand_of_mol(raw.test_data, nmol)
                    meth_strand = strand_of_mol(raw.meth_data,
                                                meth_arr.shape[0])
                    unmeth_strand = strand_of_mol(raw.unmeth_data,
                                                  unmeth_arr.shape[0])
                    meth_avg_pos = streamed_nanmean(
                        meth_arr, meth_strand == "+")
                    meth_avg_neg = streamed_nanmean(
                        meth_arr, meth_strand == "-")
                    unmeth_avg_pos = streamed_nanmean(
                        unmeth_arr, unmeth_strand == "+")
                    unmeth_avg_neg = streamed_nanmean(
                        unmeth_arr, unmeth_strand == "-")
                else:
                    meth_avg = streamed_nanmean(meth_arr)
                    unmeth_avg = streamed_nanmean(unmeth_arr)

            def normalized_batch(start, stop):
                batch = test_arr[start:stop, :]
                if not has_controls:
                    return batch
                if norm_by_strand:
                    labels = test_strand[start:stop]
                    out = np.empty_like(batch)
                    pos, neg = labels == "+", labels == "-"
                    out[pos] = apply_norm(batch[pos], meth_avg_pos,
                                          unmeth_avg_pos)
                    out[neg] = apply_norm(batch[neg], meth_avg_neg,
                                          unmeth_avg_neg)
                    return out
                return apply_norm(batch, meth_avg, unmeth_avg)

            # Normalize the data so that all values are between 0 and 1
            # Exclude trailing edge created by rolling window
            end_idx = -(binsize-1) if binsize > 1 else None

            def normalized_sample(arr, strand_labels):
                # Random subsample of an array's molecules, normalized
                # the same way as `normalized_batch` when controls are
                # available (or left raw otherwise), sampled
                # proportionally within each contiguously-read batch.
                n_total = arr.shape[0]
                sample_size = min(percentile_sample_size, n_total)
                sample_frac = sample_size / n_total
                chunks = []
                for start in range(0, n_total, batch_size):
                    stop = min(start + batch_size, n_total)
                    batch = arr[start:stop, :]
                    if has_controls:
                        if norm_by_strand:
                            labels = strand_labels[start:stop]
                            normed = np.empty_like(batch)
                            pos, neg = labels == "+", labels == "-"
                            normed[pos] = apply_norm(
                                batch[pos], meth_avg_pos, unmeth_avg_pos)
                            normed[neg] = apply_norm(
                                batch[neg], meth_avg_neg, unmeth_avg_neg)
                            batch = normed
                        else:
                            batch = apply_norm(batch, meth_avg, unmeth_avg)
                    n_take = min(stop - start,
                                max(1, round((stop - start) * sample_frac)))
                    rows = rng.choice(stop - start, size=n_take,
                                      replace=False)
                    chunks.append(batch[rows, :end_idx])
                return np.concatenate(chunks, axis=0)

            # Estimate vmin/vmax from the normalized control sources
            # when available, so bounds depend only on the controls
            # (and are therefore comparable across test conditions
            # sharing the same controls) rather than on the test
            # signal's own, condition-dependent dynamic range. Fall
            # back to the test signal itself when there are no controls.
            if has_controls:
                unmeth_sample = normalized_sample(
                    unmeth_arr, unmeth_strand if norm_by_strand else None)
                meth_sample = normalized_sample(
                    meth_arr, meth_strand if norm_by_strand else None)
                vmin = np.nanpercentile(unmeth_sample, clip_low)
                vmax = np.nanpercentile(meth_sample, clip_high)
            else:
                test_sample = normalized_sample(test_arr, None)
                vmin = np.nanpercentile(test_sample, clip_low)
                vmax = np.nanpercentile(test_sample, clip_high)

            # Use the difference between percentiles as the scaling factor
            denom = np.maximum(vmax-vmin, self._EPSILON)

            # Map to probability [0,1] and stream the result to disk.
            # Clip to ensure that outliers outside the percentile bounds
            # do not result in probabilities < 0 or > 1.
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = normalized_batch(start, stop)
                prob = np.clip((batch-vmin)/denom, 0.0, 1.0)
                out.write_batch(start, stop, prob)
            ana[prob_name] = out

    def model_prob(self, *, exp : MethPrintExperiment,
                   prob_name : str = "meth_prob",
                   pi0 : float = 0.5,
                   eta : float | dict[str, float] | None = None,
                   eta_max_lag : int = 10,
                   store_rho : bool = False,
                   nu : float = 10.0,
                   rho_leak : float = 0.1,
                   min_gap : float = 0.05,
                   l_nuc : int = 147,
                   n_min : int = 10,
                   iters : int = 200,
                   init_prot : float = 0.05,
                   init_acc : float = 0.95,
                   tol : float = 1e-8,
                   fill_edge : float = np.nan,
                   norm_by_strand : bool = False,
                   batch_size : int = 20000,
                   mask_name : str | None = None):
        """
        Convert multi-channel methylation footprinting calls into a
        methylation probability profile with values ranging between 0
        and 1, using a calibrated Bayesian log-odds model.

        Iterate through all chromosomes, classify each reference
        position into a footprinting context (m6A/GpC/CpG/GCG) from
        `exp.raw[chrom].refseq`, estimate position-specific call rates
        under "protected" and "accessible" hypotheses -- from meth/
        unmeth controls when available, or by expectation-maximization
        on the test sample otherwise -- and convert each read's
        per-position log-odds into a left-aligned, window-summed
        probability. Molecules are processed in batches and streamed to
        a disk-backed array so that peak memory scales with
        `batch_size` rather than the total number of molecules,
        including during rate calibration.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        prob_name : str, default "meth_prob"
            The key used to store the resulting probabilities in
            `exp.analysis`.
        pi0 : float, default 0.5
            Prior probability that an assayable site is methylated, used
            to convert each site's `mod_qual` confidence score into a
            likelihood ratio. 0.5 is the uninformative choice used when
            the base caller's training prior is unknown.
        eta : float, dict of str to float, or None, default None
            Per-channel multiplicative correction for inflated log-
            likelihood ratios from correlated nearby sites (e.g.
            palindromic CpG/GpC positions). If None (default), each
            channel's ("M6A"/"GCH"/"HCG"/"GCG") eta is auto-estimated
            from lag-k autocorrelation in its log-odds -- from the
            methylated control when available, or from the test data
            otherwise. A float pins every channel to that value; a
            dict pins only the named channels, leaving any not
            mentioned to be auto-estimated. The value(s) actually
            applied, along with every other parameter below, are
            stored in `exp.global_analysis[f"{prob_name}_params"]`
            afterward.
        eta_max_lag : int, default 10
            Maximum lag (bp) summed over when auto-estimating eta.
        store_rho : bool, default False
            If True, store the lag-k autocorrelation `rho(k)` used in
            each channel's eta estimation to
            `exp.global_analysis[f"{prob_name}_rho"]`, one column per
            channel that was actually auto-estimated (channels pinned
            via `eta`, or with too little data to estimate, are
            omitted).
        nu : float, default 10.0
            Pseudo-count strength for shrinking each position's call
            rate toward its context group's mean; larger values shrink
            harder at low depth.
        rho_leak : float, default 0.1
            Leak fraction in [0, 1] interpolating the protected-state
            call rate between the unmethylated control's false-positive
            rate (0 for perfect protection) and the accessible-state rate.
            Used only when meth/unmeth controls are available.
        min_gap : float, default 0.05
            Minimum required gap between the accessible and protected
            call rates for a position to be treated as informative;
            positions below this gap contribute no evidence.
        l_nuc : int, default 147
            Nucleosome footprint size (bp): the expectation-maximization
            window size (no-controls path) and the output window-sum
            size.
        n_min : int, default 10
            Minimum number of context-eligible sites a window must have
            to be used in the no-controls expectation-maximization fit.
            Used only when no meth/unmeth controls are available.
        iters : int, default 200
            Maximum number of expectation-maximization iterations for
            the no-controls rate fit.
        init_prot : float, default 0.05
            Initial guess for the protected-state call rate in the
            no-controls expectation-maximization fit.
        init_acc : float, default 0.95
            Initial guess for the accessible-state call rate in the
            no-controls expectation-maximization fit.
        tol : float, default 1e-8
            Relative log-likelihood convergence tolerance for the
            no-controls expectation-maximization fit.
        fill_edge : float, default nan
            Probability used to fill the trailing `l_nuc - 1` positions
            of the result, which have no full window to summarize. The
            nan default leaves those positions unfilled.
        norm_by_strand : bool, default False
            Whether to calibrate theta_prot/theta_acc separately per
            strand before scoring. If True, the source(s) feeding
            calibration (meth/unmeth controls when available,
            otherwise the test data via expectation-maximization) are
            split by strand, and each test read is scored against its
            own strand's calibration. `eta` is still auto-estimated
            pooled across strands, since the crosstalk it corrects for
            is an assay-chemistry property rather than a strand-
            specific one.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per
            batch.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `MethPrintExperiment.filter_dropout(mask_name=mask_name)`
            call are excluded from rate calibration, eta estimation,
            and the output, per source.

        Raises
        ------
        ValueError
            If a chromosome has no reference sequence (`refseq`), if
            the no-controls path cannot find enough windows with
            `n_min` context-eligible sites, if `eta` is a dict with an
            unrecognized channel name, or if `norm_by_strand` is True
            but molecules with unmapped strands ('.') exist.
        KeyError
            If `mask_name` is given but no matching mask is found for
            some source/chromosome.

        Warns
        -----
        UserWarning
            If `exp.wrap` is True and many wrap-mirrored position
            pairs disagree on context (folded to no context there).

        Notes
        -----
        Position `i` in the result summarizes the window
        `[i, i + l_nuc)`. If `exp.wrap` is True, `refseq` (always
        stored full-length) is folded to length `nbp` before use:
        positions `i` and `length-1-i` must classify to the same
        context to be kept; where they disagree, the folded position
        is treated as having no context.
        """
        from scipy.special import expit, logit

        tmp_dir = exp.resolve_tmp_dir()
        log_odds_fill = -logit(fill_edge)

        overrides = _resolve_eta_overrides(eta)
        need_auto = set(_ETA_CHANNELS) - set(overrides)
        channel_eta = dict(overrides)

        calib_kwargs = dict(nu=nu, rho_leak=rho_leak, min_gap=min_gap,
                            l_nuc=l_nuc, n_min=n_min, iters=iters,
                            init_prot=init_prot, init_acc=init_acc,
                            tol=tol, batch_size=batch_size,
                            mask_name=mask_name,
                            norm_by_strand=norm_by_strand)

        calib_cache = {}
        if need_auto:
            stats = _init_autocorr_stats(eta_max_lag)
            for chrom in exp.chroms:
                ctx, theta_prot, theta_acc, informative, has_controls = \
                    _chrom_calibration(exp, chrom, **calib_kwargs)
                calib_cache[chrom] = (ctx, theta_prot, theta_acc,
                                      informative)
                # meth control isolates the crosstalk artifact from
                # real, occupancy-driven correlation in the test data
                source = "meth" if has_controls else "test"
                strand_labels = None
                if norm_by_strand:
                    raw = exp.raw[chrom]
                    src_df = getattr(raw, f"{source}_data")
                    src_mol_id = getattr(raw, f"{source}_mol_id")
                    strand_labels = _strand_of_mol(src_df, len(src_mol_id))
                _accumulate_eta_source(
                    exp, chrom, source, ctx, theta_prot, theta_acc,
                    informative, pi0, batch_size, mask_name, stats,
                    eta_max_lag, need_auto, norm_by_strand=norm_by_strand,
                    strand_labels=strand_labels)
            estimated, rho_by_channel = _finalize_channel_eta(
                stats, eta_max_lag)
            channel_eta.update({c: estimated[c] for c in need_auto})

            if store_rho and rho_by_channel:
                exp.global_analysis[f"{prob_name}_rho"] = pd.DataFrame({
                    "lag": np.arange(1, eta_max_lag + 1),
                    **{f"rho_{CONTEXT_NAMES[c]}": rho_by_channel[c]
                      for c in _ETA_CHANNELS if c in rho_by_channel}})

        exp.global_analysis[f"{prob_name}_params"] = pd.DataFrame([{
            "pi0": pi0, "eta_max_lag": eta_max_lag, "nu": nu,
            "rho_leak": rho_leak, "min_gap": min_gap, "l_nuc": l_nuc,
            "n_min": n_min, "iters": iters, "init_prot": init_prot,
            "init_acc": init_acc, "tol": tol, "fill_edge": fill_edge,
            "norm_by_strand": norm_by_strand,
            "mask_name": mask_name or "",
            **{f"eta_{CONTEXT_NAMES[c]}": channel_eta[c]
              for c in _ETA_CHANNELS}}])

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            if chrom in calib_cache:
                ctx, theta_prot, theta_acc, informative = \
                    calib_cache[chrom]
            else:
                ctx, theta_prot, theta_acc, informative, _ = \
                    _chrom_calibration(exp, chrom, **calib_kwargs)

            eta_by_pos = np.ones(len(ctx))
            for c in _ETA_CHANNELS:
                eta_by_pos[ctx == c] = channel_eta[c]

            nmol = len(raw.test_mol_id)
            out = H5Array.create((nmol, raw.nbp), dtype=np.float64,
                                 dir=tmp_dir)
            test_arr = exp.to_dense(chrom, which="test", as_h5array=True,
                                    batch_size=batch_size,
                                    mask_name=mask_name)
            ctx_ok = ctx[None, :] != NONE
            test_strand = None
            if norm_by_strand:
                test_strand = _strand_of_mol(raw.test_data, nmol)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = test_arr[start:stop, :]
                # mask_name marks dropped molecules as all-nan rows (see
                # _apply_keep_mask); propagate that to the output too.
                masked = np.isnan(batch).all(axis=1)
                q = np.where(ctx_ok, batch, np.nan)
                if norm_by_strand:
                    labels = test_strand[start:stop]
                    pos, neg = labels == "+", labels == "-"
                    log_odds = np.empty_like(q)
                    log_odds[pos] = _per_base_log_odds(
                        q[pos], theta_prot["+"], theta_acc["+"],
                        informative["+"], pi0=pi0, eta=eta_by_pos)
                    log_odds[neg] = _per_base_log_odds(
                        q[neg], theta_prot["-"], theta_acc["-"],
                        informative["-"], pi0=pi0, eta=eta_by_pos)
                else:
                    log_odds = _per_base_log_odds(q, theta_prot, theta_acc,
                                                  informative, pi0=pi0,
                                                  eta=eta_by_pos)
                log_odds_win = _window_sum_log_odds(
                    log_odds, l_nuc, fill_edge=log_odds_fill)
                prob = expit(-log_odds_win)
                prob[masked, :] = np.nan
                out.write_batch(start, stop, prob)
            exp.analysis[chrom][prob_name] = out

    def sort_by_linkage(self, *,
                        exp : MethPrintExperiment,
                        data_name : str = "test_smoothed",
                        raw_which : str | None = None,
                        mask_name : str | None = None,
                        sorted_name : str | None = None,
                        store_link_mat : bool = True,
                        link_mat_name : str | None = None,
                        metric : str = "euclidean",
                        method : str = "ward",
                        batch_size : int = 20000,
                        fill_nan : str | float | None = None
                        ) -> dict[str, np.ndarray]:
        """
        Sort molecules by hierarchical-clustering similarity.

        Iterate through all chromosomes, reordering rows (molecules)
        of a dense methylation signal so that similar molecules sit
        next to each other, matching the leaf order of a
        hierarchical-clustering dendrogram. By default, sorts the
        smoothed test signal if `smooth` has been run, or the raw
        test signal otherwise -- so this works whether or not `smooth`
        has been called first.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        data_name : str, default "test_smoothed"
            The key of the dense analysis array to sort, looked up in
            `exp.analysis[chrom]` (e.g. a `smooth` or `meth_prob`
            output). Ignored if `raw_which` is given. If left at its
            default and not found, falls back to raw test data (see
            above); any other missing `data_name` raises `KeyError`.
        raw_which : {"test", "meth", "unmeth"}, optional
            If given, source the data to sort from the raw long-form
            table instead of `exp.analysis`, via
            `exp.to_dense(chrom, which=raw_which)`. Takes priority
            over `data_name`.
        mask_name : str, optional
            Only used with `raw_which`. If given, forwarded to
            `exp.to_dense(chrom, which=raw_which, mask_name=mask_name)`
            so molecules flagged as dropout by a prior
            `filter_dropout(mask_name=mask_name)` call are excluded
            (returned as all-NaN rows) before sorting.
        sorted_name : str, optional
            The key used to store the sorted result. If None
            (default), `f"{data_name}_sorted"` is used, or
            `f"{raw_which}_sorted"` if `raw_which` is given.
        store_link_mat : bool, default True
            Whether to also persist each chromosome's linkage matrix
            into `exp.analysis[chrom][link_mat_name]`. If False, the
            linkage matrix is only returned, not stored.
        link_mat_name : str, optional
            The key used to store the linkage matrix if
            `store_link_mat` is True. If None (default),
            `f"{data_name}_linkage"` is used, or
            `f"{raw_which}_linkage"` if `raw_which` is given.
        metric : str, default "euclidean"
            Distance metric, forwarded to `utils.compute_linkage`.
        method : str, default "ward"
            Linkage method, forwarded to `utils.compute_linkage`.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per
            batch.
        fill_nan : {"mean"}, float, or None, default None
            How to handle `nan` values, forwarded to
            `utils.compute_linkage`.

        Returns
        -------
        dict of str to np.ndarray
            A mapping from chromosome name to that chromosome's
            linkage matrix, for optional immediate use (e.g. passing
            straight to `MethPlot.plot_methmap`'s `link_mat` argument).
            Also persisted into `exp.analysis[chrom][link_mat_name]` if
            `store_link_mat` is True.

        Raises
        ------
        ValueError
            If `mask_name` is given but `raw_which` is None.
        KeyError
            If `raw_which` is None, `data_name` is not `"test_smoothed"`
            (its default), and not found in `exp.analysis[chrom]` for
            some chromosome.
        """
        if mask_name is not None and raw_which is None:
            raise ValueError(
                "'mask_name' requires 'raw_which' to be given.")
        tmp_dir = exp.resolve_tmp_dir()
        link_mats = {}
        for chrom in exp.chroms:
            ana = exp.analysis[chrom]
            if raw_which is not None:
                data = exp.to_dense(chrom, which=raw_which,
                                    as_h5array=True, batch_size=batch_size,
                                    mask_name=mask_name)
                base_name = raw_which
            elif data_name not in ana and data_name == "test_smoothed":
                # Default target not computed yet -- fall back to the
                # raw test signal rather than requiring smooth() first.
                data = exp.to_dense(chrom, which="test",
                                    as_h5array=True, batch_size=batch_size)
                base_name = "test"
            else:
                if data_name not in ana:
                    raise KeyError(
                        f"'{data_name}' not found in "
                        f"exp.analysis['{chrom}']. Run 'smooth' or "
                        "'meth_prob' first (matching the name/prob_name "
                        "used there), or pass 'raw_which' to source "
                        "from the raw long-form data instead.")
                data = ana[data_name]
                base_name = data_name
            name = sorted_name if sorted_name is not None \
                else f"{base_name}_sorted"

            order, link_mat = utils.compute_linkage(
                data, metric=metric, method=method, batch_size=batch_size,
                dir=tmp_dir, fill_nan=fill_nan)
            if isinstance(data, H5Array):
                sorted_data = data.reorder_rows(order, dir=tmp_dir,
                                                batch_size=batch_size)
            else:
                sorted_data = data.iloc[order]
            exp.analysis[chrom][name] = sorted_data
            if store_link_mat:
                lname = link_mat_name if link_mat_name is not None \
                    else f"{base_name}_linkage"
                exp.analysis[chrom][lname] = pd.DataFrame(link_mat)
            link_mats[chrom] = link_mat
        return link_mats
