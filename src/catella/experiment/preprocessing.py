# preprocessing.py

import warnings
import numpy as np
import pandas as pd
from typing import List
from catella.experiment.methdata import MethPrintExperiment, _apply_keep_mask
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


NONE, M6A, GCH, HCG, GCG = 0, 1, 2, 3, 4
CONTEXT_NAMES = {NONE: "none", M6A: "M6A", GCH: "GCH", HCG: "HCG",
                 GCG: "GCG"}

# Warn (not raise) if more than this fraction of wrap-mirrored position
# pairs disagree on context when folding ctx in model_prob.
_WRAP_CTX_DISAGREE_WARN_FRAC = 0.02


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


def _streamed_call_count(exp, chrom, which, ctx, batch_size, mask_name):
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

    Returns
    -------
    k : np.ndarray
        float64, length L, per-position sum of `mod_qual` confidence
        scores across every molecule in the sample (nan/no-call
        positions excluded).
    n_mol : int
        Molecule count in the sample.
    """
    arr = exp.to_dense(chrom, which=which, as_h5array=True,
                       batch_size=batch_size, mask_name=mask_name)
    n_total, L = arr.shape
    ctx_ok = ctx != NONE
    k = np.zeros(L, dtype=np.float64)
    n_mol = 0
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = arr[start:stop, :]
        # mask_name marks dropped molecules as all-nan rows (see
        # _apply_keep_mask); exclude them so they don't inflate n_mol.
        kept = ~np.isnan(batch).all(axis=1)
        q = np.where(ctx_ok[None, :], batch[kept], np.nan)
        k += np.nansum(q, axis=0)
        n_mol += int(kept.sum())
    return k, n_mol


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
                           mask_name):
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

    Returns
    -------
    K, N : np.ndarray
        float64, (n_ctx, n_mol * n_kept_windows). `K` is the per-window,
        per-read sum of `mod_qual` confidence scores (nan/no-call
        positions excluded).
    codes : list of int
        Context codes present in `ctx`, in the same order as `K`/`N`'s
        first axis.

    Raises
    ------
    ValueError
        If `ctx` has no assayable contexts, or too few windows have
        `n_min` context-eligible sites.
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
    K_batches = {c: [] for c in codes}
    n_mol = 0
    for start in range(0, n_total, batch_size):
        stop = min(start + batch_size, n_total)
        batch = arr[start:stop, :]
        # mask_name marks dropped molecules as all-nan rows (see
        # _apply_keep_mask); exclude them so they don't inflate n_mol.
        batch = batch[~np.isnan(batch).all(axis=1)]
        n_mol += batch.shape[0]
        q = np.where(ctx_ok[None, :], batch, np.nan)
        for c in codes:
            sel = ctx == c
            k_c = np.stack(
                [np.nansum(q[:, s:s + l_nuc][:, sel[s:s + l_nuc]], axis=1)
                for s in kept_starts], axis=1)
            K_batches[c].append(k_c)

    if n_mol * keep.sum() < 10:
        raise ValueError("too few windows with >= n_min context-eligible "
                         "sites")

    K = np.array([np.concatenate(K_batches[c], axis=0).ravel()
                 for c in codes]).astype(np.float64)
    N = np.array([np.tile(n_win[c][keep], n_mol)
                 for c in codes]).astype(np.float64)
    return K, N, codes


def _calibrate_from_data(K, N, ctx, codes, iters=200, eps=1e-3,
                         init_prot=0.05, init_acc=0.60, tol=1e-8,
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
    init_acc : float, default 0.60
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
    eta : float, default 1.0
        Multiplicative correction for inflated log-likelihood ratios
        from correlated nearby sites (e.g. palindromic CpG/GpC); 1.0
        leaves the log-odds unscaled.

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


class MethPrintAnalysis:
    """
    Normalization and smoothing suite for MethPrintExperiment data.

    Process raw methylation signals into probability scores [0, 1] using
    rolling average smoothing and molecule-wise percentile scaling. Support
    relative normalization against unmethylated and fully methylated controls.
    """
    
    _EPSILON = np.finfo(float).eps  # Smallest float to avoid DivByZero

    def __init__(self):
        self._binsize = None # Cache the binsize used for smoothing

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

        self._binsize = binsize
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
            If None, use the cached binsize a previous `smooth` call.
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
        nan_method : {"mean", "interpolate"}, default "mean"
            Passed through to `smooth` if smoothing is triggered lazily.
        fill_edge : float or "mean", default np.nan
            Passed through to `smooth` if smoothing is triggered lazily.
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
            from control-based normalization statistics, per source.
            Also passed through to `smooth` if smoothing is triggered
            lazily.

        Raises
        ------
        ValueError
            If `binsize` is not provided and no cached `binsize` exists.
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
            labels[first["mol_index"].to_numpy()] = first["strand"].to_numpy()
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

        # Check for cached binsize or perform smoothing if data missing
        if binsize is None:
            binsize = self._binsize

        if f"test_{smoothed_name}" not in exp.analysis[exp.chroms[0]]:
            if binsize is None:
                raise ValueError("'binsize' must be specified if data are not "
                                 "already smoothed.")
            self.smooth(binsize=binsize, exp=exp, name=smoothed_name,
                       nan_method=nan_method, fill_edge=fill_edge,
                       batch_size=batch_size, mask_name=mask_name)

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
                   eta : float = 1.0,
                   nu : float = 10.0,
                   rho_leak : float = 0.1,
                   min_gap : float = 0.05,
                   l_nuc : int = 147,
                   n_min : int = 10,
                   iters : int = 200,
                   init_prot : float = 0.05,
                   init_acc : float = 0.60,
                   tol : float = 1e-8,
                   fill_edge : float = np.nan,
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
        eta : float, default 1.0
            Multiplicative correction for inflated log-likelihood ratios
            from correlated nearby sites (e.g. palindromic CpG/GpC
            positions); 1.0 leaves the log-odds unscaled.
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
        init_acc : float, default 0.60
            Initial guess for the accessible-state call rate in the
            no-controls expectation-maximization fit.
        tol : float, default 1e-8
            Relative log-likelihood convergence tolerance for the
            no-controls expectation-maximization fit.
        fill_edge : float, default nan
            Probability used to fill the trailing `l_nuc - 1` positions
            of the result, which have no full window to summarize. The
            nan default leaves those positions unfilled.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per
            batch.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `MethPrintExperiment.filter_dropout(mask_name=mask_name)`
            call are excluded from rate calibration and from the
            output, per source.

        Raises
        ------
        ValueError
            If a chromosome has no reference sequence (`refseq`), or if
            the no-controls path cannot find enough windows with
            `n_min` context-eligible sites.
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

        for chrom in exp.chroms:
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
                        "loop/junction region.", stacklevel=2)
            else:
                ctx = full_ctx
            has_controls = (raw.meth_data is not None
                            and raw.unmeth_data is not None)

            if has_controls:
                meth_k, meth_n = _streamed_call_count(
                    exp, chrom, "meth", ctx, batch_size, mask_name)
                unmeth_k, unmeth_n = _streamed_call_count(
                    exp, chrom, "unmeth", ctx, batch_size, mask_name)
                theta_prot, theta_acc, informative = \
                    _calibrate_from_controls(
                        meth_k, meth_n, unmeth_k, unmeth_n, ctx,
                        nu=nu, rho_leak=rho_leak, min_gap=min_gap)
            else:
                K, N, codes = _streamed_window_count(
                    exp, chrom, ctx, l_nuc, n_min, batch_size, mask_name)
                theta_prot, theta_acc, informative = _calibrate_from_data(
                    K, N, ctx, codes, iters=iters, init_prot=init_prot,
                    init_acc=init_acc, tol=tol, min_gap=min_gap)

            nmol = len(raw.test_mol_id)
            out = H5Array.create((nmol, raw.nbp), dtype=np.float64,
                                 dir=tmp_dir)
            test_arr = exp.to_dense(chrom, which="test", as_h5array=True,
                                    batch_size=batch_size,
                                    mask_name=mask_name)
            ctx_ok = ctx[None, :] != NONE
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = test_arr[start:stop, :]
                # mask_name marks dropped molecules as all-nan rows (see
                # _apply_keep_mask); propagate that to the output too.
                masked = np.isnan(batch).all(axis=1)
                q = np.where(ctx_ok, batch, np.nan)
                log_odds = _per_base_log_odds(q, theta_prot, theta_acc,
                                              informative, pi0=pi0, eta=eta)
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
        KeyError
            If `raw_which` is None, `data_name` is not `"test_smoothed"`
            (its default), and not found in `exp.analysis[chrom]` for
            some chromosome.
        """
        tmp_dir = exp.resolve_tmp_dir()
        link_mats = {}
        for chrom in exp.chroms:
            ana = exp.analysis[chrom]
            if raw_which is not None:
                data = exp.to_dense(chrom, which=raw_which,
                                    as_h5array=True, batch_size=batch_size)
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
