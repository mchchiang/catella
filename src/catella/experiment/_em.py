# _em.py

from contextlib import closing

import numpy as np

from catella.experiment._contexts import _ALL_CHANNELS, NONE
from catella.experiment._reads import _iter_read_sites, _read_weights

# Mixing fractions outside this range mark a degenerate fit
_PI_RANGE = (0.02, 0.98)
# Quantiles of the per-window rate that split the data-driven starts
_START_QUANTILES = (0.25, 0.5, 0.75)


def _streamed_window_count(
    exp,
    chrom,
    ctx,
    assay,
    lnuc,
    n_min,
    batch_size,
    mask_name,
    by_strand=False,
    channels=None,
    span="full",
):
    """
    Per-window, per-read summed methylation-calling confidence and
    trial counts for the test sample, streamed in batches so peak
    memory is O(batch_size x L) rather than O(n_mol x L).

    Windows are non-overlapping. A window's trial count on a read is
    its number of sites read by that read (see `_read_sites`), so it
    varies by read, and a read site without a call is a trial with
    confidence 0. A blank read counts its sites with their expected
    strand weights (see `_read_weights`). Read-windows below `n_min`
    read sites are dropped.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    ctx : np.ndarray
        int8, length L, context code per position.
    assay : dict of str to np.ndarray
        From `_strand_assay_masks`.
    lnuc : int
        Nucleosome footprint size (bp), used as the window size.
    n_min : int
        Minimum read sites (summed over all contexts) a read-window
        needs to be kept.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to `_iter_read_sites`.
    by_strand : bool, default False
        If True, `K`/`N`/`codes` are accumulated separately per strand
        from a single streamed pass, and the return value is keyed by
        strand.
    channels : iterable of int, optional
        Context channels eligible to be counted. Defaults to all of
        `_ALL_CHANNELS`.
    span : {"full", "calls"}, default "full"
        Forwarded to `_iter_read_sites`.

    Returns
    -------
    K, N : np.ndarray
        float64, (n_ctx, n_kept), over the kept read-windows in
        read-major order. `K` is the sum of `mod_qual` confidence
        scores and `N` the (expected) number of read sites, per
        context. Only if `by_strand` is False.
    codes : list of int
        Channels from `channels` present in `ctx`, in the same order
        as `K`/`N`'s first axis. Only if `by_strand` is False.
    counts : dict of str to (K, N, codes)
        One `(K, N, codes)` triple per strand ("+"/"-"). Only if
        `by_strand` is True.

    Raises
    ------
    ValueError
        If `ctx` has no assayable contexts within `channels`, or too
        few read-windows have `n_min` read sites (per strand, when
        `by_strand` is True).
    """
    if channels is None:
        channels = _ALL_CHANNELS
    codes = [c for c in channels if (ctx == c).any()]
    if not codes:
        raise ValueError("no assayable contexts in ctx")
    n_win = len(ctx) // lnuc
    width = n_win * lnuc
    in_code = np.stack([ctx[:width] == c for c in codes])[:, None, :]

    keys = ["+", "-"] if by_strand else [None]
    K_batches = {key: [] for key in keys}
    N_batches = {key: [] for key in keys}
    batches = _iter_read_sites(
        exp,
        chrom,
        "test",
        ctx,
        assay,
        batch_size=batch_size,
        mask_name=mask_name,
        span=span,
    )
    with closing(batches):
        for b in batches:
            for key in keys:
                sel = b.keep if key is None else b.keep & (b.labels == key)
                sel = sel | b.blank
                m = int(sel.sum())
                if m == 0 or n_win == 0:
                    continue
                shape = (len(codes), m, n_win, lnuc)
                q_k = np.nan_to_num(b.q[sel, :width], nan=0.0)[None]
                w_k = _read_weights(b, assay, key)[sel, :width][None]
                K = (q_k * in_code).reshape(shape).sum(axis=3)
                N = (w_k * in_code).reshape(shape).sum(axis=3)
                ok = N.sum(axis=0) >= n_min
                K_batches[key].append(K[:, ok])
                N_batches[key].append(N[:, ok])

    def _build(key):
        if sum(b.shape[1] for b in N_batches[key]) < 10:
            suffix = "" if key is None else f" for strand {key!r}"
            raise ValueError(
                f"too few windows with >= n_min read sites{suffix}"
            )
        K = np.concatenate(K_batches[key], axis=1)
        N = np.concatenate(N_batches[key], axis=1)
        return K, N, codes

    if not by_strand:
        return _build(None)
    return {key: _build(key) for key in keys}


def _log_binom_coef_sum(K, N):
    """
    Sum of the log binomial coefficients over all windows and channels.

    `K` is a confidence-score sum, usually fractional, so the
    Gamma-function extension of the coefficient is used; it agrees with
    `scipy.stats.binom.logpmf` for integer `K`. The sum does not depend
    on the rates, so it is computed once per fit.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs), from `_streamed_window_count`.

    Returns
    -------
    float
    """
    from scipy.special import gammaln

    return float((gammaln(N + 1) - gammaln(K + 1) - gammaln(N - K + 1)).sum())


def _mixture_loglik(K, N, tp, ta, pi, const):
    """
    Log-likelihood of the shared two-state mixture, and the posterior
    probability that each window is protected.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    tp, ta : np.ndarray
        float64, length n_ctx, protected and accessible rates.
    pi : float
        Mixing fraction of the protected state.
    const : float
        From `_log_binom_coef_sum`.

    Returns
    -------
    ll : float
    r : np.ndarray
        float64, length n_obs, P(protected | window).
    """
    lp = np.log(pi) + (
        K * np.log(tp)[:, None] + (N - K) * np.log1p(-tp)[:, None]
    ).sum(axis=0)
    la = np.log1p(-pi) + (
        K * np.log(ta)[:, None] + (N - K) * np.log1p(-ta)[:, None]
    ).sum(axis=0)
    m = np.maximum(lp, la)
    ll = const + (m + np.log(np.exp(lp - m) + np.exp(la - m))).sum()
    r = 1.0 / (1.0 + np.exp(la - lp))
    return float(ll), r


def _run_em(K, N, tp, ta, pi, const, *, max_iters, eps, tol):
    """
    Run expectation-maximization from one start.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    tp, ta : np.ndarray
        float64, length n_ctx, starting rates.
    pi : float
        Starting mixing fraction.
    const : float
        From `_log_binom_coef_sum`.
    max_iters, eps, tol
        As in `_calibrate_from_data`.

    Returns
    -------
    tuple
        `(tp, ta, pi, ll, n_iter)`, with `ll` the log-likelihood at the
        returned parameters.
    """
    tp, ta = tp.copy(), ta.copy()
    prev = -np.inf
    n_iter = 0
    for it in range(max_iters):
        n_iter = it + 1
        ll, r = _mixture_loglik(K, N, tp, ta, pi, const)
        dp, da = (r * N).sum(axis=1), ((1 - r) * N).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            new_tp = np.clip((r * K).sum(axis=1) / dp, eps, 1 - eps)
            new_ta = np.clip(((1 - r) * K).sum(axis=1) / da, eps, 1 - eps)
        tp = np.where(dp > 0, new_tp, tp)
        ta = np.where(da > 0, new_ta, ta)
        pi = float(r.mean())
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    ll, _ = _mixture_loglik(K, N, tp, ta, pi, const)
    return tp, ta, pi, ll, n_iter


def _em_starts(K, N, init_prot, init_acc, eps):
    """
    Starting values for the expectation-maximization fit.

    The first start is the user's guess (`init_prot`, `init_acc`, pi
    0.75). For each q in `_START_QUANTILES`, a data-driven start ranks
    windows by their rate pooled over channels, puts the windows at or
    below the q-quantile in the protected state and the rest in the
    accessible state, and takes each channel's pooled rate within each
    group, with pi = q. A channel without sites in a group falls back
    to its overall pooled rate.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    init_prot, init_acc, eps
        As in `_calibrate_from_data`.

    Returns
    -------
    list of tuple
        `(tp, ta, pi)` per start, with `tp`/`ta` float64 arrays of
        length n_ctx.
    """
    n_ctx = K.shape[0]
    starts = [(np.full(n_ctx, init_prot), np.full(n_ctx, init_acc), 0.75)]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = K.sum(axis=0) / N.sum(axis=0)
        pooled = K.sum(axis=1) / N.sum(axis=1)

    def _group_rate(sel):
        k, n = K[:, sel].sum(axis=1), N[:, sel].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(n > 0, k / n, pooled)
        return np.clip(np.nan_to_num(g, nan=0.5), eps, 1 - eps)

    for q in _START_QUANTILES:
        low = rate <= np.nanquantile(rate, q)
        starts.append((_group_rate(low), _group_rate(~low), q))
    return starts


def _calibrate_from_data(
    K,
    N,
    ctx,
    codes,
    max_iters=200,
    eps=1e-3,
    init_prot=0.05,
    init_acc=0.95,
    tol=1e-8,
    min_gap=0.05,
):
    """
    Fit theta_prot/theta_acc per context with expectation-maximization,
    from precomputed per-window call counts, when no meth/unmeth
    controls are available.

    Fits a two-component binomial mixture (a shared latent protection
    state per window, pooled across reads), since without controls the
    two rates must be estimated from the test sample alone. The state
    and its mixing fraction are shared by all channels, while each
    channel has its own pair of rates. The fit is run from every start
    of `_em_starts`, and the one with the largest log-likelihood is
    kept, among those whose mixing fraction lies within `_PI_RANGE`
    (among all starts if none does). If the kept fit has the labels
    swapped, the rates are swapped and the mixing fraction is flipped
    to `1 - pi`.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs), from `_streamed_window_count`.
    ctx : np.ndarray
        int8, length L, context code per position.
    codes : list of int
        Context codes present in `ctx`, in the same order as `K`/`N`'s
        first axis.
    max_iters : int, default 200
        Maximum number of expectation-maximization iterations.
    eps : float, default 1e-3
        Clip bound keeping rates away from exactly 0 or 1.
    init_prot : float, default 0.05
        Initial guess for theta_prot in the first start.
    init_acc : float, default 0.95
        Initial guess for theta_acc in the first start.
    tol : float, default 1e-8
        Relative log-likelihood convergence tolerance.
    min_gap : float, default 0.05
        Forwarded to the informative-mask computation.

    Returns
    -------
    theta_prot, theta_acc : np.ndarray
        float64, length L. Rates are constant within a context, since
        position-specific efficiency is not resolvable without
        controls.
    informative : np.ndarray
        bool, length L.
    em_diag : dict
        Fit diagnostics of the chosen start: `n_iter` (int, actual
        iterations run, capped at `max_iters`), `log_likelihood`
        (float, at the returned parameters),
        `frac_protected` (float, fitted mixing fraction of windows in
        the protected state), `codes` (as passed in), and `theta_prot`/
        `theta_acc` (float64, length `len(codes)`, the final per-context
        rates before broadcasting to `theta_prot`/`theta_acc` above).
    """
    const = _log_binom_coef_sum(K, N)
    fits = [
        _run_em(K, N, tp, ta, pi, const, max_iters=max_iters, eps=eps, tol=tol)
        for tp, ta, pi in _em_starts(K, N, init_prot, init_acc, eps)
    ]
    lo, hi = _PI_RANGE
    proper = [f for f in fits if lo <= f[2] <= hi]
    tp, ta, pi, ll, n_iter = max(proper or fits, key=lambda f: f[3])

    # Label switching: "protected" must be the low-methylation component.
    if np.average(tp, weights=N.sum(axis=1)) > np.average(
        ta, weights=N.sum(axis=1)
    ):
        tp, ta = ta, tp
        pi = 1.0 - pi

    theta_prot = np.full(len(ctx), np.nan)
    theta_acc = np.full(len(ctx), np.nan)
    for c, code in enumerate(codes):
        sel = ctx == code
        theta_prot[sel], theta_acc[sel] = tp[c], ta[c]

    informative = (
        (ctx != NONE)
        & np.isfinite(theta_acc)
        & ((theta_acc - theta_prot) > min_gap)
    )
    em_diag = {
        "n_iter": n_iter,
        "log_likelihood": ll,
        "frac_protected": pi,
        "codes": codes,
        "theta_prot": tp,
        "theta_acc": ta,
    }
    return theta_prot, theta_acc, informative, em_diag
