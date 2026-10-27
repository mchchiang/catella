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
    lbin,
    batch_size,
    mask_name,
    by_strand=False,
    channels=None,
    span="full",
    profiles=None,
):
    """
    Per-window, per-read summed methylation-calling confidence and
    trial counts for the test sample, streamed in batches so peak
    memory is O(batch_size x L) rather than O(n_mol x L).

    Windows are non-overlapping. A window's trial count on a read is
    its number of sites read by that read (see `_read_sites`), so it
    varies by read, and a read site without a call is a trial with
    confidence 0. A blank read counts its sites with their expected
    strand weights (see `_read_weights`). Read-windows without read
    sites carry no information and are dropped.

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
    lbin : int
        Bin width (bp) of the non-overlapping windows.
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
    profiles : dict, optional
        Stratum (None, or "+"/"-" if `by_strand`) -> `(F, D, mask)`,
        length-L rate profiles from the controls and the sites to
        count. If given, only sites in `mask` are counted, and the
        per-window sums of `F` and `D` over the read sites are
        accumulated as well.

    Returns
    -------
    K, N : np.ndarray
        float64, (n_ctx, n_kept), over the kept read-windows in
        read-major order. `K` is the sum of `mod_qual` confidence
        scores and `N` the (expected) number of read sites, per
        context. Only if `by_strand` is False.
    SF, SD : np.ndarray or None
        float64, (n_ctx, n_kept), per-window sums of `F` and `D` over
        the read sites (weighted like `N`); None without `profiles`.
        Only if `by_strand` is False.
    codes : list of int
        Channels from `channels` present in `ctx`, in the same order
        as `K`/`N`'s first axis. Only if `by_strand` is False.
    counts : dict of str to (K, N, SF, SD, codes)
        One `(K, N, SF, SD, codes)` tuple per strand ("+"/"-"). Only if
        `by_strand` is True.

    Raises
    ------
    ValueError
        If `ctx` has no assayable contexts within `channels`, or too
        few read-windows have read sites (per strand, when `by_strand`
        is True).
    """
    if channels is None:
        channels = _ALL_CHANNELS
    codes = [c for c in channels if (ctx == c).any()]
    if not codes:
        raise ValueError("no assayable contexts in ctx")
    n_win = len(ctx) // lbin
    width = n_win * lbin
    in_code = np.stack([ctx[:width] == c for c in codes])[:, None, :]

    keys = ["+", "-"] if by_strand else [None]
    names = ("K", "N", "SF", "SD") if profiles else ("K", "N")
    parts = {key: {n: [] for n in names} for key in keys}
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
                shape = (len(codes), m, n_win, lbin)
                q_k = np.nan_to_num(b.q[sel, :width], nan=0.0)[None]
                w_k = _read_weights(b, assay, key)[sel, :width][None]
                if profiles:
                    F, D, mask = (
                        np.nan_to_num(a[:width]) for a in profiles[key]
                    )
                    q_k, w_k = q_k * mask, w_k * mask
                sums = {"K": q_k, "N": w_k}
                if profiles:
                    sums.update(SF=w_k * F, SD=w_k * D)
                sums = {
                    n: (v * in_code).reshape(shape).sum(axis=3)
                    for n, v in sums.items()
                }
                ok = sums["N"].sum(axis=0) > 0
                for n in names:
                    parts[key][n].append(sums[n][:, ok])

    def _build(key):
        if sum(b.shape[1] for b in parts[key]["N"]) < 10:
            suffix = "" if key is None else f" for strand {key!r}"
            raise ValueError(f"too few windows with read sites{suffix}")
        out = {n: np.concatenate(parts[key][n], axis=1) for n in names}
        return out["K"], out["N"], out.get("SF"), out.get("SD"), codes

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


def _bin_rates(N, prof, scale):
    """
    Per-window call rates for each channel at a given efficiency.

    Parameters
    ----------
    N : np.ndarray
        float64, (n_ctx, n_obs).
    prof : tuple of np.ndarray or None
        `(SF, SD)` from `_streamed_window_count`, or None without
        controls (F = 0, D = 1).
    scale : np.ndarray
        float64, length n_ctx, tau or kappa per channel.

    Returns
    -------
    np.ndarray
        float64, broadcastable to (n_ctx, n_obs): `scale` itself
        without `prof`, else `(SF + scale * SD) / N` (0.5 where `N` is
        0, which carries no weight).
    """
    if prof is None:
        return scale[:, None]
    SF, SD = prof
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = (SF + scale[:, None] * SD) / N
    return np.where(N > 0, rate, 0.5)


def _mixture_loglik(K, N, rp, ra, pi, const):
    """
    Log-likelihood of the shared two-state mixture, and the posterior
    probability that each window is protected.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    rp, ra : np.ndarray
        float64, protected and accessible rates, broadcastable to
        (n_ctx, n_obs), from `_bin_rates`.
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
    lp = np.log(pi) + (K * np.log(rp) + (N - K) * np.log1p(-rp)).sum(axis=0)
    la = np.log1p(-pi) + (K * np.log(ra) + (N - K) * np.log1p(-ra)).sum(axis=0)
    m = np.maximum(lp, la)
    ll = const + (m + np.log(np.exp(lp - m) + np.exp(la - m))).sum()
    r = 1.0 / (1.0 + np.exp(la - lp))
    return float(ll), r


def _update_scale(w, K, N, prof, prev, lo, hi):
    """
    M-step for one efficiency (tau or kappa) per channel.

    Without `prof` the maximum is the closed form `sum w K / sum w N`.
    With `prof` it is approached by one Newton step on the concave
    expected log-likelihood from `prev`, halved until that likelihood
    does not decrease, so the expectation-maximization fit still
    increases the likelihood at every iteration.

    Parameters
    ----------
    w : np.ndarray
        float64, length n_obs, posterior weight of the state.
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    prof : tuple of np.ndarray or None
        As in `_bin_rates`.
    prev : np.ndarray
        float64, length n_ctx, current values.
    lo, hi : np.ndarray
        float64, length n_ctx, bounds.

    Returns
    -------
    np.ndarray
        float64, length n_ctx.
    """
    if prof is None:
        den = (w * N).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            new = np.clip((w * K).sum(axis=1) / den, lo, hi)
        return np.where(den > 0, new, prev)

    SD = prof[1]

    def q_terms(scale):
        p = _bin_rates(N, prof, scale)
        q = (w * (K * np.log(p) + (N - K) * np.log1p(-p))).sum(axis=1)
        return p, q

    p, q_prev = q_terms(prev)
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(N > 0, SD / N, 0.0)
    grad = (w * d * (K - N * p) / (p * (1 - p))).sum(axis=1)
    hess = -(w * d**2 * (K / p**2 + (N - K) / (1 - p) ** 2)).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        step = np.where(hess < 0, -grad / hess, 0.0)
    new = prev.copy()
    for _ in range(30):
        cand = np.clip(prev + step, lo, hi)
        _, q_cand = q_terms(cand)
        better = q_cand >= q_prev
        new = np.where(better, cand, new)
        if better.all():
            break
        step = np.where(better, 0.0, step / 2)
    return new


def _run_em(K, N, prof, tp, ta, pi, const, *, fit_acc, lo, hi, max_iters, tol):
    """
    Run expectation-maximization from one start.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    prof : tuple of np.ndarray or None
        As in `_bin_rates`.
    tp, ta : np.ndarray
        float64, length n_ctx, starting tau and kappa (the protected
        and accessible rates without `prof`).
    pi : float
        Starting mixing fraction.
    const : float
        From `_log_binom_coef_sum`.
    fit_acc : bool
        Whether `ta` is fitted; if False it stays fixed.
    lo, hi : np.ndarray
        float64, length n_ctx, bounds on `tp` and `ta`.
    max_iters, tol
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
        rp, ra = _bin_rates(N, prof, tp), _bin_rates(N, prof, ta)
        ll, r = _mixture_loglik(K, N, rp, ra, pi, const)
        tp = _update_scale(r, K, N, prof, tp, lo, hi)
        if fit_acc:
            ta = _update_scale(1 - r, K, N, prof, ta, lo, hi)
        pi = float(r.mean())
        if abs(ll - prev) < tol * max(1.0, abs(ll)):
            break
        prev = ll
    rp, ra = _bin_rates(N, prof, tp), _bin_rates(N, prof, ta)
    ll, _ = _mixture_loglik(K, N, rp, ra, pi, const)
    return tp, ta, pi, ll, n_iter


def _em_starts(K, N, prof, init_prot, init_acc, fit_acc, lo, hi):
    """
    Starting values for the expectation-maximization fit.

    The first start is the user's guess (`init_prot`, `init_acc`, pi
    0.75). For each q in `_START_QUANTILES`, a data-driven start ranks
    windows by their rate pooled over channels, puts the windows at or
    below the q-quantile in the protected state and the rest in the
    accessible state, and takes each channel's pooled estimate within
    each group, `sum (K - SF) / sum SD` (the pooled rate without
    `prof`), with pi = q. A channel without sites in a group falls back
    to its overall pooled estimate. If `fit_acc` is False, the
    accessible value of every start is `init_acc`.

    Parameters
    ----------
    K, N : np.ndarray
        float64, (n_ctx, n_obs).
    prof : tuple of np.ndarray or None
        As in `_bin_rates`.
    init_prot, init_acc : float
        As in `_calibrate_from_data`.
    fit_acc : bool
        As in `_run_em`.
    lo, hi : np.ndarray
        float64, length n_ctx, bounds.

    Returns
    -------
    list of tuple
        `(tp, ta, pi)` per start, with `tp`/`ta` float64 arrays of
        length n_ctx.
    """
    n_ctx = K.shape[0]
    fixed = np.full(n_ctx, init_acc)
    starts = [(np.full(n_ctx, init_prot), fixed, 0.75)]
    num, den = (K, N) if prof is None else (K - prof[0], prof[1])
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = K.sum(axis=0) / N.sum(axis=0)
        pooled = num.sum(axis=1) / den.sum(axis=1)

    def _group_scale(sel):
        k, n = num[:, sel].sum(axis=1), den[:, sel].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            g = np.where(n > 0, k / n, pooled)
        return np.clip(np.nan_to_num(g, nan=0.5), lo, hi)

    for q in _START_QUANTILES:
        low = rate <= np.nanquantile(rate, q)
        acc = _group_scale(~low) if fit_acc else fixed
        starts.append((_group_scale(low), acc, q))
    return starts


def _fit_mixture(
    K, N, prof, *, fit_acc, lo, hi, init_prot, init_acc, max_iters, tol
):
    """
    Fit the mixture from every start of `_em_starts` and keep the best.

    The fit with the largest log-likelihood is kept, among those whose
    mixing fraction lies within `_PI_RANGE` (among all starts if none
    does). If `fit_acc` and the kept fit has the labels swapped, the
    two efficiencies are swapped and the mixing fraction is flipped to
    `1 - pi`.

    Parameters
    ----------
    K, N, prof, fit_acc, lo, hi
        As in `_run_em`.
    init_prot, init_acc, max_iters, tol
        As in `_calibrate_from_data`.

    Returns
    -------
    tuple
        `(tp, ta, pi, ll, n_iter)` of the kept fit.
    """
    const = _log_binom_coef_sum(K, N)
    starts = _em_starts(K, N, prof, init_prot, init_acc, fit_acc, lo, hi)
    fits = [
        _run_em(
            K,
            N,
            prof,
            tp,
            ta,
            pi,
            const,
            fit_acc=fit_acc,
            lo=lo,
            hi=hi,
            max_iters=max_iters,
            tol=tol,
        )
        for tp, ta, pi in starts
    ]
    pi_lo, pi_hi = _PI_RANGE
    proper = [f for f in fits if pi_lo <= f[2] <= pi_hi]
    tp, ta, pi, ll, n_iter = max(proper or fits, key=lambda f: f[3])

    # Label switching: "protected" must be the low-methylation component.
    if fit_acc and np.average(tp, weights=N.sum(axis=1)) > np.average(
        ta, weights=N.sum(axis=1)
    ):
        tp, ta = ta, tp
        pi = 1.0 - pi
    return tp, ta, pi, ll, n_iter


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
    tp, ta, pi, ll, n_iter = _fit_mixture(
        K,
        N,
        None,
        fit_acc=True,
        lo=np.full(len(codes), eps),
        hi=np.full(len(codes), 1 - eps),
        init_prot=init_prot,
        init_acc=init_acc,
        max_iters=max_iters,
        tol=tol,
    )

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


def _calibrate_from_profiles(
    K,
    N,
    SF,
    SD,
    codes,
    ctx,
    fpr,
    acc_ctrl,
    informative,
    *,
    fit_kappa=False,
    max_iters=200,
    init_prot=0.05,
    init_acc=0.95,
    tol=1e-8,
    eps=1e-4,
):
    """
    Fit tau (and optionally kappa) per context with
    expectation-maximization on the test sample, given the rate
    profiles from the meth/unmeth controls.

    The rates are `theta_prot = F + tau * D` and `theta_acc = F + kappa
    * D`, with `F` the unmethylated control's call rate and `D` the
    methylated control's rate minus `F`. Each window's rates average
    `F` and `D` over its read sites. The latent state and its mixing
    fraction are shared by all channels, while tau and kappa are
    fitted per channel. With `fit_kappa` False, kappa is 1 and tau is
    bounded to [0, 1], so no label swap is possible.

    Parameters
    ----------
    K, N, SF, SD : np.ndarray
        float64, (n_ctx, n_obs), from `_streamed_window_count` with
        `profiles`.
    codes : list of int
        Context codes in the same order as the first axis of `K`.
    ctx : np.ndarray
        int8, length L, context code per position.
    fpr, acc_ctrl : np.ndarray
        float64, length L, call rates of the unmeth and meth controls
        (`F` and `F + D`).
    informative : np.ndarray
        bool, length L, returned unchanged.
    fit_kappa : bool, default False
        Whether to fit kappa as well; otherwise kappa is 1.
    max_iters, init_prot, tol
        As in `_calibrate_from_data`; `init_prot` is the starting tau.
    init_acc : float, default 0.95
        Starting kappa, used only if `fit_kappa`.
    eps : float, default 1e-4
        Clip bound keeping rates away from exactly 0 or 1.

    Returns
    -------
    theta_prot, theta_acc : np.ndarray
        float64, length L.
    informative : np.ndarray
        bool, length L.
    em_diag : dict
        `n_iter`, `log_likelihood`, `frac_protected` and `codes` as in
        `_calibrate_from_data`, plus `tau`, `kappa` and `tau_max`
        (float64, length `len(codes)`, the fitted values and the upper
        bound on tau).
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        f_w, d_w = SF / N, SD / N
        bound = np.where((N > 0) & (d_w > 0), (1 - eps - f_w) / d_w, np.inf)
    hi = bound.min(axis=1)
    if not fit_kappa:
        hi = np.minimum(hi, 1.0)
    lo = np.zeros(len(codes))
    tau, kappa, pi, ll, n_iter = _fit_mixture(
        K,
        N,
        (SF, SD),
        fit_acc=fit_kappa,
        lo=lo,
        hi=hi,
        init_prot=init_prot,
        init_acc=init_acc if fit_kappa else 1.0,
        max_iters=max_iters,
        tol=tol,
    )

    D = acc_ctrl - fpr
    theta_prot = np.full(len(ctx), np.nan)
    theta_acc = acc_ctrl.copy() if not fit_kappa else np.full(len(ctx), np.nan)
    for c, code in enumerate(codes):
        sel = ctx == code
        theta_prot[sel] = fpr[sel] + tau[c] * D[sel]
        if fit_kappa:
            theta_acc[sel] = fpr[sel] + kappa[c] * D[sel]
    em_diag = {
        "n_iter": n_iter,
        "log_likelihood": ll,
        "frac_protected": pi,
        "codes": codes,
        "tau": tau,
        "kappa": kappa,
        "tau_max": hi,
    }
    return (
        np.clip(theta_prot, eps, 1 - eps),
        np.clip(theta_acc, eps, 1 - eps),
        informative,
        em_diag,
    )
