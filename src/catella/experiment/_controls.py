# _controls.py

from contextlib import closing

import numpy as np

from catella.experiment._contexts import _ALL_CHANNELS, NONE
from catella.experiment._reads import (
    _accumulate_call_frac,
    _iter_read_sites,
    _read_weights,
)

# Search bounds of the empirical-Bayes nu
_NU_BOUNDS = (1e-2, 1e6)
# nu used when a channel has too few sites to estimate it
_NU_FALLBACK = 10.0
_NU_MIN_SITES = 10
# Exact P(X > Y) within this distance of the decision threshold
_EXACT_WINDOW = 0.1
_QUAD_NODES = 32


def _fit_nu(k, n):
    """
    Empirical-Bayes concentration nu of a beta prior on per-site rates.

    Maximizes the beta-binomial log-evidence
    `sum_x [betaln(k_x + nu m, n_x - k_x + nu (1 - m))
    - betaln(nu m, nu (1 - m))]` over `log nu` within `_NU_BOUNDS`,
    with `m` fixed to the pooled rate `sum k / sum n`. `k` is a sum of
    confidence scores, so this is a quasi-likelihood. `k` is clipped
    to `[0, n]` per site.

    Parameters
    ----------
    k, n : np.ndarray
        float64, per-site summed confidence and number of reads, for
        the sites of one channel.

    Returns
    -------
    nu : float
        `_NU_FALLBACK` if fewer than `_NU_MIN_SITES` sites are read or
        `m` is 0 or 1.
    m : float
        Pooled rate, nan if no site is read.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import betaln

    read = n > 0
    k, n = np.clip(k[read], 0.0, n[read]), n[read]
    if not read.any():
        return _NU_FALLBACK, np.nan
    m = float(k.sum() / n.sum())
    if read.sum() < _NU_MIN_SITES or not 0.0 < m < 1.0:
        return _NU_FALLBACK, m

    def neg_evidence(log_nu):
        nu = np.exp(log_nu)
        a, b = nu * m, nu * (1.0 - m)
        return -(betaln(k + a, n - k + b) - betaln(a, b)).sum()

    lo, hi = np.log(_NU_BOUNDS)
    res = minimize_scalar(neg_evidence, bounds=(lo, hi), method="bounded")
    return float(np.exp(res.x)), m


def _fit_channel_nu(k, n, ctx, channels=None):
    """
    Fit nu per channel with `_fit_nu`.

    Parameters
    ----------
    k, n : np.ndarray
        float64, length L, from `_streamed_call_count`.
    ctx : np.ndarray
        int8, length L, context code per position.
    channels : iterable of int, optional
        Defaults to all of `_ALL_CHANNELS`.

    Returns
    -------
    nu, m : dict of int to float
        Channel code -> fitted nu and pooled rate, for every channel in
        `channels` present in `ctx`.
    """
    if channels is None:
        channels = _ALL_CHANNELS
    nu, m = {}, {}
    for code in channels:
        sel = ctx == code
        if sel.any():
            nu[code], m[code] = _fit_nu(k[sel], n[sel])
    return nu, m


def _channel_mean_rate(k, n, ctx, channels=None):
    """
    Pooled call rate `sum k / sum n` per channel, the value
    `_context_prior_rate` shrinks each position toward.

    Parameters
    ----------
    k, n : np.ndarray
        float64, length L, from `_streamed_call_count`.
    ctx : np.ndarray
        int8, length L, context code per position.
    channels : iterable of int, optional
        Defaults to all of `_ALL_CHANNELS`.

    Returns
    -------
    dict of int to float
        Channel code -> pooled rate, for every channel in `channels`
        with at least one read site.
    """
    if channels is None:
        channels = _ALL_CHANNELS
    rate = {}
    for code in channels:
        sel = ctx == code
        n_sel = n[sel].sum()
        if sel.any() and n_sel > 0:
            rate[code] = float(k[sel].sum() / n_sel)
    return rate


def _context_prior_rate(k, n, ctx, nu=10.0, eps=1e-4, channels=None):
    """
    Per-position call rate, shrunk toward its context group's mean.

    `nu` is the pseudo-count strength borrowed from the context
    mean: larger shrinks harder at low depth. Used on the meth control,
    the unmeth control, or the test set, so it is not controls-specific.

    Parameters
    ----------
    k : np.ndarray
        float64, length L, per-position sum of `mod_qual` confidence
        scores across every molecule reading the position. Build this
        with `_streamed_call_count` so the full sample never needs to
        be held in memory at once.
    n : np.ndarray
        float64, length L, per-position number of molecules reading
        the position, whether or not they carry a call there.
    ctx : np.ndarray
        int8, length L, context code per position.
    nu : float or dict of int to float, default 10.0
        Pseudo-count strength, for all channels or per channel code
        (e.g., from `_fit_channel_nu`). 0 disables shrinkage.
    eps : float, default 1e-4
        Clip bound keeping rates away from exactly 0 or 1.
    channels : iterable of int, optional
        Context channels to compute a rate for; positions in any other
        channel are left nan. Defaults to all of `_ALL_CHANNELS`.

    Returns
    -------
    np.ndarray
        float64, length L, in [eps, 1 - eps], nan outside `channels`
        and where `n + nu` is 0.
    """
    if channels is None:
        channels = _ALL_CHANNELS
    theta = np.full(len(ctx), np.nan)
    for code in channels:
        sel = ctx == code
        n_sel = n[sel].sum()
        if not sel.any() or n_sel == 0:
            continue
        p0 = k[sel].sum() / n_sel
        nu_c = nu.get(code, _NU_FALLBACK) if isinstance(nu, dict) else nu
        with np.errstate(invalid="ignore", divide="ignore"):
            theta[sel] = (k[sel] + nu_c * p0) / (n[sel] + nu_c)
    return np.clip(theta, eps, 1 - eps)


def _streamed_call_count(
    exp,
    chrom,
    which,
    ctx,
    assay,
    batch_size,
    mask_name,
    by_strand=False,
    span="full",
    call_stats=None,
):
    """
    Per-position summed methylation-calling confidence and read count
    for one raw source, streamed in batches so peak memory is
    O(batch_size x L) rather than O(n_mol x L). Blank molecules count
    with their expected strand weights (see `_read_weights`).

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
    assay : dict of str to np.ndarray
        From `_strand_assay_masks`.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to `_iter_read_sites`.
    by_strand : bool, default False
        If True, counts are accumulated separately per strand from a
        single streamed pass, and the return value is keyed by strand.
    span : {"full", "calls"}, default "full"
        Forwarded to `_iter_read_sites`.
    call_stats : dict, optional
        If given, per-channel call and read-site counts are added to
        it in place (see `_accumulate_call_frac`), over all channels.

    Returns
    -------
    k : np.ndarray
        float64, length L, per-position sum of `mod_qual` confidence
        scores, with a read site without a call counting as 0. Only if
        `by_strand` is False.
    n : np.ndarray
        float64, length L, per-position (expected) number of kept
        molecules reading the position. Only if `by_strand` is False.
    counts : dict of str to (k, n)
        One `(k, n)` pair per strand ("+"/"-"). Only if `by_strand` is
        True.
    """
    L = len(ctx)
    keys = ["+", "-"] if by_strand else [None]
    acc = {key: [np.zeros(L), np.zeros(L)] for key in keys}
    batches = _iter_read_sites(
        exp,
        chrom,
        which,
        ctx,
        assay,
        batch_size=batch_size,
        mask_name=mask_name,
        span=span,
    )
    with closing(batches):
        for b in batches:
            for key in keys:
                q = b.q if key is None else b.q[b.labels == key]
                acc[key][0] += np.nansum(q, axis=0)
                acc[key][1] += _read_weights(b, assay, key).sum(axis=0)
            if call_stats is not None:
                _accumulate_call_frac(call_stats, b, assay, ctx, _ALL_CHANNELS)
    if not by_strand:
        return tuple(acc[None])
    return {key: tuple(acc[key]) for key in keys}


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


def _beta_posterior(k, n, ctx, nu, mean_rate):
    """
    Per-site Beta posterior parameters of a control's call rate under
    the prior Beta(nu m, nu (1 - m)) of its channel.

    Parameters
    ----------
    k, n : np.ndarray
        float64, length L, from `_streamed_call_count`; `k` is clipped
        to `[0, n]`.
    ctx : np.ndarray
        int8, length L, context code per position.
    nu : float or dict of int to float
        As in `_context_prior_rate`.
    mean_rate : dict of int to float
        From `_channel_mean_rate`; channels absent from it are left
        nan.

    Returns
    -------
    a, b : np.ndarray
        float64, length L.
    """
    a = np.full(len(ctx), np.nan)
    b = np.full(len(ctx), np.nan)
    for code, m in mean_rate.items():
        sel = ctx == code
        nu_c = nu.get(code, _NU_FALLBACK) if isinstance(nu, dict) else nu
        k_c = np.clip(k[sel], 0.0, n[sel])
        a[sel] = k_c + nu_c * m
        b[sel] = n[sel] - k_c + nu_c * (1.0 - m)
    return a, b


def _prob_greater(a_x, b_x, a_y, b_y, near=None):
    """
    P(X > Y) for independent X ~ Beta(a_x, b_x), Y ~ Beta(a_y, b_y).

    A normal approximation of X - Y is used where its value is far from
    `near`; elsewhere, where the approximation can err by a few percent
    for small parameters, P is computed exactly as
    `int_0^1 [1 - I(G_y^-1(u); a_x, b_x)] du` by `_QUAD_NODES`-point
    Gauss-Legendre quadrature, with `I` the regularized incomplete beta
    function and `G_y^-1` the quantile function of Y.

    Parameters
    ----------
    a_x, b_x, a_y, b_y : np.ndarray
        float64, same shape.
    near : float, optional
        Decision threshold; exact values are computed where the
        approximation lies within `_EXACT_WINDOW` of it. If None, they
        are computed everywhere.

    Returns
    -------
    np.ndarray
        float64, nan where a parameter is not positive.
    """
    from scipy.special import betainc, betaincinv, ndtr

    params = np.stack([a_x, b_x, a_y, b_y])
    valid = (np.isfinite(params) & (params > 0)).all(axis=0)

    def mean_var(a, b):
        s = a + b
        return a / s, a * b / (s * s * (s + 1.0))

    with np.errstate(invalid="ignore", divide="ignore"):
        m_x, v_x = mean_var(a_x, b_x)
        m_y, v_y = mean_var(a_y, b_y)
        prob = ndtr((m_x - m_y) / np.sqrt(v_x + v_y))
    prob = np.where(valid, prob, np.nan)

    exact = valid
    if near is not None:
        with np.errstate(invalid="ignore"):
            exact = valid & (np.abs(prob - near) < _EXACT_WINDOW)
    if exact.any():
        x, w = np.polynomial.legendre.leggauss(_QUAD_NODES)
        u, w = 0.5 * (x + 1.0), 0.5 * w
        y = betaincinv(a_y[exact, None], b_y[exact, None], u)
        cdf_x = betainc(a_x[exact, None], b_x[exact, None], y)
        prob[exact] = ((1.0 - cdf_x) * w).sum(axis=1)
    return prob


def _control_profiles(
    meth_k,
    meth_n,
    unmeth_k,
    unmeth_n,
    ctx,
    nu=None,
    min_site_prob=0.95,
    channels=None,
):
    """
    Rate profiles from the meth/unmeth controls.

    Parameters
    ----------
    meth_k, meth_n : np.ndarray
        From `_streamed_call_count` on the fully-methylated control.
    unmeth_k, unmeth_n : np.ndarray
        From `_streamed_call_count` on the unmethylated control.
    ctx : np.ndarray
        int8, length L, context code per position.
    nu : float, optional
        Forwarded to `_context_prior_rate`. If None, nu is fitted per
        channel for each control separately with `_fit_channel_nu`.
    min_site_prob : float, default 0.95
        A site is informative if the posterior probability that its
        meth-control rate exceeds its unmeth-control rate, from
        `_beta_posterior` and `_prob_greater`, is at least this.
    channels : iterable of int, optional
        Forwarded to `_context_prior_rate`. Defaults to all of
        `_ALL_CHANNELS`; positions outside `channels` come back
        uninformative, since their rates are left nan.

    Returns
    -------
    fpr, acc_ctrl, informative : np.ndarray
        Each length L: the shrunk call rates of the unmeth and meth
        controls (`F` and `F + D`), and the informative mask.
    prior : dict
        `"mean_rate_meth"` and `"mean_rate_unmeth"` (from
        `_channel_mean_rate`), and, if `nu` is None, `"nu_meth"` and
        `"nu_unmeth"`; each a dict channel code -> float.
    """
    prior = {}
    nus = {"meth": nu, "unmeth": nu}
    for src, k, n in (
        ("meth", meth_k, meth_n),
        ("unmeth", unmeth_k, unmeth_n),
    ):
        prior[f"mean_rate_{src}"] = _channel_mean_rate(k, n, ctx, channels)
        if nu is None:
            nus[src], _ = _fit_channel_nu(k, n, ctx, channels)
            prior[f"nu_{src}"] = nus[src]
    acc_ctrl = _context_prior_rate(
        meth_k, meth_n, ctx, nu=nus["meth"], channels=channels
    )
    fpr = _context_prior_rate(
        unmeth_k, unmeth_n, ctx, nu=nus["unmeth"], channels=channels
    )
    a_acc, b_acc = _beta_posterior(
        meth_k, meth_n, ctx, nus["meth"], prior["mean_rate_meth"]
    )
    a_fpr, b_fpr = _beta_posterior(
        unmeth_k, unmeth_n, ctx, nus["unmeth"], prior["mean_rate_unmeth"]
    )
    prob = _prob_greater(a_acc, b_acc, a_fpr, b_fpr, near=min_site_prob)
    informative = (
        (ctx != NONE)
        & np.isfinite(acc_ctrl)
        & np.isfinite(fpr)
        & (np.nan_to_num(prob) >= min_site_prob)
    )
    return fpr, acc_ctrl, informative, prior
