# _em.py

from contextlib import closing

import numpy as np

from catella.experiment._contexts import _ALL_CHANNELS, NONE
from catella.experiment._reads import _iter_read_sites, _read_weights


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
    max_iters : int, default 200
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
    theta_prot, theta_acc : np.ndarray
        float64, length L. Rates are constant within a context, since
        position-specific efficiency is not resolvable without
        controls.
    informative : np.ndarray
        bool, length L.
    em_diag : dict
        Fit diagnostics: `n_iter` (int, actual iterations run, capped
        at `max_iters`), `log_likelihood` (float, at the iteration
        that triggered convergence or exhausted `max_iters`),
        `frac_protected` (float, fitted mixing fraction of windows in
        the protected state), `codes` (as passed in), and `theta_prot`/
        `theta_acc` (float64, length `len(codes)`, the final per-context
        rates before broadcasting to `theta_prot`/`theta_acc` above).
    """
    from scipy.special import gammaln

    def _log_binom_pmf(k, n, p):
        # k here is a confidence-score sum, usually fractional.
        # scipy.stats.binom.logpmf requires integer k and returns
        # -inf otherwise, so use the Gamma-function extension of the
        # binomial coefficient instead; it agrees with
        # scipy.stats.binom.logpmf for integer k.
        coef = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
        return coef + k * np.log(p) + (n - k) * np.log1p(-p)

    tp = np.full(len(codes), init_prot)
    ta = np.full(len(codes), init_acc)
    pi = 0.75
    prev = -np.inf
    n_iter = 0

    for it in range(max_iters):
        n_iter = it + 1
        lp = np.log(pi) + sum(
            _log_binom_pmf(K[c], N[c], tp[c]) for c in range(len(codes))
        )
        la = np.log1p(-pi) + sum(
            _log_binom_pmf(K[c], N[c], ta[c]) for c in range(len(codes))
        )
        m = np.maximum(lp, la)
        ll = (m + np.log(np.exp(lp - m) + np.exp(la - m))).sum()
        r = 1.0 / (1.0 + np.exp(la - lp))  # P(protected | window)

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
    if np.average(tp, weights=N.sum(axis=1)) > np.average(
        ta, weights=N.sum(axis=1)
    ):
        tp, ta = ta, tp

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
