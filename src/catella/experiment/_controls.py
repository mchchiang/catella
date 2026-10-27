# _controls.py

from contextlib import closing

import numpy as np

from catella.experiment._contexts import _ALL_CHANNELS, NONE
from catella.experiment._reads import (
    _accumulate_call_frac,
    _iter_read_sites,
    _read_weights,
)


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
    nu : float, default 10.0
        Pseudo-count strength. 0 disables shrinkage.
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
        with np.errstate(invalid="ignore", divide="ignore"):
            theta[sel] = (k[sel] + nu * p0) / (n[sel] + nu)
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


def _informative_mask(theta_acc, fpr, ctx, min_gap=0.05):
    """Positions the assay can discriminate; elsewhere contributes zero."""
    return (
        (ctx != NONE) & np.isfinite(theta_acc) & ((theta_acc - fpr) > min_gap)
    )


def _control_profiles(
    meth_k,
    meth_n,
    unmeth_k,
    unmeth_n,
    ctx,
    nu=10.0,
    min_gap=0.05,
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
    nu : float, default 10.0
        Forwarded to `_context_prior_rate`.
    min_gap : float, default 0.05
        Forwarded to `_informative_mask`.
    channels : iterable of int, optional
        Forwarded to `_context_prior_rate`. Defaults to all of
        `_ALL_CHANNELS`; positions outside `channels` come back
        uninformative, since their rates are left nan.

    Returns
    -------
    tuple of np.ndarray
        `(fpr, acc_ctrl, informative)`, each length L: the shrunk call
        rates of the unmeth and meth controls (`F` and `F + D`), and
        the informative mask.
    """
    acc_ctrl = _context_prior_rate(
        meth_k, meth_n, ctx, nu=nu, channels=channels
    )
    fpr = _context_prior_rate(
        unmeth_k, unmeth_n, ctx, nu=nu, channels=channels
    )
    informative = _informative_mask(acc_ctrl, fpr, ctx, min_gap=min_gap)
    return fpr, acc_ctrl, informative
