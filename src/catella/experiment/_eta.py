# _eta.py

from contextlib import closing

import numpy as np

from catella.experiment._calibration import (
    _calib_rows,
    _chrom_calibration,
    _per_base_log_odds,
)
from catella.experiment._contexts import _ALL_CHANNELS, _CHANNEL_NAME_TO_CODE
from catella.experiment._reads import _iter_read_sites


def _resolve_eta_overrides(eta):
    """
    Validate and normalize a user-supplied `eta` into per-channel
    overrides.

    Parameters
    ----------
    eta : float, dict of str to float, or None
        `None` requests auto-estimation for every channel; a float
        pins every channel to that value; a dict pins only the named
        channels (keys from `CONTEXT_NAMES`, e.g. "A"), leaving any
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
                    f"of {sorted(_CHANNEL_NAME_TO_CODE)}."
                )
            overrides[_CHANNEL_NAME_TO_CODE[name]] = float(val)
        return overrides
    return {c: float(eta) for c in _ALL_CHANNELS}


def _init_autocorr_stats(max_lag):
    """Zeroed per-channel accumulators for `_accumulate_autocorr_stats`."""
    return {
        c: {
            "n": 0.0,
            "s1": 0.0,
            "s2": 0.0,
            "npair": np.zeros(max_lag),
            "pprod": np.zeros(max_lag),
            "psumA": np.zeros(max_lag),
            "psumB": np.zeros(max_lag),
        }
        for c in _ALL_CHANNELS
    }


def _accumulate_autocorr_stats(
    stats, log_odds, called, ctx, max_lag, channels=None
):
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
        bool, (n_mol, L). True where a site is both informative and
        read by the molecule (see `_read_sites`), with or without a
        call.
    ctx : np.ndarray
        int8, length L, context code per position.
    max_lag : int
        Maximum lag (bp) to accumulate pair sums for.
    channels : iterable of int, optional
        Channel codes to accumulate. Defaults to all of
        `_ALL_CHANNELS`.
    """
    if channels is None:
        channels = _ALL_CHANNELS
    L = log_odds.shape[1]
    for c in channels:
        mask_c = called & (ctx == c)[None, :]
        if not mask_c.any():
            continue
        lam = log_odds * mask_c
        st = stats[c]
        st["n"] += mask_c.sum()  # count of called sites
        st["s1"] += lam.sum()  # sum(lambda_x)
        st["s2"] += (lam * lam).sum()  # sum(lambda_x^2)
        for k in range(1, min(max_lag, L - 1) + 1):
            left, right = mask_c[:, : L - k], mask_c[:, k:]
            pair = left & right  # both x and x+k called
            if not pair.any():
                continue
            lo_left, lo_right = log_odds[:, : L - k], log_odds[:, k:]
            i = k - 1
            st["npair"][i] += pair.sum()  # count of pairs
            # sum(lambda_x * lambda_x+k) over pairs
            st["pprod"][i] += (lo_left * lo_right * pair).sum()
            # sum(lambda_x) over pairs
            st["psumA"][i] += (lo_left * pair).sum()
            # sum(lambda_x+k) over pairs
            st["psumB"][i] += (lo_right * pair).sum()


def _finalize_channel_eta(stats, max_lag, n_min=None):
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
    n_min : int, optional
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
    if n_min is None:
        n_min = max_lag
    eta = {}
    rho = {}
    for c, st in stats.items():
        n = st["n"]
        if n <= n_min:
            eta[c] = 1.0
            continue
        bar = st["s1"] / n  # global channel mean
        denom = st["s2"] - n * bar * bar  # sum((lambda_x - bar)^2)
        if denom <= 0:
            eta[c] = 1.0
            continue
        npair, pprod = st["npair"], st["pprod"]
        # sum((lambda_x - bar)(lambda_x+k - bar)) over valid pairs
        numerator = (
            pprod - bar * st["psumA"] - bar * st["psumB"] + npair * bar * bar
        )
        rho_c = np.where(npair > 0, numerator / denom, 0.0)
        # pair counts per lag are already in rho_c, so no (1 - k/n) factor
        v = 1.0 + 2.0 * np.sum(rho_c)
        eta[c] = 1.0 / max(v, 1e-3)
        rho[c] = rho_c
    return eta, rho


def _accumulate_eta_source(
    exp,
    chrom,
    which,
    ctx,
    assay,
    theta_prot,
    theta_acc,
    informative,
    pi0,
    batch_size,
    mask_name,
    stats,
    max_lag,
    channels,
    norm_by_strand=False,
    span="full",
):
    """
    Stream one raw source and feed its raw (`eta=1`) per-read
    log-odds over the sites each read reads into
    `_accumulate_autocorr_stats`.

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
    assay : dict of str to np.ndarray
        From `_chrom_calibration` (see `_strand_assay_masks`).
    pi0 : float
        Forwarded to `_per_base_log_odds`.
    batch_size : int
        Number of molecules processed per batch.
    mask_name : str or None
        Forwarded to `_iter_read_sites`.
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
    span : {"full", "calls"}, default "full"
        Forwarded to `_iter_read_sites`.
    """
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
            strands = ("+", "-") if norm_by_strand else (None,)
            for s in strands:
                sel = b.keep if s is None else b.keep & (b.labels == s)
                if not sel.any():
                    continue
                prot = theta_prot if s is None else theta_prot[s]
                acc = theta_acc if s is None else theta_acc[s]
                info = informative if s is None else informative[s]
                log_odds = _per_base_log_odds(
                    b.q[sel], prot, acc, info, pi0=pi0, eta=1.0
                )
                _accumulate_autocorr_stats(
                    stats,
                    log_odds,
                    info[None, :] & b.read[sel],
                    ctx,
                    max_lag,
                    channels=channels,
                )


def _estimate_channel_eta(exp, calib_kwargs, channels, *, pi0, max_lag):
    """
    Calibrate every chromosome and estimate eta for `channels` from the
    autocorrelation of the log-odds, pooled across chromosomes.

    The methylated control is used when available, since it isolates
    the crosstalk artifact from genuine, occupancy-driven correlation;
    otherwise the test data.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    calib_kwargs : dict
        Keyword arguments of `_chrom_calibration`.
    channels : set of int
        Channel codes to estimate.
    pi0 : float
        Forwarded to `_per_base_log_odds`.
    max_lag : int
        Maximum lag (bp) summed over.

    Returns
    -------
    eta, rho : dict
        From `_finalize_channel_eta`.
    calib_cache : dict of str to tuple
        Chromosome -> `(ctx, assay, theta_prot, theta_acc,
        informative)`, for reuse in scoring.
    calib_records : list of dict
        Calibration rows from `_calib_rows`, over all chromosomes.
    """
    norm_by_strand = calib_kwargs["norm_by_strand"]
    stats = _init_autocorr_stats(max_lag)
    calib_cache = {}
    calib_records = []
    for chrom in exp.chroms:
        *calib, has_controls, calib_info = _chrom_calibration(
            exp, chrom, **calib_kwargs
        )
        calib_cache[chrom] = tuple(calib)
        calib_records.extend(_calib_rows(chrom, calib_info, norm_by_strand))
        _accumulate_eta_source(
            exp,
            chrom,
            "meth" if has_controls else "test",
            *calib,
            pi0,
            calib_kwargs["batch_size"],
            calib_kwargs["mask_name"],
            stats,
            max_lag,
            channels,
            norm_by_strand=norm_by_strand,
            span=calib_kwargs["span"],
        )
    eta, rho = _finalize_channel_eta(stats, max_lag)
    return eta, rho, calib_cache, calib_records
