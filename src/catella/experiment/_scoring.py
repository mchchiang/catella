# _scoring.py

from contextlib import closing

import numpy as np
from scipy.special import expit, logit

from catella.experiment._calibration import (
    _per_base_log_odds,
    _window_sum_log_odds,
)
from catella.experiment._reads import _accumulate_call_frac, _iter_read_sites
from catella.h5_array import H5Array


def _strand_log_odds(
    q, labels, theta_prot, theta_acc, informative, *, norm_by_strand, pi0, eta
):
    """
    Per-base log-odds of a batch, each read scored against its own
    strand's calibration if `norm_by_strand`.

    Parameters
    ----------
    q : np.ndarray
        float64, (n_mol, L), confidence scores (see `_read_sites`).
    labels : np.ndarray
        Strand label per molecule. Only read if `norm_by_strand`.
    theta_prot, theta_acc, informative : np.ndarray or dict
        From `_chrom_calibration`; per-strand dicts if `norm_by_strand`.
    norm_by_strand : bool
        Whether the calibration is keyed by strand.
    pi0 : float
        Forwarded to `_per_base_log_odds`.
    eta : float or np.ndarray
        Forwarded to `_per_base_log_odds`.

    Returns
    -------
    np.ndarray
        float64, (n_mol, L). Rows with neither strand are zero if
        `norm_by_strand`.
    """
    if not norm_by_strand:
        return _per_base_log_odds(
            q, theta_prot, theta_acc, informative, pi0=pi0, eta=eta
        )
    log_odds = np.zeros_like(q)
    for s in ("+", "-"):
        sel = labels == s
        log_odds[sel] = _per_base_log_odds(
            q[sel],
            theta_prot[s],
            theta_acc[s],
            informative[s],
            pi0=pi0,
            eta=eta,
        )
    return log_odds


def _score_test_reads(
    exp,
    chrom,
    ctx,
    assay,
    theta_prot,
    theta_acc,
    informative,
    *,
    channel_eta,
    channels,
    pi0,
    lnuc,
    fill_edge,
    norm_by_strand,
    batch_size,
    mask_name,
    span,
    tmp_dir,
):
    """
    Window-summed probability of each test read being accessible,
    streamed in batches to a disk-backed array.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    ctx, assay, theta_prot, theta_acc, informative
        From `_chrom_calibration`.
    channel_eta : dict of int to float
        Eta per channel code; positions outside `channels` use 1.0.
    channels : iterable of int
        Channel codes actually assayed.
    pi0, lnuc, fill_edge, norm_by_strand, batch_size, mask_name, span
        As in `MethPrintAnalysis.model_prob`.
    tmp_dir : str or pathlib.Path
        Directory for the scratch file backing the result.

    Returns
    -------
    prob : H5Array
        float64, (n_mol, nbp); all nan for molecules that are dropped,
        read no site, or (if `norm_by_strand`) have no strand.
    call_stats : dict
        Per-channel call and read-site counts of the test reads (see
        `_accumulate_call_frac`).
    """
    eta_by_pos = np.ones(len(ctx))
    for c in channels:
        eta_by_pos[ctx == c] = channel_eta[c]
    log_odds_fill = -logit(fill_edge)

    raw = exp.raw[chrom]
    out = H5Array.create(
        (len(raw.test_mol_id), raw.nbp), dtype=np.float64, dir=tmp_dir
    )
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
    call_stats = {}
    with closing(batches):
        for b in batches:
            _accumulate_call_frac(call_stats, b, assay, ctx, channels)
            masked = ~b.keep | ~b.read.any(axis=1)
            if norm_by_strand:
                masked |= ~((b.labels == "+") | (b.labels == "-"))
            log_odds = _strand_log_odds(
                b.q,
                b.labels,
                theta_prot,
                theta_acc,
                informative,
                norm_by_strand=norm_by_strand,
                pi0=pi0,
                eta=eta_by_pos,
            )
            log_odds_win = _window_sum_log_odds(
                log_odds, lnuc, fill_edge=log_odds_fill
            )
            prob = expit(-log_odds_win)
            prob[masked, :] = np.nan
            out.write_batch(b.start, b.stop, prob)
    return out, call_stats
