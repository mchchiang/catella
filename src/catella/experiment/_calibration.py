# _calibration.py

import warnings

import numpy as np
import pandas as pd

from catella.experiment._contexts import (
    _ALL_CHANNELS,
    CONTEXT_NAMES,
    NONE,
    _reference_contexts,
    _strand_assay_masks,
)
from catella.experiment._controls import (
    _calibrate_from_controls,
    _streamed_call_count,
)
from catella.experiment._em import (
    _calibrate_from_data,
    _streamed_window_count,
)
from catella.experiment._reads import _check_mapped_strands

# Warn (not raise) if more than this fraction of wrap-mirrored position
# pairs disagree on context when folding ctx in model_prob.
_WRAP_CTX_DISAGREE_WARN_FRAC = 0.02

# Call-fraction check in model_prob: channels whose controls both have
# at least _EXPLICIT_MIN_CALL_FRAC are taken as explicit-mode, and warn
# when the test fraction lies more than _CALL_FRAC_WARN_GAP outside the
# range spanned by the two controls.
_CALL_SOURCES = ("test", "meth", "unmeth")
_EXPLICIT_MIN_CALL_FRAC = 0.5
_CALL_FRAC_WARN_GAP = 0.05


def _per_base_log_odds(
    q, theta_prot, theta_acc, informative, pi0=0.5, eta=1.0
):
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


def _window_sum_log_odds(log_odds, lnuc, fill_edge=0.0):
    """
    Left-aligned rolling sum of `log_odds` over a window of `lnuc`
    positions: position i summarizes `[i, i + lnuc)`. The trailing
    `lnuc - 1` positions, which have no full window, are filled with
    `fill_edge`.

    Parameters
    ----------
    log_odds : np.ndarray
        float64, (n_mol, L).
    lnuc : int
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
    n_win = L - lnuc + 1
    if n_win <= 0:
        return out
    cum = np.concatenate(
        [np.zeros((n_mol, 1)), np.cumsum(log_odds, axis=1)], axis=1
    )
    out[:, :n_win] = cum[:, lnuc:] - cum[:, :-lnuc]
    return out


def _call_frac_info(call_stats, channels):
    """
    Fraction of read sites carrying a call, per source and channel.

    Parameters
    ----------
    call_stats : dict of str to dict
        Source name -> per-channel `[n_called, n_read]` counts, as
        accumulated by `_accumulate_call_frac`.
    channels : iterable of int
        Channel codes to report.

    Returns
    -------
    dict of str to float
        `f"call_frac_{source}_{CONTEXT_NAMES[c]}"` -> fraction, for
        every source and channel with read sites.
    """
    return {
        f"call_frac_{source}_{CONTEXT_NAMES[c]}": float(st[c][0] / st[c][1])
        for source, st in call_stats.items()
        for c in channels
        if c in st and st[c][1] > 0
    }


def _warn_call_frac_mismatch(chrom, info, channels):
    """
    Warn when the test call fraction of a channel lies outside the
    range spanned by the controls, for channels whose controls look
    explicit-mode.

    The probability of emitting a call may depend on the methylation
    state, so the controls can differ, and a test sample, which mixes
    both states, is expected to lie between them. A test fraction
    outside this range (beyond `_CALL_FRAC_WARN_GAP`) means that calls
    are missed at a different rate than in the controls.

    In implicit mode, a base without a call is canonical, so the call
    fraction tracks the methylation level and is low in the
    unmethylated control; such channels are skipped.

    Parameters
    ----------
    chrom : str
        Chromosome identifier, for the message.
    info : dict
        Holds `call_frac_{test,meth,unmeth}_{channel}` entries (see
        `_call_frac_info`).
    channels : iterable of int
        Channel codes to check.
    """
    for c in channels:
        name = CONTEXT_NAMES[c]
        vals = [info.get(f"call_frac_{s}_{name}") for s in _CALL_SOURCES]
        if any(v is None for v in vals):
            continue
        test, meth, unmeth = vals
        lo, hi = min(meth, unmeth), max(meth, unmeth)
        if lo < _EXPLICIT_MIN_CALL_FRAC:
            continue
        if lo - _CALL_FRAC_WARN_GAP <= test <= hi + _CALL_FRAC_WARN_GAP:
            continue
        bias = "protection" if test < lo else "accessibility"
        warnings.warn(
            f"Channel {name} on chrom '{chrom}': the fraction of read "
            f"sites with a call is {test:.2f} in the test sample, outside "
            f"the range {lo:.2f}-{hi:.2f} of the meth/unmeth controls. "
            "Missing calls are treated as unmethylated, so the test "
            f"sample may be biased toward {bias}.",
            stacklevel=3,
        )


def _frac_informative_by_context(ctx, informative, channels=None):
    """
    Fraction of `informative` positions within each context's own sites.

    Parameters
    ----------
    ctx : np.ndarray
        int8, length L, context code per position.
    informative : np.ndarray
        bool, length L.
    channels : iterable of int, optional
        Context channels to report. Defaults to all of `_ALL_CHANNELS`.

    Returns
    -------
    dict of str to float
        `f"frac_informative_{CONTEXT_NAMES[c]}"` -> fraction, for every
        channel in `channels` present in `ctx`.
    """
    if channels is None:
        channels = _ALL_CHANNELS
    return {
        f"frac_informative_{CONTEXT_NAMES[c]}": float(
            informative[ctx == c].mean()
        )
        for c in channels
        if (ctx == c).any()
    }


def _fold_contexts(exp, chrom):
    """
    Context code per position of `chrom`, folded to `nbp` if `exp.wrap`.

    Wrap-mirrored positions that disagree on context are set to NONE.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.

    Returns
    -------
    np.ndarray
        int8, length L.

    Raises
    ------
    ValueError
        If `chrom` has no reference sequence.

    Warns
    -----
    UserWarning
        If more than `_WRAP_CTX_DISAGREE_WARN_FRAC` of the mirrored
        position pairs disagree.
    """
    raw = exp.raw[chrom]
    if raw.refseq is None:
        raise ValueError(
            f"No reference sequence for chrom '{chrom}'; "
            "model_prob needs 'fasta_file' at load_raw."
        )
    full_ctx = _reference_contexts(raw.refseq)
    if not exp.wrap:
        return full_ctx
    nbp = raw.nbp
    length = len(raw.refseq)
    lower = full_ctx[:nbp]
    upper = full_ctx[length - nbp :][::-1]
    agree = lower == upper
    frac_disagree = 1.0 - agree.mean()
    if frac_disagree > _WRAP_CTX_DISAGREE_WARN_FRAC:
        warnings.warn(
            f"refseq for chrom '{chrom}' disagrees on "
            f"context at {(~agree).sum()}/{nbp} "
            f"({frac_disagree:.1%}) wrap-mirrored position "
            "pairs (folded to NONE there); this may "
            "indicate the wrong chromsize length or a "
            "non-symmetric reference, rather than a small "
            "loop/junction region.",
            stacklevel=4,
        )
    return np.where(agree, lower, NONE).astype(lower.dtype)


def _controls_calibration(
    exp,
    chrom,
    ctx,
    assay,
    strata,
    *,
    nu,
    rho_leak,
    min_gap,
    batch_size,
    mask_name,
    channels,
    span,
):
    """
    Calibrate each stratum from the meth/unmeth controls.

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
    strata : tuple
        `("+", "-")` to calibrate each strand, or `(None,)` to pool.
    nu, rho_leak, min_gap, batch_size, mask_name, channels, span
        As in `_chrom_calibration`.

    Returns
    -------
    dict
        Stratum -> `(theta_prot, theta_acc, informative, calib_info)`.
    """
    by_strand = strata != (None,)
    call_stats = {"meth": {}, "unmeth": {}}
    counts = {}
    for src in ("meth", "unmeth"):
        c = _streamed_call_count(
            exp,
            chrom,
            src,
            ctx,
            assay,
            batch_size,
            mask_name,
            by_strand=by_strand,
            span=span,
            call_stats=call_stats[src],
        )
        counts[src] = c if by_strand else {None: c}

    results = {}
    for s in strata:
        theta_prot, theta_acc, informative = _calibrate_from_controls(
            *counts["meth"][s],
            *counts["unmeth"][s],
            ctx,
            nu=nu,
            rho_leak=rho_leak,
            min_gap=min_gap,
            channels=channels,
        )
        calib_info = {
            "has_controls": True,
            **_frac_informative_by_context(
                ctx, informative, channels=channels
            ),
            **_call_frac_info(call_stats, channels),
        }
        results[s] = (theta_prot, theta_acc, informative, calib_info)
    return results


def _em_calibration(
    exp,
    chrom,
    ctx,
    assay,
    strata,
    *,
    lbin,
    max_iters,
    init_prot,
    init_acc,
    tol,
    min_gap,
    batch_size,
    mask_name,
    channels,
    span,
):
    """
    Calibrate each stratum by expectation-maximization on the test data.

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
    strata : tuple
        `("+", "-")` to calibrate each strand, or `(None,)` to pool.
    lbin, max_iters, init_prot, init_acc, tol, min_gap, batch_size,
    mask_name, channels, span
        As in `_chrom_calibration`.

    Returns
    -------
    dict
        Stratum -> `(theta_prot, theta_acc, informative, calib_info)`.
    """
    by_strand = strata != (None,)
    counts = _streamed_window_count(
        exp,
        chrom,
        ctx,
        assay,
        lbin,
        batch_size,
        mask_name,
        by_strand=by_strand,
        channels=channels,
        span=span,
    )
    if not by_strand:
        counts = {None: counts}

    results = {}
    for s in strata:
        K, N, _, _, codes = counts[s]
        theta_prot, theta_acc, informative, em_diag = _calibrate_from_data(
            K,
            N,
            ctx,
            codes,
            max_iters=max_iters,
            init_prot=init_prot,
            init_acc=init_acc,
            tol=tol,
            min_gap=min_gap,
        )
        calib_info = {
            "has_controls": False,
            **_frac_informative_by_context(
                ctx, informative, channels=channels
            ),
            **em_diag,
        }
        results[s] = (theta_prot, theta_acc, informative, calib_info)
    return results


def _chrom_calibration(
    exp,
    chrom,
    *,
    nu,
    rho_leak,
    min_gap,
    lbin,
    max_iters,
    init_prot,
    init_acc,
    tol,
    batch_size,
    mask_name,
    norm_by_strand=False,
    channels=None,
    span="full",
):
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
    nu, rho_leak, min_gap, lbin, max_iters, init_prot, init_acc, tol
        As in `model_prob`.
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
    channels : iterable of int, optional
        Context channels actually assayed, from `_active_channels`.
        Defaults to all of `_ALL_CHANNELS`.
    span : {"full", "calls"}, default "full"
        Forwarded to the streamed counting helpers.

    Returns
    -------
    ctx : np.ndarray
        int8, length L, context code per position.
    assay : dict of str to np.ndarray
        From `_strand_assay_masks`.
    theta_prot, theta_acc : np.ndarray or dict of str to np.ndarray
        float64, length L, or (if `norm_by_strand`) a `{"+": ...,
        "-": ...}` dict of such arrays.
    informative : np.ndarray or dict of str to np.ndarray
        bool, length L, or (if `norm_by_strand`) a `{"+": ...,
        "-": ...}` dict of such arrays.
    has_controls : bool
        Whether meth/unmeth controls were used.
    calib_info : dict or dict of str to dict
        Calibration diagnostics -- `{"has_controls": has_controls,
        **frac_informative_by_context}` (see
        `_frac_informative_by_context`), merged with the `em_diag`
        dict from `_calibrate_from_data` when the no-controls path was
        used (empty for the controls path). If `norm_by_strand`, a
        `{"+": ..., "-": ...}` dict of such dicts.

    Raises
    ------
    ValueError
        If `chrom` has no reference sequence, if the no-controls
        path cannot find enough bins with context-eligible read
        sites, or if `norm_by_strand` is True but molecules with
        unmapped strands ('.') exist.

    Warns
    -----
    UserWarning
        If `exp.wrap` is True and many wrap-mirrored position pairs
        disagree on context.
    """
    if channels is None:
        channels = _ALL_CHANNELS
    raw = exp.raw[chrom]
    ctx = _fold_contexts(exp, chrom)
    assay = _strand_assay_masks(exp, chrom, ctx)
    has_controls = raw.meth_data is not None and raw.unmeth_data is not None
    if norm_by_strand:
        _check_mapped_strands(raw, has_controls)

    strata = ("+", "-") if norm_by_strand else (None,)
    common = {
        "min_gap": min_gap,
        "batch_size": batch_size,
        "mask_name": mask_name,
        "channels": channels,
        "span": span,
    }
    if has_controls:
        results = _controls_calibration(
            exp, chrom, ctx, assay, strata, nu=nu, rho_leak=rho_leak, **common
        )
    else:
        results = _em_calibration(
            exp,
            chrom,
            ctx,
            assay,
            strata,
            lbin=lbin,
            max_iters=max_iters,
            init_prot=init_prot,
            init_acc=init_acc,
            tol=tol,
            **common,
        )

    fields = [dict(zip(strata, f)) for f in zip(*results.values())]
    if not norm_by_strand:
        fields = [f[None] for f in fields]
    theta_prot, theta_acc, informative, calib_info = fields
    return (
        ctx,
        assay,
        theta_prot,
        theta_acc,
        informative,
        has_controls,
        calib_info,
    )


def _calib_rows(chrom, info, norm_by_strand):
    """
    Flatten `_chrom_calibration`'s `calib_info` into table rows.

    Parameters
    ----------
    chrom : str
        Chromosome identifier.
    info : dict
        `calib_info` from `_chrom_calibration`; per-strand dicts if
        `norm_by_strand`.
    norm_by_strand : bool
        Whether `info` is keyed by strand.

    Returns
    -------
    list of dict
        One row per stratum, with `chrom`, `has_controls`, the
        `frac_informative_*`/`call_frac_*` entries, `strand` (only if
        `norm_by_strand`), and, for the EM path, `iters`,
        `log_likelihood`, `frac_protected` and per-context
        `theta_prot_*`/`theta_acc_*`.
    """
    strata = info.items() if norm_by_strand else [(None, info)]
    rows = []
    for strand, sub in strata:
        row = {"chrom": chrom, "has_controls": sub["has_controls"]}
        row.update(
            {
                k: v
                for k, v in sub.items()
                if k.startswith(("frac_informative_", "call_frac_"))
            }
        )
        if strand is not None:
            row["strand"] = strand
        if "n_iter" in sub:
            row["iters"] = sub["n_iter"]
            row["log_likelihood"] = sub["log_likelihood"]
            row["frac_protected"] = sub["frac_protected"]
            for key in ("theta_prot", "theta_acc"):
                row.update(
                    {
                        f"{key}_{CONTEXT_NAMES[c]}": v
                        for c, v in zip(sub["codes"], sub[key])
                    }
                )
        rows.append(row)
    return rows


def _theta_frame(theta_prot, theta_acc, informative, norm_by_strand):
    """
    Per-position calibration as a table.

    Parameters
    ----------
    theta_prot, theta_acc, informative : np.ndarray or dict
        From `_chrom_calibration`; per-strand dicts if `norm_by_strand`.
    norm_by_strand : bool
        Whether the inputs are keyed by strand.

    Returns
    -------
    pd.DataFrame
        Columns `theta_prot`, `theta_acc`, `informative`, each suffixed
        with `_pos`/`_neg` per strand if `norm_by_strand`.
    """
    if not norm_by_strand:
        return pd.DataFrame(
            {
                "theta_prot": theta_prot,
                "theta_acc": theta_acc,
                "informative": informative,
            }
        )
    cols = {}
    for s, suffix in (("+", "pos"), ("-", "neg")):
        cols[f"theta_prot_{suffix}"] = theta_prot[s]
        cols[f"theta_acc_{suffix}"] = theta_acc[s]
        cols[f"informative_{suffix}"] = informative[s]
    return pd.DataFrame(cols)
