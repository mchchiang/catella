# preprocessing.py

from collections.abc import Iterable
from contextlib import closing

import numpy as np
import pandas as pd

from catella import utils
from catella.experiment._calibration import (
    _call_frac_info,
    _chrom_calibration,
    _per_base_log_odds,
    _warn_call_frac_mismatch,
    _window_sum_log_odds,
)
from catella.experiment._contexts import (
    CONTEXT_NAMES,
    GCG,
    GCH,
    HCG,
    NONE,
    A,
    _active_channels,
)
from catella.experiment._dropout import _strand_of_mol
from catella.experiment._empirical import (
    _clip_bounds,
    _control_means,
    _masked_smoothed,
    _smooth_source,
    _write_empirical_prob,
)
from catella.experiment._eta import (
    _accumulate_eta_source,
    _finalize_channel_eta,
    _init_autocorr_stats,
    _resolve_eta_overrides,
)
from catella.experiment._reads import (
    _accumulate_call_frac,
    _check_mapped_strands,
    _iter_read_sites,
    _lookup_mask,
)
from catella.experiment.methdata import MethPrintExperiment
from catella.h5_array import H5Array

__all__ = [
    "CONTEXT_NAMES",
    "GCG",
    "GCH",
    "HCG",
    "NONE",
    "A",
    "MethPrintAnalysis",
]

_VALID_SPAN = {"full", "calls"}


class MethPrintAnalysis:
    """
    Normalization and smoothing suite for MethPrintExperiment data.

    Process raw methylation signals into probability scores [0, 1] using
    rolling average smoothing and molecule-wise percentile scaling. Support
    relative normalization against unmethylated and fully methylated controls.
    """

    _EPSILON = np.finfo(float).eps  # Smallest float to avoid DivByZero

    def smooth(
        self,
        exp: MethPrintExperiment,
        *,
        lnuc: int,
        name: str = "smoothed",
        nan_method: str = "mean",
        fill_edge: float | str = np.nan,
        batch_size: int = 20000,
        mask_name: str | None = None,
    ):
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
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        lnuc : int
            Window size in base pairs for the rolling average smoothing.
        name : str, default "smoothed"
            The suffix used to store the resulting array in `exp.analysis`.
            Results are stored as 'test_{name}', 'meth_{name}', etc.
        nan_method : {"mean", "interpolate", "none"}, default "mean"
            How to fill interior nan values (gaps with valid data on both
            sides): molecule mean, linear interpolation, or "none" to
            leave them unfilled.
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
            If `nan_method` is not "mean", "interpolate", or "none".
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

        if nan_method not in ("mean", "interpolate", "none"):
            raise ValueError(
                "'nan_method' must be 'mean', 'interpolate', "
                f"or 'none', got {nan_method!r}."
            )

        exp.global_analysis[f"{name}_params"] = pd.DataFrame(
            [
                {
                    "lnuc": lnuc,
                    "nan_method": nan_method,
                    "fill_edge": fill_edge,
                    "mask_name": mask_name or "",
                }
            ]
        )
        smooth_kwargs = {
            "lnuc": lnuc,
            "nan_method": nan_method,
            "fill_edge": fill_edge,
            "batch_size": batch_size,
            "tmp_dir": exp.resolve_tmp_dir(),
        }
        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            for src in ("test", "meth", "unmeth"):
                df = getattr(raw, f"{src}_data")
                if df is None:
                    continue
                keep = (
                    _lookup_mask(exp, chrom, src, mask_name)
                    if mask_name is not None
                    else None
                )
                exp.analysis[chrom][f"{src}_{name}"] = _smooth_source(
                    df,
                    raw.nbp,
                    len(getattr(raw, f"{src}_mol_id")),
                    keep,
                    **smooth_kwargs,
                )

    def empirical_prob(
        self,
        exp: MethPrintExperiment,
        *,
        lnuc: int | None = None,
        smoothed_name: str = "smoothed",
        prob_name: str = "meth_prob",
        clip_low: float = 0.1,
        clip_high: float = 99.9,
        norm_by_strand: bool = False,
        resmooth: bool = False,
        fill_edge: float = np.nan,
        batch_size: int = 20000,
        percentile_sample_size: int = 100000,
        seed: int | None = None,
        mask_name: str | None = None,
    ):
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
        lnuc : int, optional
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
            name}` already exists in `exp.analysis`. Smoothing always
            runs with `nan_method="none"` here (see `fill_edge`).
        fill_edge : float, default np.nan
            Probability used to fill the trailing `lnuc - 1`
            positions, which have no full window -- mirrors
            `model_prob`'s `fill_edge`. Interior gaps are always
            filled with 0.5 (`logit(0.5) == 0`), the same neutral,
            no-evidence value `model_prob` gives uninformative
            positions.
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
            If `lnuc` is not provided and no cached `lnuc` exists.
            If `test_{smoothed_name}` already exists and an
            explicitly-given `lnuc` does not match what was used to
            produce it, or it was not smoothed with `nan_method="none"`.
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

        tmp_dir = exp.resolve_tmp_dir()
        lnuc = self._ensure_smoothed(
            exp, lnuc, smoothed_name, resmooth, batch_size, mask_name
        )
        end_idx = -(lnuc - 1) if lnuc > 1 else None
        sample_kw = {
            "sample_size": percentile_sample_size,
            "batch_size": batch_size,
            "end_idx": end_idx,
            "rng": np.random.default_rng(seed),
            "eps": self._EPSILON,
        }

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            has_controls = (
                raw.meth_data is not None and raw.unmeth_data is not None
            )
            srcs = ("test", "meth", "unmeth") if has_controls else ("test",)
            arrs = {
                src: _masked_smoothed(
                    exp, chrom, src, smoothed_name, mask_name, batch_size
                )
                for src in srcs
            }
            labels = dict.fromkeys(srcs)
            means = None
            if has_controls:
                if norm_by_strand:
                    _check_mapped_strands(raw, has_controls)
                    labels = {
                        src: _strand_of_mol(
                            getattr(raw, f"{src}_data"), arrs[src].shape[0]
                        )
                        for src in srcs
                    }
                means = _control_means(
                    arrs, labels, norm_by_strand, batch_size
                )

            vmin, vmax = _clip_bounds(
                arrs,
                labels,
                means,
                clip_low=clip_low,
                clip_high=clip_high,
                **sample_kw,
            )
            exp.analysis[chrom][prob_name] = _write_empirical_prob(
                arrs["test"],
                labels["test"],
                means,
                vmin,
                vmax,
                end_idx=end_idx,
                fill_edge=fill_edge,
                batch_size=batch_size,
                tmp_dir=tmp_dir,
                eps=self._EPSILON,
            )

    def _ensure_smoothed(
        self, exp, lnuc, smoothed_name, resmooth, batch_size, mask_name
    ):
        """
        Smooth with `nan_method="none"` unless a matching cached result
        exists, and return the window size actually used.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        lnuc : int or None
            Requested window size; None uses the cached one.
        smoothed_name, resmooth, batch_size, mask_name
            As in `empirical_prob`.

        Returns
        -------
        int
            The window size of `test_{smoothed_name}`.

        Raises
        ------
        ValueError
            If smoothing is needed but no `lnuc` is known, or the cached
            result was made with a different `lnuc` or `nan_method`.
        """
        already_smoothed = (
            f"test_{smoothed_name}" in exp.analysis[exp.chroms[0]]
        )
        cached_params = exp.global_analysis.get(f"{smoothed_name}_params")
        cached_lnuc = None
        if cached_params is not None:
            cached_lnuc = int(cached_params["lnuc"].iloc[0])

        if resmooth or not already_smoothed:
            if lnuc is None:
                lnuc = cached_lnuc
            if lnuc is None:
                raise ValueError(
                    "'lnuc' must be specified if data are not already "
                    "smoothed."
                )
            self.smooth(
                lnuc=lnuc,
                exp=exp,
                name=smoothed_name,
                nan_method="none",
                batch_size=batch_size,
                mask_name=mask_name,
            )
            return lnuc

        if cached_params is not None:
            mismatched = []
            if lnuc is not None and lnuc != cached_lnuc:
                mismatched.append("lnuc")
            if cached_params["nan_method"].iloc[0] != "none":
                mismatched.append("nan_method")
            # mask_name is not compared: it is reapplied to the output
            # on every call.
            if mismatched:
                raise ValueError(
                    f"'{smoothed_name}' was already smoothed with "
                    f"different {', '.join(mismatched)}; pass "
                    "resmooth=True to recompute, or use a "
                    "different 'smoothed_name'."
                )
            return cached_lnuc
        return lnuc

    def model_prob(
        self,
        exp: MethPrintExperiment,
        *,
        prob_name: str = "meth_prob",
        pi0: float = 0.5,
        eta: float | dict[str, float] | None = None,
        eta_max_lag: int = 10,
        store_rho: bool = False,
        nu: float = 10.0,
        rho_leak: float = 0.1,
        min_gap: float = 0.05,
        lnuc: int = 147,
        n_min: int = 10,
        max_iters: int = 200,
        init_prot: float = 0.05,
        init_acc: float = 0.95,
        tol: float = 1e-8,
        fill_edge: float = np.nan,
        norm_by_strand: bool = False,
        span: str = "full",
        batch_size: int = 20000,
        mask_name: str | None = None,
    ):
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
            channel's ("A"/"GCH"/"HCG"/"GCG") eta is auto-estimated
            from lag-k autocorrelation in its log-odds -- from the
            methylated control when available, or from the test data
            otherwise. A float pins every channel to that value; a
            dict pins only the named channels, leaving any not
            mentioned to be auto-estimated. Only channels actually
            assayed by `exp.mtase` are estimated and reported (all
            four if `exp.mtase` is unset). The value(s) actually
            applied are stored in
            `exp.global_analysis[f"{prob_name}_eta"]` afterward.
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
        lnuc : int, default 147
            Nucleosome footprint size (bp): the expectation-maximization
            window size (no-controls path) and the output window-sum
            size.
        n_min : int, default 10
            Minimum number of sites a read must read in a window for
            that read-window to be used in the no-controls
            expectation-maximization fit. Used only when no meth/unmeth
            controls are available.
        max_iters : int, default 200
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
            Probability used to fill the trailing `lnuc - 1` positions
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
        span : {"full", "calls"}, default "full"
            Extent of each molecule, within which every assayable site
            on the molecule's strand is read, and a read site without a
            call is treated as unmethylated. "full" assumes every
            molecule covers the whole fiber, as for artificial
            constructs or native fragments filtered to fully cover the
            region. "calls" restricts each molecule to its first to
            last call, for data with partial reads; with base callers
            that omit confident canonical calls, this trims protected
            ends of the molecule.
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
            `n_min` read sites, if `eta` is a dict with an
            unrecognized channel name, if `span` is not recognized, or
            if `norm_by_strand` is True but molecules with unmapped
            strands ('.') exist.
        KeyError
            If `mask_name` is given but no matching mask is found for
            some source/chromosome.

        Warns
        -----
        UserWarning
            If `exp.wrap` is True and many wrap-mirrored position
            pairs disagree on context (folded to no context there),
            or if, for a channel whose controls report a call at most
            read sites (explicit-mode base calling), the fraction of
            read sites with a call in the test sample lies more than
            0.05 outside the range spanned by the two controls.

        Notes
        -----
        Position `i` in the result summarizes the window
        `[i, i + lnuc)`. If `exp.wrap` is True, `refseq` (always
        stored full-length) is folded to length `nbp` before use:
        positions `i` and `length-1-i` must classify to the same
        context to be kept; where they disagree, the folded position
        is treated as having no context.

        Per-chromosome (and, if `norm_by_strand`, per-strand)
        calibration diagnostics -- whether controls were used,
        fraction of informative positions per context (only channels
        actually assayed by `exp.mtase`, or all four if unset), fraction
        of read sites carrying a call per source and context
        (`call_frac_{source}_{context}`, pooled over strands), and
        (no-controls path only) actual EM iterations run, final
        log-likelihood, fitted mixing fraction, and fitted
        theta_prot/theta_acc per context -- are stored in
        `exp.global_analysis[f"{prob_name}_calib"]`. The full
        per-position theta_prot/theta_acc/informative actually used to
        score each chromosome are stored in
        `exp.analysis[chrom][f"{prob_name}_theta"]`.
        """
        from scipy.special import expit, logit

        if span not in _VALID_SPAN:
            raise ValueError(
                f"span must be one of {sorted(_VALID_SPAN)}, got {span!r}"
            )
        tmp_dir = exp.resolve_tmp_dir()
        log_odds_fill = -logit(fill_edge)

        channels = _active_channels(exp)
        overrides = _resolve_eta_overrides(eta)
        need_auto = set(channels) - set(overrides)
        channel_eta = dict(overrides)

        calib_kwargs = {
            "nu": nu,
            "rho_leak": rho_leak,
            "min_gap": min_gap,
            "lnuc": lnuc,
            "n_min": n_min,
            "max_iters": max_iters,
            "init_prot": init_prot,
            "init_acc": init_acc,
            "tol": tol,
            "batch_size": batch_size,
            "mask_name": mask_name,
            "norm_by_strand": norm_by_strand,
            "channels": channels,
            "span": span,
        }
        _strand_suffix = {"+": "pos", "-": "neg"}

        def _calib_rows(chrom, info):
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
                    row.update(
                        {
                            f"theta_prot_{CONTEXT_NAMES[c]}": v
                            for c, v in zip(sub["codes"], sub["theta_prot"])
                        }
                    )
                    row.update(
                        {
                            f"theta_acc_{CONTEXT_NAMES[c]}": v
                            for c, v in zip(sub["codes"], sub["theta_acc"])
                        }
                    )
                rows.append(row)
            return rows

        calib_cache = {}
        calib_records = []
        if need_auto:
            stats = _init_autocorr_stats(eta_max_lag)
            for chrom in exp.chroms:
                (
                    ctx,
                    assay,
                    theta_prot,
                    theta_acc,
                    informative,
                    has_controls,
                    calib_info,
                ) = _chrom_calibration(exp, chrom, **calib_kwargs)
                calib_cache[chrom] = (
                    ctx,
                    assay,
                    theta_prot,
                    theta_acc,
                    informative,
                )
                calib_records.extend(_calib_rows(chrom, calib_info))
                # meth control isolates the crosstalk artifact from
                # real, occupancy-driven correlation in the test data
                source = "meth" if has_controls else "test"
                _accumulate_eta_source(
                    exp,
                    chrom,
                    source,
                    ctx,
                    assay,
                    theta_prot,
                    theta_acc,
                    informative,
                    pi0,
                    batch_size,
                    mask_name,
                    stats,
                    eta_max_lag,
                    need_auto,
                    norm_by_strand=norm_by_strand,
                    span=span,
                )
            estimated, rho_by_channel = _finalize_channel_eta(
                stats, eta_max_lag
            )
            channel_eta.update({c: estimated[c] for c in need_auto})

            if store_rho and rho_by_channel:
                exp.global_analysis[f"{prob_name}_rho"] = pd.DataFrame(
                    {
                        "lag": np.arange(1, eta_max_lag + 1),
                        **{
                            f"rho_{CONTEXT_NAMES[c]}": rho_by_channel[c]
                            for c in channels
                            if c in rho_by_channel
                        },
                    }
                )

        exp.global_analysis[f"{prob_name}_eta"] = pd.DataFrame(
            [{f"eta_{CONTEXT_NAMES[c]}": channel_eta[c] for c in channels}]
        )

        exp.global_analysis[f"{prob_name}_params"] = pd.DataFrame(
            [
                {
                    "pi0": pi0,
                    "eta_max_lag": eta_max_lag,
                    "nu": nu,
                    "rho_leak": rho_leak,
                    "min_gap": min_gap,
                    "lnuc": lnuc,
                    "n_min": n_min,
                    "max_iters": max_iters,
                    "init_prot": init_prot,
                    "init_acc": init_acc,
                    "tol": tol,
                    "fill_edge": fill_edge,
                    "norm_by_strand": norm_by_strand,
                    "span": span,
                    "mask_name": mask_name or "",
                }
            ]
        )

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            if chrom in calib_cache:
                ctx, assay, theta_prot, theta_acc, informative = calib_cache[
                    chrom
                ]
            else:
                (
                    ctx,
                    assay,
                    theta_prot,
                    theta_acc,
                    informative,
                    _,
                    calib_info,
                ) = _chrom_calibration(exp, chrom, **calib_kwargs)
                calib_records.extend(_calib_rows(chrom, calib_info))

            if norm_by_strand:
                theta_cols = {}
                for s in ("+", "-"):
                    suffix = _strand_suffix[s]
                    theta_cols[f"theta_prot_{suffix}"] = theta_prot[s]
                    theta_cols[f"theta_acc_{suffix}"] = theta_acc[s]
                    theta_cols[f"informative_{suffix}"] = informative[s]
                theta_df = pd.DataFrame(theta_cols)
            else:
                theta_df = pd.DataFrame(
                    {
                        "theta_prot": theta_prot,
                        "theta_acc": theta_acc,
                        "informative": informative,
                    }
                )
            exp.analysis[chrom][f"{prob_name}_theta"] = theta_df

            eta_by_pos = np.ones(len(ctx))
            for c in channels:
                eta_by_pos[ctx == c] = channel_eta[c]

            nmol = len(raw.test_mol_id)
            out = H5Array.create(
                (nmol, raw.nbp), dtype=np.float64, dir=tmp_dir
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
            test_stats = {}
            with closing(batches):
                for b in batches:
                    start, stop, keep, labels, q, read = b[:6]
                    _accumulate_call_frac(test_stats, b, assay, ctx, channels)
                    masked = ~keep | ~read.any(axis=1)
                    if norm_by_strand:
                        pos, neg = labels == "+", labels == "-"
                        masked |= ~(pos | neg)
                        log_odds = np.zeros_like(q)
                        log_odds[pos] = _per_base_log_odds(
                            q[pos],
                            theta_prot["+"],
                            theta_acc["+"],
                            informative["+"],
                            pi0=pi0,
                            eta=eta_by_pos,
                        )
                        log_odds[neg] = _per_base_log_odds(
                            q[neg],
                            theta_prot["-"],
                            theta_acc["-"],
                            informative["-"],
                            pi0=pi0,
                            eta=eta_by_pos,
                        )
                    else:
                        log_odds = _per_base_log_odds(
                            q,
                            theta_prot,
                            theta_acc,
                            informative,
                            pi0=pi0,
                            eta=eta_by_pos,
                        )
                    log_odds_win = _window_sum_log_odds(
                        log_odds, lnuc, fill_edge=log_odds_fill
                    )
                    prob = expit(-log_odds_win)
                    prob[masked, :] = np.nan
                    out.write_batch(start, stop, prob)
            exp.analysis[chrom][prob_name] = out

            test_frac = _call_frac_info({"test": test_stats}, channels)
            rows = [r for r in calib_records if r["chrom"] == chrom]
            for row in rows:
                row.update(test_frac)
            if rows:
                _warn_call_frac_mismatch(chrom, rows[0], channels)

        exp.global_analysis[f"{prob_name}_calib"] = pd.DataFrame(calib_records)

    def sort_by_linkage(
        self,
        exp: MethPrintExperiment,
        *,
        chroms: str | Iterable[str] | None = None,
        data_name: str = "test_smoothed",
        raw_which: str | None = None,
        mask_name: str | None = None,
        sorted_name: str | None = None,
        store_link_mat: bool = True,
        link_mat_name: str | None = None,
        metric: str = "euclidean",
        method: str = "ward",
        batch_size: int = 20000,
        fill_nan: str | float | None = None,
    ) -> dict[str, np.ndarray]:
        """
        Sort molecules by hierarchical-clustering similarity.

        Iterate through the selected chromosomes, reordering rows
        (molecules) of a dense methylation signal so that similar
        molecules sit next to each other, matching the leaf order of a
        hierarchical-clustering dendrogram. By default, sorts the
        smoothed test signal if `smooth` has been run, or the raw
        test signal otherwise -- so this works whether or not `smooth`
        has been called first.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        chroms : str or iterable of str, optional
            The chromosome(s) to process. If None (default), all
            chromosomes in `exp.chroms` are processed.
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
            straight to `MethPlot.plot_meth_prob`'s `link_mat` argument).
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
            raise ValueError("'mask_name' requires 'raw_which' to be given.")
        chroms = utils.normalize_chroms(chroms, default_chroms=exp.chroms)
        tmp_dir = exp.resolve_tmp_dir()
        link_mats = {}
        for chrom in chroms:
            ana = exp.analysis[chrom]
            owns_data = False
            if raw_which is not None:
                data = exp.to_dense(
                    chrom,
                    which=raw_which,
                    as_h5array=True,
                    batch_size=batch_size,
                    mask_name=mask_name,
                )
                base_name = raw_which
                owns_data = True
            elif data_name not in ana and data_name == "test_smoothed":
                # Default target not computed yet -- fall back to the
                # raw test signal rather than requiring smooth() first.
                data = exp.to_dense(
                    chrom, which="test", as_h5array=True, batch_size=batch_size
                )
                base_name = "test"
                owns_data = True
            else:
                if data_name not in ana:
                    raise KeyError(
                        f"'{data_name}' not found in "
                        f"exp.analysis['{chrom}']. Run 'smooth' or "
                        "'meth_prob' first (matching the name/prob_name "
                        "used there), or pass 'raw_which' to source "
                        "from the raw long-form data instead."
                    )
                data = ana[data_name]
                base_name = data_name
            name = (
                sorted_name
                if sorted_name is not None
                else f"{base_name}_sorted"
            )

            try:
                order, link_mat = utils.compute_linkage(
                    data,
                    metric=metric,
                    method=method,
                    batch_size=batch_size,
                    dir=tmp_dir,
                    fill_nan=fill_nan,
                )
                if isinstance(data, H5Array):
                    sorted_data = data.reorder_rows(
                        order, dir=tmp_dir, batch_size=batch_size
                    )
                else:
                    sorted_data = data.iloc[order]
            finally:
                if owns_data:
                    data.close()
            exp.analysis[chrom][name] = sorted_data
            if store_link_mat:
                lname = (
                    link_mat_name
                    if link_mat_name is not None
                    else f"{base_name}_linkage"
                )
                exp.analysis[chrom][lname] = pd.DataFrame(link_mat)
            link_mats[chrom] = link_mat
        return link_mats
