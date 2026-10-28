# preprocessing.py

from collections.abc import Iterable

import numpy as np
import pandas as pd

from catella import utils
from catella.experiment._calibration import (
    _calib_rows,
    _call_frac_info,
    _chrom_calibration,
    _theta_frame,
    _warn_call_frac_mismatch,
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
    _estimate_channel_eta,
    _resolve_eta_overrides,
)
from catella.experiment._reads import _check_mapped_strands, _lookup_mask
from catella.experiment._scoring import _score_test_reads
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
                f"or 'none', got {nan_method!r}"
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
                    "smoothed"
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
                    "different 'smoothed_name'"
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
        nu: float | None = None,
        rho_leak: float | None = None,
        min_rel_contrast: float = 0.2,
        lnuc: int = 147,
        lbin: int = 30,
        max_iters: int = 200,
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
        nu : float, optional
            Pseudo-count strength for shrinking each position's control
            call rate toward its context group's mean; larger values
            shrink harder at low depth. If None (default), it is
            estimated by empirical Bayes per control and per channel
            (and per strand if `norm_by_strand`); a float pins it.
            Used only when meth/unmeth controls are available.
        rho_leak : float, optional
            Leak fraction in [0, 1] interpolating the protected-state
            call rate between the unmethylated control's false-positive
            rate (0 for perfect protection) and the accessible-state
            rate. If None (default), it is fitted per channel by
            expectation-maximization on the test sample (as tau, the
            protected-site efficiency, with the accessible-site
            efficiency kappa fixed to 1, i.e., to the methylated
            control); a float pins it for all channels. Used only when
            meth/unmeth controls are available.
        min_rel_contrast : float, default 0.2
            Minimum relative contrast `(kappa - tau) / kappa` between
            the fitted accessible and protected efficiencies of a
            context (its call rates without controls) for its positions
            to be treated as informative; the context must also pass a
            BIC test of the two states against one. Applies after any
            expectation-maximization fit. With controls, a position is
            also informative only if its meth-control rate exceeds its
            unmeth-control rate. Uninformative positions contribute no
            evidence.
        lnuc : int, default 147
            Nucleosome footprint size (bp), used as the output
            window-sum size.
        lbin : int, default 30
            Bin width (bp) of the non-overlapping windows in the
            expectation-maximization fit (without controls, or with
            controls when `rho_leak` is None). Bins narrower than
            `lnuc` mix linker and nucleosomal DNA less, which sharpens
            the contrast between the fitted rates, but hold fewer sites
            each. A read-bin without read sites is left out of the fit.
        max_iters : int, default 200
            Maximum number of expectation-maximization iterations.
        tol : float, default 1e-8
            Relative log-likelihood convergence tolerance of the
            expectation-maximization fit.
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
            If `lbin` is less than 1, if `nu` is negative, if
            `min_rel_contrast` is outside [0, 1], if a chromosome has
            no reference sequence
            (`refseq`), if an expectation-maximization fit cannot find
            enough bins with read sites, if `eta` is a dict with an
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
            0.05 outside the range spanned by the two controls, or if
            a controls fit of tau ends at a boundary (tau at its upper
            bound, or a degenerate protected fraction).

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
        (`call_frac_{source}_{context}`, pooled over strands), and,
        for an EM fit, actual EM iterations run, final
        log-likelihood, fitted mixing fraction, and per context the
        fitted theta_prot/theta_acc (no controls) or tau, kappa and
        rho = tau / kappa (controls), and, if nu was fitted, the
        fitted nu and pooled rate m per control and context -- are
        stored in
        `exp.global_analysis[f"{prob_name}_calib"]`. The full
        per-position theta_prot/theta_acc/informative actually used to
        score each chromosome are stored in
        `exp.analysis[chrom][f"{prob_name}_theta"]`.
        """
        if span not in _VALID_SPAN:
            raise ValueError(
                f"span must be one of {sorted(_VALID_SPAN)}, got {span!r}"
            )
        if lbin < 1:
            raise ValueError(f"lbin must be at least 1, got {lbin}")
        if nu is not None and nu < 0:
            raise ValueError(f"nu must be non-negative, got {nu}")
        if not 0.0 <= min_rel_contrast <= 1.0:
            raise ValueError(
                f"min_rel_contrast must be in [0, 1], got {min_rel_contrast}"
            )
        tmp_dir = exp.resolve_tmp_dir()

        channels = _active_channels(exp)
        overrides = _resolve_eta_overrides(eta)
        need_auto = set(channels) - set(overrides)
        channel_eta = dict(overrides)

        calib_kwargs = {
            "nu": nu,
            "rho_leak": rho_leak,
            "min_rel_contrast": min_rel_contrast,
            "lbin": lbin,
            "max_iters": max_iters,
            "tol": tol,
            "batch_size": batch_size,
            "mask_name": mask_name,
            "norm_by_strand": norm_by_strand,
            "channels": channels,
            "span": span,
        }

        calib_cache = {}
        calib_records = []
        if need_auto:
            estimated, rho_by_channel, calib_cache, calib_records = (
                _estimate_channel_eta(
                    exp, calib_kwargs, need_auto, pi0=pi0, max_lag=eta_max_lag
                )
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
                    "nu": np.nan if nu is None else nu,
                    "rho_leak": np.nan if rho_leak is None else rho_leak,
                    "min_rel_contrast": min_rel_contrast,
                    "lnuc": lnuc,
                    "lbin": lbin,
                    "max_iters": max_iters,
                    "tol": tol,
                    "fill_edge": fill_edge,
                    "norm_by_strand": norm_by_strand,
                    "span": span,
                    "mask_name": mask_name or "",
                }
            ]
        )

        for chrom in exp.chroms:
            if chrom in calib_cache:
                calib = calib_cache[chrom]
            else:
                *calib, _, calib_info = _chrom_calibration(
                    exp, chrom, **calib_kwargs
                )
                calib_records.extend(
                    _calib_rows(chrom, calib_info, norm_by_strand)
                )
            exp.analysis[chrom][f"{prob_name}_theta"] = _theta_frame(
                calib[0], *calib[2:], norm_by_strand
            )
            prob, test_stats = _score_test_reads(
                exp,
                chrom,
                *calib,
                channel_eta=channel_eta,
                channels=channels,
                pi0=pi0,
                lnuc=lnuc,
                fill_edge=fill_edge,
                norm_by_strand=norm_by_strand,
                batch_size=batch_size,
                mask_name=mask_name,
                span=span,
                tmp_dir=tmp_dir,
            )
            exp.analysis[chrom][prob_name] = prob

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
            raise ValueError("'mask_name' requires 'raw_which' to be given")
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
                        f"exp.analysis['{chrom}']; run 'smooth' or "
                        "'meth_prob' first (matching the name/prob_name "
                        "used there), or pass 'raw_which' to source "
                        "from the raw long-form data instead"
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
