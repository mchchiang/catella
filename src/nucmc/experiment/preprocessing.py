# preprocessing.py

import numpy as np
import pandas as pd
from typing import List
from .methdata import MethPrintExperiment, _apply_keep_mask
from ..h5_array import H5Array
from .. import utils
import matplotlib.pyplot as plt


def _lookup_mask(exp, chrom, channel, mask_name):
    """Fetch a filter_dropout() 'keep' array, or raise KeyError."""
    key = f"{channel}_{mask_name}"
    if key not in exp.analysis[chrom]:
        raise KeyError(
            f"'{key}' not found in exp.analysis['{chrom}']. Run "
            "filter_dropout() for this channel first, matching "
            "mask_name.")
    return exp.analysis[chrom][key]["keep"].to_numpy()


class MethPrintAnalysis:
    """
    Normalization and smoothing suite for MethPrintExperiment data.

    Process raw methylation signals into probability scores [0, 1] using
    rolling average smoothing and molecule-wise percentile scaling. Support
    relative normalization against unmethylated and fully methylated controls.
    """
    
    _EPSILON = np.finfo(float).eps  # Smallest float to avoid DivByZero

    def __init__(self):
        self._binsize = None # Cache the binsize used for smoothing

    def smooth(self, *, binsize : int,
               exp : MethPrintExperiment,
               name : str = "smoothed",
               nan_method : str = "mean",
               fill_edge : float | str = np.nan,
               batch_size : int = 20000,
               mask_name : str | None = None):
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
        binsize : int
            Window size in base pairs for the rolling average smoothing.
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        name : str, default "smoothed"
            The suffix used to store the resulting array in `exp.analysis`.
            Results are stored as 'test_{name}', 'meth_{name}', etc.
        nan_method : {"mean", "interpolate"}, default "mean"
            How to fill interior nan values (gaps with valid data on both
            sides): molecule mean, or linear interpolation.
        fill_edge : float or "mean", default np.nan
            How to fill leading/trailing edge nan values, independent of
            `nan_method`. A literal float must lie within the data range.
            See `nucmc.mapping.CoordsTransform.fill_edge`.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        mask_name : str, optional
            If given, molecules flagged as dropout by a prior
            `MethPrintExperiment.filter_dropout(mask_name=mask_name)`
            call are set to all-NaN in the smoothed output, per
            channel. Looked up as `exp.analysis[chrom][f"{channel}_
            {mask_name}"]`.

        Raises
        ------
        ValueError
            If `nan_method` is not "mean" or "interpolate".
            If `fill_edge` is a literal float outside the data range.
        KeyError
            If `mask_name` is given but no matching mask is found for
            some channel/chromosome.

        Notes
        -----
        Scratch files backing the resulting `H5Array` objects are
        written to `exp`'s scratch directory (see
        `MethPrintExperiment.resolve_tmp_dir`), shared with any other
        scratch files from the same experiment (e.g. from `load_raw` or
        `meth_prob`).
        """

        if nan_method not in ("mean", "interpolate"):
            raise ValueError("'nan_method' must be 'mean' or 'interpolate', "
                             f"got {nan_method!r}.")

        self._binsize = binsize
        tmp_dir = exp.resolve_tmp_dir()
        
        # Some helper functions
        def smooth_df(df, nbp, nmol, keep):
            if not isinstance(fill_edge, str) and not np.isnan(fill_edge):
                vmin, vmax = df["mod_qual"].min(), df["mod_qual"].max()
                if not (vmin <= fill_edge <= vmax):
                    raise ValueError(
                        f"'fill_edge'={fill_edge} is outside the data "
                        f"range [{vmin}, {vmax}].")

            all_pos = pd.Index(range(0, nbp))
            df = df.sort_values("mol_index", kind="stable")
            mol_index = df["mol_index"].to_numpy()
            out = H5Array.create((nmol, nbp), dtype=np.float64,
                                 dir=tmp_dir)

            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                lo, hi = np.searchsorted(mol_index, [start, stop])
                batch_df = df.iloc[lo:hi]

                # Pivot and reindex to ensure all positions are represented
                df_piv = batch_df.pivot(
                    index="pos", columns="mol_index",
                    values="mod_qual").reindex(all_pos)

                # Rolling mean centered by shifting and store the results
                # in a left-aligned manner
                res = df_piv.rolling(
                    window=binsize, min_periods=1).mean().shift(
                        -(binsize-1))

                # Interior nans (bounded by valid data on both sides) are
                # filled per nan_method; interpolate() with
                # limit_area="inside" only fills those, leaving edge nans
                # untouched so they can be probed for below.
                interior_interp = res.interpolate(method="linear",
                                                   limit_area="inside")
                if nan_method == "interpolate":
                    res = interior_interp
                else:
                    interior_mask = res.isna() & interior_interp.notna()
                    res = res.where(~interior_mask, res.fillna(res.mean()))

                # Fill remaining edge nans independently of nan_method
                if isinstance(fill_edge, str) and fill_edge == "mean":
                    res = res.fillna(res.mean())
                else:
                    res = res.fillna(fill_edge)

                out_batch = res.to_numpy().T
                if keep is not None:
                    out_batch = _apply_keep_mask(out_batch, keep[start:stop])
                out.write_batch(start, stop, out_batch)
            return out

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            nbp = raw.nbp
            def chan_keep(ch):
                return _lookup_mask(exp, chrom, ch, mask_name) \
                    if mask_name is not None else None
            exp.analysis[chrom][f"test_{name}"] = smooth_df(
                raw.test_data, nbp, len(raw.test_mol_id),
                chan_keep("test"))
            if raw.meth_data is not None:
                exp.analysis[chrom][f"meth_{name}"] = smooth_df(
                    raw.meth_data, nbp, len(raw.meth_mol_id),
                    chan_keep("meth"))
            if raw.unmeth_data is not None:
                exp.analysis[chrom][f"unmeth_{name}"] = smooth_df(
                    raw.unmeth_data, nbp, len(raw.unmeth_mol_id),
                    chan_keep("unmeth"))
                
            
    def meth_prob(self, *, exp : MethPrintExperiment,
                  binsize : int | None = None,
                  smoothed_name : str = "smoothed",
                  prob_name : str = "meth_prob",
                  clip_low : float = 0.1,
                  clip_high : float = 99.9,
                  norm_by_strand : bool = False,
                  nan_method : str = "mean",
                  fill_edge : float | str = np.nan,
                  batch_size : int = 20000,
                  percentile_sample_size : int = 100000,
                  seed : int | None = None,
                  mask_name : str | None = None):

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
        binsize : int, optional
            Window size in base pairs used for the rolling average smoothing.
            If None, use the cached binsize a previous `smooth` call.
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
            control channel (making the bound condition-independent);
            otherwise it is estimated from the test signal itself.
        clip_high : float, default 99.9
            Upper percentile bound for signal clipping. Values above this
            percentile are set to 1. If `exp` has meth/unmeth controls,
            this percentile is estimated from the normalized meth
            control channel (making the bound condition-independent);
            otherwise it is estimated from the test signal itself.
        norm_by_strand : bool, default False
            Whether to perform normalization separately based on strandedness.
        nan_method : {"mean", "interpolate"}, default "mean"
            Passed through to `smooth` if smoothing is triggered lazily.
        fill_edge : float or "mean", default np.nan
            Passed through to `smooth` if smoothing is triggered lazily.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        percentile_sample_size : int, default 100000
            Approximate number of molecules used to estimate `clip_low`/
            `clip_high` percentile bounds. If the relevant channel (the
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
            from control-based normalization statistics, per channel.
            Also passed through to `smooth` if smoothing is triggered
            lazily.

        Raises
        ------
        ValueError
            If `binsize` is not provided and no cached `binsize` exists.
            If `norm_by_strand` is True but molecules with unmapped strands
            ('.') exist.
        KeyError
            If `mask_name` is given but no matching mask is found for
            some channel/chromosome.

        Notes
        -----
        The scratch file backing the resulting probability `H5Array`,
        and any scratch files from smoothing triggered lazily, are
        written to `exp`'s scratch directory (see
        `MethPrintExperiment.resolve_tmp_dir`), shared with any other
        scratch files from the same experiment.
        """

        # Some helper functions
        def strands(df):
            pve = df[df["strand"] == "+"]["mol_index"].unique()
            nve = df[df["strand"] == "-"]["mol_index"].unique()
            uni = df[df["strand"] == "."]["mol_index"].unique()
            return pve, nve, uni

        def strand_of_mol(df, nmol):
            # One strand label per molecule, looked up by mol_index
            labels = np.full(nmol, ".", dtype="<U1")
            first = df.drop_duplicates("mol_index")[["mol_index", "strand"]]
            labels[first["mol_index"].to_numpy()] = first["strand"].to_numpy()
            return labels

        def streamed_nanmean(arr, row_mask=None):
            # Per-position mean over molecules (rows), streamed in
            # batches, delegating to H5Array's shared reduction logic
            return arr._streamed_reduce("mean", 0, batch_size,
                                        row_mask=row_mask)

        def apply_norm(test_batch, meth_avg, unmeth_avg):
            denom = meth_avg - unmeth_avg
            # Avoid division by zero
            denom = np.where(np.abs(denom) < self._EPSILON,
                             self._EPSILON, denom)
            return (test_batch - unmeth_avg) / denom

        tmp_dir = exp.resolve_tmp_dir()

        # Check for cached binsize or perform smoothing if data missing
        if binsize is None:
            binsize = self._binsize

        if f"test_{smoothed_name}" not in exp.analysis[exp.chroms[0]]:
            if binsize is None:
                raise ValueError("'binsize' must be specified if data are not "
                                 "already smoothed.")
            self.smooth(binsize=binsize, exp=exp, name=smoothed_name,
                       nan_method=nan_method, fill_edge=fill_edge,
                       batch_size=batch_size, mask_name=mask_name)

        rng = np.random.default_rng(seed)

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            ana = exp.analysis[chrom]
            test_arr = ana[f"test_{smoothed_name}"]
            if mask_name is not None:
                test_arr = _apply_keep_mask(
                    test_arr, _lookup_mask(exp, chrom, "test", mask_name),
                    batch_size)
            nmol, nbp = test_arr.shape

            has_controls = (raw.meth_data is not None and
                            raw.unmeth_data is not None)
            test_strand = None
            meth_avg = unmeth_avg = None
            meth_avg_pos = meth_avg_neg = None
            unmeth_avg_pos = unmeth_avg_neg = None

            if has_controls:
                meth_arr = ana[f"meth_{smoothed_name}"]
                unmeth_arr = ana[f"unmeth_{smoothed_name}"]
                if mask_name is not None:
                    meth_arr = _apply_keep_mask(
                        meth_arr,
                        _lookup_mask(exp, chrom, "meth", mask_name),
                        batch_size)
                    unmeth_arr = _apply_keep_mask(
                        unmeth_arr,
                        _lookup_mask(exp, chrom, "unmeth", mask_name),
                        batch_size)
                if norm_by_strand:
                    tp, tn, tu = strands(raw.test_data)
                    mp, mn, mu = strands(raw.meth_data)
                    up, un, uu = strands(raw.unmeth_data)
                    if len(tu) > 0 or len(mu) > 0 or len(uu) > 0:
                        raise ValueError("Cannot do normalization by strand "
                                         "with unmapped strands '.'.")
                    test_strand = strand_of_mol(raw.test_data, nmol)
                    meth_strand = strand_of_mol(raw.meth_data,
                                                meth_arr.shape[0])
                    unmeth_strand = strand_of_mol(raw.unmeth_data,
                                                  unmeth_arr.shape[0])
                    meth_avg_pos = streamed_nanmean(
                        meth_arr, meth_strand == "+")
                    meth_avg_neg = streamed_nanmean(
                        meth_arr, meth_strand == "-")
                    unmeth_avg_pos = streamed_nanmean(
                        unmeth_arr, unmeth_strand == "+")
                    unmeth_avg_neg = streamed_nanmean(
                        unmeth_arr, unmeth_strand == "-")
                else:
                    meth_avg = streamed_nanmean(meth_arr)
                    unmeth_avg = streamed_nanmean(unmeth_arr)

            def normalized_batch(start, stop):
                batch = test_arr[start:stop, :]
                if not has_controls:
                    return batch
                if norm_by_strand:
                    labels = test_strand[start:stop]
                    out = np.empty_like(batch)
                    pos, neg = labels == "+", labels == "-"
                    out[pos] = apply_norm(batch[pos], meth_avg_pos,
                                          unmeth_avg_pos)
                    out[neg] = apply_norm(batch[neg], meth_avg_neg,
                                          unmeth_avg_neg)
                    return out
                return apply_norm(batch, meth_avg, unmeth_avg)

            # Normalize the data so that all values are between 0 and 1
            # Exclude trailing edge created by rolling window
            end_idx = -(binsize-1) if binsize > 1 else None

            def normalized_sample(arr, strand_labels):
                # Random subsample of an array's molecules, normalized
                # the same way as `normalized_batch` when controls are
                # available (or left raw otherwise), sampled
                # proportionally within each contiguously-read batch.
                n_total = arr.shape[0]
                sample_size = min(percentile_sample_size, n_total)
                sample_frac = sample_size / n_total
                chunks = []
                for start in range(0, n_total, batch_size):
                    stop = min(start + batch_size, n_total)
                    batch = arr[start:stop, :]
                    if has_controls:
                        if norm_by_strand:
                            labels = strand_labels[start:stop]
                            normed = np.empty_like(batch)
                            pos, neg = labels == "+", labels == "-"
                            normed[pos] = apply_norm(
                                batch[pos], meth_avg_pos, unmeth_avg_pos)
                            normed[neg] = apply_norm(
                                batch[neg], meth_avg_neg, unmeth_avg_neg)
                            batch = normed
                        else:
                            batch = apply_norm(batch, meth_avg, unmeth_avg)
                    n_take = min(stop - start,
                                max(1, round((stop - start) * sample_frac)))
                    rows = rng.choice(stop - start, size=n_take,
                                      replace=False)
                    chunks.append(batch[rows, :end_idx])
                return np.concatenate(chunks, axis=0)

            # Estimate vmin/vmax from the normalized control channels
            # when available, so bounds depend only on the controls
            # (and are therefore comparable across test conditions
            # sharing the same controls) rather than on the test
            # signal's own, condition-dependent dynamic range. Fall
            # back to the test signal itself when there are no controls.
            if has_controls:
                unmeth_sample = normalized_sample(
                    unmeth_arr, unmeth_strand if norm_by_strand else None)
                meth_sample = normalized_sample(
                    meth_arr, meth_strand if norm_by_strand else None)
                vmin = np.nanpercentile(unmeth_sample, clip_low)
                vmax = np.nanpercentile(meth_sample, clip_high)
            else:
                test_sample = normalized_sample(test_arr, None)
                vmin = np.nanpercentile(test_sample, clip_low)
                vmax = np.nanpercentile(test_sample, clip_high)

            # Use the difference between percentiles as the scaling factor
            denom = np.maximum(vmax-vmin, self._EPSILON)

            # Map to probability [0,1] and stream the result to disk.
            # Clip to ensure that outliers outside the percentile bounds
            # do not result in probabilities < 0 or > 1.
            out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = normalized_batch(start, stop)
                prob = np.clip((batch-vmin)/denom, 0.0, 1.0)
                out.write_batch(start, stop, prob)
            ana[prob_name] = out

    def sort_by_linkage(self, *,
                        exp : MethPrintExperiment,
                        data_name : str = "test_smoothed",
                        raw_which : str | None = None,
                        sorted_name : str | None = None,
                        store_link_mat : bool = True,
                        link_mat_name : str | None = None,
                        metric : str = "euclidean",
                        method : str = "ward",
                        batch_size : int = 20000,
                        fill_nan : str | float | None = None
                        ) -> dict[str, np.ndarray]:
        """
        Sort molecules by hierarchical-clustering similarity.

        Iterate through all chromosomes, reordering rows (molecules)
        of a dense methylation signal so that similar molecules sit
        next to each other, matching the leaf order of a
        hierarchical-clustering dendrogram. By default, sorts the
        smoothed test signal if `smooth` has been run, or the raw
        test signal otherwise -- so this works whether or not `smooth`
        has been called first.

        Parameters
        ----------
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
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
            straight to `MethPlot.plot_methmap`'s `link_mat` argument).
            Also persisted into `exp.analysis[chrom][link_mat_name]` if
            `store_link_mat` is True.

        Raises
        ------
        KeyError
            If `raw_which` is None, `data_name` is not `"test_smoothed"`
            (its default), and not found in `exp.analysis[chrom]` for
            some chromosome.
        """
        tmp_dir = exp.resolve_tmp_dir()
        link_mats = {}
        for chrom in exp.chroms:
            ana = exp.analysis[chrom]
            if raw_which is not None:
                data = exp.to_dense(chrom, which=raw_which,
                                    as_h5array=True, batch_size=batch_size)
                base_name = raw_which
            elif data_name not in ana and data_name == "test_smoothed":
                # Default target not computed yet -- fall back to the
                # raw test signal rather than requiring smooth() first.
                data = exp.to_dense(chrom, which="test",
                                    as_h5array=True, batch_size=batch_size)
                base_name = "test"
            else:
                if data_name not in ana:
                    raise KeyError(
                        f"'{data_name}' not found in "
                        f"exp.analysis['{chrom}']. Run 'smooth' or "
                        "'meth_prob' first (matching the name/prob_name "
                        "used there), or pass 'raw_which' to source "
                        "from the raw long-form data instead.")
                data = ana[data_name]
                base_name = data_name
            name = sorted_name if sorted_name is not None \
                else f"{base_name}_sorted"

            order, link_mat = utils.compute_linkage(
                data, metric=metric, method=method, batch_size=batch_size,
                dir=tmp_dir, fill_nan=fill_nan)
            if isinstance(data, H5Array):
                sorted_data = data.reorder_rows(order, dir=tmp_dir,
                                                batch_size=batch_size)
            else:
                sorted_data = data.iloc[order]
            exp.analysis[chrom][name] = sorted_data
            if store_link_mat:
                lname = link_mat_name if link_mat_name is not None \
                    else f"{base_name}_linkage"
                exp.analysis[chrom][lname] = pd.DataFrame(link_mat)
            link_mats[chrom] = link_mat
        return link_mats

