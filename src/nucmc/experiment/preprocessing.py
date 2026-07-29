# preprocessing.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .methdata import MethPrintExperiment
from ..h5_array import H5Array
import matplotlib.pyplot as plt

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
               batch_size : int = 20000,
               tmp_dir : str | Path | None = None):
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
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        tmp_dir : str or Path, optional
            Directory used for the scratch files backing the resulting
            `H5Array` objects. If None, the system default temporary
            directory is used.
        """

        self._binsize = binsize

        # Some helper functions
        def smooth_df(df, nbp, nmol):
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

                # Fill nan values with the mean score for each molecule
                res = res.fillna(res.mean())
                out.write_batch(start, stop, res.to_numpy().T)
            return out

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            nbp = raw.nbp
            exp.analysis[chrom][f"test_{name}"] = smooth_df(
                raw.test_data, nbp, len(raw.test_mol_id))
            if raw.meth_data is not None:
                exp.analysis[chrom][f"meth_{name}"] = smooth_df(
                    raw.meth_data, nbp, len(raw.meth_mol_id))
            if raw.unmeth_data is not None:
                exp.analysis[chrom][f"unmeth_{name}"] = smooth_df(
                    raw.unmeth_data, nbp, len(raw.unmeth_mol_id))
                
            
    def meth_prob(self, *, exp : MethPrintExperiment,
                  binsize : int | None = None,
                  smoothed_name : str = "smoothed",
                  prob_name : str = "meth_prob",
                  clip_low : float = 0.1,
                  clip_high : float = 99.9,
                  norm_by_strand : bool = False,
                  batch_size : int = 20000,
                  percentile_sample_size : int = 100000,
                  tmp_dir : str | Path | None = None,
                  seed : int | None = None):

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
            percentile are set to 0.
        clip_high : float, default 99.9
            Upper percentile bound for signal clipping. Values above this
            percentile are set to 1.
        norm_by_strand : bool, default False
            Whether to perform normalization separately based on strandedness.
        batch_size : int, default 20000
            Number of molecules processed (and held in memory) per batch.
        percentile_sample_size : int, default 100000
            Approximate number of molecules used to estimate `clip_low`/
            `clip_high` percentile bounds. If the chromosome has fewer
            molecules than this, all of them are used (exact bounds).
        tmp_dir : str or Path, optional
            Directory used for the scratch file backing the resulting
            probability `H5Array`, and forwarded to `smooth` if smoothing
            is triggered lazily. If None, the system default temporary
            directory is used.
        seed : int, optional
            Seed for the random number generator used for percentile
            subsampling.

        Raises
        ------
        ValueError
            If `binsize` is not provided and no cached `binsize` exists.
            If `norm_by_strand` is True but molecules with unmapped strands
            ('.') exist.
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
            # Per-position mean over molecules (rows), streamed in batches
            nmol, nbp = arr.shape
            total_sum = np.zeros(nbp)
            total_count = np.zeros(nbp)
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = arr[start:stop, :]
                if row_mask is not None:
                    sel = row_mask[start:stop]
                    if not np.any(sel):
                        continue
                    batch = batch[sel]
                total_sum += np.nansum(batch, axis=0)
                total_count += (~np.isnan(batch)).sum(axis=0)
            with np.errstate(invalid="ignore"):
                return total_sum / total_count

        def apply_norm(test_batch, meth_avg, unmeth_avg):
            denom = meth_avg - unmeth_avg
            # Avoid division by zero
            denom = np.where(np.abs(denom) < self._EPSILON,
                             self._EPSILON, denom)
            return (test_batch - unmeth_avg) / denom

        # Check for cached binsize or perform smoothing if data missing
        if binsize is None:
            binsize = self._binsize

        if f"test_{smoothed_name}" not in exp.analysis[exp.chroms[0]]:
            if binsize is None:
                raise ValueError("'binsize' must be specified if data are not "
                                 "already smoothed.")
            self.smooth(binsize=binsize, exp=exp, name=smoothed_name,
                       batch_size=batch_size, tmp_dir=tmp_dir)

        rng = np.random.default_rng(seed)

        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            ana = exp.analysis[chrom]
            test_arr = ana[f"test_{smoothed_name}"]
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

            # Estimate percentile bounds from a subsample of molecules,
            # sampled proportionally within each contiguously-read batch
            sample_size = min(percentile_sample_size, nmol)
            sample_frac = sample_size / nmol
            sample_chunks = []
            for start in range(0, nmol, batch_size):
                stop = min(start + batch_size, nmol)
                batch = normalized_batch(start, stop)
                n_take = min(stop - start,
                            max(1, round((stop - start) * sample_frac)))
                rows = rng.choice(stop - start, size=n_take, replace=False)
                sample_chunks.append(batch[rows, :end_idx])
            valid = np.concatenate(sample_chunks, axis=0)

            # Calculate the floor and ceiling from the subsample
            vmin = np.percentile(valid, clip_low)
            vmax = np.percentile(valid, clip_high)

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
            
