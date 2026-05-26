# preprocessing.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .methdata import MethPrintExperiment
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
               name : str = "smoothed"):
        """
        Smooth methylation signals across an experiment using a rolling
        average.

        Iterate through all chromosomes to perform rolling average smoothing.
        Results are stored in-place within the experiment's analysis map.

        Parameters
        ----------
        binsize : int
            Window size in base pairs for the rolling average smoothing.
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        name : str, default "smoothed"
            The suffix used to store the resulting DataFrame in `exp.analysis`.
            Results are stored as 'test_{name}', 'meth_{name}', etc.
        """

        self._binsize = binsize
        
        # Some helper functions
        def smooth_df(df, nbp):
            # Pivot and reindex to ensure all positions are represented
            all_pos = pd.Index(range(0,nbp))
            df_piv = df.pivot(index="pos", columns="mol_index",
                              values="mod_qual").reindex(all_pos)

            # Rolling mean centered by shifting and store the results in a
            # left-aligned manner
            res = df_piv.rolling(
                window=binsize, min_periods=1).mean().shift(-(binsize-1))
            
            # Fill nan values with the mean score for each molecule
            return res.fillna(res.mean()).T
        
        for chrom in exp.chroms:
            nbp = exp.raw[chrom].nbp
            df_test = smooth_df(exp.raw[chrom].test_data, nbp)
            exp.analysis[chrom][f"test_{name}"] = df_test            
            if exp.raw[chrom].meth_data is not None:
                df_meth = smooth_df(exp.raw[chrom].meth_data, nbp)
                exp.analysis[chrom][f"meth_{name}"] = df_meth
            if exp.raw[chrom].unmeth_data is not None:
                df_unmeth = smooth_df(exp.raw[chrom].unmeth_data, nbp)
                exp.analysis[chrom][f"unmeth_{name}"] = df_unmeth
                
            
    def meth_prob(self, *, exp : MethPrintExperiment,
                  binsize : int | None = None,
                  smoothed_name : str = "smoothed",
                  prob_name : str = "meth_prob",
                  clip_low : float = 0.1,
                  clip_high : float = 99.9,
                  norm_by_strand : bool = False):

        """
        Convert the smoothed methylation signal into a methylation probability
        profile with values ranging between 0 and 1.

        Iterate through all chromosomes to perform optional control-based 
        relative normalization and percentile-based probability scaling.
        
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

        def norm(test, meth, unmeth):
            meth_avg = np.nanmean(meth, axis=0)
            unmeth_avg = np.nanmean(unmeth, axis=0)
            denom = meth_avg - unmeth_avg
            # Avoid divsion by zero
            denom[np.abs(denom) < self._EPSILON] = self._EPSILON
            return (test - unmeth_avg) / denom

        # Check for cached binsize or perform smoothing if data missing
        if binsize is None:
            binsize = self._binsize
            
        if f"test_{smoothed_name}" not in exp.analysis[exp.chroms[0]]:
            if binsize is None:
                raise ValueError("'binsize' must be specified if data are not "
                                 "already smoothed.")
            self.smooth(binsize=binsize, exp=exp, name=smoothed_name)
        
        for chrom in exp.chroms:
            raw = exp.raw[chrom]
            ana = exp.analysis[chrom]
            test = ana[f"test_{smoothed_name}"].to_numpy().copy()
            tp, tn, tu = strands(raw.test_data)
            if raw.meth_data is not None and raw.unmeth_data is not None:
                mp, mn, mu = strands(raw.meth_data)
                up, un, uu = strands(raw.unmeth_data)
                meth = ana[f"meth_{smoothed_name}"].to_numpy()
                unmeth = ana[f"unmeth_{smoothed_name}"].to_numpy()
                if norm_by_strand:
                    if len(tu) > 0 or len(mu) > 0 or len(uu) > 0:
                        raise ValueError("Cannot do normalization by strand "
                                         "with unmapped strands '.'.")
                    test[tp,:] = norm(test[tp,:], meth[mp,:], unmeth[up,:])
                    test[tn,:] = norm(test[tn,:], meth[mn,:], unmeth[un,:])
                else:
                    test[:] = norm(test, meth, unmeth)
        
            # Normalize the data so that all values are between 0 and 1
            # Exclude trailing edge created by rolling window
            end_idx = -(binsize-1) if binsize > 1 else None        
            valid = test[:,:end_idx]
        
            # Calculate the floor and ceiling globally
            vmin = np.percentile(valid, clip_low)
            vmax = np.percentile(valid, clip_high)
        
            # Use the difference between percentiles as the scaling factor
            denom = np.maximum(vmax-vmin, self._EPSILON)
        
            # Map to probability [0,1]. Clip to ensure that outliers outside
            # the percentile bounds do not result in probabilities < 0 or > 1.
            prob = np.clip((test-vmin)/denom, 0.0, 1.0)
            ana[prob_name] = pd.DataFrame(
                prob, index=ana[f"test_{smoothed_name}"].index)
            
