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

            # Rolling mean centered by shifting
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
            
    
    def normalize(self, *, binsize : int,
                  exp : MethPrintExperiment,
                  name : str = "meth_prob",
                  clip_low : float = 0.1,
                  clip_high : float = 99.9,
                  by_strand : bool = False):
        """
        Normalize and smooth methylation signals across an experiment.

        Iterate through all chromosomes to perform rolling average 
        smoothing and optional control-based scaling. Results are 
        stored in-place within the experiment's analysis map.

        Parameters
        ----------
        binsize : int
            Window size in base pairs for the rolling average smoothing.
        exp : MethPrintExperiment
            The experiment object containing raw data and analysis maps.
        name : str, optional
            The dictionary key used to store the resulting DataFrame 
            in `exp.analysis`. Default is "meth_prob".
        clip_low : float, default 0.1
            Lower percentile bound for signal clipping. Values below this
            percentile are set to 0. Help remove background noise and ground
            the signal. 
        clip_high : float, default 99.9
            Upper percentile bound for signal clipping. Values above this
            percentile are set to 1. Protect the dynamic range from extreme
            technical outliers.
        by_strand : bool, default False
            Whether to do the normalization separately based on the
            strandedness of the molecule.
        
        """
        for chrom in exp.chroms:
            df_test = exp.raw[chrom].test_data
            df_unmeth = exp.raw[chrom].unmeth_data
            df_meth = exp.raw[chrom].meth_data
            nbp = exp.raw[chrom].nbp
            norm = self._normalize(binsize=binsize, nbp=nbp, df_test=df_test,
                                   df_unmeth=df_unmeth, df_meth=df_meth,
                                   clip_low=clip_low, clip_high=clip_high,
                                   by_strand=by_strand)
            exp.analysis[chrom][name] = pd.DataFrame(norm)
        
    def _normalize(self, *,
                   binsize : int,
                   nbp : int,
                   df_test : pd.DataFrame,
                   df_unmeth : pd.DataFrame | None = None,
                   df_meth : pd.DataFrame | None = None,
                   clip_low : float = 0.1,
                   clip_high : float = 99.9,
                   by_strand : bool = False):
        """
        Internal engine for smoothing and control-based scaling.

        Perform data pivoting, apply a rolling average mean, and map signals
        to a [0, 1] probability range using global percentile clipping.

        Parameters
        ----------
        binsize : int
            Window size for the rolling average smoothing.
        nbp : int
            Total number of base pairs in the chromatin fiber, used for 
            reindexing the signal.
        df_test : pd.DataFrame
            The raw test data containing 'pos', 'mol_index', and 'mod_qual'.
        df_unmeth : pd.DataFrame, optional
            Unmethylated control data used to define the signal baseline.
        df_meth : pd.DataFrame, optional
            Fully methylated control data used to define the signal ceiling.
        clip_low : float, default 0.1
            Lower percentile bound for signal clipping. Values below this
            percentile are set to 0. Help remove background noise and ground
            the signal. 
        clip_high : float, default 99.9
            Upper percentile bound for signal clipping. Values above this
            percentile are set to 1. Protect the dynamic range from extreme
            technical outliers.
        by_strand : bool, default False
            Whether to do the normalization separately based on the
            strandedness of the molecule.
        
        Returns
        -------
        prob : np.ndarray
            A 2D array of normalized methylation probabilities with shape 
            (n_molecules, n_base_pairs).
        """
        
        # Some helper functions
        # Pivot data by position
        def piv(df):    
            pivot = df.pivot(
                index="pos", columns="mol_index", values="mod_qual")
            all_pos = pd.Index(range(0,nbp))
            pivot = pivot.reindex(all_pos)
            return pivot

        # Compute averages
        def get_rolling_avg(df_piv, binsize):
            res = df_piv.rolling(
                window=binsize, min_periods=1).mean().shift(-(binsize-1))
            # Fill nan values with the mean score for each molecule
            res = res.fillna(res.mean())
            return res
        
        def get_overall_avg(df_piv, binsize, mols=None):
            if mols is not None and len(mols) > 0:
                df_piv = df_piv.iloc[:,mols]
            res = df_piv.agg(["mean", "count"], axis=1)
            res = res.rename(columns={"mean":"mod_qual_mean",
                                      "count":"mod_qual_count"}).reset_index()
            res = res.rename(columns={"index":"pos"})            
            res["mod_qual_mean_smooth"] = \
                res["mod_qual_mean"].rolling(
                    window=binsize, min_periods=1).mean().shift(-(binsize-1))
            # Fill nan values with the mean score    
            res["mod_qual_mean_smooth"] = \
                res["mod_qual_mean_smooth"].fillna(
                    res["mod_qual_mean_smooth"].mean())
            return res

        def get_mol_strands(df):
            pve = df[df["strand"] == "+"]["mol_index"].unique()
            nve = df[df["strand"] == "-"]["mol_index"].unique()
            nth = df[df["strand"] == "."]["mol_index"].unique()
            return pve, nve, nth
        
        piv_test = piv(df_test)
        test_rollavg = get_rolling_avg(piv_test, binsize).to_numpy().T
        test_pve, test_nve, test_nth = get_mol_strands(df_test)
        
        # Normalize against control samples if they are present
        if (df_unmeth is not None and df_meth is not None):
            print("Normalizing against control samples ...")
            piv_unmeth = piv(df_unmeth)
            piv_meth = piv(df_meth)
            unmeth_pve, unmeth_nve, unmeth_nth = get_mol_strands(df_unmeth)
            meth_pve, meth_nve, meth_nth = get_mol_strands(df_meth)
            def norm(test, unmeth, meth):
                unmeth_avg = get_overall_avg(unmeth, binsize)
                unmeth_avg = unmeth_avg["mod_qual_mean_smooth"].to_numpy()
                meth_avg = get_overall_avg(meth, binsize)
                meth_avg = meth_avg["mod_qual_mean_smooth"].to_numpy()
                # Avoid divsion by zero in controls
                denom = meth_avg - unmeth_avg
                denom[denom == 0] = self._EPSILON
                test[:] = (test - unmeth_avg) / denom                
            if by_strand:
                if len(test_nth) > 0 or len(unmeth_nth) > 0 or \
                   len(meth_nth) > 0:
                    raise ValueError("Cannot do normalization by strand if "
                                     "there are molecules with unmapped "
                                     "strand '.'.")
                norm(test_rollavg[test_pve,:], piv_unmeth.iloc[:,unmeth_pve],
                     piv_meth.iloc[:,meth_pve])
                norm(test_rollavg[test_nve,:], piv_unmeth.iloc[:,unmeth_nve],
                     piv_meth.iloc[:,meth_nve])
            else:
                norm(test_rollavg, piv_unmeth, piv_meth)
                
        # Normalize the data so that all values are between 0 and 1
        # Identify valid genomic range to avoid biases from the trailing edge
        end_idx = -(binsize-1) if binsize > 1 else None        
        valid = test_rollavg[:,:end_idx]
        
        # Calculate the floor and ceiling globally
        vmin = np.percentile(valid, clip_low)
        vmax = np.percentile(valid, clip_high)
        
        # Use the difference between percentiles as the scaling factor
        denom = np.maximum(vmax-vmin, self._EPSILON)
        
        # Map to probability [0,1]. Clip to ensure that outliers outside the
        # percentile bounds do not result in probabilities < 0 or > 1.
        prob = np.clip((test_rollavg-vmin)/denom, 0.0, 1.0)
        
        return prob
    
    
