# preprocessing.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .methdata import MethPrintExperiment
import matplotlib.pyplot as plt

class MethPrintAnalysis:
    """
    Normalization and smoothing suite for MethPrint experimental data.

    Process raw methylation signals into probability scores [0, 1] using 
    rolling average smoothing and molecule-wise percentile scaling. Support 
    relative normalization against unmethylated and fully methylated controls.
    """
    
    _EPSILON = np.finfo(float).eps  # Smallest float to avoid DivByZero
    
    def normalize(self, *, binsize : int,
                  exp : MethPrintExperiment,
                  name : str = "meth_prob",
                  clip_low : float = 0.1,
                  clip_high : float = 99.9):
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
        """
        for chrom in exp.chroms:
            df_test = exp.raw[chrom].test_data
            df_unmeth = exp.raw[chrom].unmeth_data
            df_meth = exp.raw[chrom].meth_data
            nbp = exp.raw[chrom].nbp
            norm = self._normalize(binsize=binsize, nbp=nbp, df_test=df_test,
                                   df_unmeth=df_unmeth, df_meth=df_meth,
                                   clip_low=clip_low, clip_high=clip_high)
            exp.analysis[chrom][name] = pd.DataFrame(norm)
        
    def _normalize(self, *,
                   binsize : int,
                   nbp : int,
                   df_test : pd.DataFrame,
                   df_unmeth : pd.DataFrame | None = None,
                   df_meth : pd.DataFrame | None = None,
                   clip_low : float = 0.1,
                   clip_high : float = 99.9):
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
        
        def get_overall_avg(df_piv, binsize):
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
        
        piv_test = piv(df_test)
        test_rollavg = get_rolling_avg(piv_test, binsize).to_numpy().T
        
        # Normalize against control samples if they are present
        if (df_unmeth is not None and df_meth is not None):
            print("Normalizing against control samples ...")
            piv_unmeth = piv(df_unmeth)
            piv_meth = piv(df_meth)
            unmeth_avg = get_overall_avg(piv_unmeth, binsize)
            unmeth_avg = unmeth_avg["mod_qual_mean_smooth"].to_numpy()
            meth_avg = get_overall_avg(piv_meth, binsize)
            meth_avg = meth_avg["mod_qual_mean_smooth"].to_numpy()
            # Avoid divsion by zero in controls
            denom = meth_avg - unmeth_avg
            denom[denom == 0] = self._EPSILON
            test_rollavg = (test_rollavg - unmeth_avg) / denom
            
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
    
    
