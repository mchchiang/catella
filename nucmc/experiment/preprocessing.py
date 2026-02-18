# preprocessing.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .methydata import MethyPrintExperiment

class MethyPrintAnalysis:
    """
    A suite of normalization tools for MethyPrint experimental data.

    This class provides methods to process raw methylation signals, including 
    rolling average smoothing and normalization against unmethylated and 
    fully methylated control samples.
    """
    
    def normalize(self, binsize : int, exp : MethyPrintExperiment,
                  name : str = "norm"):
        """
        Normalize and smooth methylation signals across an experiment.

        This method processes the test data for each chromosome in the 
        experiment. If control samples (unmethylated and methylated) are 
        available, it performs a relative normalization. Results are stored 
        directly in the experiment's analysis map.

        Parameters
        ----------
        binsize : int
            The window size (in base pairs) for the rolling average smoothing.
        exp : MethyPrintExperiment
            The experiment object containing the raw data and analysis maps.
        name : str, default "norm"
            The key name used to store the resulting DataFrame in 
            `exp.analysis`.
        """
        for chrom in exp.chroms:
            df_test = exp.raw[chrom].test_data
            df_unmeth = exp.raw[chrom].unmeth_data
            df_meth = exp.raw[chrom].meth_data
            nbp = exp.raw[chrom].nbp
            norm = self._normalize(binsize, nbp, df_test, df_unmeth, df_meth)
            exp.analysis[chrom][name] = pd.DataFrame(norm)
        
    def _normalize(self, binsize, nbp, df_test, df_unmeth=None, df_meth=None):
        """
        Internal normalization engine for processing methylation dataframes.

        This method handles the pivot operations, rolling averages, and 
        the optional control-based scaling.

        Parameters
        ----------
        binsize : int
            The window size for rolling average smoothing.
        nbp : int
            The total number of base pairs in the chromatin fiber.
        df_test : pd.DataFrame
            The test data to be normalized.
        df_unmeth : pd.DataFrame, optional
            The unmethylated control data.
        df_meth : pd.DataFrame, optional
            The fully methylated control data.

        Returns
        -------
        np.ndarray
            A 2D NumPy array of normalized and smoothed methylation scores 
            with shape (n_molecules, n_base_pairs).

        Notes
        -----
        If both `df_unmeth` and `df_meth` are provided, the test data is 
        scaled using the formula:
        
        .. math::
           T_{norm} = \\frac{T - U_{avg}}{M_{avg} - U_{avg}}

        where :math:`T` is the test signal, :math:`U_{avg}` is the 
        unmethylated average, and :math:`M_{avg}` is the methylated average.
        Finally, the global mean is subtracted from the results.
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
            test_rollavg = (test_rollavg-unmeth_avg)/(meth_avg-unmeth_avg)
        test_rollavg -= np.mean(test_rollavg)        
        return test_rollavg
    
