# prep.py

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List
from .seq_data import FiberSeqExperiment

class FiberSeqAnalysis:

    def normalize(self, binsize : int, exp : FiberSeqExperiment,
                  name : str = "norm"):
        for chrom in exp.chroms:
            df_test = exp.raw[chrom].test_data
            df_unmeth = exp.raw[chrom].unmeth_data
            df_meth = exp.raw[chrom].meth_data
            nbp = exp.raw[chrom].nbp
            norm = self._normalize(binsize, nbp, df_test, df_unmeth, df_meth)
            exp.analysis[chrom][name] = pd.DataFrame(norm)
        
    def _normalize(self, binsize, nbp, df_test, df_unmeth=None, df_meth=None):
        # Helper functions
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
    
