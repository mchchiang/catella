# preprocess.py

import numpy as np
import pandas as pd
from pathlib import Path
from .data import PrepData

# Helper functions
def _read_data(data_file, sizes, wrap=False):
    print(f"Reading {data_file} ...")
    colnames = ["read_id", "upos", "chrom", "mod_qual", "mod_code"]
    df = pd.read_csv(data_file, header=None, sep="\t", names=colnames)
    df = pd.merge(df, sizes, on="chrom") # Add chromosome length column
    if (wrap):
        df["pos"] = np.where(df["upos"] > (df["length"]-1)/2,
                             df["length"]-df["upos"]-1, df["upos"])
    else:
        df["pos"] = df["upos"]

    # Split the data by chromosomes
    chroms = df["chrom"].unique()        
    dfs = {c:df[df["chrom"] == c] for c in chroms}        
        
    # Sort by position. Note that each molecule only has one chrom
    for c in chroms:
        dfs[c] = dfs[c].sort_values(["read_id","pos"])
    return dfs

# Pivot data by position
def _piv(data):    
    pivot = data.pivot(index="pos", columns="read_id", values="mod_qual")
    all_pos = pd.Index(range(0,data["pos"].max()+1))
    pivot = pivot.reindex(all_pos)
    return pivot

# Compute averages
def _get_rolling_avg(piv_data, binsize):
    res = piv_data.rolling(window=binsize,
                           min_periods=1).mean().shift(-(binsize-1))
    # Fill nan values with the mean score for each molecule
    res = res.fillna(res.mean())
    return res
    
def _get_overall_avg(piv_data, binsize):
    res = piv_data.agg(["mean", "count"], axis=1)
    res = res.rename(columns={"mean":"mod_qual_mean",
                              "count":"mod_qual_count"}).reset_index()
    res = res.rename(columns={"index":"pos"})

    res["mod_qual_mean_smooth"] = \
        res["mod_qual_mean"].rolling(window=binsize,
                                     min_periods=1).mean().shift(-(binsize-1))
    # Fill nan values with the mean score    
    res["mod_qual_mean_smooth"] = \
        res["mod_qual_mean_smooth"].fillna(res["mod_qual_mean_smooth"].mean())
    return res

def preprocess(binsize : int,
               chromsize : str,
               test_file : str,
               out_dir : str,
               unmeth_file : str = None,
               meth_file : str = None,
               wrap = False) -> PrepData:

    # Check if control samples are present
    has_control = unmeth_file is not None and meth_file is not None
    
    # Read chromosome sizes
    sizes = pd.read_csv(chromsize, header=None, sep="\t",
                        names=["chrom", "length"])

    # Read fibre-seq data and re-orientate the data with pos as index and
    # mod_qual score at each pos for each molecule as columns
    dfs_test = _read_data(test_file, sizes, wrap)
    chroms = dfs_test.keys()

    dfs_unmeth = None
    dfs_meth = None
    if (has_control):
        dfs_unmeth = _read_data(unmeth_file, sizes, wrap)
        dfs_meth = _read_data(meth_file, sizes, wrap)

    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True, parents=True)

    def trim(df):
        id2idx = {rid:i for i,rid in enumerate(df["read_id"].unique())}
        out = df[["pos","mod_qual","mod_code"]].copy()
        out["molid"] = df["read_id"].map(id2idx)
        out = out[["molid","pos","mod_qual","mod_code"]]
        return out
    
    for c in chroms:
        piv_test = _piv(dfs_test[c])
        test_rollavg = _get_rolling_avg(piv_test, binsize).to_numpy().T

        # Normalise the data if control samples are present
        if (has_control):
            piv_unmeth = _piv(dfs_unmeth[c])
            piv_meth = _piv(dfs_meth[c])
            unmeth_avg = _get_overall_avg(piv_unmeth, binsize)
            unmeth_avg = unmeth_avg["mod_qual_mean_smooth"].to_numpy()
            meth_avg = _get_overall_avg(piv_meth, binsize)
            meth_avg = meth_avg["mod_qual_mean_smooth"].to_numpy()
            test_rollavg = (test_rollavg-unmeth_avg)/(meth_avg-unmeth_avg)
        
        test_rollavg -= np.mean(test_rollavg)
 
        # Store the result       
        nmols = test_rollavg.shape[0]
        nbps = test_rollavg.shape[1]
        molids = dfs_test[c]["read_id"].unique()
        
        test_data = trim(dfs_test[c])
        unmeth_data = None
        meth_data = None
        if (has_control):
            unmeth_data = trim(dfs_unmeth[c])
            meth_data = trim(dfs_meth[c])        
        data = FibreSeqData(c, nbps, nmols, molids, test_rollavg,
                            unmeth_data, meth_data, test_data)
        data.save(out_path/("fbseq_"+c+".h5"))
