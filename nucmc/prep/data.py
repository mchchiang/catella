# data.py

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import h5py

@dataclass
class FibreSeqData:
    chrom : str
    nbps : int
    nmols : int
    molids : np.ndarray    
    norm_data : np.ndarray    
    unmeth_data : pd.DataFrame
    meth_data : pd.DataFrame
    test_data : pd.DataFrame
    
    def save(self, path: str | Path):
        # Fix string array so items are not objects and have fixed length
        def fix_strarr(arr):
            max_len = max(len(s) for s in arr)
            return arr.astype(f"S{max_len}")
        with h5py.File(path, "w") as f:
            f.attrs["chrom"] = self.chrom
            f.attrs["nbps"] = self.nbps            
            f.attrs["nmols"] = self.nmols
            f.create_dataset("molids", data=fix_strarr(self.molids),
                             compression="gzip")
            f.create_dataset("norm_data", data=self.norm_data,
                             compression="gzip")
            for name, df in [("unmeth", self.unmeth_data),
                             ("meth", self.meth_data),
                             ("test", self.test_data)]:
                if (df is not None):
                    dtypes = []
                    g = f.create_group(name)
                    g.attrs["columns"] = fix_strarr(df.columns.to_numpy())
                    for col in df.columns:
                        data = df[col]
                        if (data.dtype.kind == "O"):
                            data = data.fillna("").astype(
                                f"S{data.str.len().max()}")
                        g.create_dataset(col, data=data.to_numpy(),
                                         compression="gzip")
    
    @classmethod
    def load(cls, path: str | Path):
        with h5py.File(path, "r") as f:
            chrom = f.attrs["chrom"]
            nbps = int(f.attrs["nbps"])
            nmols = int(f.attrs["nmols"])
            molids = f["molids"][:].astype("U")
            norm_data = f["norm_data"][:]
            def load_df(group):
                if (group in f.keys()):
                    g = f[group]
                    colnames = [c.decode("utf-8") for c in g.attrs["columns"]]
                    loaded = {}
                    for col in colnames:
                        data = g[col][()]
                        if (data.dtype.kind == "S"):
                            data = data.astype(str)
                        loaded[col] = data                    
                    return pd.DataFrame(loaded, columns=colnames)
                else:
                    return None

            unmeth_data = load_df("unmeth")
            meth_data = load_df("meth")
            test_data = load_df("test")
    
        return cls(chrom, nbps, nmols, molids, norm_data, unmeth_data,
                   meth_data, test_data)
    
