# h5_utils.py

import pandas as pd
import numpy as np
import h5py
    
# Helper functions for loading and saving data frames in h5 files
def save_df(name, df, group):
    g = group.create_group(name)
    dt = h5py.string_dtype(encoding="utf-8") # For storing strings
    # Store the master order of all columns
    order = df.columns.astype(str).tolist()
    g.create_dataset("_column_order", data=order, dtype=dt)
    g.attrs["_column_index_dtype"] = str(df.columns.dtype)

    # Split numeric and string data
    num_df = df.select_dtypes(include=[np.number])
    str_df = df.select_dtypes(exclude=[np.number])

    # Save numeric block
    if not num_df.empty:
        g.create_dataset("num_values", data=num_df.to_numpy(),
                         compression="gzip")
        g.create_dataset("num_names", dtype=dt, compression="gzip",
                         data=num_df.columns.astype(str).tolist())

    # Save string columns individually
    if not str_df.empty:
        for col in str_df.columns:
            sdata = str_df[col].astype(str).tolist()
            g.create_dataset(f"str_{col}", data=sdata, dtype=dt,
                             compression="gzip")

def load_df(name, group):
    if name in group:
        g = group[name]
        orig_dtype = g.attrs.get("_column_index_dtype", "object")
        
        # Load the numeric data
        data = g["num_values"][:]
        cols = [c.decode("utf-8") if isinstance(c, bytes) else c
                for c in g["num_names"][:]]
        df = pd.DataFrame(data, columns=cols)
        
        # Load the string data
        for k in g.keys():
            if k.startswith("str_"):
                colname = k.replace("str_", "")
                df[colname] = [s.decode('utf-8') if isinstance(s,bytes) else s
                               for s in g[k][:]]

        # Reorder to the original state
        order = [c.decode() if isinstance(c,bytes) else c
                 for c in g["_column_order"][:]]
        df = df[order]

        # Convert column names to their original type
        try:
            df.columns = df.columns.astype(orig_dtype)
        except: pass
        return df
    return None
