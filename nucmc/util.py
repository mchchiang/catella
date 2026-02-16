# util.py

from collections.abc import MutableMapping
from dataclasses import fields
from typing import Dict, Any, Iterator, Sequence
import pandas as pd
import numpy as np
import h5py

IndexType = int | slice | Sequence[int]

class DataFrameMap(MutableMapping):
    def __init__(self, data: Dict[str, Any] = None, **kwargs):
        self._data: Dict[str, pd.DataFrame] = {}
        # Use the update method so the type check runs during initialization
        if data is not None:
            self.update(data)
        if kwargs:
            self.update(kwargs)    
        
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        # Strict type enforcement
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Value for key '{key}' must be a pandas"
                            "DataFrame")        
        self._data[key] = value
        
    def __delitem__(self, key: str) -> None:
        del self._data[key]
        
    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        # Clean repr showing keys only (since DataFrames are huge)
        return f"{self.__class__.__name__}(keys={list(self._data.keys())})"
    
class FixedKeyMap(MutableMapping):
    def __init__(self, data: Dict[str, Any]):
        # Internal storage
        self._data = dict(data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setitem__(self, key: str, value: Any) -> None:
        if key not in self._data:
            raise KeyError(f"Key '{key}' is not predefined. "
                           "New keys are not allowed.")
        self._data[key] = value

    def __delitem__(self, key: str) -> None:
        raise TypeError("Deletion is disabled for FixedKeyMap.")

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._data!r})"

    
def copy_array_properties(cls):
    for f in fields(cls):
        if f.name.startswith("_"):
            public_name = f.name[1:]
            private_name = f.name
            def getter(self, _private_name=private_name):
                value = getattr(self, _private_name)
                # Copy numpy arrays
                if isinstance(value, np.ndarray):
                    return value.copy()
                # Copy pandas objects
                if isinstance(value, (pd.DataFrame, pd.Series)):
                    return value.copy()
                return value
            setattr(cls, public_name, property(getter))
    return cls

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
                 for c in g["_columns_order"][:]]
        df = df[order]

        # Convert column names to their original type
        try:
            df.columns = df.columns.astype(orig_dtype)
        except: pass
        return df
    return None
