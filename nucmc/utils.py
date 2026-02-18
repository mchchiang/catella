# utils.py

from dataclasses import fields
from typing import List, Iterable, Sequence
import pandas as pd
import numpy as np

IndexType = int | slice | Sequence[int]
    
def copy_array_properties(cls):
    for f in fields(cls):
        if f.name.startswith("_"):
            public_name = f.name[1:]
            private_name = f.name
            field_doc = f.metadata.get("doc","")
            def getter(self, _private_name=private_name):
                value = getattr(self, _private_name)
                # Copy numpy arrays and pandas objects
                if isinstance(value, (np.ndarray, pd.DataFrame, pd.Series)):
                    return value.copy()
                return value
            prop = property(getter)
            prop.__get__(None, cls).__doc__ = field_doc
            setattr(cls, public_name, prop)
    return cls


# Standard utility to normalize chromosome inputs into a list
def normalize_chroms(chroms: str | Iterable[str] | None = None, 
                     default_chroms: Iterable[str] | None = None) -> List[str]:
    if chroms is None:
        return list(default_chroms) if default_chroms is not None else []
    if isinstance(chroms, str):
        return [chroms]
    if not isinstance(chroms, Iterable):
        raise TypeError("chroms must be a str or an iterable")
    return list(chroms)
