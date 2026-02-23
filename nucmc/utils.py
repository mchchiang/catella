# utils.py

from dataclasses import fields, Field
from typing import List, Iterable, Sequence, Any
import pandas as pd
import numpy as np
import copy

IndexType = int | slice | Sequence[int]

def add_frozen_properties(cls):
    keyword = "_frozen_"
    for f in fields(cls):
        if not f.name.startswith(keyword):
            continue
        
        public_name = f.name.removeprefix(keyword)
        private_name = f.name
        doc_str = f.metadata.get("doc", "")
        
        def make_getter(p_name):
            def getter(self):
                val = getattr(self, p_name)
                # NumPy: return read-only view
                if isinstance(val, np.ndarray):
                    view = val.view()
                    view.flags.writeable = False
                    return view
                # Pandas 3.0: standard copy is O(1) and safe
                if isinstance(val, (pd.DataFrame, pd.Series)):
                    return val.copy()
                # Nested lists/tuples: convert to tuple of views
                if isinstance(val, (tuple, list)):
                    return tuple(v.view() if isinstance(v, np.ndarray)
                                 else v for v in val)
                return val
            return getter
        # Attach property + docstring. Properties don't need a slot.
        prop = property(make_getter(private_name))
        prop.__doc__ = doc_str
        setattr(cls, public_name, prop)
    return cls

# Standard utility to normalize chromosome inputs into a list
def normalize_chroms(chroms: str | Iterable[str] | None = None, 
                     default_chroms: Iterable[str] | None = None) -> List[str]:
    if chroms is None:
        return list(default_chroms) if default_chroms is not None else []
    if isinstance(chroms, str):
        if chroms == "":
            raise ValueError("chroms must not be an empty string.")
        return [chroms]
    if not isinstance(chroms, Iterable):
        raise TypeError("chroms must be a str or an iterable.")
    return list(chroms)
