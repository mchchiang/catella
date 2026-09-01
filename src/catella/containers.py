# containers.py

from collections.abc import MutableMapping, Mapping
from typing import Dict, Any, Iterator
from dataclasses import fields, is_dataclass
import pandas as pd
from catella.h5_array import H5Array

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
        if not isinstance(value, (pd.DataFrame, H5Array)):
            raise TypeError(f"Value for key '{key}' must be a pandas "
                            "DataFrame or H5Array")
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

class DataclassPublicProxy(Mapping):
    def __init__(self, obj):
        self._obj = obj
        # Identify public fields once during initialization
        self._keys = [f.name for f in fields(obj)
                      if not f.name.startswith('_')]

    def __getitem__(self, key):
        if key in self._keys:
            return getattr(self._obj, key)
        raise KeyError(key)

    def __iter__(self):
        yield from self._keys

    def __len__(self):
        return len(self._keys)

    def __repr__(self) -> str:
        if not is_dataclass(self._obj):
            return super().__repr__()        
        # Get field-value pairs dynamically
        items = [f"{f.name}={getattr(self._obj, f.name)!r}" 
                 for f in fields(self._obj)]        
        return f"{self.__class__.__name__}" + \
            f"({self._obj.__class__.__name__}({', '.join(items)}))"
