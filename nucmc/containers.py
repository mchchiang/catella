# containers.py

from collections.abc import MutableMapping
from typing import Dict, Any, Iterator
import pandas as pd

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
            raise TypeError(f"Value for key '{key}' must be a pandas "
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
