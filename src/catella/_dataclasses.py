# _dataclasses.py

# Helpers for frozen dataclasses whose data fields are named
# `_frozen_<name>` and exposed as read-only properties `<name>`

from dataclasses import fields

import numpy as np
import pandas as pd

_FROZEN = "_frozen_"


def _init_frozen_fields(obj, kwargs, msg):
    """
    Set the fields of a frozen dataclass instance from `kwargs`.

    A field named `_frozen_<name>` is set from `kwargs[<name>]` if
    `kwargs` has no entry for the field name itself. Missing fields are
    set to None.

    Parameters
    ----------
    obj : dataclass instance
        The object to initialize.
    kwargs : dict
        The field values; must contain `_internal=True`, which is removed.
    msg : str
        The error message if `_internal` is missing.

    Raises
    ------
    TypeError
        If `kwargs` lacks `_internal=True`.
    """
    if not kwargs.pop("_internal", False):
        raise TypeError(msg)
    for f in fields(obj):
        val = kwargs.get(f.name)
        if val is None:
            val = kwargs.get(f.name.removeprefix(_FROZEN))
        object.__setattr__(obj, f.name, val)


def _add_frozen_properties(cls):
    """
    Add a property for each `_frozen_` field of a dataclass.

    For every field named `_frozen_<name>`, a property `<name>` is added
    whose docstring is the field's `metadata["doc"]`. The property
    returns a read-only view of a NumPy array, a copy of a pandas
    object, a tuple of the elements (with views of any arrays) for a
    list or tuple, and any other value as is.

    Parameters
    ----------
    cls : type
        A dataclass.

    Returns
    -------
    type
        `cls`, with the properties added.
    """
    for f in fields(cls):
        if not f.name.startswith(_FROZEN):
            continue

        public_name = f.name.removeprefix(_FROZEN)
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
                    return tuple(
                        v.view() if isinstance(v, np.ndarray) else v
                        for v in val
                    )
                return val

            return getter

        # Attach property + docstring. Properties do not need a slot.
        prop = property(make_getter(private_name))
        prop.__doc__ = doc_str
        setattr(cls, public_name, prop)
    return cls
