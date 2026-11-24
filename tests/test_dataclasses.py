# test_dataclasses.py

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import pandas as pd
import pytest
from catella._dataclasses import _add_frozen_properties, _init_frozen_fields


class _Point(NamedTuple):
    x: int
    y: int


@_add_frozen_properties
@dataclass(frozen=True, slots=True, init=False)
class _Data:
    name: str = None
    _frozen_arr: np.ndarray = field(
        default=None, metadata={"doc": "An array."}
    )
    _frozen_df: pd.DataFrame = field(default=None)
    _frozen_seq: list = field(default=None)
    _frozen_value: float = field(default=None)

    def __init__(self, **kwargs):
        _init_frozen_fields(self, kwargs, "use the loader")


class TestInitFrozenFields:
    def test_sets_frozen_fields_from_public_names(self):
        arr = np.arange(3)
        data = _Data(_internal=True, name="a", arr=arr)
        assert data.name == "a"
        assert data._frozen_arr is arr

    def test_sets_frozen_fields_from_field_names(self):
        data = _Data(_internal=True, _frozen_value=2.0)
        assert data.value == 2.0

    def test_missing_fields_are_none(self):
        data = _Data(_internal=True)
        assert data.name is None
        assert data.value is None

    def test_requires_internal(self):
        with pytest.raises(TypeError, match="use the loader"):
            _Data(name="a")

    def test_removes_internal_from_kwargs(self):
        kwargs = {"_internal": True, "name": "a"}
        _init_frozen_fields(object.__new__(_Data), kwargs, "msg")
        assert kwargs == {"name": "a"}


class TestAddFrozenProperties:
    def test_returns_the_class(self):
        @dataclass(frozen=True)
        class Plain:
            _frozen_x: int = 0

        assert _add_frozen_properties(Plain) is Plain
        assert Plain().x == 0

    def test_array_is_read_only_view(self):
        arr = np.arange(3)
        data = _Data(_internal=True, arr=arr)
        view = data.arr
        np.testing.assert_array_equal(view, arr)
        with pytest.raises(ValueError):
            view[0] = 10
        assert arr.flags.writeable

    def test_dataframe_is_copy(self):
        df = pd.DataFrame({"a": [1, 2]})
        data = _Data(_internal=True, df=df)
        got = data.df
        got.loc[0, "a"] = 10
        assert data.df.loc[0, "a"] == 1

    def test_list_becomes_tuple_of_read_only_views(self):
        arrs = [np.arange(2), np.arange(3)]
        data = _Data(_internal=True, seq=[*arrs, "x"])
        got = data.seq
        assert isinstance(got, tuple)
        assert got[2] == "x"
        for view, arr in zip(got[:2], arrs):
            assert np.shares_memory(view, arr)
            with pytest.raises(ValueError):
                view[0] = 10
            assert arr.flags.writeable

    def test_other_values_returned_as_is(self):
        data = _Data(_internal=True, value=1.5)
        assert data.value == 1.5

    def test_named_tuple_keeps_its_type(self):
        point = _Point(1, 2)
        data = _Data(_internal=True, seq=point)
        assert data.seq is point
        assert data.seq.y == 2

    def test_property_docstrings_from_metadata(self):
        assert _Data.arr.__doc__ == "An array."
        assert _Data.value.__doc__ == ""

    def test_only_frozen_fields_get_properties(self):
        assert not isinstance(vars(_Data).get("name"), property)
        assert isinstance(vars(_Data)["arr"], property)
