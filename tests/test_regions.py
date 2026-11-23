# test_regions.py

import pytest
from catella._regions import _normalize_regions, _per_region, _region_list


class TestNormalizeRegions:
    def test_str_gives_one_element_list(self):
        assert _normalize_regions("chr1") == ["chr1"]

    def test_iterable_keeps_order(self):
        assert _normalize_regions(("chr2", "chr1")) == ["chr2", "chr1"]

    def test_none_uses_default(self):
        assert _normalize_regions(None, default_regions=["a", "b"]) == [
            "a",
            "b",
        ]

    def test_none_without_default_is_empty(self):
        assert _normalize_regions() == []

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty string"):
            _normalize_regions("")

    def test_non_iterable_raises(self):
        with pytest.raises(TypeError, match="str or an iterable"):
            _normalize_regions(3)


class TestRegionList:
    def test_str_gives_one_element_list(self):
        assert _region_list("chr1") == ["chr1"]

    def test_iterable_keeps_order(self):
        assert _region_list(iter(["chr2", "chr1"])) == ["chr2", "chr1"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _region_list([])


class TestPerRegion:
    def test_none_gives_none_per_region(self):
        assert _per_region(None, ["a", "b"], "x") == {"a": None, "b": None}

    def test_dict_fills_missing_and_ignores_extra(self):
        got = _per_region({"a": 1, "z": 9}, ["a", "b"], "x")
        assert got == {"a": 1, "b": None}

    def test_single_value_with_one_region(self):
        assert _per_region([0, 2], ["a"], "x") == {"a": [0, 2]}

    def test_single_value_with_several_regions_raises(self):
        with pytest.raises(ValueError, match="x must be a dict keyed"):
            _per_region([0, 2], ["a", "b"], "x")
