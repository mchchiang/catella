# test_chroms.py

import pytest
from catella._chroms import _chrom_list, _normalize_chroms, _per_chrom


class TestNormalizeChroms:
    def test_str_gives_one_element_list(self):
        assert _normalize_chroms("chr1") == ["chr1"]

    def test_iterable_keeps_order(self):
        assert _normalize_chroms(("chr2", "chr1")) == ["chr2", "chr1"]

    def test_none_uses_default(self):
        assert _normalize_chroms(None, default_chroms=["a", "b"]) == [
            "a",
            "b",
        ]

    def test_none_without_default_is_empty(self):
        assert _normalize_chroms() == []

    def test_empty_string_raises(self):
        with pytest.raises(ValueError, match="empty string"):
            _normalize_chroms("")

    def test_non_iterable_raises(self):
        with pytest.raises(TypeError, match="str or an iterable"):
            _normalize_chroms(3)


class TestChromList:
    def test_str_gives_one_element_list(self):
        assert _chrom_list("chr1") == ["chr1"]

    def test_iterable_keeps_order(self):
        assert _chrom_list(iter(["chr2", "chr1"])) == ["chr2", "chr1"]

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _chrom_list([])


class TestPerChrom:
    def test_none_gives_none_per_chrom(self):
        assert _per_chrom(None, ["a", "b"], "x") == {"a": None, "b": None}

    def test_dict_fills_missing_and_ignores_extra(self):
        got = _per_chrom({"a": 1, "z": 9}, ["a", "b"], "x")
        assert got == {"a": 1, "b": None}

    def test_single_value_with_one_chrom(self):
        assert _per_chrom([0, 2], ["a"], "x") == {"a": [0, 2]}

    def test_single_value_with_several_chroms_raises(self):
        with pytest.raises(ValueError, match="x must be a dict keyed"):
            _per_chrom([0, 2], ["a", "b"], "x")
