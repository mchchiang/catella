# test_regions.py

from pathlib import Path

import pandas as pd
import pytest
from catella._regions import (
    Region,
    _check_region_lengths,
    _check_region_names,
    _make_region,
    _normalize_regions,
    _parse_regions,
    _per_region,
    _read_bed,
    _region_list,
    _RegionMap,
)


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


class TestCheckRegionNames:
    def test_valid_names_pass(self):
        _check_region_names(["SOX2", "chr3_100_200", "601-PciI-loop2"])

    @pytest.mark.parametrize("name", ["", ".", ".."])
    def test_reserved_names_raise(self, name):
        with pytest.raises(ValueError, match="invalid region name"):
            _check_region_names([name])

    @pytest.mark.parametrize("name", [" a", "a ", "a\t"])
    def test_surrounding_whitespace_raises(self, name):
        with pytest.raises(ValueError, match="whitespace"):
            _check_region_names([name])

    @pytest.mark.parametrize("char", list('/\\:*?"<>|'))
    def test_unsafe_character_raises(self, char):
        with pytest.raises(ValueError, match="must not contain"):
            _check_region_names([f"a{char}b"])

    def test_long_name_raises(self):
        with pytest.raises(ValueError, match="longer than 100"):
            _check_region_names(["a" * 101])

    def test_name_of_100_characters_passes(self):
        _check_region_names(["a" * 100])

    def test_case_insensitive_duplicate_raises(self):
        with pytest.raises(ValueError, match="unique ignoring case"):
            _check_region_names(["Sox2", "SOX2"])


class TestCheckRegionLengths:
    def test_short_regions_pass_silently(self, recwarn):
        _check_region_lengths({"a": 5000, "b": 100000})
        assert not recwarn

    def test_over_100_kb_warns(self):
        with pytest.warns(UserWarning, match="1 region"):
            _check_region_lengths({"a": 5000, "b": 100001})

    def test_over_1_mb_raises(self):
        with pytest.raises(ValueError, match="'chr1' is 248956422 bp"):
            _check_region_lengths({"chr1": 248956422})

    def test_exactly_1_mb_warns_only(self):
        with pytest.warns(UserWarning):
            _check_region_lengths({"a": 1000000})


class TestMakeRegion:
    def test_defaults(self):
        assert _make_region("chr1", 10, 20) == Region(
            "chr1", 10, 20, "chr1_10_20", "."
        )

    def test_name_and_strand(self):
        assert _make_region("chr1", 0, 5, "g", "-") == Region(
            "chr1", 0, 5, "g", "-"
        )

    def test_dot_name_gives_default(self):
        assert _make_region("c", 1, 2, ".").name == "c_1_2"

    @pytest.mark.parametrize("start, end", [(-1, 5), (5, 5), (6, 5)])
    def test_bad_interval_raises(self, start, end):
        with pytest.raises(ValueError, match="0 <= start < end"):
            _make_region("c", start, end)

    @pytest.mark.parametrize("value", [1.5, "3", True])
    def test_non_integer_coordinate_raises(self, value):
        with pytest.raises(TypeError, match="must be an integer"):
            _make_region("c", value, 10)

    def test_bad_strand_raises(self):
        with pytest.raises(ValueError, match="strand must be one of"):
            _make_region("c", 0, 5, "g", "x")

    def test_too_many_fields_raises(self):
        with pytest.raises(ValueError, match="got 6 fields"):
            _make_region("c", 0, 5, "g", "+", "extra")


def _bed(tmp_path, text):
    path = tmp_path / "regions.bed"
    path.write_text(text)
    return path


class TestReadBed:
    def test_bed3_gets_default_names(self, tmp_path):
        path = _bed(tmp_path, "chr1\t10\t20\nchr2\t0\t5\n")
        assert _read_bed(path) == [
            Region("chr1", 10, 20, "chr1_10_20", "."),
            Region("chr2", 0, 5, "chr2_0_5", "."),
        ]

    def test_bed6_name_and_strand(self, tmp_path):
        path = _bed(tmp_path, "chr3\t100\t200\tSOX2\t0\t-\n")
        assert _read_bed(path) == [Region("chr3", 100, 200, "SOX2", "-")]

    def test_header_and_blank_lines_skipped(self, tmp_path):
        text = (
            "track name=x\nbrowser position chr1\n# note\n\n"
            "chr1\t1\t2\tA\n"
        )
        assert [r.name for r in _read_bed(_bed(tmp_path, text))] == ["A"]

    def test_numeric_names_and_chroms_stay_text(self, tmp_path):
        parsed = _read_bed(_bed(tmp_path, "1\t0\t5\t007\n"))
        assert parsed == [Region("1", 0, 5, "007", ".")]

    def test_too_few_fields_raises(self, tmp_path):
        with pytest.raises(ValueError, match="line 1: expected at least 3"):
            _read_bed(_bed(tmp_path, "chr1\t10\n"))

    def test_non_integer_coordinate_raises(self, tmp_path):
        with pytest.raises(ValueError, match="must be integers"):
            _read_bed(_bed(tmp_path, "chr1\tx\t20\n"))

    def test_empty_file_raises(self, tmp_path):
        with pytest.raises(ValueError, match="holds no regions"):
            _read_bed(_bed(tmp_path, "# only a comment\n"))

    def test_duplicate_names_hint_at_column_4(self, tmp_path):
        text = "chr1\t0\t5\tGENE\nchr1\t9\t20\tgene\n"
        with pytest.raises(ValueError, match="column 4"):
            _read_bed(_bed(tmp_path, text))


class TestParseRegions:
    def test_none_is_all(self):
        assert _parse_regions(None) == ("all", None)

    def test_single_name(self):
        assert _parse_regions("601") == ("names", ["601"])

    def test_list_of_names(self):
        assert _parse_regions(["601", "R1"]) == ("names", ["601", "R1"])

    def test_tuples_are_regions(self):
        kind, parsed = _parse_regions([("chr1", 0, 5, "a"), ("chr1", 3, 9)])
        assert kind == "regions"
        assert [r.name for r in parsed] == ["a", "chr1_3_9"]

    @pytest.mark.parametrize("as_path", [False, True])
    def test_existing_file_is_bed(self, tmp_path, as_path):
        path = _bed(tmp_path, "chr1\t0\t5\tA\n")
        kind, parsed = _parse_regions(path if as_path else str(path))
        assert kind == "regions"
        assert parsed == [Region("chr1", 0, 5, "A", ".")]

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(ValueError, match="not found"):
            _parse_regions(Path(tmp_path / "missing.bed"))

    def test_empty_iterable_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            _parse_regions([])

    def test_mixed_iterable_raises(self):
        with pytest.raises(TypeError, match="not a mix"):
            _parse_regions(["601", ("chr1", 0, 5)])

    def test_unsupported_type_raises(self):
        with pytest.raises(TypeError, match="regions must be None"):
            _parse_regions(3)

    def test_unsafe_region_name_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            _parse_regions([("chr1", 0, 5, "a:b")])


def _rows(upos):
    return pd.DataFrame(
        {
            "chrom": "chr1",
            "mol_id": [f"m{i}" for i in range(len(upos))],
            "upos": upos,
        }
    )


class TestRegionMap:
    def _map(self):
        return _RegionMap(
            [
                Region("chr1", 10, 20, "B", "."),
                Region("chr1", 0, 12, "A", "+"),
                Region("chr2", 0, 5, "C", "."),
            ]
        )

    def test_chroms(self):
        assert self._map().chroms == ("chr1", "chr2")

    def test_rows_split_and_shifted(self):
        out = self._map().split(_rows([11, 5, 15, 30, 0]), "chr1")
        assert out["chrom"].tolist() == ["A", "A", "A", "B", "B"]
        assert out["mol_id"].tolist() == ["m0", "m1", "m4", "m0", "m2"]
        assert out["upos"].tolist() == [11, 5, 0, 1, 5]

    def test_half_open_bounds(self):
        out = self._map().split(_rows([9, 10, 19, 20]), "chr1")
        b_rows = out[out["chrom"] == "B"]
        assert b_rows["upos"].tolist() == [0, 9]

    def test_file_order_kept_within_region(self):
        out = self._map().split(_rows([4, 1, 3]), "chr1")
        assert out["mol_id"].tolist() == ["m0", "m1", "m2"]

    def test_chrom_without_regions_gives_no_rows(self):
        assert self._map().split(_rows([1, 2]), "chr9").empty

    def test_input_not_modified(self):
        rows = _rows([11, 15])
        self._map().split(rows, "chr1")
        assert rows["chrom"].tolist() == ["chr1", "chr1"]
        assert rows["upos"].tolist() == [11, 15]
