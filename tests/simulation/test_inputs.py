# test_inputs.py

import numpy as np
import pandas as pd
import pytest
from catella.simulation._inputs import (
    _check_llink,
    _check_mcmc_settings,
    _count_mols,
    _is_valid_llink,
    _normalize_meth_prob,
    _normalize_mols,
    _parse_out_types,
    _prepare_rows,
    _resolve_settings,
    _start_positions,
)
from catella.simulation.config import SimSettings
from catella_cpp import Dump


def _make_settings(**overrides):
    defaults = {
        "nucbp": 147,
        "llink": 20,
        "elink": 1.0,
        "mu": -1.0,
        "start_temp": 1.0,
        "end_temp": 0.1,
        "cool_option": "linear",
        "nsweep": 5,
        "print_freq": 1,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


class TestIsValidLlink:
    @pytest.mark.parametrize("value", [0, 5, np.int64(3)])
    def test_accepts_non_negative_integers(self, value):
        assert _is_valid_llink(value)

    @pytest.mark.parametrize("value", [-1, 2.0, True, "3", None])
    def test_rejects_others(self, value):
        assert not _is_valid_llink(value)


class TestNormalizeMethProb:
    def test_array_for_single_chrom(self):
        arr = np.random.default_rng(0).random((3, 5))
        out = _normalize_meth_prob(arr, ["chr1"])
        assert list(out) == ["chr1"]
        np.testing.assert_array_equal(out["chr1"], arr)

    def test_array_for_multiple_chroms_raises(self):
        with pytest.raises(TypeError):
            _normalize_meth_prob(np.zeros((2, 3)), ["chr1", "chr2"])

    def test_mapping_selects_chroms_in_order(self):
        data = {"a": np.zeros((1, 2)), "b": np.ones((2, 2)), "c": None}
        out = _normalize_meth_prob(data, ["b", "a"])
        assert list(out) == ["b", "a"]
        np.testing.assert_array_equal(out["b"], np.ones((2, 2)))

    def test_mapping_missing_chrom_raises(self):
        with pytest.raises(KeyError):
            _normalize_meth_prob({"a": np.zeros((1, 2))}, ["b"])

    def test_dataframe_converted_to_array(self):
        df = pd.DataFrame(np.arange(6.0).reshape(2, 3))
        out = _normalize_meth_prob({"a": df}, ["a"])
        assert isinstance(out["a"], np.ndarray)
        np.testing.assert_array_equal(out["a"], df.to_numpy())

    def test_1d_array_made_2d(self):
        out = _normalize_meth_prob(np.zeros(4), ["a"])
        assert out["a"].shape == (1, 4)

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_meth_prob([[0.1, 0.2]], ["a"])

    def test_input_mapping_not_modified(self):
        df = pd.DataFrame(np.zeros((2, 2)))
        data = {"a": df}
        _normalize_meth_prob(data, ["a"])
        assert data["a"] is df


class TestStartPositions:
    def test_keeps_first_nbp_minus_nucbp_plus_one_columns(self):
        block = np.arange(20.0).reshape(2, 10)
        assert _start_positions(block, 4).shape == (2, 7)

    def test_one_dimensional_row(self):
        assert _start_positions(np.arange(10.0), 4).shape == (7,)

    def test_short_molecules_keep_all_columns(self):
        block = np.zeros((2, 3))
        assert _start_positions(block, 4).shape == (2, 3)


class TestPrepareRows:
    def test_all_nan_rows_have_no_data(self):
        block = np.full((3, 10), 0.5)
        block[1] = np.nan
        out, no_data = _prepare_rows(block, 4, "chr1")
        assert out is block
        np.testing.assert_array_equal(no_data, [False, True, False])

    def test_nan_only_at_start_positions_has_no_data(self):
        block = np.full((1, 10), 0.5)
        block[0, :7] = np.nan
        _, no_data = _prepare_rows(block, 4, "chr1")
        assert no_data[0]

    def test_trailing_nan_is_allowed(self):
        block = np.full((2, 10), 0.5)
        block[:, 7:] = np.nan
        _, no_data = _prepare_rows(block, 4, "chr1")
        assert not no_data.any()

    def test_partial_nan_raises(self):
        block = np.full((3, 10), 0.5)
        block[2, 3] = np.nan
        with pytest.raises(ValueError, match="chr1.*1 molecule.*first: 2"):
            _prepare_rows(block, 4, "chr1")

    def test_short_molecules_never_raise(self):
        block = np.full((2, 3), 0.5)
        block[0, 1] = np.nan
        block[1] = np.nan
        _, no_data = _prepare_rows(block, 4, "chr1")
        np.testing.assert_array_equal(no_data, [False, True])

    def test_input_not_modified(self):
        block = np.full((2, 10), 0.5)
        block[0] = np.nan
        copy = block.copy()
        _prepare_rows(block, 4, "chr1")
        np.testing.assert_array_equal(block, copy)


class TestNormalizeMols:
    @pytest.mark.parametrize("mols", [2, slice(0, 3), [0, 2]])
    def test_selection_applied_to_every_chrom(self, mols):
        out = _normalize_mols(mols, ["a", "b"])
        assert out == {"a": mols, "b": mols}

    def test_mapping_selects_chroms(self):
        out = _normalize_mols({"a": [0], "b": [1], "c": [2]}, ["a", "b"])
        assert out == {"a": [0], "b": [1]}

    def test_mapping_missing_chrom_raises(self):
        with pytest.raises(KeyError):
            _normalize_mols({"a": [0]}, ["b"])

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _normalize_mols(1.5, ["a"])


class TestCountMols:
    def test_counts_over_chroms(self):
        mols = {"a": slice(None), "b": [0, 2], "c": 1}
        nmol = {"a": 4, "b": 3, "c": 2}
        assert _count_mols(mols, nmol) == 7

    def test_out_of_range_raises(self):
        with pytest.raises(IndexError, match="'a'"):
            _count_mols({"a": [5]}, {"a": 3})

    def test_empty_selection_raises(self):
        with pytest.raises(ValueError, match="'a'"):
            _count_mols({"a": slice(2, 2)}, {"a": 3})


class TestResolveSettings:
    def test_settings_returned_as_is(self):
        settings = _make_settings()
        assert _resolve_settings(settings) is settings

    def test_loads_from_path(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            '{"nucbp": 147, "llink": 20, "elink": 1.0, "mu": -1.0, '
            '"end_temp": 0.1, "emax": 5.0}'
        )
        out = _resolve_settings(path)
        assert isinstance(out, SimSettings)
        assert out.nucbp == 147
        assert _resolve_settings(str(path)) == out

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _resolve_settings({"nucbp": 147})


class TestCheckLlink:
    def test_valid_scalar_passes(self):
        _check_llink(_make_settings(llink=20), ["a", "b"])

    @pytest.mark.parametrize("llink", [25.0, 25.5, -1, True])
    def test_invalid_scalar_raises(self, llink):
        with pytest.raises(ValueError, match="non-negative integer"):
            _check_llink(_make_settings(llink=llink), ["a"])

    def test_mapping_covering_chroms_passes(self):
        _check_llink(_make_settings(llink={"a": 10, "b": 0}), ["a", "b"])

    def test_missing_chrom_raises(self):
        with pytest.raises(KeyError, match="'b'"):
            _check_llink(_make_settings(llink={"a": 10}), ["a", "b"])

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError, match="'a'"):
            _check_llink(_make_settings(llink={"a": -1}), ["a"])


class TestCheckMCMCSettings:
    def test_valid_settings_pass(self):
        _check_mcmc_settings(_make_settings())

    @pytest.mark.parametrize(
        "name", ["start_temp", "cool_option", "nsweep", "print_freq"]
    )
    def test_missing_field_raises(self, name):
        with pytest.raises(ValueError, match=name):
            _check_mcmc_settings(_make_settings(**{name: None}))

    def test_end_temp_above_start_temp_raises(self):
        with pytest.raises(ValueError):
            _check_mcmc_settings(
                _make_settings(start_temp=0.5, end_temp=1.0)
            )

    def test_unknown_cool_option_raises(self):
        with pytest.raises(KeyError):
            _check_mcmc_settings(_make_settings(cool_option="cubic"))


class TestParseOutTypes:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("energy", Dump.OutputType.Energy),
            ("position", Dump.OutputType.Position),
            ("temp", Dump.OutputType.Temp),
            ("all", Dump.OutputType.All),
        ],
    )
    def test_single_name(self, name, expected):
        assert _parse_out_types(name) == expected

    def test_names_combined(self):
        out = _parse_out_types(["energy", "temp"])
        expected = int(Dump.OutputType.Energy) | int(Dump.OutputType.Temp)
        assert int(out) == expected

    def test_accepts_any_iterable(self):
        out = _parse_out_types(n for n in ("position",))
        assert out == Dump.OutputType.Position

    @pytest.mark.parametrize("value", ["bogus", ["energy", "bogus"]])
    def test_unknown_name_raises(self, value):
        with pytest.raises(ValueError, match="bogus"):
            _parse_out_types(value)

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _parse_out_types([])

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            _parse_out_types(3)
