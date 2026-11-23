# test_analysis_io.py

import h5py
import numpy as np
import pandas as pd
import pytest
from catella._analysis_io import (
    _analysis_group,
    _read_entries,
    _replace_group,
    _save_analysis,
    _write_entries,
)
from catella.h5_array import H5Array


def _arr(tmp_path, values, **labels):
    arr = H5Array.create(values.shape, dir=tmp_path, **labels)
    arr.write_batch(0, values.shape[0], values)
    return arr


def _load(path):
    with h5py.File(path, "r") as f:
        g = _analysis_group(f, "analysis")
        analysis = {chrom: _read_entries(g[chrom], path) for chrom in g}
        glob = _read_entries(_analysis_group(f, "global_analysis"), path)
    return analysis, glob


A = np.arange(6.0).reshape(2, 3)
B = np.ones((2, 3))


class TestAnalysisGroup:
    def test_returns_group(self, tmp_path):
        with h5py.File(tmp_path / "f.h5", "w") as f:
            f.create_group("analysis")
            f.create_group("analysis.new")
            assert _analysis_group(f, "analysis").name == "/analysis"

    def test_falls_back_to_staged(self, tmp_path):
        with h5py.File(tmp_path / "f.h5", "w") as f:
            f.create_group("analysis.new")
            assert _analysis_group(f, "analysis").name == "/analysis.new"


class TestSaveAnalysis:
    @pytest.fixture
    def path(self, tmp_path):
        path = tmp_path / "exp.h5"
        _save_analysis(
            path,
            {"c1": {"a": _arr(tmp_path, A), "b": _arr(tmp_path, B)}},
            {"df": pd.DataFrame({"v": [1.0, 2.0]})},
        )
        return path

    def test_new_file_round_trip(self, path):
        analysis, glob = _load(path)
        assert set(analysis["c1"]) == {"a", "b"}
        np.testing.assert_array_equal(analysis["c1"]["a"].to_numpy(), A)
        pd.testing.assert_frame_equal(
            glob["df"], pd.DataFrame({"v": [1.0, 2.0]})
        )

    def test_resave_does_not_copy_stored_arrays(self, tmp_path):
        path = tmp_path / "big.h5"
        rng = np.random.default_rng(0)
        big = rng.random((2000, 500))
        _save_analysis(path, {"c1": {"x": _arr(tmp_path, big)}}, {})
        size = path.stat().st_size
        analysis, glob = _load(path)
        _save_analysis(path, analysis, glob)
        assert path.stat().st_size < 1.5 * size
        analysis, _ = _load(path)
        np.testing.assert_array_equal(analysis["c1"]["x"].to_numpy(), big)

    def test_swapped_keys(self, path):
        analysis, glob = _load(path)
        c1 = analysis["c1"]
        analysis["c1"] = {"a": c1["b"], "b": c1["a"]}
        _save_analysis(path, analysis, glob)
        analysis, _ = _load(path)
        np.testing.assert_array_equal(analysis["c1"]["a"].to_numpy(), B)
        np.testing.assert_array_equal(analysis["c1"]["b"].to_numpy(), A)

    def test_same_entry_under_two_keys(self, path):
        analysis, glob = _load(path)
        analysis["c1"]["a2"] = analysis["c1"]["a"]
        _save_analysis(path, analysis, glob)
        analysis, _ = _load(path)
        np.testing.assert_array_equal(analysis["c1"]["a2"].to_numpy(), A)

    def test_dropped_entry_removed(self, path):
        analysis, glob = _load(path)
        del analysis["c1"]["b"]
        _save_analysis(path, analysis, glob)
        analysis, _ = _load(path)
        assert set(analysis["c1"]) == {"a"}

    def test_entry_moved_to_global(self, path):
        analysis, glob = _load(path)
        glob["g"] = analysis["c1"].pop("a")
        _save_analysis(path, analysis, glob)
        analysis, glob = _load(path)
        assert set(analysis["c1"]) == {"b"}
        np.testing.assert_array_equal(glob["g"].to_numpy(), A)

    def test_labels_kept_when_linked(self, tmp_path):
        path = tmp_path / "exp.h5"
        arr = _arr(tmp_path, A, index=["r0", "r1"], columns=[5, 6, 7])
        _save_analysis(path, {"c1": {"a": arr}}, {})
        analysis, glob = _load(path)
        analysis["c1"]["z"] = analysis["c1"].pop("a")
        _save_analysis(path, analysis, glob)
        analysis, _ = _load(path)
        assert set(analysis["c1"]) == {"z"}
        assert list(analysis["c1"]["z"].index) == ["r0", "r1"]
        assert list(analysis["c1"]["z"].columns) == [5, 6, 7]

    def test_held_reference_usable_after_save(self, path):
        analysis, glob = _load(path)
        ref = analysis["c1"]["a"]
        analysis["c1"] = {"renamed": ref}
        _save_analysis(path, analysis, glob)
        assert ref.dataset_path == "/analysis/c1/renamed"
        np.testing.assert_array_equal(ref.to_numpy(), A)

    def test_write_extra_called_with_open_file(self, path):
        def write_extra(f):
            f.attrs["extra"] = 1

        analysis, glob = _load(path)
        _save_analysis(path, analysis, glob, write_extra=write_extra)
        with h5py.File(path, "r") as f:
            assert f.attrs["extra"] == 1

    def test_failure_keeps_file_and_references(self, path):
        def write_extra(f):
            raise RuntimeError("boom")

        analysis, glob = _load(path)
        ref = analysis["c1"]["a"]
        analysis["c1"] = {"renamed": ref}
        with pytest.raises(RuntimeError, match="boom"):
            _save_analysis(path, analysis, glob, write_extra=write_extra)
        np.testing.assert_array_equal(ref.to_numpy(), A)
        loaded, _ = _load(path)
        assert set(loaded["c1"]) == {"a", "b"}

    def test_recovers_from_staged_only_file(self, path):
        with h5py.File(path, "a") as f:
            f.move("analysis", "analysis.new")
        analysis, glob = _load(path)
        _save_analysis(path, analysis, glob)
        with h5py.File(path, "r") as f:
            assert "analysis" in f
            assert "analysis.new" not in f
        analysis, _ = _load(path)
        np.testing.assert_array_equal(analysis["c1"]["a"].to_numpy(), A)

    def test_clears_stale_staged_group(self, path):
        with h5py.File(path, "a") as f:
            f.create_group("analysis.new/junk")
        analysis, glob = _load(path)
        _save_analysis(path, analysis, glob)
        with h5py.File(path, "r") as f:
            assert "analysis.new" not in f
        analysis, _ = _load(path)
        assert set(analysis["c1"]) == {"a", "b"}

    def test_file_open_elsewhere_raises(self, path):
        analysis, glob = _load(path)
        other = H5Array.load_from(path, "analysis/c1/b")
        with pytest.raises(OSError, match="open elsewhere"):
            _save_analysis(path, analysis, glob)
        other.close()
        np.testing.assert_array_equal(analysis["c1"]["a"].to_numpy(), A)


class TestReplaceGroup:
    def test_creates_empty_group(self, tmp_path):
        with h5py.File(tmp_path / "f.h5", "w") as f:
            g = f.create_group("a")
            g.create_dataset("x", data=[1])
            new = _replace_group(f, "a")
            assert len(new) == 0
            assert _replace_group(f, "b").name == "/b"


class TestEntries:
    def test_round_trip(self, tmp_path):
        path = tmp_path / "d.h5"
        arr = H5Array.create((2, 3), dtype=np.float64, dir=tmp_path)
        arr.write_batch(0, 2, np.arange(6.0).reshape(2, 3))
        df = pd.DataFrame({"v": [1.0, 2.0]})
        with h5py.File(path, "w") as f:
            parent = f.create_group("analysis")
            _write_entries(parent.create_group("c1"), {"arr": arr, "df": df})
        with h5py.File(path, "r") as f:
            entries = _read_entries(f["analysis/c1"], path)
        assert set(entries) == {"arr", "df"}
        assert isinstance(entries["arr"], H5Array)
        np.testing.assert_array_equal(
            entries["arr"].to_numpy(), np.arange(6.0).reshape(2, 3)
        )
        pd.testing.assert_frame_equal(entries["df"], df)

    def test_round_trip_labeled_array_is_one_entry(self, tmp_path):
        path = tmp_path / "d.h5"
        arr = H5Array.create(
            (2, 3), dir=tmp_path, index=["a", "b"], columns=[10, 20, 30]
        )
        with h5py.File(path, "w") as f:
            _write_entries(f.create_group("c1"), {"arr": arr})
        with h5py.File(path, "r") as f:
            entries = _read_entries(f["c1"], path)
        assert set(entries) == {"arr"}
        assert list(entries["arr"].index) == ["a", "b"]
        assert list(entries["arr"].columns) == [10, 20, 30]
