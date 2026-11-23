# test_analysis_io.py

import h5py
import numpy as np
import pandas as pd
from catella._analysis_io import _read_entries, _replace_group, _write_entries
from catella.h5_array import H5Array


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
