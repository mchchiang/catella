# test_h5_utils.py

import os
import re
import shutil
import subprocess
import sys
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
from catella import h5_utils


class TestSaveLoadDf:
    def test_mixed_numeric_and_string_round_trip(self, tmp_path):
        df = pd.DataFrame(
            {"a": [1.0, 2.0, 3.0], "i": [4, 5, 6], "b": ["x", "y", "z"]}
        )
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        np.testing.assert_array_equal(loaded["a"].to_numpy(), df["a"])
        assert loaded["i"].dtype == df["i"].dtype
        assert loaded["i"].tolist() == [4, 5, 6]
        assert loaded["b"].tolist() == df["b"].tolist()
        assert list(loaded.columns) == ["a", "i", "b"]

    def test_all_string_round_trip(self, tmp_path):
        # No numeric columns at all -- save_df skips creating
        # 'num_values'; load_df must not assume it exists.
        df = pd.DataFrame({"b": ["x", "y", "z"]})
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert loaded["b"].tolist() == df["b"].tolist()
        assert len(loaded) == 3

    def test_bool_column_round_trips_as_bool(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "keep": [True, False, True]})
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert loaded["keep"].dtype == bool
        assert loaded["keep"].tolist() == [True, False, True]
        assert list(loaded.columns) == ["a", "keep"]

    def test_strings_stored_at_fixed_width(self, tmp_path):
        df = pd.DataFrame({"b": ["x", "yyy"]})
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            assert f["df/str_b"].dtype == np.dtype("S3")

    def test_non_ascii_and_empty_strings_round_trip(self, tmp_path):
        df = pd.DataFrame({"b": ["naïve", "", "β"], "c": ["", "", ""]})
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert loaded["b"].tolist() == ["naïve", "", "β"]
        assert loaded["c"].tolist() == ["", "", ""]

    def test_old_vlen_float_layout_still_loads(self, tmp_path):
        dt = h5py.string_dtype(encoding="utf-8")
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            g = f.create_group("df")
            g.create_dataset("_column_order", data=["pos", "s"], dtype=dt)
            g.attrs["_column_index_dtype"] = "object"
            g.create_dataset("num_values", data=np.array([[1.0], [2.0]]))
            g.create_dataset("num_names", data=["pos"], dtype=dt)
            g.create_dataset("str_s", data=["+", "-"], dtype=dt)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert loaded["pos"].tolist() == [1.0, 2.0]
        assert loaded["s"].tolist() == ["+", "-"]


class TestAppendableDF:
    SPEC = {
        "i": "integer",
        "x": "numeric",
        "s": "S4",
    }

    def _write(self, path, chunks, spec=None):
        spec = self.SPEC if spec is None else spec
        with h5py.File(path, "w") as f:
            writer = h5_utils.AppendableDF(
                f, "df", columns=list(spec), dtypes=spec
            )
            for chunk in chunks:
                writer.append(chunk)
        return writer

    def test_integer_columns_round_trip_over_appends(self, tmp_path):
        path = tmp_path / "df.h5"
        writer = self._write(
            path,
            [
                pd.DataFrame({"i": [1, 2], "x": [0.5, 1.5], "s": ["a", "b"]}),
                pd.DataFrame({"i": [3], "x": [2.5], "s": ["c"]}),
            ],
        )
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert writer.nrows == 3
        assert list(loaded.columns) == ["i", "x", "s"]
        assert np.issubdtype(loaded["i"].dtype, np.integer)
        assert loaded["i"].tolist() == [1, 2, 3]
        assert loaded["x"].tolist() == [0.5, 1.5, 2.5]
        assert loaded["s"].tolist() == ["a", "b", "c"]

    def test_only_integer_columns(self, tmp_path):
        path = tmp_path / "df.h5"
        self._write(
            path,
            [pd.DataFrame({"i": [7, 8]})],
            spec={"i": "integer"},
        )
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert loaded["i"].tolist() == [7, 8]

    def test_zero_rows_still_loads(self, tmp_path):
        path = tmp_path / "df.h5"
        writer = self._write(path, [])
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert writer.nrows == 0
        assert len(loaded) == 0
        assert list(loaded.columns) == ["i", "x", "s"]

    def test_strings_stored_at_fixed_width(self, tmp_path):
        path = tmp_path / "df.h5"
        self._write(
            path,
            [
                pd.DataFrame({"i": [1], "x": [0.5], "s": ["a"]}),
                pd.DataFrame({"i": [2], "x": [0.5], "s": ["abcd"]}),
            ],
        )
        with h5py.File(path, "r") as f:
            assert f["df/str_s"].dtype == np.dtype("S4")
            loaded = h5_utils.load_df("df", f)
        assert loaded["s"].tolist() == ["a", "abcd"]

    def test_value_over_width_raises(self, tmp_path):
        with pytest.raises(ValueError, match="fixed width 4"):
            self._write(
                tmp_path / "df.h5",
                [pd.DataFrame({"i": [1], "x": [0.5], "s": ["abcde"]})],
            )

    @pytest.mark.parametrize("kind", ["string", "S0", "S", "float", "U4"])
    def test_unknown_kind_raises(self, tmp_path, kind):
        with h5py.File(tmp_path / "df.h5", "w") as f:
            with pytest.raises(ValueError, match="unknown kind"):
                h5_utils.AppendableDF(
                    f, "df", columns=["s"], dtypes={"s": kind}
                )
            assert "df" not in f


class TestFreshTmpDir:
    def test_registers_atexit_fallback_cleanup(self):
        # Guards against a leaked dir if the caller is interrupted
        # before registering its own cleanup (e.g. weakref.finalize).
        with patch("catella.h5_utils.atexit.register") as mock_register:
            path = h5_utils.fresh_tmp_dir()
        try:
            assert os.path.isdir(path)
            mock_register.assert_called_once_with(
                shutil.rmtree, path, ignore_errors=True
            )
        finally:
            shutil.rmtree(path, ignore_errors=True)

    def test_name_encodes_host_and_pid(self, tmp_path):
        path = h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        name = os.path.basename(path)
        assert os.path.dirname(path) == str(tmp_path)
        assert re.fullmatch(
            r"catella_\d{8}_\d{6}_[A-Za-z0-9-]+_\d+_[0-9a-f]{8}", name
        )
        assert f"_{h5_utils._host_tag()}_{os.getpid()}_" in name


def _dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    return proc.pid


def _make(base, host, pid):
    path = base / f"catella_20260101_000000_{host}_{pid}_0123abcd"
    path.mkdir()
    return path


class TestSweepStale:
    @pytest.fixture(autouse=True)
    def _fresh_swept(self, monkeypatch):
        monkeypatch.setattr(h5_utils, "_SWEPT", set())

    def test_removes_dead_pid_same_host(self, tmp_path):
        stale = _make(tmp_path, h5_utils._host_tag(), _dead_pid())
        h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        assert not stale.exists()

    def test_keeps_live_pid(self, tmp_path):
        live = _make(tmp_path, h5_utils._host_tag(), os.getppid())
        h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        assert live.exists()

    def test_keeps_other_host(self, tmp_path):
        other = _make(tmp_path, "some-other-host", _dead_pid())
        h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        assert other.exists()

    def test_keeps_legacy_and_unrelated_dirs(self, tmp_path):
        legacy = tmp_path / "catella_20260916_201042_a4f8f4ed"
        unrelated = tmp_path / "not_catella"
        legacy.mkdir()
        unrelated.mkdir()
        h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        assert legacy.exists()
        assert unrelated.exists()

    def test_sweeps_once_per_base(self, tmp_path):
        h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        stale = _make(tmp_path, h5_utils._host_tag(), _dead_pid())
        h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        assert stale.exists()

    def test_scandir_error_is_swallowed(self, tmp_path):
        with patch("catella.h5_utils.os.scandir", side_effect=OSError):
            path = h5_utils.fresh_tmp_dir(base_dir=tmp_path)
        assert os.path.isdir(path)
