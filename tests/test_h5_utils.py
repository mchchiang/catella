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
