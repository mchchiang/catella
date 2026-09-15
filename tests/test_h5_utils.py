# test_h5_utils.py

import os
import shutil
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
from catella import h5_utils


class TestSaveLoadDf:
    def test_mixed_numeric_and_string_round_trip(self, tmp_path):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        np.testing.assert_array_equal(loaded["a"].to_numpy(), df["a"])
        assert loaded["b"].tolist() == df["b"].tolist()

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

    def test_bool_column_round_trips_as_string_not_bool(self, tmp_path):
        # bool isn't a numpy 'number' dtype, so it's stored in the
        # string block and comes back as literal "True"/"False" text,
        # not a bool -- documenting this so callers know to store
        # int (0/1) instead if they need bool semantics preserved.
        df = pd.DataFrame({"keep": [True, False, True]})
        path = tmp_path / "df.h5"
        with h5py.File(path, "w") as f:
            h5_utils.save_df("df", df, f)
        with h5py.File(path, "r") as f:
            loaded = h5_utils.load_df("df", f)
        assert loaded["keep"].tolist() == ["True", "False", "True"]


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
