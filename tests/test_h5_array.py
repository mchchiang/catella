# test_h5_array.py

from pathlib import Path

import numpy as np
import pytest

from nucmc.h5_array import H5Array


def test_write_batch_and_read_back():
    arr = H5Array.create((10, 4))
    values = np.arange(40, dtype=np.float64).reshape(10, 4)
    arr.write_batch(0, 5, values[0:5])
    arr.write_batch(5, 10, values[5:10])
    np.testing.assert_array_equal(arr.to_numpy(), values)
    assert arr.shape == (10, 4)
    assert arr.dtype == np.float64


def test_getitem_slicing():
    arr = H5Array.create((6, 3))
    values = np.arange(18, dtype=np.float64).reshape(6, 3)
    arr.write_batch(0, 6, values)
    np.testing.assert_array_equal(arr[2:4, :], values[2:4, :])


def test_scratch_file_cleanup_on_close():
    arr = H5Array.create((3, 3))
    path = Path(arr.path)
    assert path.exists()
    arr.close()
    assert not path.exists()


def test_permanent_path_survives_close(tmp_path):
    path = tmp_path / "permanent.h5"
    arr = H5Array.create((3, 3), path=path)
    arr.write_batch(0, 3, np.ones((3, 3)))
    arr.close()
    assert path.exists()


def test_save_to_and_load_from_round_trip(tmp_path):
    import h5py

    src = H5Array.create((4, 2))
    values = np.arange(8, dtype=np.float64).reshape(4, 2)
    src.write_batch(0, 4, values)

    dest_path = tmp_path / "dest.h5"
    with h5py.File(dest_path, "w") as f:
        group = f.create_group("analysis")
        src.save_to(group, "test_smoothed")

    loaded = H5Array.load_from(dest_path, "analysis/test_smoothed")
    np.testing.assert_array_equal(loaded.to_numpy(), values)


def test_create_with_dir_uses_given_directory(tmp_path):
    arr = H5Array.create((2, 2), dir=tmp_path)
    assert Path(arr.path).parent == tmp_path
    arr.close()
