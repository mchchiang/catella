# h5_array.py

import os
import tempfile
import weakref
from pathlib import Path
import numpy as np
import h5py

_DATASET_NAME = "data"


class H5Array:
    """
    Disk-backed 2D numeric array stored in an HDF5 file.

    Provide bounded-memory storage for large matrices (e.g. smoothed
    methylation signals) by keeping the data on disk and writing or
    reading it in row batches instead of holding it fully in memory.
    Backed by a scratch temporary file by default (removed once this
    object is closed or garbage-collected), or by a user-specified
    permanent path.
    """

    def __init__(self, path, file, dataset, owns_file):
        self._path = path
        self._file = file
        self._dataset = dataset
        self._owns_file = owns_file
        self._finalizer = weakref.finalize(
            self, H5Array._cleanup, file, path, owns_file)

    @staticmethod
    def _cleanup(file, path, owns_file):
        try:
            file.close()
        except Exception:
            pass
        if owns_file:
            Path(path).unlink(missing_ok=True)

    @classmethod
    def create(cls, shape, *, dtype=np.float64, path=None, dir=None):
        """
        Create a new disk-backed array.

        Parameters
        ----------
        shape : tuple of int
            Shape of the array, (nrow, ncol).
        dtype : data-type, default np.float64
            Numeric dtype of the stored data.
        path : str or pathlib.Path, optional
            Location of the backing HDF5 file. If None, a scratch file is
            created and removed once this object is closed or
            garbage-collected. If given, the file is kept and `dir` is
            ignored.
        dir : str or pathlib.Path, optional
            Directory in which to create the scratch file when `path` is
            None. If None, the system default temporary directory is
            used.

        Returns
        -------
        H5Array
            The newly created disk-backed array.
        """
        owns_file = path is None
        if path is None:
            fd, path = tempfile.mkstemp(suffix=".h5", dir=dir)
            os.close(fd)
        else:
            path = str(path)
        file = h5py.File(path, "w")
        dataset = file.create_dataset(_DATASET_NAME, shape=shape,
                                      dtype=dtype)
        return cls(path=path, file=file, dataset=dataset,
                   owns_file=owns_file)

    @classmethod
    def load_from(cls, path, dataset_path):
        """
        Open an existing dataset within an HDF5 file as a read-only
        H5Array.

        Parameters
        ----------
        path : str or pathlib.Path
            Path to the HDF5 file.
        dataset_path : str
            Path of the dataset within the file (e.g. as produced by a
            previous call to `save_to`).

        Returns
        -------
        H5Array
            A read-only view onto the on-disk data.
        """
        file = h5py.File(str(path), "r")
        dataset = file[dataset_path]
        return cls(path=str(path), file=file, dataset=dataset,
                   owns_file=False)

    @property
    def path(self) -> str:
        """str: Path to the backing HDF5 file."""
        return self._path

    @property
    def shape(self):
        """tuple of int: Shape of the array, (nrow, ncol)."""
        return self._dataset.shape

    @property
    def dtype(self):
        """np.dtype: Numeric dtype of the stored data."""
        return self._dataset.dtype

    def write_batch(self, start, stop, values):
        """
        Write a contiguous block of rows.

        Parameters
        ----------
        start : int
            First row index (inclusive).
        stop : int
            Last row index (exclusive).
        values : np.ndarray
            Array of shape (stop - start, ncol) to write.
        """
        self._dataset[start:stop, :] = values

    def __getitem__(self, key):
        return self._dataset[key]

    def to_numpy(self) -> np.ndarray:
        """
        Read the full array into memory.

        Returns
        -------
        np.ndarray
            The complete array.
        """
        return self._dataset[:]

    def save_to(self, group, name):
        """
        Stream-copy the underlying dataset into another HDF5 group.

        The copy happens at the HDF5 layer without materializing the
        full array in memory.

        Parameters
        ----------
        group : h5py.Group
            Destination group.
        name : str
            Name of the dataset to create within `group`.
        """
        group.copy(self._dataset, name)

    def close(self):
        """Close the backing file, deleting it if it is a scratch file."""
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
