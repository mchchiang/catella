# h5_array.py

import os
import tempfile
import weakref
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
from . import h5_utils

_DATASET_NAME = "data"

_ROW_REDUCERS = {
    "mean": lambda b: np.nanmean(b, axis=1),
    "sum": lambda b: np.nansum(b, axis=1),
    "min": lambda b: np.nanmin(b, axis=1),
    "max": lambda b: np.nanmax(b, axis=1),
}


def _acc_mean(acc, batch):
    s, c = acc if acc is not None else (0.0, 0.0)
    return (s + np.nansum(batch, axis=0),
            c + (~np.isnan(batch)).sum(axis=0))


def _acc_sum(acc, batch):
    s = acc if acc is not None else 0.0
    return s + np.nansum(batch, axis=0)


def _acc_min(acc, batch):
    m = np.nanmin(batch, axis=0)
    return m if acc is None else np.fmin(acc, m)


def _acc_max(acc, batch):
    m = np.nanmax(batch, axis=0)
    return m if acc is None else np.fmax(acc, m)


_COL_ACCUMULATORS = {"mean": _acc_mean, "sum": _acc_sum,
                     "min": _acc_min, "max": _acc_max}

_DOWNSAMPLE_HOW = ("mean", "sum", "min", "max", "stride")


def _finalize_mean(acc, ncol):
    if acc is None:
        return np.full(ncol, np.nan)
    s, c = acc
    with np.errstate(invalid="ignore"):
        return s / c


def _finalize_sum(acc, ncol):
    return acc if acc is not None else np.zeros(ncol)


def _finalize_minmax(acc, ncol):
    return acc if acc is not None else np.full(ncol, np.nan)


_COL_FINALIZERS = {"mean": _finalize_mean, "sum": _finalize_sum,
                   "min": _finalize_minmax, "max": _finalize_minmax}


class _ILocIndexer:
    """Positional 2D indexing proxy, mirroring pandas' `.iloc`."""

    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, key):
        return self._dataset[key]


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
    def create(cls, shape, *, dtype=np.float64, path=None, dir=None,
              index=None, columns=None):
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
            None. If None, a fresh `nucmc_<timestamp>_<hex>` subfolder
            is created under the system default temporary directory and
            used instead.
        index : array-like, optional
            Row labels, length `shape[0]`. If None, `.index` falls back
            to a `pd.RangeIndex`.
        columns : array-like, optional
            Column labels, length `shape[1]`. If None, `.columns` falls
            back to a `pd.RangeIndex`.

        Returns
        -------
        H5Array
            The newly created disk-backed array.
        """
        owns_file = path is None
        if path is None:
            if dir is None:
                dir = h5_utils.fresh_tmp_dir()
            fd, path = tempfile.mkstemp(suffix=".h5", dir=dir)
            os.close(fd)
        else:
            path = str(path)
        file = h5py.File(path, "w")
        dataset = file.create_dataset(_DATASET_NAME, shape=shape,
                                      dtype=dtype)
        if index is not None:
            H5Array._write_label(file, f"{_DATASET_NAME}__index", index)
        if columns is not None:
            H5Array._write_label(file, f"{_DATASET_NAME}__columns", columns)
        return cls(path=path, file=file, dataset=dataset,
                   owns_file=owns_file)

    @staticmethod
    def _write_label(parent_group, key, values):
        arr = np.asarray(values)
        if arr.dtype.kind in ("U", "O", "S"):
            dt = h5py.string_dtype(encoding="utf-8")
            parent_group.create_dataset(key, data=[str(v) for v in arr],
                                        dtype=dt)
        else:
            parent_group.create_dataset(key, data=arr)

    @staticmethod
    def _read_label(dataset):
        if h5py.check_string_dtype(dataset.dtype) is not None:
            return dataset.asstr()[:]
        return dataset[:]

    def _label_dataset(self, suffix):
        parent = self._dataset.parent
        base_name = self._dataset.name.rsplit("/", 1)[-1]
        key = f"{base_name}{suffix}"
        return parent[key] if key in parent else None

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

    @property
    def index(self) -> pd.Index:
        """pd.Index: Row labels, or a RangeIndex if none were given."""
        ds = self._label_dataset("__index")
        if ds is None:
            return pd.RangeIndex(self.shape[0])
        return pd.Index(H5Array._read_label(ds))

    @property
    def columns(self) -> pd.Index:
        """pd.Index: Column labels, or a RangeIndex if none were given."""
        ds = self._label_dataset("__columns")
        if ds is None:
            return pd.RangeIndex(self.shape[1])
        return pd.Index(H5Array._read_label(ds))

    @property
    def iloc(self) -> _ILocIndexer:
        """_ILocIndexer: Positional 2D indexing, mirroring `.iloc`."""
        return _ILocIndexer(self._dataset)

    def __repr__(self):
        kind = "scratch" if self._owns_file else "permanent"
        header = f"H5Array(shape={self.shape}, dtype={self.dtype}, {kind})"
        return f"{header}\n{self._preview_str()}"

    def _repr_html_(self):
        kind = "scratch" if self._owns_file else "permanent"
        header = (f"<b>H5Array</b> shape={self.shape} dtype={self.dtype} "
                 f"({kind})")
        return f"{header}<pre>{self._preview_str()}</pre>"

    def _preview_str(self, edgeitems=3) -> str:
        # Numpy-style preview (first/last `edgeitems` rows and columns,
        # with '...' in between) built from a handful of bounded disk
        # reads -- never touches the middle of the array. A dummy
        # all-zero row/column stands in for the skipped region; numpy's
        # own summarization (forced via threshold=0) never formats it,
        # since it only prints the first/last `edgeitems` per axis.
        nrow, ncol = self.shape
        if nrow == 0 or ncol == 0:
            return "[]"

        row_trunc = nrow > 2 * edgeitems
        col_trunc = ncol > 2 * edgeitems

        row_idx = (list(range(edgeitems)) + list(range(nrow-edgeitems, nrow))
                  if row_trunc else list(range(nrow)))
        block = self._dataset[row_idx, :]

        if col_trunc:
            left, right = block[:, :edgeitems], block[:, -edgeitems:]
            gap_col = np.zeros((block.shape[0], 1), dtype=block.dtype)
            block = np.concatenate([left, gap_col, right], axis=1)

        if row_trunc:
            top, bottom = block[:edgeitems], block[edgeitems:]
            gap_row = np.zeros((1, block.shape[1]), dtype=block.dtype)
            block = np.concatenate([top, gap_row, bottom], axis=0)

        return np.array2string(block, threshold=0, edgeitems=edgeitems)

    def __len__(self):
        return self.shape[0]

    def __array__(self, dtype=None):
        arr = self.to_numpy()
        return arr if dtype is None else arr.astype(dtype)

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

    def to_pandas(self) -> pd.DataFrame:
        """
        Materialize the full array as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            The complete array, labeled with `index`/`columns` if they
            were provided at creation time, otherwise a RangeIndex.
        """
        return pd.DataFrame(self.to_numpy(), index=self.index,
                            columns=self.columns)

    def mean(self, axis=0, batch_size=20000) -> np.ndarray:
        """
        Compute the mean along an axis without materializing the array.

        Parameters
        ----------
        axis : {0, 1}, default 0
            Axis along which to reduce. 0 reduces over rows (one value
            per column, matching pandas' default); 1 reduces over
            columns (one value per row).
        batch_size : int, default 20_000
            Number of rows read (and held in memory) per batch.

        Returns
        -------
        np.ndarray
            The reduced values.
        """
        return self._streamed_reduce("mean", axis, batch_size)

    def sum(self, axis=0, batch_size=20000) -> np.ndarray:
        """Sum along an axis without materializing the array. See `mean`
        for parameter/return details."""
        return self._streamed_reduce("sum", axis, batch_size)

    def min(self, axis=0, batch_size=20000) -> np.ndarray:
        """Minimum along an axis without materializing the array. See
        `mean` for parameter/return details."""
        return self._streamed_reduce("min", axis, batch_size)

    def max(self, axis=0, batch_size=20000) -> np.ndarray:
        """Maximum along an axis without materializing the array. See
        `mean` for parameter/return details."""
        return self._streamed_reduce("max", axis, batch_size)

    def _streamed_reduce(self, op, axis, batch_size, row_mask=None):
        if axis not in (0, 1):
            raise ValueError("axis must be 0 or 1")
        nrow, ncol = self.shape

        if axis == 1:
            chunks = []
            for start in range(0, nrow, batch_size):
                stop = min(start + batch_size, nrow)
                batch = self._dataset[start:stop, :]
                if row_mask is not None:
                    batch = batch[row_mask[start:stop]]
                if batch.shape[0] == 0:
                    continue
                chunks.append(_ROW_REDUCERS[op](batch))
            return np.concatenate(chunks) if chunks else np.array([])

        acc = None
        for start in range(0, nrow, batch_size):
            stop = min(start + batch_size, nrow)
            batch = self._dataset[start:stop, :]
            if row_mask is not None:
                batch = batch[row_mask[start:stop]]
            if batch.shape[0] == 0:
                continue
            acc = _COL_ACCUMULATORS[op](acc, batch)
        return _COL_FINALIZERS[op](acc, ncol)

    def save_to(self, group, name):
        """
        Stream-copy the underlying dataset into another HDF5 group.

        The copy happens at the HDF5 layer without materializing the
        full array in memory. Row/column labels, if present, are copied
        alongside it.

        Parameters
        ----------
        group : h5py.Group
            Destination group.
        name : str
            Name of the dataset to create within `group`.
        """
        group.copy(self._dataset, name)
        for suffix in ("__index", "__columns"):
            ds = self._label_dataset(suffix)
            if ds is not None:
                group.copy(ds, f"{name}{suffix}")

    def close(self):
        """Close the backing file, deleting it if it is a scratch file."""
        self._finalizer()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
