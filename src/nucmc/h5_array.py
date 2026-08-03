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

_DEFAULT_CHUNK_TARGET_BYTES = 1 << 20  # ~1 MiB per chunk


def _default_chunk_shape(shape, dtype):
    nrow, ncol = shape
    row_bytes = ncol * np.dtype(dtype).itemsize
    chunk_rows = max(1, min(nrow, _DEFAULT_CHUNK_TARGET_BYTES // row_bytes))
    return (chunk_rows, ncol)


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
              index=None, columns=None,
              compression="gzip", compression_opts=None,
              shuffle=True, chunks=None):
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
        compression : str, optional
            HDF5 compression filter for the backing dataset, e.g.
            "gzip" or "lzf". Default "gzip". Pass None to disable
            compression and store the dataset contiguous and
            uncompressed.
        compression_opts : int, optional
            Filter-specific compression setting (e.g. gzip level
            0-9). If None (default), HDF5's own default for the
            chosen filter is used (level 4 for gzip). Ignored if
            `compression` is None.
        shuffle : bool, default True
            Whether to apply HDF5's byte-shuffle filter before
            compression. Reorders bytes so same-significance bytes
            across elements are grouped together, which improves
            compression of correlated numeric data (e.g. smoothed
            signal arrays, where neighboring values are similar but
            not bit-identical) at negligible cost. Ignored if
            `compression` is None.
        chunks : tuple of int, optional
            Chunk shape for the backing dataset. If None (default)
            and `compression` is given, a chunk shape spanning the
            full column width is chosen automatically, sized to
            ~1 MiB and matching this class's row-batch access
            pattern (`write_batch`/`downsample`/streamed reducers all
            read or write contiguous row ranges across every column).
            Ignored if `compression` is None and `chunks` is not
            explicitly given.

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

        dataset_kwargs = {}
        nrow, ncol = shape
        if compression is not None and nrow > 0 and ncol > 0:
            dataset_kwargs["compression"] = compression
            if compression_opts is not None:
                dataset_kwargs["compression_opts"] = compression_opts
            dataset_kwargs["shuffle"] = shuffle
            dataset_kwargs["chunks"] = chunks if chunks is not None \
                else _default_chunk_shape(shape, dtype)
        elif chunks is not None:
            dataset_kwargs["chunks"] = chunks

        dataset = file.create_dataset(_DATASET_NAME, shape=shape,
                                      dtype=dtype, **dataset_kwargs)
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

    def downsample(self, max_rows, *, how="mean", batch_size=20000):
        """
        Collapse rows to at most `max_rows` without materializing the
        full array in memory.

        Parameters
        ----------
        max_rows : int
            Target number of rows in the output.
        how : {"mean", "sum", "min", "max", "stride"}, default "mean"
            How to collapse groups of consecutive rows into one output
            row. "mean"/"sum"/"min"/"max" aggregate every contiguous
            bin of rows, so no rows are dropped. "stride" instead reads
            every k-th row directly from disk, skipping the rest --
            cheaper, but can miss features between sampled rows.
        batch_size : int, default 20000
            Number of rows read (and held in memory) per streamed
            chunk within a bin, when `how` aggregates bins. Ignored
            for "stride".

        Returns
        -------
        np.ndarray
            Array of shape (min(nrow, max_rows), ncol).

        Raises
        ------
        ValueError
            If `how` is not a recognized option.
        """
        if how not in _DOWNSAMPLE_HOW:
            raise ValueError(f"'how' must be one of {_DOWNSAMPLE_HOW}")
        nrow, ncol = self.shape

        if nrow <= max_rows:
            return self._dataset[:]

        if how == "stride":
            step = -(-nrow // max_rows)
            return self._dataset[0:nrow:step, :]

        acc_fn = _COL_ACCUMULATORS[how]
        fin_fn = _COL_FINALIZERS[how]
        edges = np.linspace(0, nrow, max_rows + 1).astype(int)
        out = np.empty((max_rows, ncol))
        for i in range(max_rows):
            start, stop = edges[i], edges[i+1]
            acc = None
            for sub_start in range(start, stop, batch_size):
                sub_stop = min(sub_start + batch_size, stop)
                batch = self._dataset[sub_start:sub_stop, :]
                acc = acc_fn(acc, batch)
            out[i] = fin_fn(acc, ncol)
        return out

    def reorder_rows(self, order, *, path=None, dir=None, batch_size=20000):
        """
        Gather rows into a new disk-backed array, in an arbitrary order.

        Reads one source row at a time (the only way to support an
        arbitrary permutation, since HDF5 fancy indexing requires
        increasing order) and accumulates into `batch_size`-sized
        blocks before each write, so peak memory scales with
        `batch_size` rather than the number of rows.

        Parameters
        ----------
        order : array-like of int
            Row indices into this array, in output order. May repeat
            or omit indices; its length sets the output row count.
        path : str or pathlib.Path, optional
            Location of the new array's backing HDF5 file. See
            `create`.
        dir : str or pathlib.Path, optional
            Directory for the new scratch file when `path` is None.
            See `create`.
        batch_size : int, default 20000
            Number of rows accumulated in memory before each write.

        Returns
        -------
        H5Array
            A new array of shape (len(order), ncol) with this array's
            rows rearranged according to `order`.
        """
        order = np.asarray(order)
        nrow_out = len(order)
        ncol = self.shape[1]

        index_ds = self._label_dataset("__index")
        columns_ds = self._label_dataset("__columns")
        index = (H5Array._read_label(index_ds)[order]
                if index_ds is not None else None)
        columns = (H5Array._read_label(columns_ds)
                  if columns_ds is not None else None)

        out = H5Array.create((nrow_out, ncol), dtype=self.dtype,
                             path=path, dir=dir, index=index,
                             columns=columns)
        for start in range(0, nrow_out, batch_size):
            stop = min(start + batch_size, nrow_out)
            block = np.empty((stop - start, ncol), dtype=self.dtype)
            for i, src in enumerate(order[start:stop]):
                block[i] = self._dataset[int(src), :]
            out.write_batch(start, stop, block)
        return out

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
