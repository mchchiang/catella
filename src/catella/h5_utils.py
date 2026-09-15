# h5_utils.py

import atexit
import os
import shutil
import tempfile
import uuid
from datetime import UTC, datetime

import h5py
import numpy as np
import pandas as pd


def fresh_tmp_dir(base_dir=None):
    """
    Create a fresh scratch directory for temporary HDF5 files.

    Registers an atexit fallback that removes the directory, in case
    the caller is interrupted before setting up its own cleanup.

    Parameters
    ----------
    base_dir : str or pathlib.Path, optional
        Parent directory in which to create the scratch directory. If
        None, the system default temporary directory is used.

    Returns
    -------
    str
        Path to the newly created scratch directory, named
        `catella_<timestamp>_<hex>`.
    """
    base = str(base_dir) if base_dir is not None else tempfile.gettempdir()
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    path = os.path.join(base, f"catella_{stamp}_{suffix}")
    os.makedirs(path)
    atexit.register(shutil.rmtree, path, ignore_errors=True)
    return path


# Helper functions for loading and saving data frames in h5 files
def save_df(name, df, group):
    g = group.create_group(name)
    dt = h5py.string_dtype(encoding="utf-8")  # For storing strings
    # Store the master order of all columns
    order = df.columns.astype(str).tolist()
    g.create_dataset("_column_order", data=order, dtype=dt)
    g.attrs["_column_index_dtype"] = str(df.columns.dtype)

    # Split numeric and string data
    num_df = df.select_dtypes(include=[np.number])
    str_df = df.select_dtypes(exclude=[np.number])

    # Save numeric block
    if not num_df.empty:
        g.create_dataset(
            "num_values", data=num_df.to_numpy(), compression="gzip"
        )
        g.create_dataset(
            "num_names",
            dtype=dt,
            compression="gzip",
            data=num_df.columns.astype(str).tolist(),
        )

    # Save string columns individually
    if not str_df.empty:
        for col in str_df.columns:
            sdata = str_df[col].astype(str).tolist()
            g.create_dataset(
                f"str_{col}", data=sdata, dtype=dt, compression="gzip"
            )


class AppendableDF:
    """
    Incrementally write a DataFrame to an HDF5 group, in the same
    on-disk layout `save_df` produces, without holding the full table
    in memory. Read back by the unmodified `load_df`.

    The column schema (order, and which columns are numeric vs. string)
    is fixed up front and does not change across `append` calls.
    """

    def __init__(self, group, name, *, columns, dtypes):
        """
        Parameters
        ----------
        group : h5py.Group
            Parent group; a new sub-group `name` is created within it.
        name : str
            Name of the sub-group to create.
        columns : list of str
            Full, fixed column order.
        dtypes : dict of str to {"numeric", "string"}
            Per-column kind, used to route values into the numeric block
            or a per-column string dataset.
        """
        dt = h5py.string_dtype(encoding="utf-8")
        self._group = group.create_group(name)
        self._columns = list(columns)
        self._num_cols = [c for c in self._columns if dtypes[c] == "numeric"]
        self._str_cols = [c for c in self._columns if dtypes[c] == "string"]

        self._group.create_dataset(
            "_column_order", data=[str(c) for c in self._columns], dtype=dt
        )
        self._group.attrs["_column_index_dtype"] = "object"

        self._num_ds = None
        if self._num_cols:
            self._num_ds = self._group.create_dataset(
                "num_values",
                shape=(0, len(self._num_cols)),
                maxshape=(None, len(self._num_cols)),
                dtype=np.float64,
                compression="gzip",
            )
            self._group.create_dataset(
                "num_names", dtype=dt, data=[str(c) for c in self._num_cols]
            )
        self._str_ds = {}
        for col in self._str_cols:
            self._str_ds[col] = self._group.create_dataset(
                f"str_{col}",
                shape=(0,),
                maxshape=(None,),
                dtype=dt,
                compression="gzip",
            )
        self._nrows = 0

    def append(self, df_chunk):
        """
        Append rows from a DataFrame chunk.

        Parameters
        ----------
        df_chunk : pd.DataFrame
            Rows to append; must contain all columns passed to the
            constructor. Extra columns are ignored.
        """
        n = len(df_chunk)
        if n == 0:
            return
        if self._num_cols:
            block = df_chunk[self._num_cols].to_numpy(dtype=np.float64)
            old = self._num_ds.shape[0]
            self._num_ds.resize(old + n, axis=0)
            self._num_ds[old : old + n, :] = block
        for col in self._str_cols:
            ds = self._str_ds[col]
            old = ds.shape[0]
            ds.resize(old + n, axis=0)
            ds[old : old + n] = df_chunk[col].astype(str).to_numpy()
        self._nrows += n

    @property
    def nrows(self) -> int:
        """int: Number of rows appended so far."""
        return self._nrows

    def close(self):
        """No-op; reserved for symmetry with other streaming writers."""


def load_df(name, group):
    if name in group:
        g = group[name]
        orig_dtype = g.attrs.get("_column_index_dtype", "object")

        # Load the numeric data, if any (absent for all-string frames)
        df = pd.DataFrame()
        if "num_values" in g:
            data = g["num_values"][:]
            cols = [
                c.decode("utf-8") if isinstance(c, bytes) else c
                for c in g["num_names"][:]
            ]
            df = pd.DataFrame(data, columns=cols)

        # Load the string data
        for k in g:
            if k.startswith("str_"):
                colname = k.replace("str_", "")
                df[colname] = [
                    s.decode("utf-8") if isinstance(s, bytes) else s
                    for s in g[k][:]
                ]

        # Reorder to the original state
        order = [
            c.decode() if isinstance(c, bytes) else c
            for c in g["_column_order"][:]
        ]
        df = df[order]

        # Convert column names to their original type
        try:
            df.columns = df.columns.astype(orig_dtype)
        except (ValueError, TypeError):
            pass
        return df
    return None
