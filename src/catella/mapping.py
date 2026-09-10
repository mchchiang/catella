# mapping.py

import numpy as np
import pandas as pd
from dataclasses import dataclass

from catella.h5_array import H5Array

@dataclass(slots=True)
class CoordsTransform:
    """
    Handle alignment transformations for smoothed data.
    """
    lnuc : int
    """The smoothing window size used to calculate the original mean."""

    fill_edge : float = np.nan
    """The value used to fill empty spaces created by shifting data. Default
       np.nan."""

    # Helper function to perform a non-circular shift on DataFrames or Arrays.
    def _apply_shift(self, x, delta, axis):
        # Get absolute axis
        axis = axis if axis >= 0 else x.ndim + axis
        # For Pandas DataFrame
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.shift(delta, axis=axis, fill_value=self.fill_edge)
        # For NumPy arrays
        res = np.roll(x, delta, axis=axis)
        idx = [slice(None)] * x.ndim
        if delta > 0:
            idx[axis] = slice(0, delta)
        elif delta < 0:
            idx[axis] = slice(delta, None)
        res[tuple(idx)] = self.fill_edge
        return res

    # Stream the shift across an H5Array's rows, writing the result to a
    # new disk-backed array so peak memory scales with `batch_size` rather
    # than the array's full size. A row-axis (axis 0) shift is a
    # contiguous block move (unlike an arbitrary permutation), so it too
    # is streamed without materializing the full array.
    def _apply_shift_h5array(self, x, delta, axis, batch_size, path, dir):
        nrow, ncol = x.shape
        axis = axis if axis >= 0 else x.ndim + axis
        out = H5Array.create(x.shape, dtype=x.dtype, path=path, dir=dir,
                             index=x.index, columns=x.columns)
        if axis != 0:
            for start in range(0, nrow, batch_size):
                stop = min(start + batch_size, nrow)
                out.write_batch(
                    start, stop,
                    self._apply_shift(x[start:stop, :], delta, axis))
            return out

        shift = min(abs(delta), nrow)
        if delta >= 0:
            src_of = lambda start, stop: (start - shift, stop - shift)
            fill_range = (0, shift)
        else:
            src_of = lambda start, stop: (start + shift, stop + shift)
            fill_range = (nrow - shift, nrow)

        if fill_range[1] > fill_range[0]:
            fill_block = np.full((fill_range[1] - fill_range[0], ncol),
                                 self.fill_edge)
            out.write_batch(*fill_range, fill_block)

        copy_range = (fill_range[1], nrow) if delta >= 0 else \
            (0, fill_range[0])
        for start in range(copy_range[0], copy_range[1], batch_size):
            stop = min(start + batch_size, copy_range[1])
            src_start, src_stop = src_of(start, stop)
            out.write_batch(start, stop, x[src_start:src_stop, :])
        return out

    def left_to_center_aligned(self,
                               x : np.ndarray | pd.DataFrame | pd.Series
                                   | H5Array,
                               axis : int = -1,
                               trim : bool = False,
                               batch_size : int = 20000,
                               path : str | None = None,
                               dir : str | None = None):
        """
        Shift left-aligned data to the center.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame | pd.Series | H5Array
            The input data that was previously smoothed and left-aligned.
        axis : int, optional
            The axis along which to perform the shift. Default -1.
        trim : bool, optional
            If True, symmetrically trims `lnuc // 2` elements from both ends
            of the specified axis. Default False.
        batch_size : int, optional
            Number of rows streamed at a time when `x` is an `H5Array`.
            Ignored otherwise. Default 20000.
        path : str, optional
            Location of the result's backing HDF5 file, when `x` is an
            `H5Array`. See `H5Array.create`. Ignored otherwise.
        dir : str, optional
            Directory for the result's scratch file, when `x` is an
            `H5Array` and `path` is None. See `H5Array.create`. Ignored
            otherwise.

        Returns
        -------
        np.ndarray | pd.DataFrame | pd.Series | H5Array
            The center-aligned data.
        """
        delta = int(self.lnuc//2)
        res = self._apply_shift_h5array(x, delta, axis, batch_size, path,
                                        dir) if isinstance(x, H5Array) \
            else self._apply_shift(x, delta, axis)
        if trim and delta > 0:
            slc = [slice(None)] * x.ndim
            slc[axis] = slice(delta, -delta)
            return res.iloc[tuple(slc)] if hasattr(res, "iloc") else \
                res[tuple(slc)]
        return res

    def center_to_left_aligned(self,
                               x : np.ndarray | pd.DataFrame | pd.Series
                                   | H5Array,
                               axis : int = -1,
                               trim : bool = False,
                               batch_size : int = 20000,
                               path : str | None = None,
                               dir : str | None = None):
        """
        Shift center-aligned data back to the left.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame | pd.Series | H5Array
            The input data that is currently center-aligned.
        axis : int, optional
            The axis along which to perform the shift. Default -1.
        trim : bool, optional
            If True, trims the trailing shift-padding from the end of the
            specified axis. Default False.
        batch_size : int, optional
            Number of rows streamed at a time when `x` is an `H5Array`.
            Ignored otherwise. Default 20000.
        path : str, optional
            Location of the result's backing HDF5 file, when `x` is an
            `H5Array`. See `H5Array.create`. Ignored otherwise.
        dir : str, optional
            Directory for the result's scratch file, when `x` is an
            `H5Array` and `path` is None. See `H5Array.create`. Ignored
            otherwise.

        Returns
        -------
        np.ndarray | pd.DataFrame | pd.Series | H5Array
            The left-aligned data.
        """
        delta = -int(self.lnuc//2)
        res = self._apply_shift_h5array(x, delta, axis, batch_size, path,
                                        dir) if isinstance(x, H5Array) \
            else self._apply_shift(x, delta, axis)
        if trim and abs(delta) > 0:
            slc = [slice(None)] * x.ndim
            slc[axis] = slice(None, delta)
            return res.iloc[tuple(slc)] if hasattr(res, "iloc") else \
                res[tuple(slc)]
        return res

