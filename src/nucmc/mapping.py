# mapping.py

import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass(slots=True)
class CoordsTransform:
    """
    Handle alignment transformations for smoothed data.
    """
    binsize : int
    """The smoothing binsize size used to calculate the original mean."""

    fill_value : float = np.nan
    """The value used to fill empty spaces created by shifting data. Default
       np.nan."""

    # Helper function to perform a non-circular shift on DataFrames or Arrays.
    def _apply_shift(self, x, delta, axis):
        # Get absolute axis
        axis = axis if axis >= 0 else x.ndim + axis
        # For Pandas DataFrame
        if isinstance(x, (pd.DataFrame, pd.Series)):
            return x.shift(delta, axis=axis, fill_value=self.fill_value)
        # For NumPy arrays
        res = np.roll(x, delta, axis=axis)
        idx = [slice(None)] * x.ndim
        if delta > 0:
            idx[axis] = slice(0, delta)
        elif delta < 0:
            idx[axis] = slice(delta, None)
        res[tuple(idx)] = self.fill_value
        return res        

    def left_to_center_aligned(self,
                               x : np.ndarray | pd.DataFrame | pd.Series,
                               axis : int = -1,
                               trim : bool = False):
        """
        Shift left-aligned data to the center.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame | pd.Series
            The input data that was previously smoothed and left-aligned.
        axis : int, optional
            The axis along which to perform the shift. Default -1.
        trim : bool, optional
            If True, symmetrically trims `binsize // 2` elements from both ends 
            of the specified axis. Default False.

        Returns
        -------
        np.ndarray | pd.DataFrame | pd.Series
            The center-aligned data.
        """
        delta = int(self.binsize//2)
        res = self._apply_shift(x, delta, axis)
        if trim and delta > 0:
            slc = [slice(None)] * x.ndim
            slc[axis] = slice(delta, -delta)
            return res.iloc[tuple(slc)] if hasattr(res, "iloc") else \
                res[tuple(slc)]
        return res

    def center_to_left_aligned(self,
                               x : np.ndarray | pd.DataFrame | pd.Series,
                               axis : int = -1,
                               trim : bool = False):
        """
        Shift center-aligned data back to the left.

        Parameters
        ----------
        x : np.ndarray | pd.DataFrame | pd.Series
            The input data that is currently center-aligned.
        axis : int, optional
            The axis along which to perform the shift. Default -1.
        trim : bool, optional
            If True, trims the trailing shift-padding from the end of the 
            specified axis. Default False.

        Returns
        -------
        np.ndarray | pd.DataFrame | pd.Series
            The left-aligned data.
        """
        delta = -int(self.binsize//2)
        res = self._apply_shift(x, delta, axis)
        if trim and abs(delta) > 0:
            slc = [slice(None)] * x.ndim
            slc[axis] = slice(None, delta)
            return res.iloc[tuple(slc)] if hasattr(res, "iloc") else \
                res[tuple(slc)]
        return res        
    
