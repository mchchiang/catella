# _inputs.py

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral, Real
from pathlib import Path

import numpy as np
import pandas as pd
from catella_cpp import Dump

from catella.simulation.config import SimSettings

# Recognized names of the data recorded by an MCMC run
_OUT_MAP = {
    "energy": Dump.OutputType.Energy,
    "position": Dump.OutputType.Position,
    "temp": Dump.OutputType.Temp,
    "all": Dump.OutputType.All,
}

# Protocol fields that are optional on SimSettings but required for MCMC
_MCMC_FIELDS = ("start_temp", "cool_option", "nsweep", "print_freq")


def _is_valid_llink(value) -> bool:
    """Return True if `value` is a non-negative integer (bools excluded)."""
    return (
        isinstance(value, Integral)
        and not isinstance(value, bool)
        and value >= 0
    )


def _normalize_meth_prob(meth_prob, regions):
    """
    Return the methylation probability of each region as a 2D array.

    Parameters
    ----------
    meth_prob : np.ndarray or mapping of str to np.ndarray or pd.DataFrame
        A single array, allowed only if `regions` has one region, or a
        mapping from region to data.
    regions : sequence of str
        The regions to select.

    Returns
    -------
    dict of str to np.ndarray
        The 2D array (molecules by base pairs) of each region in
        `regions`.

    Raises
    ------
    TypeError
        If `meth_prob` is an array and `regions` has more than one
        region, or `meth_prob` is neither an array nor a mapping.
    KeyError
        If a mapping `meth_prob` lacks a region of `regions`.
    """
    if isinstance(meth_prob, np.ndarray):
        if len(regions) != 1:
            raise TypeError(
                "a single 'meth_prob' array requires exactly one region, "
                f"got {len(regions)}"
            )
        meth_prob = {regions[0]: meth_prob}
    elif not isinstance(meth_prob, Mapping):
        raise TypeError("'meth_prob' must be a numpy array or a mapping")
    out = {}
    for region in regions:
        data = meth_prob[region]
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        out[region] = np.atleast_2d(data)
    return out


def _start_positions(block, nucbp):
    """
    Return the columns of `block` where a nucleosome can start.

    Parameters
    ----------
    block : np.ndarray
        2D array, molecules by base pairs.
    nucbp : int
        The nucleosome footprint size.

    Returns
    -------
    np.ndarray
        The first `nbp - nucbp + 1` columns, or all columns if the
        molecules are shorter than a nucleosome.
    """
    npos = block.shape[-1] - nucbp + 1
    return block[..., :npos] if npos > 0 else block


def _check_fill_nan(fill_nan):
    """
    Check a `fill_nan` value.

    Parameters
    ----------
    fill_nan : None, "mean", or float
        The value to check.

    Raises
    ------
    ValueError
        If `fill_nan` is not None, "mean", or a number in [0, 1].
    """
    if fill_nan is None or (isinstance(fill_nan, str) and fill_nan == "mean"):
        return
    if (
        isinstance(fill_nan, bool)
        or not isinstance(fill_nan, Real)
        or not 0 <= fill_nan <= 1
    ):
        raise ValueError(
            "fill_nan must be None, 'mean', or a number in [0, 1], got "
            f"{fill_nan!r}"
        )


def _prepare_rows(block, nucbp, region, fill_nan=None):
    """
    Find the molecules of `block` with no data and fill the others.

    A molecule has no data if its methylation probability is nan at
    every nucleosome start position. Nan beyond the last start
    position is left as is, since those base pairs never set a
    nucleosome's energy.

    Parameters
    ----------
    block : np.ndarray
        2D methylation probability, molecules by base pairs.
    nucbp : int
        The nucleosome footprint size.
    region : str
        The region of `block`, for error messages.
    fill_nan : None, "mean", or float, default None
        How to treat a molecule that is nan at some but not all start
        positions: None raises, "mean" fills the nan with the mean of
        the molecule's other start positions, and a number in [0, 1]
        fills them with that probability (0.5 adds no information).

    Returns
    -------
    block : np.ndarray
        `block` if nothing is filled, else a filled copy.
    no_data : np.ndarray
        bool, True for each molecule with no data.

    Raises
    ------
    ValueError
        If `fill_nan` is invalid, or is None and a molecule is nan at
        some but not all start positions.
    """
    _check_fill_nan(fill_nan)
    nan = np.isnan(_start_positions(block, nucbp))
    no_data = nan.all(axis=1)
    if block.shape[1] < nucbp:
        return block, no_data
    partial = np.flatnonzero(nan.any(axis=1) & ~no_data)
    if not partial.size:
        return block, no_data
    if fill_nan is None:
        raise ValueError(
            f"meth_prob of region '{region}' has nan at nucleosome start "
            f"positions in {partial.size} molecule(s) (first: "
            f"{partial[0]}); pass fill_nan (0.5 for windows a read does "
            "not cover, e.g. with span='read' or long gaps), or fill "
            "them upstream (e.g., fill_edge or nan_method)"
        )
    block = np.array(block, dtype=np.float64)
    starts = _start_positions(block, nucbp)
    sub = starts[partial]
    if isinstance(fill_nan, str):
        value = np.nanmean(sub, axis=1, keepdims=True)
    else:
        value = float(fill_nan)
    sub_nan = np.isnan(sub)
    sub[sub_nan] = np.broadcast_to(value, sub.shape)[sub_nan]
    starts[partial] = sub
    return block, no_data


def _normalize_mols(mols, regions):
    """
    Return the molecule selection of each region.

    Parameters
    ----------
    mols : int, slice, sequence of int, or mapping
        One selection applied to every region, or a mapping from
        region to selection.
    regions : sequence of str
        The regions to select.

    Returns
    -------
    dict
        The selection of each region in `regions`.

    Raises
    ------
    TypeError
        If `mols` is neither a selection nor a mapping.
    KeyError
        If a mapping `mols` lacks a region of `regions`.
    """
    if isinstance(mols, (int, slice, Sequence)):
        return {region: mols for region in regions}
    if isinstance(mols, Mapping):
        return {region: mols[region] for region in regions}
    raise TypeError("'mols' must be an IndexType or a mapping")


def _count_mols(mols, nmol):
    """
    Check the molecule selections and count the selected molecules.

    Parameters
    ----------
    mols : mapping of str to int, slice, or sequence of int
        The selection of each region.
    nmol : mapping of str to int
        The number of molecules of each region.

    Returns
    -------
    int
        The total number of selected molecules over all regions.

    Raises
    ------
    IndexError
        If a selection is out of range.
    ValueError
        If a selection is empty.
    """
    total = 0
    for region, size in nmol.items():
        try:
            # Index a small dummy array instead of building the index list
            count = np.empty(size, dtype=np.int8)[mols[region]].size
        except IndexError as e:
            raise IndexError(
                f"mols index for region '{region}': {e}"
            ) from None
        if count == 0:
            raise ValueError(
                f"selection 'mols' for region '{region}' is empty"
            )
        total += count
    return total


def _resolve_settings(settings):
    """
    Return `settings` as a SimSettings, loading it from a file if needed.

    Parameters
    ----------
    settings : str, Path, or SimSettings
        The settings, or the path to a settings file.

    Returns
    -------
    SimSettings
        The resolved settings.

    Raises
    ------
    TypeError
        If `settings` is neither a path nor a SimSettings.
    """
    if isinstance(settings, (str, Path)):
        return SimSettings.load(settings)
    if not isinstance(settings, SimSettings):
        raise TypeError(
            "'settings' must be either a path to the simulation setting "
            "file or a SimSettings object"
        )
    return settings


def _check_llink(settings, regions):
    """
    Check that `settings.llink` is valid for `regions`.

    A scalar `settings.llink` must be a non-negative integer. A mapping
    must cover `regions` with non-negative integer values.

    Parameters
    ----------
    settings : SimSettings
        The settings to check.
    regions : iterable of str
        The regions that must be covered.

    Raises
    ------
    KeyError
        If the mapping lacks a region of `regions`.
    ValueError
        If the scalar or a value of the mapping is not a non-negative
        integer.
    """
    if not isinstance(settings.llink, Mapping):
        if not _is_valid_llink(settings.llink):
            raise ValueError(
                "llink must be a non-negative integer, "
                f"got {settings.llink!r}"
            )
        return
    for region in regions:
        if region not in settings.llink:
            raise KeyError(f"llink has no value for region '{region}'")
        value = settings.llink[region]
        if not _is_valid_llink(value):
            raise ValueError(
                "llink must be a non-negative integer for region "
                f"'{region}', got {value!r}"
            )


def _check_mcmc_settings(settings):
    """
    Check that `settings` has a valid MCMC annealing protocol.

    Parameters
    ----------
    settings : SimSettings
        The settings to check.

    Raises
    ------
    ValueError
        If a protocol field is missing, or `end_temp` exceeds
        `start_temp`.
    KeyError
        If `cool_option` is not a recognized cooling schedule.
    """
    for name in _MCMC_FIELDS:
        if getattr(settings, name) is None:
            raise ValueError(f"'{name}' is required for method='mcmc'")
    if settings.end_temp > settings.start_temp:
        raise ValueError("expect 'end_temp' <= 'start_temp'")
    if settings.cool_option not in SimSettings._cool_map:
        raise KeyError(
            "invalid value for 'cool_option'; expect either "
            "'linear', 'geometric', or 'constant'"
        )


def _parse_out_types(out_types):
    """
    Combine output type names into a single Dump.OutputType.

    Parameters
    ----------
    out_types : str or iterable of str
        One or more of 'energy', 'position', 'temp', or 'all'.

    Returns
    -------
    Dump.OutputType
        The union of the named output types.

    Raises
    ------
    TypeError
        If `out_types` is neither a str nor an iterable.
    ValueError
        If a name is not recognized, or `out_types` is empty.
    """
    if isinstance(out_types, str):
        out_types = [out_types]
    elif not isinstance(out_types, Iterable):
        raise TypeError("'out_types' must be a str or an iterable")
    combined = None
    for name in out_types:
        if name not in _OUT_MAP:
            raise ValueError(f"output type '{name}' is not a valid option")
        # The bound C++ enum has no '|', so combine the int values
        value = int(_OUT_MAP[name])
        combined = value if combined is None else combined | value
    if combined is None:
        raise ValueError("'out_types' must not be empty")
    return Dump.OutputType(combined)
