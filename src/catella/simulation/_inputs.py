# _inputs.py

from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
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


def _normalize_meth_prob(meth_prob, chroms):
    """
    Return the methylation probability of each chromosome as a 2D array.

    Parameters
    ----------
    meth_prob : np.ndarray or mapping of str to np.ndarray or pd.DataFrame
        A single array, allowed only if `chroms` has one chromosome, or a
        mapping from chromosome to data.
    chroms : sequence of str
        The chromosomes to select.

    Returns
    -------
    dict of str to np.ndarray
        The 2D array (molecules by base pairs) of each chromosome in
        `chroms`.

    Raises
    ------
    TypeError
        If `meth_prob` is an array and `chroms` has more than one
        chromosome, or `meth_prob` is neither an array nor a mapping.
    KeyError
        If a mapping `meth_prob` lacks a chromosome of `chroms`.
    """
    if isinstance(meth_prob, np.ndarray):
        if len(chroms) != 1:
            raise TypeError(
                "a single 'meth_prob' array requires exactly one chrom, "
                f"got {len(chroms)}"
            )
        meth_prob = {chroms[0]: meth_prob}
    elif not isinstance(meth_prob, Mapping):
        raise TypeError("'meth_prob' must be a numpy array or a mapping")
    out = {}
    for chrom in chroms:
        data = meth_prob[chrom]
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        out[chrom] = np.atleast_2d(data)
    return out


def _normalize_mols(mols, chroms):
    """
    Return the molecule selection of each chromosome.

    Parameters
    ----------
    mols : int, slice, sequence of int, or mapping
        One selection applied to every chromosome, or a mapping from
        chromosome to selection.
    chroms : sequence of str
        The chromosomes to select.

    Returns
    -------
    dict
        The selection of each chromosome in `chroms`.

    Raises
    ------
    TypeError
        If `mols` is neither a selection nor a mapping.
    KeyError
        If a mapping `mols` lacks a chromosome of `chroms`.
    """
    if isinstance(mols, (int, slice, Sequence)):
        return {chrom: mols for chrom in chroms}
    if isinstance(mols, Mapping):
        return {chrom: mols[chrom] for chrom in chroms}
    raise TypeError("'mols' must be an IndexType or a mapping")


def _count_mols(mols, nmol):
    """
    Check the molecule selections and count the selected molecules.

    Parameters
    ----------
    mols : mapping of str to int, slice, or sequence of int
        The selection of each chromosome.
    nmol : mapping of str to int
        The number of molecules of each chromosome.

    Returns
    -------
    int
        The total number of selected molecules over all chromosomes.

    Raises
    ------
    IndexError
        If a selection is out of range.
    ValueError
        If a selection is empty.
    """
    total = 0
    for chrom, size in nmol.items():
        try:
            # Index a small dummy array instead of building the index list
            count = np.empty(size, dtype=np.int8)[mols[chrom]].size
        except IndexError as e:
            raise IndexError(
                f"mols index for chromosome '{chrom}': {e}"
            ) from None
        if count == 0:
            raise ValueError(
                f"selection 'mols' for chromosome '{chrom}' is empty"
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


def _check_llink(settings, chroms):
    """
    Check that a mapping `settings.llink` covers `chroms` with valid values.

    A scalar `settings.llink` is not checked.

    Parameters
    ----------
    settings : SimSettings
        The settings to check.
    chroms : iterable of str
        The chromosomes that must be covered.

    Raises
    ------
    KeyError
        If the mapping lacks a chromosome of `chroms`.
    ValueError
        If a value of the mapping is not a non-negative integer.
    """
    if not isinstance(settings.llink, Mapping):
        return
    for chrom in chroms:
        if chrom not in settings.llink:
            raise KeyError(f"llink has no value for chromosome '{chrom}'")
        value = settings.llink[chrom]
        if not _is_valid_llink(value):
            raise ValueError(
                "llink must be a non-negative integer for chromosome "
                f"'{chrom}', got {value!r}"
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
