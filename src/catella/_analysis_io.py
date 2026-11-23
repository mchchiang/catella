# _analysis_io.py

import h5py

from catella import h5_utils
from catella.h5_array import H5Array, _is_label


def _replace_group(parent, name):
    """
    Create an empty group, deleting any existing one of the same name.

    Parameters
    ----------
    parent : h5py.Group
        The parent group.
    name : str
        The name of the group.

    Returns
    -------
    h5py.Group
        The new, empty group.
    """
    if name in parent:
        del parent[name]
    return parent.create_group(name)


def _write_entries(group, entries):
    """
    Write analysis entries (H5Array or DataFrame) into `group`.

    Parameters
    ----------
    group : h5py.Group
        The group to write to.
    entries : mapping of str to H5Array or pd.DataFrame
        The entries, by name.
    """
    for name, entry in entries.items():
        if isinstance(entry, H5Array):
            entry.save_to(group, name)
        else:
            h5_utils.save_df(name, entry, group)


def _read_entries(group, h5_path):
    """
    Read analysis entries written by `_write_entries`.

    Datasets are opened lazily as H5Array; groups are read as DataFrames.

    Parameters
    ----------
    group : h5py.Group
        The group to read from.
    h5_path : str or Path
        The file holding `group`, used to open each H5Array.

    Returns
    -------
    dict of str to H5Array or pd.DataFrame
        The entries, by name.
    """
    prefix = group.name.strip("/")
    entries = {}
    for name in group:
        if _is_label(group, name):
            continue
        if isinstance(group[name], h5py.Dataset):
            entries[name] = H5Array.load_from(h5_path, f"{prefix}/{name}")
        else:
            entries[name] = h5_utils.load_df(name, group)
    return entries
