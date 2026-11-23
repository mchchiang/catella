# _analysis_io.py

import os
from pathlib import Path

import h5py

from catella import h5_utils
from catella.h5_array import _LABEL_SUFFIXES, H5Array, _is_label

_GROUPS = ("analysis", "global_analysis")
_STAGED = ".new"


def _analysis_group(h5stream, name):
    """
    Return the analysis group `name`, falling back to its staged copy.

    A save interrupted between deleting `name` and renaming its staged
    copy leaves only `<name>.new`, which holds the complete new state.

    Parameters
    ----------
    h5stream : h5py.File
        The open file.
    name : str
        The group name, e.g. "analysis".

    Returns
    -------
    h5py.Group
        The group, or its staged copy if the group is missing.
    """
    if name in h5stream:
        return h5stream[name]
    return h5stream[name + _STAGED]


def _save_analysis(
    path, analysis, global_analysis, *, write_extra=None, keep_unlisted=False
):
    """
    Write the analysis groups to `path` in place.

    Permanent H5Array entries already stored in `path` are hard-linked
    into the new groups instead of copied, so no stored data is
    duplicated. Both groups are first built as `<group>.new` and then
    swapped in, so an interrupted save leaves the old groups intact.
    The released entries are reopened afterwards, so references to
    them stay valid.

    Parameters
    ----------
    path : str or pathlib.Path
        The HDF5 file to write to, created if missing.
    analysis : mapping of str to mapping of str to H5Array or DataFrame
        Per-region entries, written to `analysis/<region>/<name>`.
    global_analysis : mapping of str to H5Array or DataFrame
        Entries written to `global_analysis/<name>`.
    write_extra : callable, optional
        Called with the open `h5py.File` before the analysis groups are
        written, e.g. to write raw data or metadata.
    keep_unlisted : bool, default False
        Carry over region groups of the existing `analysis` group
        that are not keys of `analysis` (e.g. regions not loaded).

    Raises
    ------
    OSError
        If `path` is open elsewhere (e.g. by another loaded experiment
        or dataset), or cannot be written to.
    """
    linked = _stored_in(path, analysis, global_analysis)
    sources = {key: src for key, (_, src) in linked.items()}
    staged = {}
    swapped = set()
    for arr, _ in linked.values():
        arr._release()
    try:
        with _open_for_write(path) as f:
            if write_extra is not None:
                write_extra(f)
            for name in _GROUPS:
                _recover(f, name, sources)
            gnew = f.create_group("analysis" + _STAGED)
            for region, data in analysis.items():
                _link_or_write(
                    f, gnew.create_group(region), data, sources, staged
                )
            if keep_unlisted and "analysis" in f:
                for region, gold in f["analysis"].items():
                    if region not in analysis:
                        gnew[region] = gold
            gnew = f.create_group("global_analysis" + _STAGED)
            _link_or_write(f, gnew, global_analysis, sources, staged)
            f.flush()
            for name in _GROUPS:
                if name in f:
                    del f[name]
                f.move(name + _STAGED, name)
                swapped.add(name)
    finally:
        _reopen_all(path, linked, sources, staged, swapped)


def _stored_in(path, analysis, global_analysis):
    """Return `{id: (entry, dataset_path)}` of arrays stored in `path`."""
    if not Path(path).exists():
        return {}
    linked = {}
    for data in [*analysis.values(), global_analysis]:
        for entry in data.values():
            if (
                isinstance(entry, H5Array)
                and not entry._owns_file
                and id(entry) not in linked
                and os.path.samefile(entry.path, path)
            ):
                linked[id(entry)] = (entry, entry.dataset_path)
    return linked


def _open_for_write(path):
    """Open `path` for appending, with a clear error if open elsewhere."""
    try:
        return h5py.File(str(path), "a")
    except OSError as err:
        if "already open" not in str(err):
            raise
        msg = (
            f"cannot save to {path}; it is open elsewhere (e.g. by "
            "another loaded experiment or dataset); close it first"
        )
        raise OSError(msg) from err


def _recover(f, name, sources):
    """Promote or clear a staged group left by an interrupted save."""
    staged_name = name + _STAGED
    if staged_name not in f:
        return
    if name in f:
        del f[staged_name]
        return
    f.move(staged_name, name)
    old, new = f"/{staged_name}/", f"/{name}/"
    for key, src in sources.items():
        if src.startswith(old):
            sources[key] = new + src[len(old) :]


def _link_or_write(f, group, entries, sources, staged):
    """Hard-link stored arrays into `group`; write the other entries."""
    for name, entry in entries.items():
        src = sources.get(id(entry))
        if src is None:
            _write_entries(group, {name: entry})
            continue
        group[name] = f[src]
        for suffix in _LABEL_SUFFIXES:
            if src + suffix in f:
                group[name + suffix] = f[src + suffix]
        staged.setdefault(id(entry), f"{group.name}/{name}")


def _reopen_all(path, linked, sources, staged, swapped):
    """Reopen released arrays at wherever their data now lives."""
    if not linked:
        return
    with h5py.File(str(path), "r") as f:
        targets = {}
        for key in linked:
            candidates = [sources[key]]
            if key in staged:
                top, rest = staged[key][1:].split("/", 1)
                final = f"/{top.removesuffix(_STAGED)}/{rest}"
                if top.removesuffix(_STAGED) in swapped:
                    candidates.insert(0, final)
                candidates.append(staged[key])
            targets[key] = next(p for p in candidates if p in f)
    for key, (arr, _) in linked.items():
        arr._reopen(arr.path, targets[key])


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
    A "chrom" column or index level, written before the analysis unit
    was renamed, is read as "region".

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
            entries[name] = _legacy_region_names(
                h5_utils.load_df(name, group)
            )
    return entries


def _legacy_region_names(df):
    """Rename a "chrom" column or index level of `df` to "region"."""
    if "chrom" in df.columns and "region" not in df.columns:
        df = df.rename(columns={"chrom": "region"})
    if "chrom" in df.index.names:
        df.index = df.index.set_names(
            ["region" if n == "chrom" else n for n in df.index.names]
        )
    return df
