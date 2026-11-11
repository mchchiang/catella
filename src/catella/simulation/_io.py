# _io.py

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
from catella_cpp import Dump

from catella import h5_utils
from catella.h5_array import H5Array
from catella.simulation.config import SimSettings

# Largest chunk (in elements) of a compressed MCMC run dataset
_MAX_CHUNK_SIZE = 1024

# Physical parameters stored as attrs on the '/params' group of a run file
_PARAM_NAMES = ("nucbp", "nbp", "llink", "elink", "mu")


def _read_attrs(group, names):
    """
    Read the named attrs of an HDF5 group or dataset.

    Parameters
    ----------
    group : h5py.Group or h5py.Dataset
        The object holding the attrs.
    names : iterable of str
        The attr names.

    Returns
    -------
    dict of str to Any
        The value of each named attr.

    Raises
    ------
    KeyError
        If an attr is missing.
    """
    return {name: group.attrs[name] for name in names}


def _pack_jagged(arrays, dtype=np.int32):
    """
    Pack a sequence of 1D arrays into flat/offset form.

    Parameters
    ----------
    arrays : sequence of array_like
        The arrays to pack.
    dtype : data-type, default np.int32
        The data type of the flat array.

    Returns
    -------
    flat : np.ndarray
        All arrays concatenated, of type `dtype`.
    offset : np.ndarray
        A uint64 array of length `len(arrays) + 1`; array `i` is
        `flat[offset[i]:offset[i + 1]]`.
    """
    offset = np.zeros(len(arrays) + 1, dtype=np.uint64)
    offset[1:] = np.cumsum([len(a) for a in arrays])
    if len(arrays) == 0:
        return np.array([], dtype=dtype), offset
    return np.concatenate(arrays).astype(dtype), offset


def _unpack_jagged(flat, offset):
    """
    Split a flat/offset pair back into its arrays.

    Parameters
    ----------
    flat : np.ndarray
        The concatenated values.
    offset : np.ndarray
        The start of each array in `flat`, followed by the total length.

    Returns
    -------
    tuple of np.ndarray
        The `len(offset) - 1` arrays, with the data type of `flat`.
    """
    return tuple(
        np.asarray(flat[offset[i] : offset[i + 1]], dtype=flat.dtype)
        for i in range(len(offset) - 1)
    )


def _write_tm_samples(
    out_file: str | Path,
    *,
    nucbp: int,
    nbp: int,
    llink: int,
    elink: float,
    mu: float,
    logz: float,
    positions: Sequence[np.ndarray],
    energies: Sequence[float],
    seeds: Sequence[int],
) -> None:
    """
    Write a molecule's exact transfer-matrix samples to one HDF5 file.

    Uses the same flat/offset jagged layout already used for MCMC
    position trajectories, but indexed by sample instead of frame.

    Parameters
    ----------
    out_file : str or Path
        Destination HDF5 file.
    nucbp, nbp, llink, elink, mu : int or float
        Physical parameters, stored as attrs on '/params'.
    logz : float
        The log partition function for this molecule.
    positions : sequence of np.ndarray
        The sampled nucleosome start positions, one array per sample.
    energies : sequence of float
        The exact Hamiltonian energy of each sample.
    seeds : sequence of int
        The seed each sample was drawn with.

    Raises
    ------
    ValueError
        If `positions`, `energies`, and `seeds` differ in length.
    """
    if not len(positions) == len(energies) == len(seeds):
        raise ValueError(
            "positions, energies, and seeds must have the same length, "
            f"got {len(positions)}, {len(energies)}, and {len(seeds)}"
        )
    flat, offset = _pack_jagged(positions)
    with h5py.File(out_file, "w") as f:
        gparams = f.create_group("params")
        gparams.attrs["nucbp"] = nucbp
        gparams.attrs["nbp"] = nbp
        gparams.attrs["llink"] = llink
        gparams.attrs["elink"] = elink
        gparams.attrs["mu"] = mu

        gdata = f.create_group("data")
        gdata.attrs["logz"] = logz
        gdata.create_dataset("position_flat", data=flat)
        gdata.create_dataset("position_offset", data=offset)
        gdata.create_dataset("energy", data=np.asarray(energies))
        gdata.create_dataset("seed", data=np.asarray(seeds))


def _write_mcmc_params(
    h5file: h5py.File,
    *,
    nucbp: int,
    nbp: int,
    llink: int,
    elink: float,
    mu: float,
    seed: int,
) -> None:
    """
    Write an MCMC run's parameters as attrs on the '/params' group.

    Integer parameters are stored as int32 and the seed as uint64, matching
    the C++ types of the simulation.

    Parameters
    ----------
    h5file : h5py.File
        The run's HDF5 file, open for writing.
    nucbp, nbp, llink : int
        Nucleosome size, fiber length, and linker length in base pairs.
    elink, mu : float
        Linker repulsion energy and chemical potential.
    seed : int
        The seed of the run.
    """
    g = h5file.create_group("params")
    g.attrs["nucbp"] = np.int32(nucbp)
    g.attrs["nbp"] = np.int32(nbp)
    g.attrs["llink"] = np.int32(llink)
    g.attrs["elink"] = np.float64(elink)
    g.attrs["mu"] = np.float64(mu)
    g.attrs["seed"] = np.uint64(seed)


def _write_mcmc_data(
    h5file: h5py.File, values: Mapping[str, np.ndarray]
) -> None:
    """
    Write an MCMC run's recorded observables to the '/data' group.

    Each observable is stored as a 1D dataset, gzip-compressed (level 4) in
    chunks of at most 1024 elements; empty arrays are stored uncompressed.

    Parameters
    ----------
    h5file : h5py.File
        The run's HDF5 file, open for writing.
    values : mapping of str to np.ndarray
        Observable name to 1D array, as returned by ``Dump.values()``
        (e.g., 'time', 'energy', 'temp', 'position_flat', and
        'position_offset').
    """
    g = h5file.create_group("data")
    for name, arr in values.items():
        arr = np.asarray(arr)
        if len(arr) == 0:
            g.create_dataset(name, data=arr)
        else:
            g.create_dataset(
                name,
                data=arr,
                chunks=(min(len(arr), _MAX_CHUNK_SIZE),),
                compression="gzip",
                compression_opts=4,
            )


def _read_mcmc_file(sim_file):
    """
    Read the parameters and observables of one MCMC run file.

    Parameters
    ----------
    sim_file : str or Path
        The run file, as written by `_write_mcmc_params` and
        `_write_mcmc_data`.

    Returns
    -------
    dict of str to Any
        The parameters ('nucbp', 'nbp', 'llink', 'elink', 'mu', 'seed')
        and observables ('time', 'energy', 'temp' as arrays, 'position'
        as a tuple of arrays, one per frame). A missing 'time', 'energy',
        or 'temp' observable is nan.

    Raises
    ------
    OSError
        If the file cannot be opened.
    KeyError
        If the file lacks a parameter or the position datasets.
    """
    with h5py.File(sim_file, "r") as f:
        out = _read_attrs(f["params"], (*_PARAM_NAMES, "seed"))
        g = f["data"]
        for name in ("time", "energy", "temp"):
            out[name] = g[name][()] if name in g else np.nan
        out["position"] = _unpack_jagged(
            g["position_flat"][()], g["position_offset"][()]
        )
    return out


def _read_tm_file(combined_file):
    """
    Read every sample of a combined per-molecule TM file.

    Parameters
    ----------
    combined_file : str or Path
        The file written by `_write_tm_samples`.

    Returns
    -------
    dict of str to Any
        'params' (dict of the physical parameters), 'logz', 'flat' and
        'offset' (the packed positions), 'energy', and 'seed'.
    """
    with h5py.File(combined_file, "r") as f:
        gdata = f["data"]
        return {
            "params": _read_attrs(f["params"], _PARAM_NAMES),
            "logz": gdata.attrs["logz"],
            "flat": gdata["position_flat"][()],
            "offset": gdata["position_offset"][()],
            "energy": gdata["energy"][()],
            "seed": gdata["seed"][()],
        }


def _tm_sample_fields(arrays, run):
    """
    Select one sample from the contents of a combined TM file.

    Parameters
    ----------
    arrays : dict
        The contents, as returned by `_read_tm_file`.
    run : int
        The sample index.

    Returns
    -------
    dict of str to Any
        The fields of the sample: the physical parameters, 'seed',
        'position', 'energy', and 'logz'.
    """
    offset = arrays["offset"]
    return {
        **arrays["params"],
        "seed": arrays["seed"][run],
        "position": arrays["flat"][offset[run] : offset[run + 1]],
        "energy": arrays["energy"][run],
        "logz": arrays["logz"],
    }


class SimFileMapper:
    """Map simulation identifiers to raw file paths, and back."""

    def __init__(
        self,
        root_dir: str,
        total_mols: int,
        max_mols_per_dir: int = 100,
        max_runs_per_mol: int = 100,
    ):
        self.root_dir = Path(root_dir)
        self.max_mols_per_dir = max_mols_per_dir
        self.max_runs_per_mol = max_runs_per_mol
        self.mol_pad = max(1, int(np.ceil(np.log10(total_mols))))
        self.run_pad = max(1, int(np.ceil(np.log10(max_runs_per_mol))))

    def _shard_and_mol(self, molidx):
        start = int((molidx // self.max_mols_per_dir) * self.max_mols_per_dir)
        end = int(start + self.max_mols_per_dir - 1)
        z = self.mol_pad
        shard_name = f"mol_{start:0{z}d}-{end:0{z}d}"
        return shard_name, f"mol_{int(molidx):0{z}d}"

    def params_to_mcmc_file(self, chrom: str, molidx: int, run: int) -> Path:
        shard_name, mol_name = self._shard_and_mol(molidx)
        run_name = f"run_{int(run):0{self.run_pad}d}"
        sim_h5 = f"{chrom}_{mol_name}_{run_name}.h5"
        return self.root_dir / chrom / shard_name / mol_name / sim_h5

    def params_to_tm_file(self, chrom: str, molidx: int) -> Path:
        # All TM samples of a molecule share one file in the shard directory
        shard_name, mol_name = self._shard_and_mol(molidx)
        return self.root_dir / chrom / shard_name / f"{chrom}_{mol_name}_tm.h5"

    def file_to_params(self, sim_file: str | Path):
        sim_file = Path(sim_file)
        chrom = sim_file.parts[-4]
        match = re.search(r"mol_(\d+)_run_(\d+)\.h5$", sim_file.name)
        if match:
            return chrom, int(match.group(1)), int(match.group(2))
        raise ValueError(f"could not parse parameters from file: {sim_file}")


def _write_settings(group, settings):
    """
    Write SimSettings fields as attrs of `group`.

    None values are skipped (h5py cannot store them), and a mapping
    (per-chromosome `llink`) is stored as a JSON string.

    Parameters
    ----------
    group : h5py.Group
        The group to write to.
    settings : SimSettings
        The settings to write.
    """
    for field in fields(settings):
        value = getattr(settings, field.name)
        if isinstance(value, Mapping):
            value = json.dumps({str(k): int(v) for k, v in value.items()})
        if value is not None:
            group.attrs[field.name] = value


def _read_settings(group):
    """
    Read SimSettings written by `_write_settings`.

    Parameters
    ----------
    group : h5py.Group
        The group holding the settings attrs.

    Returns
    -------
    SimSettings
        The settings.
    """
    attrs = dict(group.attrs.items())
    if isinstance(attrs["llink"], str):
        attrs["llink"] = json.loads(attrs["llink"])
    return SimSettings(**attrs)


def _write_metadata(
    h5file,
    *,
    raw_dir,
    nsim,
    method,
    chroms,
    nmol,
    nbp,
    settings,
    eseq=None,
    out_type=None,
    seed=None,
    seed_table=None,
):
    """
    Write the dataset metadata to the '/metadata' group.

    If the group already exists, only `raw_dir` and `nsim` are updated,
    since the other fields are fixed once a dataset is created (a TM
    dataset may grow from nsim=0 to nsim=N).

    Parameters
    ----------
    h5file : h5py.File
        The dataset file, open for writing.
    raw_dir : Path
        The raw simulation data directory.
    nsim : int
        The number of runs per molecule.
    method : str
        The simulation method, "mcmc" or "tm".
    chroms : sequence of str
        The chromosomes.
    nmol, nbp : mapping of str to int
        The number of molecules and base pairs of each chromosome.
    settings : SimSettings
        The simulation settings.
    eseq : mapping of str to np.ndarray, optional
        The binding energy of each chromosome; not written if None.
    out_type : Dump.OutputType, optional
        The recorded MCMC data; not written if None.
    seed : int, optional
        The master seed; not written if None.
    seed_table : mapping of str to np.ndarray, optional
        The seed of each run; not written if None.
    """
    if "metadata" in h5file:
        gmeta = h5file["metadata"]
        gmeta.attrs["raw_dir"] = str(raw_dir)
        gmeta.attrs["nsim"] = nsim
        return
    gmeta = h5file.create_group("metadata")
    gmeta.attrs["raw_dir"] = str(raw_dir)
    gmeta.attrs["nsim"] = nsim
    gmeta.attrs["method"] = method
    dt = h5py.string_dtype(encoding="utf-8")
    gmeta.create_dataset("chroms", data=chroms, dtype=dt)
    gmeta.create_dataset("nmol", data=np.asarray([nmol[c] for c in chroms]))
    gmeta.create_dataset("nbp", data=np.asarray([nbp[c] for c in chroms]))
    _write_settings(gmeta.create_group("settings"), settings)
    if eseq is not None:
        geseq = gmeta.create_group("eseq")
        for chrom in chroms:
            geseq.create_dataset(chrom, data=eseq[chrom], compression="gzip")
    if out_type is not None:
        gmeta.attrs["out_type"] = int(out_type)
    if seed is not None:
        gmeta.attrs["seed"] = int(seed)
    if seed_table is not None:
        gseeds = gmeta.create_group("seed_table")
        for chrom in chroms:
            gseeds.create_dataset(chrom, data=seed_table[chrom])


def _read_metadata(h5file):
    """
    Read the dataset metadata written by `_write_metadata`.

    Parameters
    ----------
    h5file : h5py.File
        The dataset file, open for reading.

    Returns
    -------
    dict of str to Any
        'raw_dir' (Path, as stored), 'nsim', 'method', 'chroms', 'nmol',
        'nbp', 'settings', 'eseq', 'out_type', 'seed', and 'seed_table'.
        Optional fields that were not written are None; 'method' defaults
        to "mcmc".
    """
    gmeta = h5file["metadata"]
    attrs = gmeta.attrs
    chroms = list(gmeta["chroms"].asstr()[()])

    def per_chrom(name):
        if name not in gmeta:
            return None
        return {chrom: gmeta[name][chrom][()] for chrom in chroms}

    # Old datasets stored the raw directory as 'raw_path'
    raw_dir = attrs["raw_path"] if "raw_path" in attrs else attrs["raw_dir"]
    return {
        "raw_dir": Path(raw_dir),
        "nsim": attrs["nsim"],
        "method": attrs.get("method", "mcmc"),
        "chroms": chroms,
        "nmol": dict(zip(chroms, gmeta["nmol"][()], strict=True)),
        "nbp": dict(zip(chroms, gmeta["nbp"][()], strict=True)),
        "settings": _read_settings(gmeta["settings"]),
        "eseq": per_chrom("eseq"),
        "out_type": (
            Dump.OutputType(int(attrs["out_type"]))
            if "out_type" in attrs
            else None
        ),
        "seed": int(attrs["seed"]) if "seed" in attrs else None,
        "seed_table": per_chrom("seed_table"),
    }


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
        if isinstance(group[name], h5py.Dataset):
            entries[name] = H5Array.load_from(h5_path, f"{prefix}/{name}")
        else:
            entries[name] = h5_utils.load_df(name, group)
    return entries
