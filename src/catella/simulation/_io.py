# _io.py

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

import h5py
import numpy as np
from catella_cpp import Dump

from catella._jagged import _pack_jagged, _unpack_jagged
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

    def params_to_mcmc_file(self, region: str, molidx: int, run: int) -> Path:
        shard_name, mol_name = self._shard_and_mol(molidx)
        run_name = f"run_{int(run):0{self.run_pad}d}"
        sim_h5 = f"{region}_{mol_name}_{run_name}.h5"
        return self.root_dir / region / shard_name / mol_name / sim_h5

    def params_to_tm_file(self, region: str, molidx: int) -> Path:
        # All TM samples of a molecule share one file in the shard directory
        shard_name, mol_name = self._shard_and_mol(molidx)
        tm_h5 = f"{region}_{mol_name}_tm.h5"
        return self.root_dir / region / shard_name / tm_h5

    def file_to_params(self, sim_file: str | Path):
        sim_file = Path(sim_file)
        region = sim_file.parts[-4]
        match = re.search(r"mol_(\d+)_run_(\d+)\.h5$", sim_file.name)
        if match:
            return region, int(match.group(1)), int(match.group(2))
        raise ValueError(f"could not parse parameters from file: {sim_file}")


def _write_settings(group, settings):
    """
    Write SimSettings fields as attrs of `group`.

    None values are skipped (h5py cannot store them), and a mapping
    (per-region `llink`) is stored as a JSON string.

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
    regions,
    nmol,
    nbp,
    settings,
    eseq=None,
    out_type=None,
    seed=None,
    seed_table=None,
    no_data=None,
    nan_fill=None,
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
    regions : sequence of str
        The regions.
    nmol, nbp : mapping of str to int
        The number of molecules and base pairs of each region.
    settings : SimSettings
        The simulation settings.
    eseq : mapping of str to np.ndarray, optional
        The binding energy of each region; not written if None.
    out_type : Dump.OutputType, optional
        The recorded MCMC data; not written if None.
    seed : int, optional
        The master seed; not written if None.
    seed_table : mapping of str to np.ndarray, optional
        The seed of each run; not written if None.
    no_data : mapping of str to np.ndarray, optional
        bool, True for each molecule skipped for having no data; not
        written if None.
    nan_fill : "mean" or float, optional
        How partly-nan molecules were filled; not written if None.
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
    gmeta.create_dataset("chroms", data=regions, dtype=dt)
    gmeta.create_dataset("nmol", data=np.asarray([nmol[r] for r in regions]))
    gmeta.create_dataset("nbp", data=np.asarray([nbp[r] for r in regions]))
    _write_settings(gmeta.create_group("settings"), settings)
    if eseq is not None:
        geseq = gmeta.create_group("eseq")
        for region in regions:
            geseq.create_dataset(region, data=eseq[region], compression="gzip")
    if out_type is not None:
        gmeta.attrs["out_type"] = int(out_type)
    if seed is not None:
        gmeta.attrs["seed"] = int(seed)
    if nan_fill is not None:
        gmeta.attrs["nan_fill"] = nan_fill
    if seed_table is not None:
        gseeds = gmeta.create_group("seed_table")
        for region in regions:
            gseeds.create_dataset(region, data=seed_table[region])
    if no_data is not None:
        gno = gmeta.create_group("no_data")
        for region in regions:
            gno.create_dataset(region, data=np.asarray(no_data[region], bool))


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
        'raw_dir' (Path, as stored), 'nsim', 'method', 'regions', 'nmol',
        'nbp', 'settings', 'eseq', 'out_type', 'seed', 'seed_table',
        'no_data', and 'nan_fill'.
        Optional fields that were not written are None; 'method' defaults
        to "mcmc".
    """
    gmeta = h5file["metadata"]
    attrs = gmeta.attrs
    regions = list(gmeta["chroms"].asstr()[()])

    def per_region(name):
        if name not in gmeta:
            return None
        return {region: gmeta[name][region][()] for region in regions}

    # Old datasets stored the raw directory as 'raw_path'
    raw_dir = attrs["raw_path"] if "raw_path" in attrs else attrs["raw_dir"]
    return {
        "raw_dir": Path(raw_dir),
        "nsim": attrs["nsim"],
        "method": attrs.get("method", "mcmc"),
        "regions": regions,
        "nmol": dict(zip(regions, gmeta["nmol"][()], strict=True)),
        "nbp": dict(zip(regions, gmeta["nbp"][()], strict=True)),
        "settings": _read_settings(gmeta["settings"]),
        "eseq": per_region("eseq"),
        "out_type": (
            Dump.OutputType(int(attrs["out_type"]))
            if "out_type" in attrs
            else None
        ),
        "seed": int(attrs["seed"]) if "seed" in attrs else None,
        "seed_table": per_region("seed_table"),
        "no_data": per_region("no_data"),
        "nan_fill": _read_nan_fill(attrs),
    }


def _read_nan_fill(attrs):
    """Return the stored `nan_fill` ("mean", a float, or None)."""
    if "nan_fill" not in attrs:
        return None
    value = attrs["nan_fill"]
    if isinstance(value, bytes):
        value = value.decode()
    return value if isinstance(value, str) else float(value)
