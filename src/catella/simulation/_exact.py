# _exact.py

import numpy as np
import pandas as pd
from catella_cpp import NucPosTM

from catella.h5_array import H5Array
from catella.simulation._io import _pack_jagged
from catella.simulation.config import SimSettings


def _require_eseq(dataset):
    """
    Check that `dataset` stores the binding energy of each molecule.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to check.

    Raises
    ------
    ValueError
        If the dataset has no stored energy landscape.
    """
    if not dataset.has_eseq:
        raise ValueError(
            "exact computation requires the per-molecule energy "
            "landscape; recreate the dataset with store_eseq=True"
        )


def _require_tm(dataset, caller):
    """
    Check that `dataset` supports exact transfer-matrix analysis.

    Parameters
    ----------
    dataset : SimDataset
        The dataset to check.
    caller : str
        The name of the calling method, used in the error message.

    Raises
    ------
    ValueError
        If `dataset.method != "tm"`, or the dataset has no stored energy
        landscape.
    """
    if dataset.method != "tm":
        raise ValueError(f"{caller} requires a method='tm' dataset")
    _require_eseq(dataset)


def _reject_time(time, record_time):
    """
    Check that no time point is requested for a TM dataset.

    Parameters
    ----------
    time : int or None
        The requested time point.
    record_time : bool
        Whether the time point was to be added to the result name.

    Raises
    ------
    ValueError
        If `time` is given or `record_time` is True.
    """
    if time is not None or record_time:
        raise ValueError(
            "'time'/'record_time' are not meaningful for method='tm' "
            "datasets (no time axis)"
        )


def _solved_tm(dataset, chrom, mol):
    """
    Build and solve the transfer matrix of one molecule.

    The model is solved at `settings.end_temp`, the temperature of the
    exact equilibrium distribution.

    Parameters
    ----------
    dataset : SimDataset
        The dataset holding the settings and binding energy.
    chrom : str
        The chromosome of the molecule.
    mol : int
        The molecule index.

    Returns
    -------
    NucPosTM
        The solved model.
    """
    settings = SimSettings(**dict(dataset.settings)).for_chrom(chrom)
    tm = NucPosTM(
        settings.nucbp,
        dataset.nbp[chrom],
        settings.llink,
        settings.elink,
        settings.mu,
        1,
    )
    tm.setEnergy(dataset.eseq[chrom][mol].tolist(), settings.emax)
    tm.solve(settings.end_temp)
    return tm


def _tm_per_mol(dataset, chroms, name, getter):
    """
    Store one exact value per molecule as a DataFrame.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom][name]`.
    chroms : iterable of str
        The chromosomes to process.
    name : str
        The key of the result.
    getter : callable
        Called as `getter(tm)` on each solved model; returns a float.
    """
    for chrom in chroms:
        nmol = dataset.nmol[chrom]
        values = np.empty(nmol, dtype=np.float64)
        for mol in range(nmol):
            values[mol] = getter(_solved_tm(dataset, chrom, mol))
        dataset.analysis[chrom][name] = pd.DataFrame(values)


def _tm_per_bp(dataset, chroms, name, batch_size, getter):
    """
    Store one exact value per molecule and base pair as an H5Array.

    Molecules are written in batches so that peak memory scales with
    `batch_size`.

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom][name]`.
    chroms : iterable of str
        The chromosomes to process.
    name : str
        The key of the result.
    batch_size : int
        The number of molecules held in memory at a time.
    getter : callable
        Called as `getter(tm)` on each solved model; returns one value
        per base pair.
    """
    tmp_dir = dataset.resolve_tmp_dir()
    for chrom in chroms:
        nmol = dataset.nmol[chrom]
        nbp = dataset.nbp[chrom]
        out = H5Array.create((nmol, nbp), dtype=np.float64, dir=tmp_dir)
        for start in range(0, nmol, batch_size):
            stop = min(start + batch_size, nmol)
            block = np.empty((stop - start, nbp), dtype=np.float64)
            for i, mol in enumerate(range(start, stop)):
                block[i] = getter(_solved_tm(dataset, chrom, mol))
            out.write_batch(start, stop, block)
        dataset.analysis[chrom][name] = out


def _tm_configs(dataset, chroms, name, get_config):
    """
    Store one nucleosome configuration per molecule.

    For each chromosome, writes `{name}_position_flat` and
    `{name}_position_offset` (the packed positions) and `{name}_energy`
    (the energy and log posterior probability of each configuration).

    Parameters
    ----------
    dataset : SimDataset
        The TM dataset; results go to `dataset.analysis[chrom]`.
    chroms : iterable of str
        The chromosomes to process.
    name : str
        The prefix of the result keys.
    get_config : callable
        Called as `get_config(tm)` on each solved model; returns
        `(positions, energy)`.
    """
    end_temp = SimSettings(**dict(dataset.settings)).end_temp
    for chrom in chroms:
        nmol = dataset.nmol[chrom]
        positions = []
        energy = np.empty(nmol, dtype=np.float64)
        log_prob = np.empty(nmol, dtype=np.float64)
        for mol in range(nmol):
            tm = _solved_tm(dataset, chrom, mol)
            pos, energy[mol] = get_config(tm)
            positions.append(np.asarray(pos, dtype=np.int32))
            log_prob[mol] = -energy[mol] / end_temp - tm.getLogZ()
        flat, offset = _pack_jagged(positions)
        analysis = dataset.analysis[chrom]
        analysis[f"{name}_position_flat"] = pd.DataFrame({"position": flat})
        analysis[f"{name}_position_offset"] = pd.DataFrame({"offset": offset})
        analysis[f"{name}_energy"] = pd.DataFrame(
            {"energy": energy, "log_prob": log_prob}
        )
