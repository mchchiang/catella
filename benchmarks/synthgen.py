# synthgen.py

from pathlib import Path

import numpy as np
import pandas as pd

_TRUTH_COLS = ["mol_id", "chrom", "start", "end"]


def place_nucleosomes(
    nbp: int,
    nmol: int,
    *,
    chrom: str = "chr1",
    lnuc: int = 147,
    linker_mean: float = 40.0,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Place nucleosomes on synthetic molecules with random linkers.

    Each molecule spans `[0, nbp)` on `chrom`. Starting from position 0,
    linker lengths are drawn from an exponential distribution (rounded
    to integers) and nucleosome footprints of length `lnuc` are placed
    between them until no further footprint fits.

    Parameters
    ----------
    nbp : int
        Length of each molecule in bp.
    nmol : int
        Number of molecules.
    chrom : str, default "chr1"
        Chromosome name assigned to every molecule.
    lnuc : int, default 147
        Nucleosome footprint length (bp).
    linker_mean : float, default 40.0
        Mean linker length (bp).
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        Ground-truth table with columns `mol_id`, `chrom`, `start`,
        `end` (0-based, half-open), sorted by molecule then start.
    """
    if nbp <= 0:
        raise ValueError(f"nbp must be positive, got {nbp}")
    if nmol <= 0:
        raise ValueError(f"nmol must be positive, got {nmol}")
    if lnuc <= 0:
        raise ValueError(f"lnuc must be positive, got {lnuc}")
    if linker_mean < 0:
        raise ValueError(
            f"linker_mean must be non-negative, got {linker_mean}"
        )
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(nmol):
        mol_id = f"mol{i}"
        pos = int(round(rng.exponential(linker_mean)))
        while pos + lnuc <= nbp:
            rows.append((mol_id, chrom, pos, pos + lnuc))
            pos += lnuc + int(round(rng.exponential(linker_mean)))
    nucs = pd.DataFrame(rows, columns=_TRUTH_COLS)
    return nucs.astype({"start": "int64", "end": "int64"})


def write_truth_bed(nucs: pd.DataFrame, path: str | Path) -> Path:
    """
    Write ground-truth nucleosome positions as a BED file.

    Parameters
    ----------
    nucs : pandas.DataFrame
        Table with columns `mol_id`, `chrom`, `start`, `end`, e.g., from
        `place_nucleosomes`.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    nucs[["chrom", "start", "end", "mol_id"]].to_csv(
        path, sep="\t", header=False, index=False
    )
    return path
