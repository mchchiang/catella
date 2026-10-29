# synthgen.py

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_TRUTH_COLS = ["mol_id", "chrom", "start", "end"]
_CALL_COLS = ["mol_id", "chrom", "pos", "strand", "mod_qual"]
_VALID_MTASE = ("A", "CG", "GC")


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
        pos = round(rng.exponential(linker_mean))
        while pos + lnuc <= nbp:
            rows.append((mol_id, chrom, pos, pos + lnuc))
            pos += lnuc + round(rng.exponential(linker_mean))
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


def random_sequence(nbp: int, *, seed: int | None = None) -> str:
    """
    Draw a random DNA sequence with uniform base composition.

    Parameters
    ----------
    nbp : int
        Sequence length in bp.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    str
        Upper-case sequence over "ACGT" of length `nbp`.
    """
    if nbp <= 0:
        raise ValueError(f"nbp must be positive, got {nbp}")
    rng = np.random.default_rng(seed)
    return "".join(rng.choice(list("ACGT"), size=nbp))


def eligible_sites(seq: str, mtase: str, strand: str) -> np.ndarray:
    """
    Positions methylatable by `mtase` on a molecule of a given strand.

    Positions are reported in forward reference coordinates at the
    modified base, following the ModKit convention: for "A", A on the
    "+" strand and T on the "-" strand; for "CG"/"GC", the C of the
    dinucleotide on the given strand.

    Parameters
    ----------
    seq : str
        Forward reference sequence.
    mtase : {"A", "CG", "GC"}
        Methyltransferase context.
    strand : {"+", "-"}
        Strand of the molecule.

    Returns
    -------
    np.ndarray of int64
        Sorted 0-based eligible positions.
    """
    if mtase not in _VALID_MTASE:
        raise ValueError(
            f"mtase must be one of {_VALID_MTASE}, got {mtase!r}"
        )
    if strand not in ("+", "-"):
        raise ValueError(f"strand must be '+' or '-', got {strand!r}")
    arr = np.frombuffer(seq.upper().encode(), dtype="S1")
    if mtase == "A":
        base = b"A" if strand == "+" else b"T"
        return np.flatnonzero(arr == base).astype(np.int64)
    dinuc = np.flatnonzero((arr[:-1] == b"C") & (arr[1:] == b"G"))
    if mtase == "GC":
        dinuc = np.flatnonzero((arr[:-1] == b"G") & (arr[1:] == b"C"))
    # CG: + strand C at offset 0, - strand C at offset 1 (GC: opposite)
    offset = int((mtase == "CG") != (strand == "+"))
    return (dinuc + offset).astype(np.int64)


def simulate_calls(
    seq: str,
    nucs: pd.DataFrame,
    mol_ids: Sequence[str] | None = None,
    *,
    chrom: str = "chr1",
    mtase: str = "A",
    p_linker: float = 0.7,
    p_nuc: float = 0.05,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate per-site methylation calls from nucleosome positions.

    Each molecule is assigned the "+" or "-" strand with equal
    probability. Every site eligible for `mtase` on that strand is
    methylated with probability `p_nuc` if it lies inside a nucleosome
    and `p_linker` otherwise. Methylated sites get a call score
    `mod_qual` drawn uniformly from [0.5, 1) and unmethylated sites
    from [0, 0.5).

    Parameters
    ----------
    seq : str
        Forward reference sequence shared by all molecules.
    nucs : pandas.DataFrame
        Nucleosome positions with columns `mol_id`, `start`, `end`
        (0-based, half-open), e.g., from `place_nucleosomes`.
    mol_ids : sequence of str, optional
        Molecules to simulate. If None (default), the unique `mol_id`
        values in `nucs`, in order of appearance. Molecules absent from
        `nucs` are nucleosome free.
    chrom : str, default "chr1"
        Chromosome name written to the output.
    mtase : {"A", "CG", "GC"}, default "A"
        Methyltransferase context.
    p_linker : float, default 0.7
        Methylation probability of an eligible site in linker DNA.
    p_nuc : float, default 0.05
        Methylation probability of an eligible site inside a
        nucleosome.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        One row per eligible site per molecule, with columns `mol_id`,
        `chrom`, `pos`, `strand`, `mod_qual`, sorted by molecule then
        position.
    """
    for name, p in (("p_linker", p_linker), ("p_nuc", p_nuc)):
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {p}")
    nbp = len(seq)
    if len(nucs) and (nucs["start"].min() < 0 or nucs["end"].max() > nbp):
        raise ValueError(
            f"nucleosome positions must lie within [0, {nbp}]"
        )
    if mol_ids is None:
        mol_ids = list(dict.fromkeys(nucs["mol_id"]))
    rng = np.random.default_rng(seed)
    sites = {s: eligible_sites(seq, mtase, s) for s in ("+", "-")}
    by_mol = dict(tuple(nucs.groupby("mol_id", sort=False)))
    frames = []
    for mol_id in mol_ids:
        strand = "+" if rng.random() < 0.5 else "-"
        pos = sites[strand]
        occupied = np.zeros(nbp, dtype=bool)
        if mol_id in by_mol:
            grp = by_mol[mol_id]
            for start, end in zip(grp["start"], grp["end"]):
                occupied[start:end] = True
        prob = np.where(occupied[pos], p_nuc, p_linker)
        meth = rng.random(len(pos)) < prob
        qual = 0.5 * rng.random(len(pos)) + 0.5 * meth
        frames.append(
            pd.DataFrame(
                {
                    "mol_id": mol_id,
                    "chrom": chrom,
                    "pos": pos,
                    "strand": strand,
                    "mod_qual": qual,
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=_CALL_COLS)
    return pd.concat(frames, ignore_index=True)
