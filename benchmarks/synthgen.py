# synthgen.py

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

_TRUTH_COLS = ["mol_id", "chrom", "start", "end"]
_CALL_COLS = ["mol_id", "chrom", "pos", "strand", "mod_qual"]
_VALID_MTASE = ("A", "CG", "GC")
# ModKit column name for each call column, in ModKit order
_MODKIT_COLS = {
    "mol_id": "read_id",
    "pos": "ref_position",
    "chrom": "chrom",
    "strand": "ref_strand",
    "mod_qual": "mod_qual",
}
_MOD_CODE = {"A": "a", "CG": "m", "GC": "m"}
_COMPLEMENT = {"A": "T", "C": "G", "G": "C", "T": "A"}


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


def write_chromsize(sizes: dict[str, int], path: str | Path) -> Path:
    """
    Write a chromosome-size file with a `chrom`/`length` header.

    Parameters
    ----------
    sizes : dict of str to int
        Chromosome name -> length (bp).
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    with open(path, "w") as fh:
        fh.write("chrom\tlength\n")
        fh.writelines(f"{chrom}\t{nbp}\n" for chrom, nbp in sizes.items())
    return path


def write_fasta(
    seqs: dict[str, str], path: str | Path, *, width: int = 60
) -> Path:
    """
    Write sequences as a multi-FASTA file.

    Parameters
    ----------
    seqs : dict of str to str
        Record id -> sequence.
    path : str or Path
        Output file path.
    width : int, default 60
        Number of bases per line.

    Returns
    -------
    Path
        The path written.
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    path = Path(path)
    with open(path, "w") as fh:
        for name, seq in seqs.items():
            fh.write(f">{name}\n")
            fh.writelines(
                f"{seq[i : i + width]}\n" for i in range(0, len(seq), width)
            )
    return path


def write_modkit_tsv(
    calls: pd.DataFrame, path: str | Path, *, mtase: str = "A"
) -> Path:
    """
    Write methylation calls in ModKit `extract full` column format.

    Only the six columns read by catella's `load_raw` are written,
    under ModKit's own header names: `read_id`, `ref_position`,
    `chrom`, `ref_strand`, `mod_qual`, `mod_code`.

    Parameters
    ----------
    calls : pandas.DataFrame
        Calls with columns `mol_id`, `chrom`, `pos`, `strand`,
        `mod_qual`, e.g., from `simulate_calls`.
    path : str or Path
        Output file path (tab-separated, with header).
    mtase : {"A", "CG", "GC"}, default "A"
        Methyltransferase context; sets `mod_code` to "a" (6mA) for
        "A" and "m" (5mC) otherwise.

    Returns
    -------
    Path
        The path written.
    """
    if mtase not in _VALID_MTASE:
        raise ValueError(
            f"mtase must be one of {_VALID_MTASE}, got {mtase!r}"
        )
    path = Path(path)
    out = calls[list(_MODKIT_COLS)].rename(columns=_MODKIT_COLS)
    out["mod_code"] = _MOD_CODE[mtase]
    out.to_csv(path, sep="\t", index=False)
    return path


def _mm_ml(seq, pos, qual, base, strand):
    """
    MM skip counts and ML bytes for one read, in read orientation.

    Parameters
    ----------
    seq : str
        Forward reference sequence the read is aligned to.
    pos : np.ndarray of int
        Called reference positions (each a `base` on the read strand).
    qual : np.ndarray of float
        Call scores in [0, 1], parallel to `pos`.
    base : {"A", "C"}
        Unmodified base on the read strand.
    strand : {"+", "-"}
        Read strand.

    Returns
    -------
    skips : np.ndarray of int64
    ml : np.ndarray of int64
    """
    ref_base = base if strand == "+" else _COMPLEMENT[base]
    arr = np.frombuffer(seq.upper().encode(), dtype="S1")
    cand = np.flatnonzero(arr == ref_base.encode())
    order = np.argsort(pos)
    idx = np.searchsorted(cand, pos[order])
    # - strand reads run 3' to 5' along the reference
    if strand == "-":
        order = order[::-1]
        idx = len(cand) - 1 - idx[::-1]
    skips = np.diff(idx, prepend=-1) - 1
    ml = np.minimum(np.floor(qual[order] * 256), 255).astype(np.int64)
    return skips.astype(np.int64), ml


def write_sam(
    calls: pd.DataFrame,
    seqs: dict[str, str],
    path: str | Path,
    *,
    mtase: str = "A",
) -> Path:
    """
    Write methylation calls as a SAM file with MM/ML base-mod tags.

    Each molecule becomes one read aligned end to end to its
    chromosome (POS 1, CIGAR `<len>M`), with SEQ set to the forward
    reference sequence and FLAG 16 for "-" strand molecules. Calls are
    encoded in the `MM`/`ML` tags (SAM specification, "?" mode) in the
    original read orientation, so they can be read by base-mod-aware
    tools such as fibertools. Convert to BAM with `samtools view -b`.

    Parameters
    ----------
    calls : pandas.DataFrame
        Calls with columns `mol_id`, `chrom`, `pos`, `strand`,
        `mod_qual`, e.g., from `simulate_calls`. Every call position
        must be an eligible site for `mtase` on the molecule's strand.
    seqs : dict of str to str
        Chromosome name -> forward reference sequence.
    path : str or Path
        Output file path.
    mtase : {"A", "CG", "GC"}, default "A"
        Methyltransferase context; sets the MM code to "A+a" (6mA) for
        "A" and "C+m" (5mC) otherwise.

    Returns
    -------
    Path
        The path written.
    """
    if mtase not in _VALID_MTASE:
        raise ValueError(
            f"mtase must be one of {_VALID_MTASE}, got {mtase!r}"
        )
    base = "A" if mtase == "A" else "C"
    code = _MOD_CODE[mtase]
    path = Path(path)
    with open(path, "w") as fh:
        fh.write("@HD\tVN:1.6\tSO:unsorted\n")
        fh.writelines(
            f"@SQ\tSN:{chrom}\tLN:{len(seq)}\n"
            for chrom, seq in seqs.items()
        )
        for mol_id, grp in calls.groupby("mol_id", sort=False):
            chrom = grp["chrom"].iloc[0]
            strand = grp["strand"].iloc[0]
            seq = seqs[chrom]
            skips, ml = _mm_ml(
                seq,
                grp["pos"].to_numpy(),
                grp["mod_qual"].to_numpy(),
                base,
                strand,
            )
            flag = 0 if strand == "+" else 16
            mm = "".join(f",{s}" for s in skips)
            ml_str = "".join(f",{v}" for v in ml)
            fh.write(
                f"{mol_id}\t{flag}\t{chrom}\t1\t60\t{len(seq)}M\t*\t0\t0\t"
                f"{seq}\t*\tMM:Z:{base}+{code}?{mm};\tML:B:C{ml_str}\n"
            )
    return path
