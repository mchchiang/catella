# synthgen.py

from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

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
    lnuc_range: tuple[int, int] | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Place nucleosomes on synthetic molecules with random linkers.

    Each molecule spans `[0, nbp)` on `chrom`. Starting from position 0,
    linker lengths are drawn from an exponential distribution (rounded
    to integers) and nucleosome footprints of length `lnuc` (or of a
    random length in `lnuc_range`) are placed between them until no
    further footprint fits.

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
    lnuc_range : tuple of (int, int), optional
        If given, `(lo, hi)`: each nucleosome footprint length is drawn
        uniformly from the integers in `[lo, hi]`, overriding `lnuc`.
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
    if lnuc_range is not None and not 0 < lnuc_range[0] <= lnuc_range[1]:
        raise ValueError(
            f"lnuc_range must satisfy 0 < lo <= hi, got {lnuc_range}"
        )
    rng = np.random.default_rng(seed)

    def draw_len():
        if lnuc_range is None:
            return lnuc
        return int(rng.integers(lnuc_range[0], lnuc_range[1] + 1))

    rows = []
    for i in range(nmol):
        mol_id = f"test_{i}"
        pos = round(rng.exponential(linker_mean))
        length = draw_len()
        while pos + length <= nbp:
            rows.append((mol_id, chrom, pos, pos + length))
            pos += length + round(rng.exponential(linker_mean))
            length = draw_len()
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
    with open(path, "w") as fh:
        fh.write("#")
        nucs[["chrom", "start", "end", "mol_id"]].to_csv(
            fh, sep="\t", index=False
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
    p_nuc: float = 0.02,
    signal_sep: float = 2.0,
    report_sep: float | None = None,
    double_strand: bool = False,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Simulate per-site methylation calls from nucleosome positions.

    Each molecule's read is aligned to the "+" or "-" strand with equal
    probability, and has calls at the eligible sites of that strand, or
    of both strands if `double_strand`. Every called site is methylated
    with probability `p_nuc` if it lies inside a nucleosome and
    `p_linker` otherwise. Each site's call score `mod_qual` is the
    probability that it is methylated given a simulated raw signal
    (see `signal_sep`).

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
    p_nuc : float, default 0.02
        Methylation probability of an eligible site inside a
        nucleosome.
    signal_sep : float, default 2.0
        Signal separation s. Each site has a raw signal x ~ Normal(s, 1)
        if methylated and Normal(-s, 1) otherwise, and `mod_qual` is
        P(methylated | x) under a 50:50 prior,
        1 / (1 + exp(-2 * s * x)). A fraction Phi(-s) of sites (Phi the
        standard normal CDF) are scored on the wrong side of 0.5.
    report_sep : float, optional
        Signal separation the simulated caller assumes when converting x
        to `mod_qual`, i.e., 1 / (1 + exp(-2 * report_sep * x)). If None
        (default), equal to `signal_sep` (calibrated scores); larger
        values give overconfident scores and smaller values
        underconfident ones.
    double_strand : bool, default False
        If True, each read has calls on both strands, as in PacBio
        Fiber-seq; otherwise only on the strand it is aligned to, as in
        Nanopore.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    pandas.DataFrame
        One row per called site per molecule, with columns `mol_id`,
        `chrom`, `pos`, `strand` (the strand the molecule's read is
        aligned to, the same for all its rows), `mod_qual`, sorted by
        molecule then position.
    """
    for name, p in (("p_linker", p_linker), ("p_nuc", p_nuc)):
        if not 0 <= p <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {p}")
    if signal_sep <= 0:
        raise ValueError(f"signal_sep must be positive, got {signal_sep}")
    if report_sep is None:
        report_sep = signal_sep
    if report_sep <= 0:
        raise ValueError(f"report_sep must be positive, got {report_sep}")
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
        read_strand = "+" if rng.random() < 0.5 else "-"
        strands = ("+", "-") if double_strand else (read_strand,)
        pos = np.sort(np.concatenate([sites[s] for s in strands]))
        occupied = np.zeros(nbp, dtype=bool)
        if mol_id in by_mol:
            grp = by_mol[mol_id]
            for start, end in zip(grp["start"], grp["end"]):
                occupied[start:end] = True
        prob = np.where(occupied[pos], p_nuc, p_linker)
        meth = rng.random(len(pos)) < prob
        x = rng.normal(np.where(meth, signal_sep, -signal_sep), 1.0)
        qual = 1 / (1 + np.exp(-2 * report_sep * x))
        frames.append(
            pd.DataFrame(
                {
                    "mol_id": mol_id,
                    "chrom": chrom,
                    "pos": pos,
                    "strand": read_strand,
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
        Called reference positions (each a `base` in the original read).
    qual : np.ndarray of float
        Call scores in [0, 1], parallel to `pos`.
    base : {"A", "C", "T", "G"}
        Base counted by the MM skip counts, as it appears in the
        original read.
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
    reference sequence and FLAG 16 for reads aligned to the "-"
    strand. The strand of each call is given by the reference base at
    its position (e.g., A for "+" and T for "-" with 6mA). Calls on the
    read's own strand are encoded as same-strand modifications (e.g.,
    "A+a") and calls on the other strand as opposite-strand ones
    (e.g., "T-a"), as in PacBio Fiber-seq, in the `MM`/`ML` tags (SAM
    specification, "?" mode) in the original read orientation, so
    they can be read by base-mod-aware tools such as fibertools.
    Convert to BAM with `samtools view -b`.

    Parameters
    ----------
    calls : pandas.DataFrame
        Calls with columns `mol_id`, `chrom`, `pos`, `strand` (strand
        the read is aligned to), `mod_qual`, e.g., from
        `simulate_calls`. Every call position must be an eligible site
        for `mtase` on either strand.
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
            seq = seqs[chrom]
            pos = grp["pos"].to_numpy()
            qual = grp["mod_qual"].to_numpy()
            strand = grp["strand"].iloc[0]
            flag = 0 if strand == "+" else 16
            ref = np.frombuffer(seq.upper().encode(), dtype="S1")[pos]
            # A call is on the read's own strand if its base matches it
            own = (ref == base.encode()) == (strand == "+")
            comp = _COMPLEMENT[base]
            entries = [
                (f"{base}+{code}?", own, base),
                (f"{comp}-{code}?", ~own, comp),
            ]
            mm, ml_str = "", ""
            for tag, sel, b in entries:
                if not sel.any():
                    continue
                skips, ml = _mm_ml(seq, pos[sel], qual[sel], b, strand)
                mm += tag + "".join(f",{v}" for v in skips) + ";"
                ml_str += "".join(f",{v}" for v in ml)
            fh.write(
                f"{mol_id}\t{flag}\t{chrom}\t1\t60\t{len(seq)}M\t*\t0\t0\t"
                f"{seq}\t*\tMM:Z:{mm}\tML:B:C{ml_str}\n"
            )
    return path


def make_dataset(
    outdir: str | Path,
    *,
    prefix: str = "synth",
    nbp: int = 5000,
    nmol: int = 100,
    chrom: str = "chr1",
    mtase: str = "A",
    nucs: pd.DataFrame | None = None,
    lnuc: int = 147,
    linker_mean: float = 40.0,
    lnuc_range: tuple[int, int] | None = None,
    p_linker: float = 0.7,
    p_nuc: float = 0.02,
    signal_sep: float = 2.0,
    report_sep: float | None = None,
    double_strand: bool = False,
    controls: bool = False,
    seed: int | None = None,
) -> dict[str, Path]:
    """
    Generate a synthetic footprinting dataset and write it to disk.

    Draws a random reference sequence, places nucleosomes (unless
    `nucs` is given), simulates methylation calls, and writes the same
    calls for catella (ModKit TSV) and for base-mod-aware tools such as
    fibertools (SAM with MM/ML tags), along with the ground truth.
    Optionally also writes a methylated control (naked DNA treated with
    the methyltransferase: no nucleosomes, methylation probability
    `p_linker` at every site) and an unmethylated control (untreated
    DNA: methylation probability 0), each with the same number of
    molecules as the test data.

    Parameters
    ----------
    outdir : str or Path
        Output directory; created if missing.
    prefix : str, default "synth"
        File name prefix.
    nbp : int, default 5000
        Reference (and molecule) length in bp.
    nmol : int, default 100
        Number of molecules; ignored if `nucs` is given.
    chrom : str, default "chr1"
        Chromosome name.
    mtase : {"A", "CG", "GC"}, default "A"
        Methyltransferase context.
    nucs : pandas.DataFrame, optional
        User-supplied nucleosome positions with columns `mol_id`,
        `start`, `end`. If None (default), drawn with
        `place_nucleosomes`.
    lnuc : int, default 147
        Nucleosome footprint length (bp); used only if `nucs` is None.
    linker_mean : float, default 40.0
        Mean linker length (bp); used only if `nucs` is None.
    lnuc_range : tuple of (int, int), optional
        Range of random footprint lengths (bp); see `place_nucleosomes`.
        Used only if `nucs` is None.
    p_linker : float, default 0.7
        Methylation probability of an eligible site in linker DNA.
    p_nuc : float, default 0.02
        Methylation probability of an eligible site inside a
        nucleosome.
    signal_sep : float, default 2.0
        Signal separation for the call scores; see `simulate_calls`.
    report_sep : float, optional
        Signal separation assumed by the simulated caller; see
        `simulate_calls`. Applies to the controls too.
    double_strand : bool, default False
        If True, read every molecule on both strands; see
        `simulate_calls`.
    controls : bool, default False
        If True, also write the methylated and unmethylated controls.
    seed : int, optional
        Seed for the random number generator.

    Returns
    -------
    dict of str to Path
        Paths written, keyed by "chromsize", "fasta", "modkit", "sam",
        and "truth", plus "meth_modkit", "meth_sam", "unmeth_modkit",
        and "unmeth_sam" if `controls`.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    seq_seed, nuc_seed, call_seed, meth_seed, unmeth_seed = (
        np.random.default_rng(seed).integers(2**32, size=5)
    )
    seq = random_sequence(nbp, seed=seq_seed)
    if nucs is None:
        nucs = place_nucleosomes(
            nbp,
            nmol,
            chrom=chrom,
            lnuc=lnuc,
            linker_mean=linker_mean,
            lnuc_range=lnuc_range,
            seed=nuc_seed,
        )
    else:
        nucs = nucs.assign(chrom=chrom)[_TRUTH_COLS]
    calls = simulate_calls(
        seq,
        nucs,
        chrom=chrom,
        mtase=mtase,
        p_linker=p_linker,
        p_nuc=p_nuc,
        signal_sep=signal_sep,
        report_sep=report_sep,
        double_strand=double_strand,
        seed=call_seed,
    )
    stem = outdir / prefix
    paths = {
        "chromsize": write_chromsize({chrom: nbp}, f"{stem}.size"),
        "fasta": write_fasta({chrom: seq}, f"{stem}.fa"),
        "modkit": write_modkit_tsv(calls, f"{stem}.tsv", mtase=mtase),
        "sam": write_sam(calls, {chrom: seq}, f"{stem}.sam", mtase=mtase),
        "truth": write_truth_bed(nucs, f"{stem}.truth.bed"),
    }
    if controls:
        no_nucs = pd.DataFrame(columns=_TRUTH_COLS)
        ncontrol = nucs["mol_id"].nunique()
        for name, rate, ctrl_seed in (
            ("meth", p_linker, meth_seed),
            ("unmeth", 0.0, unmeth_seed),
        ):
            ctrl = simulate_calls(
                seq,
                no_nucs,
                [f"{name}_{i}" for i in range(ncontrol)],
                chrom=chrom,
                mtase=mtase,
                p_linker=rate,
                signal_sep=signal_sep,
                report_sep=report_sep,
                double_strand=double_strand,
                seed=ctrl_seed,
            )
            paths[f"{name}_modkit"] = write_modkit_tsv(
                ctrl, f"{stem}.{name}.tsv", mtase=mtase
            )
            paths[f"{name}_sam"] = write_sam(
                ctrl, {chrom: seq}, f"{stem}.{name}.sam", mtase=mtase
            )
    return paths


app = typer.Typer(help="Synthetic footprinting data generator")


@app.command()
def main(
    outdir: Annotated[
        Path,
        typer.Argument(help="Output directory", file_okay=False),
    ],
    prefix: Annotated[str, typer.Option(help="File name prefix")] = "synth",
    nbp: Annotated[int, typer.Option(help="Molecule length (bp)")] = 5000,
    nmol: Annotated[int, typer.Option(help="Number of molecules")] = 100,
    chrom: Annotated[str, typer.Option(help="Chromosome name")] = "chr1",
    mtase: Annotated[
        str, typer.Option(help="Methyltransferase context: A, CG, or GC")
    ] = "A",
    nucs: Annotated[
        Path | None,
        typer.Option(
            help="BED of nucleosome positions (chrom, start, end, mol_id)",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ] = None,
    lnuc: Annotated[
        int, typer.Option(help="Nucleosome footprint length (bp)")
    ] = 147,
    linker_mean: Annotated[
        float, typer.Option(help="Mean linker length (bp)")
    ] = 40.0,
    lnuc_range: Annotated[
        tuple[int, int] | None,
        typer.Option(help="Random footprint length range LO HI (bp)"),
    ] = None,
    p_linker: Annotated[
        float, typer.Option(help="Methylation probability in linkers")
    ] = 0.7,
    p_nuc: Annotated[
        float, typer.Option(help="Methylation probability in nucleosomes")
    ] = 0.02,
    signal_sep: Annotated[
        float, typer.Option(help="Signal separation for call scores")
    ] = 2.0,
    report_sep: Annotated[
        float | None,
        typer.Option(help="Signal separation assumed by the caller"),
    ] = None,
    double_strand: Annotated[
        bool, typer.Option(help="Read every molecule on both strands")
    ] = False,
    controls: Annotated[
        bool, typer.Option(help="Also write meth/unmeth controls")
    ] = False,
    seed: Annotated[int | None, typer.Option(help="Random seed")] = None,
):
    """Generate a synthetic footprinting dataset."""
    nuc_table = None
    if nucs is not None:
        nuc_table = pd.read_csv(
            nucs,
            sep="\t",
            header=None,
            comment="#",
            usecols=range(4),
            names=["chrom", "start", "end", "mol_id"],
        )
    paths = make_dataset(
        outdir,
        prefix=prefix,
        nbp=nbp,
        nmol=nmol,
        chrom=chrom,
        mtase=mtase,
        nucs=nuc_table,
        lnuc=lnuc,
        linker_mean=linker_mean,
        lnuc_range=lnuc_range,
        p_linker=p_linker,
        p_nuc=p_nuc,
        signal_sep=signal_sep,
        report_sep=report_sep,
        double_strand=double_strand,
        controls=controls,
        seed=seed,
    )
    for path in paths.values():
        typer.echo(path)


if __name__ == "__main__":
    app()
