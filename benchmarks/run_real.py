# run_real.py

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import pyfaidx
import synthgen
import typer
from callers import ENV, sam_to_bam

# Widom 601 nucleosome positioning sequence (147 bp core)
CORE_601 = (
    "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACG"
    "TACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGAT"
    "ATATACATCCTGT"
)
MTASE_OF_CODE = {"a": "A", "m": "GC"}
SAMPLES = ("chromatin", "methylated", "unmethylated")
_COMP = str.maketrans("ACGTN", "TGCAN")


def find_601_dyads(seq: str, anchor: int = 50) -> np.ndarray:
    """
    Dyad positions of the 601 cores in a sequence.

    A core is located by an exact match of its first `anchor` bases, in
    either orientation, so constructs carrying 601 variants that differ
    further into the core are still found.

    Parameters
    ----------
    seq : str
        Reference sequence.
    anchor : int, default 50
        Number of leading core bases that must match exactly.

    Returns
    -------
    np.ndarray of int64
        Sorted 0-based dyad positions (core start + 73).
    """
    seq = seq.upper()
    rev = CORE_601.translate(_COMP)[::-1]
    starts = [
        i
        for core in (CORE_601, rev)
        for i in range(len(seq) - len(core) + 1)
        if seq.startswith(core[:anchor], i)
    ]
    return np.sort(np.asarray(starts, dtype=np.int64) + len(CORE_601) // 2)


def read_modkit(path: str | Path, chrom: str) -> pd.DataFrame:
    """
    Read the m6A and 5mC calls of one chromosome from a ModKit table.

    Parameters
    ----------
    path : str or Path
        `modkit extract full` output (tab-separated, with header).
    chrom : str
        Chromosome to keep.

    Returns
    -------
    pandas.DataFrame
        Columns `read_id`, `ref_position`, `ref_strand`, `mod_qual`,
        `mod_code`, for `mod_code` in `MTASE_OF_CODE`.
    """
    cols = ["read_id", "ref_position", "chrom", "ref_strand", "mod_qual",
            "mod_code"]
    table = pd.read_csv(
        path,
        sep="\t",
        usecols=cols,
        dtype={"chrom": str, "ref_strand": str, "mod_code": str},
    )
    keep = (table["chrom"] == chrom) & table["mod_code"].isin(MTASE_OF_CODE)
    return table.loc[keep, [c for c in cols if c != "chrom"]].reset_index(
        drop=True
    )


def _eligible(seq, code, strand):
    mask = np.zeros(len(seq), dtype=bool)
    mask[synthgen.eligible_sites(seq, MTASE_OF_CODE[code], strand)] = True
    return mask


def unwrap_calls(calls: pd.DataFrame, doubled: str) -> pd.DataFrame:
    """
    Fold hairpin-ligated reads onto the fibre, with both strands.

    Each read holds the forward strand, a hairpin, and the reverse
    strand, aligned to a doubled reference of length `L`. Rows whose
    recorded strand is not a methylatable site of their channel on the
    doubled reference (read-to-reference mismatches) are dropped; the
    rest are folded with `p -> L - 1 - p` for `p > (L - 1) / 2`, which
    maps the reverse strand onto the opposite strand of the fibre.
    Calls landing on a fibre base that is not a site of their channel
    on either strand (the hairpin junction) are dropped, and repeated
    calls at one position are averaged.

    Parameters
    ----------
    calls : pandas.DataFrame
        From `read_modkit`.
    doubled : str
        Doubled reference sequence.

    Returns
    -------
    pandas.DataFrame
        Columns `mol_id`, `pos` (fibre coordinate), `mod_code`,
        `mod_qual`, sorted by molecule, channel, then position.
    """
    doubled = doubled.upper()
    length = len(doubled)
    fibre = doubled[: length // 2]
    pos = calls["ref_position"].to_numpy()
    keep = np.zeros(len(calls), dtype=bool)
    for code in MTASE_OF_CODE:
        for strand in ("+", "-"):
            sel = ((calls["mod_code"] == code)
                   & (calls["ref_strand"] == strand)).to_numpy()
            keep[sel] = _eligible(doubled, code, strand)[pos[sel]]
    calls = calls[keep]
    pos = calls["ref_position"].to_numpy()
    folded = np.where(pos > (length - 1) / 2, length - 1 - pos, pos)
    out = pd.DataFrame(
        {
            "mol_id": calls["read_id"].to_numpy(),
            "pos": folded,
            "mod_code": calls["mod_code"].to_numpy(),
            "mod_qual": calls["mod_qual"].to_numpy(dtype=float),
        }
    )
    ok = np.zeros(len(out), dtype=bool)
    for code in MTASE_OF_CODE:
        sel = (out["mod_code"] == code).to_numpy()
        site = _eligible(fibre, code, "+") | _eligible(fibre, code, "-")
        ok[sel] = site[out["pos"].to_numpy()[sel]]
    out = out[ok]
    keys = ["mol_id", "mod_code", "pos"]
    return out.groupby(keys, as_index=False, sort=True)["mod_qual"].mean()


def select_molecules(
    calls: pd.DataFrame,
    length: int,
    *,
    nmol: int | None,
    min_cover: float,
    seed: int,
) -> list[str]:
    """
    Pick molecules that cover most of the fibre, at random.

    Parameters
    ----------
    calls : pandas.DataFrame
        From `unwrap_calls`.
    length : int
        Fibre length in bp.
    nmol : int or None
        Number of molecules to keep; all eligible if None or larger.
    min_cover : float
        Minimum fraction of the fibre spanned by a molecule's calls.
    seed : int
        Seed for the random choice.

    Returns
    -------
    list of str
        Selected molecule ids, sorted.
    """
    if not 0 <= min_cover <= 1:
        raise ValueError(f"min_cover must be in [0, 1], got {min_cover}")
    span = calls.groupby("mol_id")["pos"].agg(lambda p: p.max() - p.min())
    ids = sorted(span.index[(span + 1) / length >= min_cover])
    if nmol is not None and nmol < len(ids):
        rng = np.random.default_rng(seed)
        ids = sorted(rng.choice(ids, size=nmol, replace=False))
    return [str(i) for i in ids]


def split_molecules(
    ids: list[str], seed: int
) -> tuple[list[str], list[str]]:
    """
    Split molecules into two random halves.

    Parameters
    ----------
    ids : list of str
        Molecule ids.
    seed : int
        Seed for the random split.

    Returns
    -------
    first, second : list of str
        Disjoint sorted halves; `first` gets the extra molecule if the
        count is odd.
    """
    order = np.random.default_rng(seed).permutation(sorted(ids))
    half = (len(order) + 1) // 2
    return sorted(order[:half].tolist()), sorted(order[half:].tolist())


def _write_tsv(calls, chrom, path):
    out = pd.DataFrame(
        {
            "read_id": calls["mol_id"],
            "ref_position": calls["pos"],
            "chrom": chrom,
            "ref_strand": "+",
            "mod_qual": calls["mod_qual"],
            "mod_code": calls["mod_code"],
        }
    )
    out.to_csv(path, sep="\t", index=False, lineterminator="\n")


def _write_bams(calls, chrom, fibre, stem, env, convert):
    for code, suffix in (("a", ""), ("m", ".gpc")):
        sel = calls[calls["mod_code"] == code]
        reads = pd.DataFrame(
            {
                "mol_id": sel["mol_id"],
                "chrom": chrom,
                "pos": sel["pos"],
                "strand": "+",
                "mod_qual": sel["mod_qual"],
            }
        )
        sam = synthgen.write_sam(
            reads, {chrom: fibre}, f"{stem}{suffix}.sam",
            mtase=MTASE_OF_CODE[code],
        )
        if convert:
            sam_to_bam(sam, env)


def write_dataset(
    ddir: str | Path,
    test: pd.DataFrame,
    meth: pd.DataFrame,
    unmeth: pd.DataFrame,
    fibre: str,
    chrom: str,
    *,
    env: str = ENV,
    convert: bool = True,
) -> Path:
    """
    Write unwrapped real data in the benchmark dataset layout.

    Writes `synth.tsv`, `synth.meth.tsv` and `synth.unmeth.tsv` (ModKit
    columns, both channels, one row per call, `ref_strand` "+" since a
    read is the whole fibre), double-strand SAM files per channel (m6A:
    `synth*.sam`; GpC: `synth*.gpc.sam`) with `synthgen.write_sam`,
    and `synth.fa`, `synth.size` and the FASTA index.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory; created if missing.
    test, meth, unmeth : pandas.DataFrame
        Unwrapped calls (see `unwrap_calls`) of the test sample and of
        the methylated and unmethylated controls.
    fibre : str
        Fibre reference sequence.
    chrom : str
        Chromosome name.
    env : str, default "catella"
        Conda environment providing samtools.
    convert : bool, default True
        If True, convert every SAM file to a sorted, indexed BAM file.

    Returns
    -------
    Path
        The dataset directory.
    """
    ddir = Path(ddir)
    ddir.mkdir(parents=True, exist_ok=True)
    synthgen.write_fasta({chrom: fibre}, ddir / "synth.fa")
    synthgen.write_chromsize({chrom: len(fibre)}, ddir / "synth.size")
    pyfaidx.Faidx(str(ddir / "synth.fa")).close()
    for name, calls in (("", test), (".meth", meth), (".unmeth", unmeth)):
        _write_tsv(calls, chrom, ddir / f"synth{name}.tsv")
        _write_bams(calls, chrom, fibre, ddir / f"synth{name}", env, convert)
    return ddir


def _read_fasta(path, chrom):
    fasta = pyfaidx.Fasta(str(path), as_raw=True)
    try:
        return str(fasta[chrom][:]).upper()
    finally:
        fasta.close()


def prepare(
    raw_dir: str | Path,
    out: str | Path,
    *,
    chrom: str = "601",
    nmol: int | None = 1000,
    min_cover: float = 0.8,
    seed: int = 0,
    env: str = ENV,
    convert: bool = True,
) -> dict:
    """
    Prepare the real-data benchmark from hairpin-ligated fibre reads.

    Reads `chromatin.tsv.gz`, `methylated.tsv.gz` and
    `unmethylated.tsv.gz` (ModKit `extract full` tables) and
    `segments.fa` (doubled reference) from `raw_dir`; unwraps every
    read (see `unwrap_calls`); keeps molecules covering at least
    `min_cover` of the fibre and samples up to `nmol` per sample; splits
    the methylated control into a calibration and an evaluation half.
    Writes two datasets (see `write_dataset`): `<out>/test` (chromatin
    against the calibration half and the unmethylated control) and
    `<out>/meth_eval` (the evaluation half against the same controls),
    and `<out>/annotation.json` with the fibre length, the 601 dyads,
    and the molecule ids used.

    Parameters
    ----------
    raw_dir : str or Path
        Directory with the ModKit tables and `segments.fa`.
    out : str or Path
        Output directory.
    chrom : str, default "601"
        Fibre (chromosome) to prepare.
    nmol : int or None, default 1000
        Molecules per sample; all eligible if None.
    min_cover : float, default 0.8
        Minimum fraction of the fibre covered by a molecule.
    seed : int, default 0
        Seed for sampling and for the control split.
    env : str, default "catella"
        Conda environment providing samtools.
    convert : bool, default True
        If True, write sorted, indexed BAM files.

    Returns
    -------
    dict
        The annotation written to `annotation.json`.
    """
    raw_dir, out = Path(raw_dir), Path(out)
    doubled = _read_fasta(raw_dir / "segments.fa", chrom)
    fibre = doubled[: len(doubled) // 2]
    calls, ids = {}, {}
    for k, sample in enumerate(SAMPLES):
        unwrapped = unwrap_calls(
            read_modkit(raw_dir / f"{sample}.tsv.gz", chrom), doubled
        )
        ids[sample] = select_molecules(
            unwrapped, len(fibre), nmol=nmol, min_cover=min_cover,
            seed=seed + k,
        )
        calls[sample] = unwrapped[unwrapped["mol_id"].isin(ids[sample])]
    calib, evaluation = split_molecules(ids["methylated"], seed)
    meth = calls["methylated"]
    meth_calib = meth[meth["mol_id"].isin(calib)]
    for name, test in (
        ("test", calls["chromatin"]),
        ("meth_eval", meth[meth["mol_id"].isin(evaluation)]),
    ):
        write_dataset(
            out / name, test, meth_calib, calls["unmethylated"], fibre,
            chrom, env=env, convert=convert,
        )
    annotation = {
        "chrom": chrom,
        "fibre_length": len(fibre),
        "dyads": find_601_dyads(fibre).tolist(),
        "molecules": ids,
        "meth_calibration": calib,
        "meth_evaluation": evaluation,
    }
    (out / "annotation.json").write_text(json.dumps(annotation, indent=2))
    return annotation


app = typer.Typer(help="Benchmark nucleosome callers on real 601 fibres")


@app.callback()
def main():
    """Benchmark nucleosome callers on real 601 fibres."""


@app.command(name="prepare")
def prepare_cmd(
    raw_dir: Annotated[
        Path, typer.Argument(help="Directory with ModKit tables")
    ],
    out: Annotated[Path, typer.Argument(help="Output directory")],
    chrom: Annotated[str, typer.Option(help="Fibre to prepare")] = "601",
    nmol: Annotated[
        int | None, typer.Option(help="Molecules per sample; default all")
    ] = 1000,
    min_cover: Annotated[
        float, typer.Option(help="Minimum fraction of the fibre covered")
    ] = 0.8,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
    env: Annotated[str, typer.Option(help="Conda env with samtools")] = ENV,
):
    """Unwrap, sample and write the real-data datasets."""
    annotation = prepare(
        raw_dir, out, chrom=chrom, nmol=nmol, min_cover=min_cover,
        seed=seed, env=env,
    )
    counts = {k: len(v) for k, v in annotation["molecules"].items()}
    typer.echo(f"wrote {out}: {counts}, {len(annotation['dyads'])} dyads")


if __name__ == "__main__":
    app()
