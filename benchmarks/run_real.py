# run_real.py

import json
import tempfile
from pathlib import Path
from typing import Annotated

import callers
import numpy as np
import pandas as pd
import pyfaidx
import synthgen
import typer
from callers import ENV, Calls, load_calls, sam_to_bam
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from plotting import (
    TOOL_STYLES,
    _occupancy_figure,
    _occupancy_per_page,
    _style_axes,
)
from run_bench import _markdown_table, _run_tasks, _set_hiddenfoot

# Widom 601 nucleosome positioning sequence (147 bp core)
CORE_601 = (
    "CTGGAGAATCCCGGTGCCGAGGCCGCTCAATTGGTCGTAGACAGCTCTAGCACCGCTTAAACGCACG"
    "TACGCGCTGTCCCCCGCGTTTTAACCGCCAAGGGGATTACTCCCTAGTCTCCAGGCACGTGTCAGAT"
    "ATATACATCCTGT"
)
MTASE_OF_CODE = {"a": "A", "m": "GC"}
SAMPLES = ("chromatin", "methylated", "unmethylated")
_COMP = str.maketrans("ACGTN", "TGCAN")

# Scoring; fixed before any real dataset is scored
TRIM = 150
CORE_HALF = 73
REPEAT = 197
DYAD_TOL = 20
SPACING_TOL = 20
MARGINS = {"precision": 0.02, "contrast": 0.02}
HIGHER_IS_BETTER = {"precision": True, "contrast": True}


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
    `min_cover` of the fibre and samples up to `nmol` per sample.
    Writes the dataset `<out>/test` (chromatin against both controls;
    see `write_dataset`) and `<out>/annotation.json` with the fibre
    length, the 601 dyads, and the molecule ids used.

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
        Seed for sampling the molecules.
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
    write_dataset(
        out / "test", calls["chromatin"], calls["methylated"],
        calls["unmethylated"], fibre, chrom, env=env, convert=convert,
    )
    annotation = {
        "chrom": chrom,
        "fibre_length": len(fibre),
        "dyads": find_601_dyads(fibre).tolist(),
        "molecules": ids,
    }
    (out / "annotation.json").write_text(json.dumps(annotation, indent=2))
    return annotation


def occupancy_runs(occ: np.ndarray) -> np.ndarray:
    """
    Intervals where occupancy is at least 0.5.

    Parameters
    ----------
    occ : np.ndarray
        Per-bp occupancy of one molecule.

    Returns
    -------
    np.ndarray of int64
        (nrun, 2) start and end (half-open) of each run.
    """
    on = np.concatenate([[0], (np.asarray(occ) >= 0.5).astype(int), [0]])
    edges = np.flatnonzero(np.diff(on))
    return edges.reshape(-1, 2).astype(np.int64)


def molecule_metrics(
    calls: Calls,
    spans: pd.DataFrame,
    dyads: np.ndarray,
    *,
    trim: int = TRIM,
) -> pd.DataFrame:
    """
    Per-molecule counts behind the real-data metrics.

    Each molecule is scored inside the span its calls cover, minus
    `trim` bp at each end. Called nucleosomes are runs of occupancy
    >= 0.5 (see `occupancy_runs`), counted if their centre lies in that
    region.

    Parameters
    ----------
    calls : Calls
        Calls of one tool on one dataset.
    spans : pandas.DataFrame
        Index `mol_id`; columns `lo`, `hi`: first and last position
        with a methylation call.
    dyads : np.ndarray
        601 dyad positions.
    trim : int, default `TRIM`
        Bases excluded at each end of the covered span.

    Returns
    -------
    pandas.DataFrame
        Index `mol_id`; columns `n_calls` (called nucleosomes),
        `n_hits` (those with centre within `DYAD_TOL` of a dyad),
        `core` and `linker` (mean occupancy over 601 cores and over the
        linkers between them), `n_pairs` and `n_regular` (adjacent call
        pairs, and those `REPEAT` or 2 * `REPEAT` apart within
        `SPACING_TOL`), `kb` (scored length), `lengths` (call lengths).
    """
    nbp = calls.occ.shape[1]
    core = np.zeros(nbp, dtype=bool)
    for dyad in dyads:
        core[max(dyad - CORE_HALF, 0) : dyad + CORE_HALF + 1] = True
    array = np.zeros(nbp, dtype=bool)
    if len(dyads):
        array[dyads.min() - CORE_HALF : dyads.max() + CORE_HALF + 1] = True
    linker = array & ~core
    rows = []
    for mol_id, occ in zip(calls.mol_ids, calls.occ):
        lo, hi = spans.loc[mol_id, ["lo", "hi"]]
        region = np.zeros(nbp, dtype=bool)
        region[lo + trim : hi + 1 - trim] = True
        runs = occupancy_runs(occ)
        centres = (runs[:, 0] + runs[:, 1] - 1) / 2
        inside = region[np.clip(centres.astype(int), 0, nbp - 1)]
        centres, runs = centres[inside], runs[inside]
        near = np.abs(centres[:, None] - dyads[None, :]).min(axis=1) \
            if len(dyads) and len(centres) else np.array([])
        gaps = np.diff(centres)
        regular = (np.abs(gaps - REPEAT) <= SPACING_TOL) | (
            np.abs(gaps - 2 * REPEAT) <= SPACING_TOL
        )
        rows.append(
            {
                "mol_id": mol_id,
                "n_calls": len(centres),
                "n_hits": int((near <= DYAD_TOL).sum()),
                "core": occ[region & core].mean()
                if (region & core).any() else np.nan,
                "linker": occ[region & linker].mean()
                if (region & linker).any() else np.nan,
                "n_pairs": len(gaps),
                "n_regular": int(regular.sum()),
                "kb": region.sum() / 1000,
                "lengths": (runs[:, 1] - runs[:, 0]).tolist(),
            }
        )
    return pd.DataFrame(rows).set_index("mol_id")


def aggregate(table: pd.DataFrame) -> dict[str, float]:
    """
    Real-data metrics from per-molecule counts.

    Parameters
    ----------
    table : pandas.DataFrame
        From `molecule_metrics` (possibly a resample of its rows).

    Returns
    -------
    dict of str to float
        `precision` (pooled hits / calls), `contrast` (mean over
        molecules of core minus linker occupancy), `spacing` (pooled
        regular / all adjacent pairs), `per_mol` (mean calls per
        molecule).
    """
    calls = table["n_calls"].sum()
    pairs = table["n_pairs"].sum()
    return {
        "precision": table["n_hits"].sum() / calls if calls else np.nan,
        "contrast": (table["core"] - table["linker"]).mean(),
        "spacing": table["n_regular"].sum() / pairs if pairs else np.nan,
        "per_mol": table["n_calls"].mean(),
    }


def bootstrap(
    tables: dict[str, pd.DataFrame], *, nboot: int, seed: int
) -> dict[str, pd.DataFrame]:
    """
    Bootstrap the metrics over molecules, paired across tools.

    Every resample draws the same molecules for every tool.

    Parameters
    ----------
    tables : dict of str to pandas.DataFrame
        Tool -> `molecule_metrics` table, all with the same molecules.
    nboot : int
        Number of resamples.
    seed : int
        Seed for the resampling.

    Returns
    -------
    dict of str to pandas.DataFrame
        Tool -> (nboot, metric) table of resampled `aggregate` values.
    """
    mol_ids = sorted(next(iter(tables.values())).index)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(mol_ids), size=(nboot, len(mol_ids)))
    out = {}
    for tool, table in tables.items():
        table = table.loc[mol_ids]
        out[tool] = pd.DataFrame(
            [aggregate(table.iloc[d]) for d in draws]
        )
    return out


def equivalence_label(
    diffs: np.ndarray, margin: float, higher_is_better: bool
) -> str:
    """
    Classify a paired difference (catella minus another tool).

    Parameters
    ----------
    diffs : np.ndarray
        Bootstrap resamples of the difference.
    margin : float
        Equivalence margin.
    higher_is_better : bool
        Whether a larger metric value is better.

    Returns
    -------
    str
        "equivalent" if the 90% interval lies inside (-margin, margin)
        (two one-sided tests at alpha 0.05); otherwise "better" or
        "worse" for catella if the 95% interval excludes 0; otherwise
        "inconclusive".
    """
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return "inconclusive"
    lo90, hi90 = np.percentile(diffs, [5, 95])
    if -margin < lo90 and hi90 < margin:
        return "equivalent"
    lo95, hi95 = np.percentile(diffs, [2.5, 97.5])
    if lo95 > 0:
        return "better" if higher_is_better else "worse"
    if hi95 < 0:
        return "worse" if higher_is_better else "better"
    return "inconclusive"


DATASET_METRICS = {"test": ("precision", "contrast", "spacing", "per_mol")}


def _spans(ddir):
    tsv = pd.read_csv(
        Path(ddir) / "synth.tsv", sep="\t", usecols=["read_id", "ref_position"]
    )
    spans = tsv.groupby("read_id")["ref_position"].agg(["min", "max"])
    return spans.rename(columns={"min": "lo", "max": "hi"})


def _interval(values, level):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    tail = (100 - level) / 2
    return tuple(np.percentile(values, [tail, 100 - tail]))


def score(out: str | Path, *, nboot: int = 1000, seed: int = 0) -> dict:
    """
    Score saved real-data calls and compare catella with each tool.

    For the chromatin dataset (`test`) and each parameter variant,
    computes the metrics of `aggregate` per tool with 95% bootstrap
    intervals over molecules,
    and, on the molecules all tools share, the paired difference
    catella minus each other tool with its `equivalence_label` (margins
    `MARGINS`). Writes `<out>/real_results.tsv`,
    `<out>/real_paired.tsv`, and `<out>/real_summary.md`.

    Parameters
    ----------
    out : str or Path
        Real-data output directory (after `prepare` and `evaluate`),
        with calls in `<out>/calls/<tool>_<variant>/<dataset>.h5`.
    nboot : int, default 1000
        Number of bootstrap resamples.
    seed : int, default 0
        Seed for the resampling.

    Returns
    -------
    dict
        "results" and "paired" tables, as written.
    """
    out = Path(out)
    annotation = json.loads((out / "annotation.json").read_text())
    dyads = np.asarray(annotation["dyads"], dtype=np.int64)
    results, paired = [], []
    for dataset, metrics in DATASET_METRICS.items():
        spans = _spans(out / dataset)
        for variant in ("tuned", "default"):
            tables = {}
            pattern = f"*_{variant}/{dataset}.h5"
            for h5 in sorted((out / "calls").glob(pattern)):
                tool = h5.parent.name[: -len(variant) - 1]
                tables[tool] = molecule_metrics(load_calls(h5), spans, dyads)
            if not tables:
                continue
            shared = sorted(
                set.intersection(*(set(t.index) for t in tables.values()))
            )
            tables = {t: table.loc[shared] for t, table in tables.items()}
            boots = bootstrap(tables, nboot=nboot, seed=seed)
            points = {t: aggregate(table) for t, table in tables.items()}
            for tool in tables:
                for metric in metrics:
                    lo, hi = _interval(boots[tool][metric], 95)
                    results.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "tool": tool,
                            "metric": metric,
                            "value": points[tool][metric],
                            "ci_low": lo,
                            "ci_high": hi,
                            "n": len(shared),
                        }
                    )
            if "catella" not in tables:
                continue
            for other in tables:
                if other == "catella":
                    continue
                for metric in metrics:
                    if metric not in MARGINS:
                        continue
                    diffs = boots["catella"][metric] - boots[other][metric]
                    lo90, hi90 = _interval(diffs, 90)
                    lo95, hi95 = _interval(diffs, 95)
                    paired.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "other": other,
                            "metric": metric,
                            "diff": points["catella"][metric]
                            - points[other][metric],
                            "ci90_low": lo90,
                            "ci90_high": hi90,
                            "ci95_low": lo95,
                            "ci95_high": hi95,
                            "margin": MARGINS[metric],
                            "label": equivalence_label(
                                diffs, MARGINS[metric],
                                HIGHER_IS_BETTER[metric],
                            ),
                        }
                    )
    results = pd.DataFrame(results)
    paired = pd.DataFrame(paired)
    results.to_csv(
        out / "real_results.tsv", sep="\t", index=False, lineterminator="\n"
    )
    paired.to_csv(
        out / "real_paired.tsv", sep="\t", index=False, lineterminator="\n"
    )
    text = [
        "# Real-data benchmark (601 fibre)",
        "",
        "## Metrics (95% bootstrap intervals over molecules)",
        "",
        _markdown_table(results) if len(results) else "(none)",
        "",
        "## catella minus other tool",
        "",
        _markdown_table(paired) if len(paired) else "(none)",
        "",
    ]
    (out / "real_summary.md").write_text("\n".join(text))
    return {"results": results, "paired": paired}


# Real-data tool -> (synthetic tool whose runner and tuned parameters
# are used, fixed runner options); each tool gets the best channel it
# supports, and the m6A variants are labelled secondary results
REAL_TOOLS = {
    "catella": ("catella", {"mtase": ["A", "GC"]}),
    "hiddenfoot": ("hiddenfoot", {"mod_code": "m"}),
    "fiberhmm": ("fiberhmm", {"retrain": True, "channel": "m"}),
    "ft": ("ft", {}),
    "hiddenfoot_m6a": ("hiddenfoot", {"mod_code": "a"}),
    "fiberhmm_pre_m6a": ("fiberhmm_pre", {"retrain": False, "channel": "a"}),
}
DATASETS = ("test",)


def run_real_tool(
    tool: str,
    ddir: str | Path,
    workdir: str | Path,
    params: dict,
    env: str = ENV,
) -> Calls:
    """
    Run one real-data tool on one dataset.

    Parameters
    ----------
    tool : str
        A key of `REAL_TOOLS`.
    ddir : str or Path
        Dataset directory (from `prepare`).
    workdir : str or Path
        Directory for intermediate files.
    params : dict
        Tuned or default parameters of the tool's runner.
    env : str, default "catella"
        Conda environment for the external tools.

    Returns
    -------
    Calls
    """
    key, fixed = REAL_TOOLS[tool]
    runner = callers.TOOLS[key][0]
    kwargs = {**fixed, **params}
    if key not in ("catella", "hiddenfoot"):
        kwargs["env"] = env
    return runner(ddir, workdir, **kwargs)


def _evaluate_unit(out, dataset, tool, variant, params, env, overwrite):
    out = Path(out)
    path = out / "calls" / f"{tool}_{variant}" / f"{dataset}.h5"
    if path.exists() and not overwrite:
        return str(path)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work) as wd:
        calls = run_real_tool(tool, out / dataset, wd, params, env)
    callers.save_calls(calls, path)
    return str(path)


def evaluate(
    out: str | Path,
    tuned: str | Path,
    *,
    env: str = ENV,
    nworker: int = 1,
    tools: list[str] | None = None,
    overwrite: bool = False,
) -> list[str]:
    """
    Run the real-data tools on the prepared dataset.

    Each (dataset, tool, variant) is one unit, run in parallel; units
    whose calls exist are skipped unless `overwrite`. The "tuned"
    variant uses the parameters chosen on synthetic data, the "default"
    variant each tool's defaults; no parameter is tuned on real data.
    Also writes the tool versions to `<out>/versions.json`.

    Parameters
    ----------
    out : str or Path
        Real-data output directory (after `prepare`).
    tuned : str or Path
        `tuned_params.json` of a synthetic benchmark run.
    env : str, default "catella"
        Conda environment for the external tools.
    nworker : int, default 1
        Number of units run in parallel.
    tools : list of str, optional
        Restrict to these keys of `REAL_TOOLS`; all if None.
    overwrite : bool, default False
        If True, rerun units whose calls exist.

    Returns
    -------
    list of str
        The calls files.
    """
    out = Path(out)
    tools = list(REAL_TOOLS) if tools is None else tools
    unknown = sorted(set(tools) - set(REAL_TOOLS))
    if unknown:
        raise ValueError(
            f"unknown tools {unknown}; expected values in "
            f"{sorted(REAL_TOOLS)}"
        )
    tuned = json.loads(Path(tuned).read_text())
    with open(out / "versions.json", "w") as fh:
        json.dump(callers.tool_versions(env), fh, indent=2)
    if nworker > 1 and "fiberhmm" in tools:
        for dataset in DATASETS:
            callers.train_fiberhmm(out / dataset, env=env, channel="m")
    units = [
        (out, dataset, tool, variant, params, env, overwrite)
        for dataset in DATASETS
        for tool in tools
        for variant, params in (
            ("tuned", tuned[REAL_TOOLS[tool][0]]),
            ("default", {}),
        )
    ]
    return _run_tasks(_evaluate_unit, units, nworker)


def _gpc_rows(ddir, mol_ids, nbp):
    tsv = pd.read_csv(Path(ddir) / "synth.tsv", sep="\t")
    tsv = tsv[tsv["mod_code"] == "m"]
    rows = np.full((len(mol_ids), nbp), np.nan)
    index = {m: i for i, m in enumerate(mol_ids)}
    for mol_id, grp in tsv.groupby("read_id", sort=False):
        if mol_id in index:
            pos = grp["ref_position"].to_numpy()
            rows[index[mol_id], pos] = grp["mod_qual"].to_numpy()
    return rows


def _style(tool):
    label, color = TOOL_STYLES[REAL_TOOLS[tool][0]]
    if tool.endswith("_m6a"):
        label = f"{label} (m6A)"
    return label, color


def plot(
    out: str | Path,
    *,
    variant: str = "tuned",
    nmol: int = 8,
    path: str | Path | None = None,
) -> Path:
    """
    Plot real-data calls on the 601 fibre.

    Page 1: mean occupancy over chromatin molecules per tool, with the
    601 dyads marked. Page 2: distances between adjacent called
    nucleosome centres per tool. Then per-molecule tracks (GpC
    `mod_qual`, the 601 cores, each tool) for the first `nmol`
    molecules.

    Parameters
    ----------
    out : str or Path
        Real-data output directory (after `evaluate`).
    variant : {"tuned", "default"}, default "tuned"
        Parameter variant plotted.
    nmol : int, default 8
        Number of molecules shown as tracks.
    path : str or Path, optional
        Output PDF; defaults to `<out>/plots/real_<variant>.pdf`.

    Returns
    -------
    Path
        The PDF written.
    """
    out = Path(out)
    annotation = json.loads((out / "annotation.json").read_text())
    dyads = np.asarray(annotation["dyads"], dtype=np.int64)
    calls = {}
    for tool in REAL_TOOLS:
        h5 = out / "calls" / f"{tool}_{variant}" / "test.h5"
        if h5.exists():
            calls[tool] = callers.load_calls(h5)
    if not calls:
        raise ValueError(
            f"no saved calls for variant {variant!r}; run evaluate first"
        )
    nbp = annotation["fibre_length"]
    if path is None:
        path = out / "plots" / f"real_{variant}.pdf"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = 8.27, 11.69
    with PdfPages(path) as pdf:
        fig = Figure(figsize=(width, height))
        axes = fig.subplots(len(calls), 1, sharex=True, squeeze=False)
        for ax, (tool, c) in zip(axes[:, 0], calls.items()):
            label, color = _style(tool)
            ax.plot(np.arange(nbp), c.occ.mean(axis=0), color=color, lw=0.8)
            for dyad in dyads:
                ax.axvline(dyad, color="#bbbbbb", lw=0.4, zorder=0)
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"{label}\nmean occ.", fontsize=7)
            _style_axes(ax)
        axes[-1, 0].set_xlabel("position on 601 fibre (bp)", fontsize=8)
        fig.suptitle(
            f"Mean nucleosome occupancy, chromatin ({variant}); grey: "
            "601 dyads",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)

        fig = Figure(figsize=(width, height))
        axes = fig.subplots(len(calls), 1, sharex=True, squeeze=False)
        bins = np.arange(0, 1001, 10)
        for ax, (tool, c) in zip(axes[:, 0], calls.items()):
            label, color = _style(tool)
            gaps = []
            for occ in c.occ:
                runs = occupancy_runs(occ)
                gaps += np.diff((runs[:, 0] + runs[:, 1]) / 2).tolist()
            ax.hist(gaps, bins=bins, color=color)
            for k in (1, 2):
                ax.axvline(k * REPEAT, color="#52514e", lw=0.6, ls="--")
            ax.set_ylabel(f"{label}\npairs", fontsize=7)
            _style_axes(ax)
        axes[-1, 0].set_xlabel(
            "distance between adjacent called centres (bp)", fontsize=8
        )
        fig.suptitle(
            f"Nucleosome spacing ({variant}); dashed: 197 and 394 bp",
            fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        pdf.savefig(fig)

        mol_ids = sorted(next(iter(calls.values())).mol_ids)[:nmol]
        qual = _gpc_rows(out / "test", mol_ids, nbp)
        core = np.zeros(nbp)
        for dyad in dyads:
            core[max(dyad - CORE_HALF, 0) : dyad + CORE_HALF + 1] = 1
        rows = {
            tool: {m: i for i, m in enumerate(c.mol_ids)}
            for tool, c in calls.items()
        }
        molecules = []
        for i, mol_id in enumerate(mol_ids):
            tracks = [("601 cores", core, TOOL_STYLES["truth"][1])]
            for tool, c in calls.items():
                label, color = _style(tool)
                occ = None
                if mol_id in rows[tool]:
                    occ = c.occ[rows[tool][mol_id]]
                tracks.append((label, occ, color))
            molecules.append((mol_id, qual[i], tracks))
        per_page = _occupancy_per_page(len(calls) + 2)
        title = (
            f"GpC methylation and nucleosome occupancy, chromatin "
            f"({variant})\n(shaded ends: not scored)"
        )
        for start in range(0, len(molecules), per_page):
            page = molecules[start : start + per_page]
            pdf.savefig(_occupancy_figure(page, title, nbp, TRIM))
    return path


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


@app.command(name="evaluate")
def evaluate_cmd(
    out: Annotated[Path, typer.Argument(help="Real-data output directory")],
    tuned: Annotated[
        Path,
        typer.Option(help="tuned_params.json of a synthetic benchmark run"),
    ],
    env: Annotated[
        str, typer.Option(help="Conda env with samtools/ft/FiberHMM")
    ] = ENV,
    nworker: Annotated[
        int, typer.Option(help="Number of parallel processes")
    ] = 1,
    tool: Annotated[
        list[str] | None, typer.Option(help="Only this tool (repeatable)")
    ] = None,
    overwrite: Annotated[
        bool, typer.Option(help="Rerun units with existing calls")
    ] = False,
    hiddenfoot: Annotated[
        str | None,
        typer.Option(help="HiddenFoot executable; default $HIDDENFOOT"),
    ] = None,
):
    """Run every tool on the prepared real datasets."""
    _set_hiddenfoot(hiddenfoot)
    paths = evaluate(
        out, tuned, env=env, nworker=nworker, tools=tool,
        overwrite=overwrite,
    )
    typer.echo(f"wrote {len(paths)} calls files under {Path(out) / 'calls'}")


@app.command(name="plot")
def plot_cmd(
    out: Annotated[Path, typer.Argument(help="Real-data output directory")],
    variant: Annotated[
        str, typer.Option(help="Parameter variant: tuned or default")
    ] = "tuned",
    nmol: Annotated[
        int, typer.Option(help="Molecules shown as tracks")
    ] = 8,
):
    """Plot occupancy profiles, spacing and per-molecule tracks."""
    typer.echo(f"wrote {plot(out, variant=variant, nmol=nmol)}")


@app.command(name="score")
def score_cmd(
    out: Annotated[Path, typer.Argument(help="Real-data output directory")],
    nboot: Annotated[int, typer.Option(help="Bootstrap resamples")] = 1000,
    seed: Annotated[int, typer.Option(help="Random seed")] = 0,
):
    """Score saved calls and write the real-data summary."""
    score(out, nboot=nboot, seed=seed)
    typer.echo(f"wrote {Path(out) / 'real_summary.md'}")


if __name__ == "__main__":
    app()
