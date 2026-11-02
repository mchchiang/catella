# callers.py

import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import catella
import h5py
import numpy as np
import pandas as pd
from catella.simulation.config import SimSettings

ENV = "catella"
# HiddenFoot runs on overlapping windows; its likelihood underflows on
# sequences longer than about 2000 bp
HF_WINDOW = 1000
HF_STEP = 500
HF_MAX_WINDOW = 2000
# FiberHMM labelling channel -> (BAM file suffix, FiberHMM mode)
FIBERHMM_CHANNELS = {"a": ("", "pacbio-fiber"), "m": (".gpc", "gpc")}


def run_in_env(env: str, args: list[str], cwd: str | Path | None = None):
    """
    Run a command inside a conda environment.

    Uses the executable in `$CONDA_EXE`, else `$MAMBA_EXE`, else
    `conda` on the `PATH`.

    Parameters
    ----------
    env : str
        Conda environment name.
    args : list of str
        Command and its arguments.
    cwd : str or Path, optional
        Working directory.

    Returns
    -------
    subprocess.CompletedProcess
        The finished process, with captured text output.
    """
    conda = os.environ.get("CONDA_EXE") or os.environ.get("MAMBA_EXE")
    conda = conda or "conda"
    proc = subprocess.run(
        [conda, "run", "-n", env, *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"command failed in env {env!r}: {' '.join(args)}; "
            f"{proc.stderr.strip()}"
        )
    return proc


VERSION_CMDS = {
    "fibertools": ["ft", "--version"],
    "fiberhmm": ["fiberhmm-call", "--version"],
    "samtools": ["samtools", "--version"],
}


def tool_versions(env: str = ENV) -> dict[str, str]:
    """
    Report the versions of catella and the external tools.

    Parameters
    ----------
    env : str, default "catella"
        Conda environment for the external tools.

    Returns
    -------
    dict of str to str
        Version number by tool; "unknown" if a tool cannot be run or
        reports no version number. HiddenFoot has no version number, so
        its entry is the short git commit of the repository holding its
        binary (see `run_hiddenfoot`).
    """
    versions = {"catella": catella.__version__}
    for tool, args in VERSION_CMDS.items():
        try:
            proc = run_in_env(env, args)
        except (RuntimeError, FileNotFoundError):
            versions[tool] = "unknown"
            continue
        match = re.search(r"\d+(?:\.\d+)+", proc.stdout)
        versions[tool] = match.group() if match else "unknown"
    versions["hiddenfoot"] = "unknown"
    binary = _hiddenfoot_binary()
    if binary is not None:
        try:
            proc = subprocess.run(
                ["git", "-C", str(Path(binary).parent), "rev-parse",
                 "--short", "HEAD"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            proc = None
        if proc is not None and proc.returncode == 0:
            versions["hiddenfoot"] = proc.stdout.strip()
    return versions


def sam_to_bam(sam: str | Path, env: str) -> Path:
    """
    Convert a SAM file to a sorted, indexed BAM file with samtools.

    Parameters
    ----------
    sam : str or Path
        Input SAM file.
    env : str
        Conda environment providing samtools.

    Returns
    -------
    Path
        The BAM file written next to `sam`.
    """
    sam = Path(sam)
    bam = sam.with_suffix(".bam")
    run_in_env(env, ["samtools", "sort", "-o", str(bam), str(sam)])
    run_in_env(env, ["samtools", "index", str(bam)])
    return bam


@dataclass
class Calls:
    """
    Nucleosome calls of one tool on one dataset.

    Attributes
    ----------
    mol_ids : np.ndarray of str
        Molecule ids, one per row.
    occ : np.ndarray of float
        (nmol, nbp) per-bp nucleosome occupancy: 0/1 for interval calls,
        a probability for catella.
    count : np.ndarray of float
        Number of called (or expected, for catella) nucleosomes per
        molecule.
    """

    mol_ids: np.ndarray
    occ: np.ndarray
    count: np.ndarray


def intervals_to_calls(
    intervals: pd.DataFrame, mol_ids: list[str], nbp: int
) -> Calls:
    """
    Convert nucleosome intervals to calls.

    Parameters
    ----------
    intervals : pandas.DataFrame
        Columns `mol_id`, `start`, `end` (0-based, half-open). Intervals
        are clipped to `[0, nbp)`.
    mol_ids : list of str
        Molecules to include, in row order; molecules without intervals
        get no nucleosomes.
    nbp : int
        Molecule length in bp.

    Returns
    -------
    Calls
        Binary occupancy and interval counts.
    """
    row = {m: i for i, m in enumerate(mol_ids)}
    occ = np.zeros((len(mol_ids), nbp), dtype=np.float32)
    count = np.zeros(len(mol_ids))
    cols = intervals[["mol_id", "start", "end"]]
    for mol_id, start, end in cols.itertuples(index=False):
        if mol_id not in row:
            continue
        occ[row[mol_id], max(start, 0) : min(end, nbp)] = 1
        count[row[mol_id]] += 1
    return Calls(np.asarray(mol_ids), occ, count)


def _int_list(field):
    text = str(field).strip(",")
    if text in ("", ".", "nan"):
        return []
    return [int(v) for v in text.split(",")]


def parse_ft_extract(path: str | Path) -> pd.DataFrame:
    """
    Read nucleosome intervals from an `ft extract --all` table.

    Parameters
    ----------
    path : str or Path
        Output of `ft extract --all`, with columns `fiber`,
        `ref_nuc_starts`, and `ref_nuc_lengths`.

    Returns
    -------
    pandas.DataFrame
        Columns `mol_id`, `start`, `end` in reference coordinates.
    """
    table = pd.read_csv(path, sep="\t")
    cols = table[["fiber", "ref_nuc_starts", "ref_nuc_lengths"]]
    rows = []
    for fiber, starts, lengths in cols.itertuples(index=False):
        for start, length in zip(_int_list(starts), _int_list(lengths)):
            rows.append((fiber, start, start + length))
    return pd.DataFrame(rows, columns=["mol_id", "start", "end"])


def _test_mol_ids(ddir):
    tsv = pd.read_csv(Path(ddir) / "synth.tsv", sep="\t", usecols=["read_id"])
    return list(dict.fromkeys(tsv["read_id"]))


def _dataset_length(ddir):
    sizes = pd.read_csv(Path(ddir) / "synth.size", sep="\t")
    return int(sizes["length"].iloc[0])


def _ft_calls(bam, ddir, env):
    table = Path(bam).with_suffix(".tsv.gz")
    run_in_env(env, ["ft", "extract", str(bam), "--all", str(table)])
    return intervals_to_calls(
        parse_ft_extract(table), _test_mol_ids(ddir), _dataset_length(ddir)
    )


def run_ft(
    ddir: str | Path,
    workdir: str | Path,
    *,
    env: str = ENV,
    nucleosome_length: int | None = None,
    combined_length: int | None = None,
) -> Calls:
    """
    Call nucleosomes with `ft add-nucleosomes`.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory (with `synth.bam`).
    workdir : str or Path
        Directory for intermediate files.
    env : str, default "catella"
        Conda environment providing fibertools.
    nucleosome_length, combined_length : int, optional
        Values for `--nucleosome-length` and
        `--combined-nucleosome-length`; tool defaults if None.

    Returns
    -------
    Calls
    """
    out = Path(workdir) / "ft.bam"
    args = ["ft", "add-nucleosomes", "--ml", "128"]
    if nucleosome_length is not None:
        args += ["--nucleosome-length", str(nucleosome_length)]
    if combined_length is not None:
        args += ["--combined-nucleosome-length", str(combined_length)]
    run_in_env(env, [*args, str(Path(ddir) / "synth.bam"), str(out)])
    return _ft_calls(out, ddir, env)


def _fiberhmm_channel(channel):
    if channel not in FIBERHMM_CHANNELS:
        raise ValueError(
            f"channel must be one of {sorted(FIBERHMM_CHANNELS)}, "
            f"got {channel!r}"
        )
    return FIBERHMM_CHANNELS[channel]


def train_fiberhmm(
    ddir: str | Path,
    *,
    env: str = ENV,
    iterations: int = 10,
    channel: str = "a",
) -> Path:
    """
    Train a FiberHMM model on a dataset, using its controls.

    Emission tables come from the methylated (accessible) and
    unmethylated (inaccessible) controls; the model is then trained on
    the test reads. No ground truth is used. The model is cached in the
    dataset directory and reused on later calls.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory (with `synth.bam`, `synth.meth.bam`,
        `synth.unmeth.bam`, or their `.gpc.bam` versions for GpC).
    env : str, default "catella"
        Conda environment providing FiberHMM.
    iterations : int, default 10
        Training iterations (random initializations).
    channel : {"a", "m"}, default "a"
        Labelling channel: "a" for m6A (FiberHMM mode `pacbio-fiber`),
        "m" for GpC 5mC (mode `gpc`, files `synth*.gpc.bam`).

    Returns
    -------
    Path
        The trained model file.
    """
    suffix, mode = _fiberhmm_channel(channel)
    ddir = Path(ddir)
    model = ddir / f"fiberhmm_model{suffix}" / "best-model.json"
    if model.exists():
        return model
    probs = ddir / f"fiberhmm_probs{suffix}"
    run_in_env(
        env,
        [
            "fiberhmm-probs",
            "-a",
            str(ddir / f"synth.meth{suffix}.bam"),
            "-u",
            str(ddir / f"synth.unmeth{suffix}.bam"),
            "-o",
            str(probs),
            "--mode",
            mode,
            "-p",
            "128",
            "-k",
            "3",
        ],
    )
    tables = probs / "tables"
    run_in_env(
        env,
        [
            "fiberhmm-train",
            "-i",
            str(ddir / f"synth{suffix}.bam"),
            "-p",
            str(next(tables.glob("*_accessible_*_k3.tsv"))),
            str(next(tables.glob("*_inaccessible_*_k3.tsv"))),
            "-o",
            str(model.parent),
            "--mode",
            mode,
            "-k",
            "3",
            "-c",
            str(iterations),
            "--prob-threshold",
            "128",
        ],
    )
    return model


def run_fiberhmm(
    ddir: str | Path,
    workdir: str | Path,
    *,
    env: str = ENV,
    retrain: bool = True,
    nuc_min_size: int | None = None,
    split_min_llr: float | None = None,
    iterations: int = 10,
    channel: str = "a",
) -> Calls:
    """
    Call nucleosomes with `fiberhmm-call`.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    env : str, default "catella"
        Conda environment providing FiberHMM and fibertools.
    retrain : bool, default True
        If True, use a model trained on this dataset (see
        `train_fiberhmm`); otherwise the pretrained Hia5 PacBio model.
    nuc_min_size : int, optional
        Value for `--nuc-min-size`; tool default if None.
    split_min_llr : float, optional
        Value for `--split-min-llr`; tool default if None.
    iterations : int, default 10
        Training iterations if `retrain`.
    channel : {"a", "m"}, default "a"
        Labelling channel; see `train_fiberhmm`. The pretrained model
        is m6A only.

    Returns
    -------
    Calls
    """
    suffix, _ = _fiberhmm_channel(channel)
    if not retrain and channel != "a":
        raise ValueError(
            f"pretrained FiberHMM models are m6A only; channel must be "
            f"'a' when retrain is False, got {channel!r}"
        )
    out = Path(workdir) / "fiberhmm.bam"
    args = ["fiberhmm-call", "-i", str(Path(ddir) / f"synth{suffix}.bam")]
    args += ["-o", str(out), "--prob-threshold", "128", "-c", "2"]
    if retrain:
        model = train_fiberhmm(
            ddir, env=env, iterations=iterations, channel=channel
        )
        args += ["-m", str(model)]
    else:
        args += ["--enzyme", "hia5", "--seq", "pacbio"]
    if nuc_min_size is not None:
        args += ["--nuc-min-size", str(nuc_min_size)]
    if split_min_llr is not None:
        args += ["--split-min-llr", str(split_min_llr)]
    run_in_env(env, args)
    return _ft_calls(out, ddir, env)


def run_catella(
    ddir: str | Path,
    workdir: str | Path,
    *,
    llink: int = 50,
    elink: float = 25.0,
    mu: float = 0.0,
    emax: float | None = None,
    seed: int = 1,
    mtase: str | list[str] = "A",
    q_thres: float | None = None,
) -> Calls:
    """
    Compute nucleosome occupancy with catella's exact (TM) solver.

    Uses both controls for the methylation probability calibration and
    reads both strands of each molecule (`ignore_strand=True`). By
    default, `emax` is estimated from the data.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    llink : int, default 50
        `SimSettings.llink`.
    elink : float, default 25.0
        `SimSettings.elink`.
    mu : float, default 0.0
        `SimSettings.mu`.
    emax : float, optional
        `SimSettings.emax`. If None (default), the 99th percentile of
        abs(logit(meth_prob)) from `catella.estimate_start_temp`.
    seed : int, default 1
        Seed passed to `catella.run` and `catella.estimate_start_temp`.
    mtase : str or list of str, default "A"
        Methyltransferase label(s) passed to `catella.load_raw`, e.g.,
        `["A", "GC"]` for m6A and GpC labelling.
    q_thres : float, optional
        Passed to `catella.compute_model_prob`; if given, confidence
        scores are binarized at this value.

    Returns
    -------
    Calls
        Occupancy probabilities and expected nucleosome counts.
    """
    ddir = Path(ddir)
    exp = catella.load_raw(
        chromsize=ddir / "synth.size",
        test_file=ddir / "synth.tsv",
        unmeth_file=ddir / "synth.unmeth.tsv",
        meth_file=ddir / "synth.meth.tsv",
        fasta_file=ddir / "synth.fa",
        mtase=mtase,
        ignore_strand=True,
        tmp_dir=workdir,
    )
    exp = catella.compute_model_prob(exp=exp, q_thres=q_thres)
    chrom = exp.chroms[0]
    if emax is None:
        emax = catella.estimate_start_temp(exp, percentile=99.0, seed=seed)
    settings = SimSettings(
        nucbp=147, llink=llink, elink=elink, mu=mu, end_temp=1.0, emax=emax
    )
    sim_dir = Path(tempfile.mkdtemp(dir=workdir)) / "sim"
    sim = catella.run(
        chroms=chrom,
        nsim=0,
        settings=settings,
        meth_prob={chrom: exp.analysis[chrom]["meth_prob"]},
        out_dir=sim_dir,
        method="tm",
        seed=seed,
        verbose=False,
    )
    catella.analyze(sim)
    occ = np.asarray(sim.analysis[chrom]["occup"][:], dtype=np.float32)
    count = np.asarray(sim.analysis[chrom]["mean_nnuc"][:]).ravel()
    mol_ids = np.asarray(exp.raw[chrom].test_mol_id)
    return Calls(mol_ids, occ, count)


def _hiddenfoot_binary(binary=None):
    name = binary or os.environ.get("HIDDENFOOT") or "hiddenfoot_omp"
    found = shutil.which(name)
    return os.path.abspath(found) if found else None


def hiddenfoot_windows(nbp: int, window: int, step: int) -> list[int]:
    """
    List the window starts used to run HiddenFoot on a molecule.

    Parameters
    ----------
    nbp : int
        Molecule length in bp.
    window : int
        Window length in bp.
    step : int
        Distance between consecutive window starts in bp.

    Returns
    -------
    list of int
        Window starts every `step` bp from 0, with the last window
        ending at `nbp`.
    """
    if not 0 < window <= nbp:
        raise ValueError(f"window must be in [1, {nbp}], got {window}")
    if not 0 < step <= window:
        raise ValueError(f"step must be in [1, {window}], got {step}")
    starts = list(range(0, nbp - window + 1, step))
    if starts[-1] != nbp - window:
        starts.append(nbp - window)
    return starts


def write_hiddenfoot_input(
    ddir: str | Path,
    workdir: str | Path,
    window: int,
    step: int,
    mod_code: str = "a",
) -> list[tuple[str, int]]:
    """
    Write HiddenFoot input files for a dataset cut into windows.

    Every window of every molecule becomes one row of the methylation
    matrix of a single `window`-bp region. Without position weight
    matrices, HiddenFoot's model does not depend on the DNA sequence,
    so the sequence file holds a dummy sequence and the weight matrix
    file is empty. Sites are coded 1 if `mod_qual` >= 0.5, 0 otherwise,
    and 2 where a row has no site.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for the files `seq.txt`, `wms.txt`, and `met.txt`.
    window : int
        Window length in bp.
    step : int
        Distance between consecutive window starts in bp.
    mod_code : str, default "a"
        Modification code of the calls used (e.g., "a" for m6A, "m" for
        5mC in GpC); other calls are ignored.

    Returns
    -------
    list of tuple of (str, int)
        Molecule id and window start of each matrix row.
    """
    workdir = Path(workdir)
    tsv = pd.read_csv(
        Path(ddir) / "synth.tsv",
        sep="\t",
        usecols=["read_id", "ref_position", "mod_qual", "mod_code"],
        dtype={"mod_code": str},
    )
    tsv = tsv[tsv["mod_code"] == mod_code].copy()
    tsv["state"] = (tsv["mod_qual"] >= 0.5).astype(int)
    mol_ids = _test_mol_ids(ddir)
    blocks, keys = [], []
    for start in hiddenfoot_windows(_dataset_length(ddir), window, step):
        pos = tsv["ref_position"]
        sel = tsv[(pos >= start) & (pos < start + window)]
        block = sel.pivot(
            index="read_id", columns="ref_position", values="state"
        ).reindex(mol_ids)
        block.columns = block.columns - start
        blocks.append(block)
        keys += [(mol_id, start) for mol_id in mol_ids]
    mat = pd.concat(blocks, ignore_index=True)
    mat = mat.reindex(columns=sorted(mat.columns)).fillna(2).astype(int)
    (workdir / "seq.txt").write_text("A" * window + "\n")
    (workdir / "wms.txt").write_text("")
    with open(workdir / "met.txt", "w") as fh:
        fh.write(" ".join(map(str, mat.columns)) + "\n")
        mat.to_csv(
            fh, sep=" ", header=False, index=False, lineterminator="\n"
        )
    return keys


def stitch_windows(
    values: np.ndarray,
    keys: list[tuple[str, int]],
    mol_ids: list[str],
    nbp: int,
    window: int,
    step: int,
) -> np.ndarray:
    """
    Combine per-window profiles into per-molecule profiles.

    Each base takes its value from the window whose centre is nearest.

    Parameters
    ----------
    values : np.ndarray
        (nrow, window) profile of each window, in the order of `keys`.
    keys : list of tuple of (str, int)
        Molecule id and window start of each row of `values`.
    mol_ids : list of str
        Molecule ids, one per output row.
    nbp : int
        Molecule length in bp.
    window : int
        Window length in bp.
    step : int
        Distance between consecutive window starts in bp.

    Returns
    -------
    np.ndarray
        (len(mol_ids), nbp) profile of each molecule.
    """
    starts = hiddenfoot_windows(nbp, window, step)
    centres = np.array(starts) + window / 2
    owner = np.abs(np.arange(nbp)[:, None] - centres).argmin(axis=1)
    owned = {s: np.flatnonzero(owner == k) for k, s in enumerate(starts)}
    row = {mol_id: i for i, mol_id in enumerate(mol_ids)}
    out = np.zeros((len(mol_ids), nbp))
    for vals, (mol_id, start) in zip(values, keys):
        x = owned[start]
        out[row[mol_id], x] = vals[x - start]
    return out


def run_hiddenfoot(
    ddir: str | Path,
    workdir: str | Path,
    *,
    nuclen: int = 147,
    window: int = HF_WINDOW,
    step: int = HF_STEP,
    threads: int = 2,
    binary: str | None = None,
    mod_code: str = "a",
) -> Calls:
    """
    Call nucleosomes with HiddenFoot in nucleosome-only mode.

    HiddenFoot fits its parameters on the dataset itself and uses no
    controls. The released implementation underflows on sequences
    longer than about 2000 bp, so molecules are cut into overlapping
    windows (see `write_hiddenfoot_input` and `stitch_windows`) and one
    parameter set is fitted to all windows. HiddenFoot seeds its random
    number generator from the clock, so calls vary slightly between
    runs.

    Parameters
    ----------
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    nuclen : int, default 147
        Nucleosome footprint length (`-nuclen`).
    window : int, default `HF_WINDOW`
        Window length in bp; at most `HF_MAX_WINDOW`.
    step : int, default `HF_STEP`
        Distance between consecutive window starts in bp.
    threads : int, default 2
        Number of threads (`-numthreads`).
    binary : str, optional
        HiddenFoot executable; defaults to `$HIDDENFOOT`, else
        `hiddenfoot_omp` on the `PATH`.
    mod_code : str, default "a"
        Modification code of the calls used; see
        `write_hiddenfoot_input`.

    Returns
    -------
    Calls
        Posterior occupancy and expected nucleosome counts.
    """
    if window > HF_MAX_WINDOW:
        raise ValueError(
            f"window must be at most {HF_MAX_WINDOW} to avoid HiddenFoot "
            f"likelihood underflow, got {window}"
        )
    exe = _hiddenfoot_binary(binary)
    if exe is None:
        raise FileNotFoundError(
            "HiddenFoot executable not found; set $HIDDENFOOT or pass "
            "binary"
        )
    workdir = Path(workdir)
    keys = write_hiddenfoot_input(ddir, workdir, window, step, mod_code)
    proc = subprocess.run(
        [
            exe,
            "run",
            "-s",
            "seq.txt",
            "-m",
            "met.txt",
            "-w",
            "wms.txt",
            "-o",
            "hf",
            "-nuclen",
            str(nuclen),
            "-numthreads",
            str(threads),
        ],
        cwd=workdir,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"HiddenFoot failed; {proc.stderr.strip()}")
    last = (workdir / "hf.params_SGD.txt").read_text().splitlines()[-1]
    if not np.isfinite(float(last.split()[1])):
        raise RuntimeError(
            "HiddenFoot likelihood underflowed; use a smaller window"
        )
    mol_ids = _test_mol_ids(ddir)
    occ, start = (
        stitch_windows(
            np.loadtxt(workdir / f"hf.{name}.nucleosome.txt", ndmin=2),
            keys,
            mol_ids,
            _dataset_length(ddir),
            window,
            step,
        )
        for name in ("hfprofile", "hfbinding")
    )
    return Calls(np.asarray(mol_ids), occ.astype(np.float32), start.sum(1))


def save_calls(calls: Calls, path: str | Path) -> Path:
    """
    Save calls to an HDF5 file.

    The file holds datasets `mol_ids` (bytes), `occ` (float16,
    gzip-compressed), and `count`.

    Parameters
    ----------
    calls : Calls
        Calls to save.
    path : str or Path
        Output file path.

    Returns
    -------
    Path
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as fh:
        fh.create_dataset("mol_ids", data=calls.mol_ids.astype("S"))
        fh.create_dataset(
            "occ", data=calls.occ.astype(np.float16), compression="gzip"
        )
        fh.create_dataset("count", data=calls.count)
    return path


def load_calls(path: str | Path) -> Calls:
    """
    Load calls saved by `save_calls`.

    Parameters
    ----------
    path : str or Path
        File written by `save_calls`.

    Returns
    -------
    Calls
    """
    with h5py.File(path, "r") as fh:
        return Calls(
            fh["mol_ids"][:].astype(str),
            fh["occ"][:].astype(np.float32),
            fh["count"][:],
        )


TOOLS = {
    "catella": (run_catella, {}),
    "catella_q05": (run_catella, {"q_thres": 0.5}),
    "ft": (run_ft, {}),
    "fiberhmm": (run_fiberhmm, {"retrain": True}),
    "fiberhmm_pre": (run_fiberhmm, {"retrain": False}),
    "hiddenfoot": (run_hiddenfoot, {}),
}
# Tools whose runners take no conda environment
NO_ENV_TOOLS = ("catella", "catella_q05", "hiddenfoot")


def run_tool(
    tool: str,
    ddir: str | Path,
    workdir: str | Path,
    params: dict,
    env: str = ENV,
) -> Calls:
    """
    Run one tool on one dataset.

    Parameters
    ----------
    tool : str
        A key of `TOOLS`.
    ddir : str or Path
        Dataset directory.
    workdir : str or Path
        Directory for intermediate files.
    params : dict
        Keyword arguments for the tool's runner (e.g., a grid point).
    env : str, default "catella"
        Conda environment for the external tools.

    Returns
    -------
    Calls
    """
    runner, fixed = TOOLS[tool]
    if tool in NO_ENV_TOOLS:
        return runner(ddir, workdir, **fixed, **params)
    return runner(ddir, workdir, env=env, **fixed, **params)
