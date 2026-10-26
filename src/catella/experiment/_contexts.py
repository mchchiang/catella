# _contexts.py

import numpy as np

from catella.experiment._ingest import _methylatable_positions

NONE, A, GCH, HCG, GCG = 0, 1, 2, 3, 4
CONTEXT_NAMES = {NONE: "none", A: "A", GCH: "GCH", HCG: "HCG", GCG: "GCG"}

# The four assayable context channels, and the reverse of
# CONTEXT_NAMES restricted to them, for parsing user eta overrides.
_ALL_CHANNELS = (A, GCH, HCG, GCG)
_CHANNEL_NAME_TO_CODE = {CONTEXT_NAMES[c]: c for c in _ALL_CHANNELS}

# Context channel(s) each mtase label actually methylates. "CG"/"GC"
# both include GCG, since that position matches both rules (see
# _reference_contexts).
_MTASE_CHANNELS = {"A": (A,), "CG": (HCG, GCG), "GC": (GCH, GCG)}


def _active_channels(exp):
    """
    Channels actually assayed by `exp.mtase`.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object.

    Returns
    -------
    tuple of int
        Subset of `_ALL_CHANNELS`, in `_ALL_CHANNELS` order. All of
        `_ALL_CHANNELS` if `exp.mtase` is unset (None).
    """
    if exp.mtase is None:
        return _ALL_CHANNELS
    active = {c for label in exp.mtase for c in _MTASE_CHANNELS[label]}
    return tuple(c for c in _ALL_CHANNELS if c in active)


def _reference_contexts(seq):
    """
    Classify every reference position into a footprinting context.

    Strand logic:
      m6A     : A on forward, or T on forward (= A on reverse)
      fwd CpG : C with G at i+1        rev CpG : G with C at i-1
      fwd GpC : C with G at i-1        rev GpC : G with C at i+1
    A position matching both CpG and GpC rules is GCG.

    Parameters
    ----------
    seq : str or bytes
        The reference nucleotide sequence.

    Returns
    -------
    np.ndarray
        int8, length `len(seq)`, one of NONE/A/GCH/HCG/GCG per
        position.
    """
    s = np.frombuffer(
        seq.upper().encode() if isinstance(seq, str) else seq.upper(),
        dtype="S1",
    )
    L = len(s)
    ctx = np.zeros(L, dtype=np.int8)

    prev = np.concatenate([[b"N"], s[:-1]])
    nxt = np.concatenate([s[1:], [b"N"]])

    ctx[(s == b"A") | (s == b"T")] = A

    fwd_C = s == b"C"
    rev_C = s == b"G"
    is_cg = (fwd_C & (nxt == b"G")) | (rev_C & (prev == b"C"))
    is_gc = (fwd_C & (prev == b"G")) | (rev_C & (nxt == b"C"))

    ctx[is_gc & ~is_cg] = GCH
    ctx[is_cg & ~is_gc] = HCG
    ctx[is_cg & is_gc] = GCG
    return ctx


def _strand_assay_masks(exp, chrom, ctx):
    """
    Positions a molecule can read, per strand label.

    A position is readable on a strand if it has a context in `ctx` and
    is methylatable on that strand by one of `exp.mtase` (all mtase
    labels if unset), following the row filter of `load_raw`.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment object containing raw data and analysis maps.
    chrom : str
        Chromosome identifier.
    ctx : np.ndarray
        int8, length L, context code per position (folded if
        `exp.wrap`).

    Returns
    -------
    dict of str to np.ndarray
        "+", "-", and "." -> bool, length L. "." (unmapped strand) is
        all False, since the sites of a molecule of unknown strand
        cannot be told; such a molecule reads only its called sites.
    """
    seq = exp.raw[chrom].refseq
    labels = exp.mtase if exp.mtase is not None else tuple(_MTASE_CHANNELS)
    full = {}
    for s in ("+", "-"):
        m = np.zeros(len(seq), dtype=bool)
        for label in labels:
            m |= _methylatable_positions(seq, label, s)
        full[s] = m
    if exp.ignore_strand:
        both = full["+"] | full["-"]
        full = {"+": both, "-": both}
    L = len(ctx)
    if exp.wrap:
        length = len(seq)
        full = {s: m[:L] | m[length - L :][::-1] for s, m in full.items()}
    ok = ctx != NONE
    masks = {s: m[:L] & ok for s, m in full.items()}
    masks["."] = np.zeros(L, dtype=bool)
    return masks
