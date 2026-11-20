# _qc.py

import numpy as np

from catella.experiment._calibration import _fold_contexts
from catella.experiment._contexts import CONTEXT_NAMES, _strand_assay_masks
from catella.experiment._controls import _streamed_call_count
from catella.experiment._dropout import (
    _channel_site_masks,
    _dropout_fracs,
    _fold_site_masks,
    _site_coverage,
    _strand_of_mol,
)

_QC_SOURCES = ("test", "meth", "unmeth")


def _check_percentiles(percentiles):
    """
    Validate dropout percentiles.

    Parameters
    ----------
    percentiles : float or iterable of float
        Percentiles in [0, 100].

    Returns
    -------
    tuple of float

    Raises
    ------
    ValueError
        If `percentiles` is empty or has a value outside [0, 100].
    """
    qs = tuple(float(q) for q in np.atleast_1d(percentiles))
    if not qs:
        raise ValueError("percentiles must not be empty")
    bad = [q for q in qs if not 0.0 <= q <= 100.0]
    if bad:
        raise ValueError(f"percentiles must be in [0, 100], got {bad}")
    return qs


def _ratio(num, den):
    """`num / den`, or NaN if `den` is 0."""
    return float(num / den) if den else float("nan")


def _source_stats(exp, chrom, src, site_masks, percentiles, mask_name):
    """
    Molecule, strand and dropout statistics of one source.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment.
    chrom : str
        Chromosome identifier.
    src : {"test", "meth", "unmeth"}
        Source to summarize.
    site_masks : dict of str to (np.ndarray, np.ndarray)
        From `_channel_site_masks`.
    percentiles : tuple of float
        Dropout percentiles to report.
    mask_name : str or None
        Dropout mask; masked molecules are left out.

    Returns
    -------
    dict
        `nmol` (kept molecules), `strand` (`frac_plus`/`frac_minus`/
        `frac_unmapped` over kept molecules; empty if
        `exp.ignore_strand`), `frac_masked`, and `dropout` (percentiles
        per channel name, plus "all" pooled).
    """
    raw = exp.raw[chrom]
    df = getattr(raw, f"{src}_data")
    nmol_all = len(getattr(raw, f"{src}_mol_id"))
    keep = exp.is_kept(chrom, which=src, mask_name=mask_name)
    strand = _strand_of_mol(df, nmol_all)
    kept = strand[keep]

    strand_fracs = {}
    if not exp.ignore_strand:
        for key, s in (("plus", "+"), ("minus", "-"), ("unmapped", ".")):
            strand_fracs[f"frac_{key}"] = _ratio((kept == s).sum(), len(kept))

    cov = _site_coverage(
        df,
        nmol_all,
        site_masks,
        "union" if exp.ignore_strand else "drop",
        ignore_strand=exp.ignore_strand,
    )
    names = list(site_masks)
    frac, pooled = _dropout_fracs(cov, names)
    sel = keep if exp.ignore_strand else keep & (strand != ".")

    def pct(x):
        if not sel.any():
            return [float("nan")] * len(percentiles)
        return [float(v) for v in np.percentile(x[sel], percentiles)]

    dropout = {name: pct(frac[i]) for i, name in enumerate(names)}
    dropout["all"] = pct(pooled)
    return {
        "nmol": int(keep.sum()),
        "strand": strand_fracs,
        "frac_masked": _ratio((~keep).sum(), nmol_all),
        "dropout": dropout,
    }


def _channel_rates(exp, chrom, codes, sources, mask_name, batch_size):
    """
    Pooled methylation rate per channel and source.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment.
    chrom : str
        Chromosome identifier.
    codes : tuple of int
        Channel codes, as from `_resolve_channels`.
    sources : list of str
        Sources present for `chrom`.
    mask_name : str or None
        Dropout mask; masked molecules are left out.
    batch_size : int
        Molecules per streamed batch.

    Returns
    -------
    dict of str to dict of str to float
        Maps each source to `{channel name: rate}`.
    """
    ctx = _fold_contexts(exp, chrom, codes)
    assay = _strand_assay_masks(exp, chrom, ctx)
    rates = {}
    for src in sources:
        k, n = _streamed_call_count(
            exp, chrom, src, ctx, assay, batch_size, mask_name
        )
        rates[src] = {
            CONTEXT_NAMES[c]: _ratio(k[ctx == c].sum(), n[ctx == c].sum())
            for c in codes
        }
    return rates


def _rate_cols(rates, name):
    """`meth_frac_<src>` and `accessible_frac` of one channel."""
    cols = {
        f"meth_frac_{src}": rates[src][name]
        for src in ("test", "unmeth", "meth")
        if src in rates
    }
    if "meth" in rates and "unmeth" in rates:
        unmeth = rates["unmeth"][name]
        cols["accessible_frac"] = _ratio(
            rates["test"][name] - unmeth, rates["meth"][name] - unmeth
        )
    return cols


def _site_cols(site_masks, parts, ignore_strand):
    """Site counts of the channels in `parts`."""
    if ignore_strand:
        return {
            "nsite": sum(
                int((site_masks[p][0] | site_masks[p][1]).sum())
                for p in parts
            )
        }
    return {
        "nsite_plus": sum(int(site_masks[p][0].sum()) for p in parts),
        "nsite_minus": sum(int(site_masks[p][1].sum()) for p in parts),
    }


def _chrom_qc_rows(exp, chrom, codes, percentiles, mask_name, batch_size):
    """
    QC table rows of one chromosome.

    Parameters
    ----------
    exp : MethPrintExperiment
        The experiment.
    chrom : str
        Chromosome identifier.
    codes : tuple of int
        Channel codes, as from `_resolve_channels`.
    percentiles : tuple of float
        Dropout percentiles to report.
    mask_name : str or None
        Dropout mask; masked molecules are left out.
    batch_size : int
        Molecules per streamed batch.

    Returns
    -------
    list of dict
        One row per channel, plus an "all" row if more than one
        channel. Columns are described in `MethPrintExperiment.qc`;
        columns of a missing control source are left out.

    Raises
    ------
    ValueError
        If `chrom` has no reference sequence or no test data.
    """
    raw = exp.raw[chrom]
    if raw.refseq is None:
        raise ValueError(
            f"no refseq available for chrom '{chrom}'; "
            "required to determine methylatable positions"
        )
    if raw.test_data is None:
        raise ValueError(f"no 'test' data available for chrom '{chrom}'")
    sources = [s for s in _QC_SOURCES if getattr(raw, f"{s}_data") is not None]

    site_masks = _channel_site_masks(raw.refseq, codes)
    if exp.wrap:
        site_masks = _fold_site_masks(site_masks, raw.nbp)
    stats = {
        s: _source_stats(exp, chrom, s, site_masks, percentiles, mask_name)
        for s in sources
    }
    rates = _channel_rates(exp, chrom, codes, sources, mask_name, batch_size)

    names = list(site_masks)
    if len(names) > 1:
        names.append("all")
    rows = []
    for name in names:
        parts = list(site_masks) if name == "all" else [name]
        row = {"chrom": chrom, "channel": name}
        row.update(_site_cols(site_masks, parts, exp.ignore_strand))
        row["nmol"] = stats["test"]["nmol"]
        for s in sources:
            for key, val in stats[s]["strand"].items():
                row[f"{key}_{s}"] = val
        for s in sources:
            row[f"frac_masked_{s}"] = stats[s]["frac_masked"]
        for s in sources:
            vals = stats[s]["dropout"][name]
            for q, val in zip(percentiles, vals, strict=True):
                row[f"dropout_{s}_p{q:g}"] = val
        if name != "all":
            row.update(_rate_cols(rates, name))
        rows.append(row)
    return rows
