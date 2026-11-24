# _regions.py

# Helpers for region arguments, shared by the experiment and
# simulation packages

from collections.abc import Iterable
from numbers import Integral
from pathlib import Path
from typing import NamedTuple

import numpy as np

from catella._warnings import _warn

# Region names become HDF5 group names and simulation run path components
_UNSAFE_NAME_CHARS = '/\\:*?"<>|'
_MAX_NAME_LEN = 100
# Dense per-molecule arrays scale with region length: warn above
# _WARN_REGION_BP, refuse above _MAX_REGION_BP (beyond realistic reads)
_WARN_REGION_BP = 100000
_MAX_REGION_BP = 1000000
_VALID_STRANDS = ("+", "-", ".")
# BED lines that carry no interval
_BED_HEADER_PREFIXES = ("#", "track", "browser")


def _check_region_names(names):
    """
    Check that region names are safe as HDF5 group and path names.

    Parameters
    ----------
    names : iterable of str
        The region names.

    Raises
    ------
    ValueError
        If a name is empty, "." or "..", has leading or trailing
        whitespace, contains one of `/ \\ : * ? " < > |`, is longer than
        100 characters, or equals another name ignoring case (such
        names would share a directory on case-insensitive file
        systems).
    """
    seen = {}
    for name in names:
        if not isinstance(name, str) or name in ("", ".", ".."):
            raise ValueError(f"invalid region name {name!r}")
        if name != name.strip():
            raise ValueError(
                f"region name {name!r} has leading or trailing whitespace"
            )
        bad = sorted({c for c in name if c in _UNSAFE_NAME_CHARS})
        if bad:
            raise ValueError(
                f"region name {name!r} contains {''.join(bad)!r}; names "
                f"must not contain any of {_UNSAFE_NAME_CHARS!r}"
            )
        if len(name) > _MAX_NAME_LEN:
            raise ValueError(
                f"region name {name!r} is longer than {_MAX_NAME_LEN} "
                "characters"
            )
        key = name.lower()
        if key in seen:
            raise ValueError(
                f"region names must be unique ignoring case, got "
                f"{seen[key]!r} and {name!r}"
            )
        seen[key] = name


def _check_region_lengths(lengths):
    """
    Check that regions are short enough for dense per-molecule arrays.

    Parameters
    ----------
    lengths : dict of str to int
        Region name -> stored length (bp).

    Raises
    ------
    ValueError
        If a region is longer than 1 Mb.

    Warns
    -----
    UserWarning
        If a region is longer than 100 kb.
    """
    too_long = {r: n for r, n in lengths.items() if n > _MAX_REGION_BP}
    if too_long:
        region, nbp = next(iter(too_long.items()))
        raise ValueError(
            f"region {region!r} is {nbp} bp; catella analyzes regions of "
            f"at most {_MAX_REGION_BP} bp, so pass regions= to select "
            "shorter ones"
        )
    long = [r for r, n in lengths.items() if n > _WARN_REGION_BP]
    if long:
        _warn(
            f"{len(long)} region(s) longer than {_WARN_REGION_BP} bp (e.g. "
            f"{long[0]!r}); memory and time grow with region length, so "
            "lower batch_size if needed"
        )


def _normalize_regions(
    regions: str | Iterable[str] | None = None,
    default_regions: Iterable[str] | None = None,
) -> list[str]:
    """
    Return a region argument as a list.

    Parameters
    ----------
    regions : str or iterable of str, optional
        One region or several. If None (default), `default_regions`
        is used.
    default_regions : iterable of str, optional
        The regions used when `regions` is None. If both are None, an
        empty list is returned.

    Returns
    -------
    list of str
        The regions, in the given order.

    Raises
    ------
    ValueError
        If `regions` is an empty string.
    TypeError
        If `regions` is neither None, a str, nor an iterable.
    """
    if regions is None:
        return list(default_regions) if default_regions is not None else []
    if isinstance(regions, str):
        if regions == "":
            raise ValueError("regions must not be an empty string")
        return [regions]
    if not isinstance(regions, Iterable):
        raise TypeError("regions must be a str or an iterable")
    return list(regions)


def _region_list(regions):
    """
    Return `regions` (a str or an iterable of str) as a list.

    Parameters
    ----------
    regions : str or iterable of str
        One region or several.

    Returns
    -------
    list of str
        The regions, in the given order.

    Raises
    ------
    ValueError
        If `regions` is empty.
    """
    regions = [regions] if isinstance(regions, str) else list(regions)
    if not regions:
        raise ValueError("regions must not be empty")
    return regions


def _per_region(value, regions, name):
    """
    Map a per-region argument onto each region of `regions`.

    Parameters
    ----------
    value : object, dict, or None
        None (None for every region), a dict keyed by region
        (missing regions get None; extra keys are ignored), or a
        single value (only allowed for one region).
    regions : list of str
        The regions.
    name : str
        The argument name, for the error message.

    Returns
    -------
    dict
        The value for each region of `regions`.

    Raises
    ------
    ValueError
        If `value` is a single value and `regions` has several entries.
    """
    if value is None:
        return dict.fromkeys(regions)
    if isinstance(value, dict):
        return {r: value.get(r) for r in regions}
    if len(regions) > 1:
        raise ValueError(
            f"{name} must be a dict keyed by region when several regions are "
            "given"
        )
    return {regions[0]: value}


class Region(NamedTuple):
    """One region: a half-open, 0-based reference interval."""

    chrom: str
    start: int
    end: int
    name: str
    strand: str


def _parse_regions(regions):
    """
    Interpret a `regions` argument of `load_raw`.

    Parameters
    ----------
    regions : None, str, pathlib.Path, or iterable
        None; a BED file (a str or Path naming an existing file); a
        region name (any other str); an iterable of names (whole
        chromosomes); or an iterable of `(chrom, start, end[, name[,
        strand]])` tuples (regions given as intervals).

    Returns
    -------
    kind : {"all", "names", "regions"}
        What `regions` holds. Names become whole-chromosome regions
        once their lengths are known.
    value : None, list of str, or list of Region
        Nothing for "all", the names, or the validated regions.

    Raises
    ------
    ValueError
        If `regions` is empty, a Path does not name an existing file, a
        region is invalid (see `_make_region`), or region names are
        unsafe (see `_check_region_names`).
    TypeError
        If `regions` has an unsupported type or mixes names and tuples.
    """
    if regions is None:
        return "all", None
    if isinstance(regions, Path):
        if not regions.is_file():
            raise ValueError(f"regions file {str(regions)!r} not found")
        return "regions", _read_bed(regions)
    if isinstance(regions, str):
        if Path(regions).is_file():
            return "regions", _read_bed(regions)
        return "names", [regions]
    if not isinstance(regions, Iterable):
        raise TypeError(
            "regions must be None, a str, a Path, or an iterable, got "
            f"{type(regions).__name__}"
        )
    items = list(regions)
    if not items:
        raise ValueError("regions must not be empty")
    if all(isinstance(r, str) for r in items):
        return "names", items
    if all(isinstance(r, (tuple, list)) for r in items):
        parsed = [_make_region(*r) for r in items]
        _check_region_names([r.name for r in parsed])
        return "regions", parsed
    raise TypeError(
        "regions must hold only names (str) or only (chrom, start, end"
        "[, name[, strand]]) tuples, not a mix"
    )


def _make_region(chrom, start, end, name=None, strand=None, *extra):
    """
    Build and validate one region given as an interval.

    Parameters
    ----------
    chrom : str
        Reference chromosome.
    start, end : int
        Half-open, 0-based interval; `0 <= start < end`.
    name : str, optional
        Region name; None or "." gives `chrom_start_end`.
    strand : {"+", "-", "."}, optional
        Region strand (annotation only); None gives ".".
    *extra
        Must be empty.

    Returns
    -------
    Region

    Raises
    ------
    ValueError
        If there are more than 5 fields, `start < 0`, `end <= start`,
        or `strand` is invalid.
    TypeError
        If `start` or `end` is not an integer.
    """
    if extra:
        raise ValueError(
            "a region must be (chrom, start, end[, name[, strand]]), got "
            f"{len(extra) + 5} fields"
        )
    chrom = str(chrom)
    for label, value in (("start", start), ("end", end)):
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise TypeError(
                f"region {chrom}:{start}-{end}: {label} must be an "
                f"integer, got {value!r}"
            )
    start, end = int(start), int(end)
    if not 0 <= start < end:
        raise ValueError(
            f"region {chrom}:{start}-{end} must have 0 <= start < end"
        )
    if name is None or name == ".":
        name = f"{chrom}_{start}_{end}"
    strand = "." if strand is None else str(strand)
    if strand not in _VALID_STRANDS:
        raise ValueError(
            f"region {name!r}: strand must be one of {_VALID_STRANDS}, "
            f"got {strand!r}"
        )
    return Region(chrom, start, end, str(name), strand)


def _read_bed(path):
    """
    Read regions given as intervals from a BED file.

    Columns 1-3 give the interval (0-based, half-open), column 4 the
    name and column 6 the strand, when present. Fields are
    tab-separated; blank lines and lines starting with `#`, `track` or
    `browser` are skipped.

    Parameters
    ----------
    path : str or pathlib.Path
        The BED file.

    Returns
    -------
    list of Region

    Raises
    ------
    ValueError
        If a line has fewer than 3 fields, a coordinate is not an
        integer, the file holds no regions, or a region or name is
        invalid.
    """
    parsed = []
    with open(path) as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.rstrip("\n\r")
            if not line.strip() or line.startswith(_BED_HEADER_PREFIXES):
                continue
            fields = line.split("\t")
            if len(fields) < 3:
                raise ValueError(
                    f"{path}, line {lineno}: expected at least 3 "
                    f"tab-separated fields, got {len(fields)}"
                )
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError:
                raise ValueError(
                    f"{path}, line {lineno}: start and end must be "
                    f"integers, got {fields[1]!r} and {fields[2]!r}"
                ) from None
            name = fields[3] if len(fields) > 3 and fields[3] else None
            strand = fields[5] if len(fields) > 5 and fields[5] else None
            parsed.append(_make_region(fields[0], start, end, name, strand))
    if not parsed:
        raise ValueError(f"{path} holds no regions")
    try:
        _check_region_names([r.name for r in parsed])
    except ValueError as err:
        raise ValueError(
            f"{err}; make the names in column 4 of {path} unique and "
            "path-safe"
        ) from None
    return parsed


class _RegionMap:
    """
    Assignment of reference rows to regions given as intervals.

    Parameters
    ----------
    regions : list of Region
        From `_parse_regions`.
    """

    def __init__(self, regions):
        self.regions = {r.name: r for r in regions}
        self._by_chrom = {}
        for chrom in dict.fromkeys(r.chrom for r in regions):
            sel = sorted(
                (r for r in regions if r.chrom == chrom),
                key=lambda r: r.start,
            )
            self._by_chrom[chrom] = (
                np.array([r.start for r in sel], dtype=np.int64),
                np.array([r.end for r in sel], dtype=np.int64),
                np.array([r.name for r in sel], dtype=object),
            )

    @property
    def chroms(self):
        """Reference chromosomes holding at least one region."""
        return tuple(self._by_chrom)

    def split(self, group, chrom):
        """
        The rows of one chromosome in each region that holds them.

        Parameters
        ----------
        group : pd.DataFrame
            Rows of `chrom`, with `chrom` and `upos` columns.
        chrom : str
            The reference chromosome.

        Returns
        -------
        pd.DataFrame
            One copy of each row per region holding it, with `chrom` set
            to the region name and `upos` relative to the region start;
            rows keep their file order within a region, and regions
            follow their start order.
        """
        if chrom not in self._by_chrom or group.empty:
            return group.iloc[:0]
        starts, ends, names = self._by_chrom[chrom]
        upos = group["upos"].to_numpy()
        order = np.argsort(upos, kind="stable")
        lo = np.searchsorted(upos[order], starts, side="left")
        hi = np.searchsorted(upos[order], ends, side="left")
        counts = hi - lo
        region_idx = np.repeat(np.arange(len(starts)), counts)
        within = np.arange(counts.sum()) - np.repeat(
            np.cumsum(counts) - counts, counts
        )
        rows = order[np.repeat(lo, counts) + within]
        keep = np.lexsort((rows, region_idx))
        rows, region_idx = rows[keep], region_idx[keep]
        out = group.iloc[rows].copy()
        out["chrom"] = names[region_idx]
        out["upos"] = upos[rows] - starts[region_idx]
        return out
