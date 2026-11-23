# _regions.py

# Helpers for region arguments, shared by the experiment and
# simulation packages

from collections.abc import Iterable


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
