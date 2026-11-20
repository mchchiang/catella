# _chroms.py

# Helpers for chromosome arguments, shared by the experiment and
# simulation packages

from collections.abc import Iterable


def _normalize_chroms(
    chroms: str | Iterable[str] | None = None,
    default_chroms: Iterable[str] | None = None,
) -> list[str]:
    """
    Return a chromosome argument as a list.

    Parameters
    ----------
    chroms : str or iterable of str, optional
        One chromosome or several. If None (default), `default_chroms`
        is used.
    default_chroms : iterable of str, optional
        The chromosomes used when `chroms` is None. If both are None, an
        empty list is returned.

    Returns
    -------
    list of str
        The chromosomes, in the given order.

    Raises
    ------
    ValueError
        If `chroms` is an empty string.
    TypeError
        If `chroms` is neither None, a str, nor an iterable.
    """
    if chroms is None:
        return list(default_chroms) if default_chroms is not None else []
    if isinstance(chroms, str):
        if chroms == "":
            raise ValueError("chroms must not be an empty string")
        return [chroms]
    if not isinstance(chroms, Iterable):
        raise TypeError("chroms must be a str or an iterable")
    return list(chroms)


def _chrom_list(chroms):
    """
    Return `chroms` (a str or an iterable of str) as a list.

    Parameters
    ----------
    chroms : str or iterable of str
        One chromosome or several.

    Returns
    -------
    list of str
        The chromosomes, in the given order.

    Raises
    ------
    ValueError
        If `chroms` is empty.
    """
    chroms = [chroms] if isinstance(chroms, str) else list(chroms)
    if not chroms:
        raise ValueError("chroms must not be empty")
    return chroms


def _per_chrom(value, chroms, name):
    """
    Map a per-chromosome argument onto each chromosome of `chroms`.

    Parameters
    ----------
    value : object, dict, or None
        None (None for every chromosome), a dict keyed by chromosome
        (missing chromosomes get None; extra keys are ignored), or a
        single value (only allowed for one chromosome).
    chroms : list of str
        The chromosomes.
    name : str
        The argument name, for the error message.

    Returns
    -------
    dict
        The value for each chromosome of `chroms`.

    Raises
    ------
    ValueError
        If `value` is a single value and `chroms` has several entries.
    """
    if value is None:
        return dict.fromkeys(chroms)
    if isinstance(value, dict):
        return {c: value.get(c) for c in chroms}
    if len(chroms) > 1:
        raise ValueError(
            f"{name} must be a dict keyed by chrom when several chroms are "
            "given"
        )
    return {chroms[0]: value}
