# _chroms.py

# Helpers for chromosome arguments, shared by the experiment and
# simulation packages


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
