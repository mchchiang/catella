# _batches.py


def _row_batches(start, stop, batch_size):
    """
    Bounds of consecutive batches covering rows `start` to `stop`.

    Parameters
    ----------
    start, stop : int
        The row range, as in `range(start, stop)`.
    batch_size : int
        The maximum number of rows per batch.

    Returns
    -------
    generator of tuple of int
        (batch_start, batch_stop) pairs; the last batch may be shorter.

    Raises
    ------
    ValueError
        If `batch_size` is not positive.
    """
    if batch_size < 1:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    return (
        (s, min(s + batch_size, stop)) for s in range(start, stop, batch_size)
    )
