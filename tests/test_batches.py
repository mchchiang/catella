# test_batches.py

import pytest
from catella._batches import _row_batches


@pytest.mark.parametrize(
    ("start", "stop", "batch_size", "expected"),
    [
        (0, 10, 4, [(0, 4), (4, 8), (8, 10)]),
        (0, 8, 4, [(0, 4), (4, 8)]),
        (3, 7, 10, [(3, 7)]),
        (0, 3, 1, [(0, 1), (1, 2), (2, 3)]),
        (0, 0, 5, []),
        (5, 2, 5, []),
    ],
)
def test_row_batches_bounds(start, stop, batch_size, expected):
    assert list(_row_batches(start, stop, batch_size)) == expected


@pytest.mark.parametrize("batch_size", [0, -5])
def test_row_batches_nonpositive_raises_on_call(batch_size):
    with pytest.raises(
        ValueError, match=f"batch_size must be positive, got {batch_size}"
    ):
        _row_batches(0, 10, batch_size)
