# test_jagged.py

import numpy as np
from catella._jagged import _interval_mask, _pack_jagged, _unpack_jagged


class TestJagged:
    def test_round_trip(self):
        arrays = [np.array([1, 5]), np.array([], dtype=int), np.array([7])]
        flat, offset = _pack_jagged(arrays)
        assert flat.dtype == np.int32
        assert offset.dtype == np.uint64
        np.testing.assert_array_equal(flat, [1, 5, 7])
        np.testing.assert_array_equal(offset, [0, 2, 2, 3])
        out = _unpack_jagged(flat, offset)
        assert len(out) == 3
        for got, expected in zip(out, arrays, strict=True):
            np.testing.assert_array_equal(got, expected)
            assert got.dtype == np.int32

    def test_empty(self):
        flat, offset = _pack_jagged([])
        assert flat.size == 0
        np.testing.assert_array_equal(offset, [0])
        assert _unpack_jagged(flat, offset) == ()

    def test_dtype(self):
        flat, _ = _pack_jagged([np.array([1.5])], dtype=np.float64)
        assert flat.dtype == np.float64


class TestIntervalMask:
    _INTERVALS = np.array([[1, 3], [4, 6], [0, 2]])
    _OFFSET = np.array([0, 2, 2, 3])

    def test_whole_range(self):
        mask = _interval_mask(self._INTERVALS, self._OFFSET, 0, 3, 6)
        np.testing.assert_array_equal(
            mask,
            [
                [False, True, True, False, True, True],
                [False] * 6,
                [True, True, False, False, False, False],
            ],
        )

    def test_sub_range(self):
        mask = _interval_mask(self._INTERVALS, self._OFFSET, 1, 3, 6)
        np.testing.assert_array_equal(
            mask, [[False] * 6, [True, True, False, False, False, False]]
        )

    def test_overlapping_intervals(self):
        mask = _interval_mask(
            np.array([[1, 4], [2, 5]]), np.array([0, 2]), 0, 1, 6
        )
        np.testing.assert_array_equal(
            mask, [[False, True, True, True, True, False]]
        )
