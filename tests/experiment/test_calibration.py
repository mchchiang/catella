from types import SimpleNamespace

import numpy as np
import pytest
from catella.experiment._calibration import _fold_contexts
from catella.experiment._contexts import NONE, _reference_contexts


def _exp(refseq, wrap, nbp=None):
    raw = SimpleNamespace(refseq=refseq, nbp=nbp)
    return SimpleNamespace(raw={"c": raw}, wrap=wrap)


class TestFoldContexts:
    def test_no_wrap_returns_reference_contexts(self):
        seq = "AACGTTGCAT"
        out = _fold_contexts(_exp(seq, False), "c")
        np.testing.assert_array_equal(out, _reference_contexts(seq))

    def test_wrap_symmetric_sequence_keeps_contexts(self):
        seq = "ATGCGCAT"
        ctx = _reference_contexts(seq)
        assert np.array_equal(ctx[:4], ctx[4:][::-1])
        out = _fold_contexts(_exp(seq, True, nbp=4), "c")
        np.testing.assert_array_equal(out, ctx[:4])

    def test_wrap_disagreeing_pairs_become_none_and_warn(self):
        seq = "ACGTAAAA"
        ctx = _reference_contexts(seq)
        with pytest.warns(UserWarning, match="disagrees on context"):
            out = _fold_contexts(_exp(seq, True, nbp=4), "c")
        agree = ctx[:4] == ctx[4:][::-1]
        assert not agree.all()
        np.testing.assert_array_equal(out, np.where(agree, ctx[:4], NONE))

    def test_missing_refseq_raises(self):
        with pytest.raises(ValueError, match="No reference sequence"):
            _fold_contexts(_exp(None, False), "c")
