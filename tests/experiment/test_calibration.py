from types import SimpleNamespace

import numpy as np
import pytest
from catella.experiment._calibration import (
    _calib_rows,
    _fold_contexts,
    _theta_frame,
)
from catella.experiment._contexts import GCH, NONE, A, _reference_contexts


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


class TestThetaFrame:
    def test_pooled_columns(self):
        df = _theta_frame(
            np.array([A, GCH], dtype=np.int8),
            np.array([0.1, 0.2]),
            np.array([0.8, 0.9]),
            np.array([True, False]),
            False,
        )
        assert list(df.columns) == [
            "pos",
            "channel",
            "theta_prot",
            "theta_acc",
            "informative",
        ]
        np.testing.assert_array_equal(df["pos"], [0, 1])
        np.testing.assert_array_equal(df["channel"], ["A", "GCH"])
        np.testing.assert_array_equal(df["theta_acc"], [0.8, 0.9])
        np.testing.assert_array_equal(df["informative"], [True, False])

    def test_by_strand_columns(self):
        df = _theta_frame(
            np.array([NONE], dtype=np.int8),
            {"+": np.array([0.1]), "-": np.array([0.2])},
            {"+": np.array([0.8]), "-": np.array([0.9])},
            {"+": np.array([True]), "-": np.array([False])},
            True,
        )
        assert list(df.columns) == [
            "pos",
            "channel",
            "theta_prot_pos",
            "theta_acc_pos",
            "informative_pos",
            "theta_prot_neg",
            "theta_acc_neg",
            "informative_neg",
        ]
        assert df["theta_prot_neg"].iloc[0] == 0.2
        assert not df["informative_neg"].iloc[0]


def _controls_info():
    return {
        "has_controls": True,
        "frac_informative_A": 0.5,
        "call_frac_meth_A": 0.9,
    }


def _em_info():
    return {
        "has_controls": False,
        "frac_informative_A": 0.4,
        "n_iter": 7,
        "log_likelihood": -12.5,
        "frac_protected": 0.6,
        "codes": [A, GCH],
        "theta_prot": np.array([0.1, 0.2]),
        "theta_acc": np.array([0.8, 0.9]),
    }


class TestCalibRows:
    def test_controls_row(self):
        rows = _calib_rows("c", _controls_info(), False)
        assert rows == [
            {
                "chrom": "c",
                "has_controls": True,
                "frac_informative_A": 0.5,
                "call_frac_meth_A": 0.9,
            }
        ]

    def test_em_row_has_fit_diagnostics(self):
        (row,) = _calib_rows("c", _em_info(), False)
        assert row["iters"] == 7
        assert row["log_likelihood"] == -12.5
        assert row["frac_protected"] == 0.6
        assert row["theta_prot_A"] == 0.1
        assert row["theta_prot_GCH"] == 0.2
        assert row["theta_acc_GCH"] == 0.9
        assert "strand" not in row
        assert "codes" not in row

    def test_by_strand_one_row_per_strand(self):
        info = {"+": _controls_info(), "-": _em_info()}
        rows = _calib_rows("c", info, True)
        assert [r["strand"] for r in rows] == ["+", "-"]
        assert "iters" not in rows[0]
        assert rows[1]["iters"] == 7
