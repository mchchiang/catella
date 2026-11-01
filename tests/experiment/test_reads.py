# test_reads.py

from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from catella.experiment._contexts import A
from catella.experiment._reads import _dense_batches, _iter_read_sites
from catella.experiment.methdata import MethPrintData, MethPrintExperiment

_NBP = 6
_NMOL = 5
_ROWS = [
    (3, 5, 0.9),
    (0, 0, 0.1),
    (0, 4, 0.2),
    (1, 1, 0.3),
    (3, 2, 0.4),
    (4, 3, 0.5),
    (1, 5, 0.6),
]


def _table(rows):
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "mod_qual"])
    df["strand"] = "+"
    df["mod_code"] = "a"
    return df


def _experiment(rows=_ROWS, nmol=_NMOL, nbp=_NBP):
    raw = MethPrintData._create(
        chrom="chr1",
        nbp=nbp,
        test_mol_id=np.array([f"mol{m}" for m in range(nmol)], dtype=object),
        test_data=_table(rows),
    )
    return MethPrintExperiment._create(_raw_data={"chr1": raw})


def _read_args(nbp=_NBP):
    ctx = np.full(nbp, A, dtype=np.int8)
    full = np.ones(nbp, dtype=bool)
    return ctx, {"+": full, "-": full, ".": full}


def _collect(batches):
    return np.concatenate([batch for _, _, batch in batches], axis=0)


class TestDenseBatches:
    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    def test_matches_to_dense(self, batch_size):
        exp = _experiment()
        batches = _dense_batches(_table(_ROWS), _NMOL, _NBP, batch_size)
        expected = exp.to_dense("chr1", batch_size=batch_size).to_numpy()
        np.testing.assert_array_equal(_collect(batches), expected)

    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    def test_row_ranges_cover_all_molecules(self, batch_size):
        batches = _dense_batches(_table(_ROWS), _NMOL, _NBP, batch_size)
        spans = [(start, stop) for start, stop, _ in batches]
        assert spans[0][0] == 0
        assert spans[-1][1] == _NMOL
        assert all(a[1] == b[0] for a, b in pairwise(spans))

    def test_molecule_without_rows_is_all_nan(self):
        dense = _collect(_dense_batches(_table(_ROWS), _NMOL, _NBP, 2))
        assert np.isnan(dense[2]).all()

    def test_out_of_range_positions_are_ignored(self):
        rows = _ROWS + [(0, _NBP, 0.7), (0, -1, 0.8)]
        dense = _collect(_dense_batches(_table(rows), _NMOL, _NBP, 2))
        expected = _collect(_dense_batches(_table(_ROWS), _NMOL, _NBP, 2))
        np.testing.assert_array_equal(dense, expected)

    def test_duplicate_rows_raise(self):
        rows = _ROWS + [(0, 0, 0.99)]
        with pytest.raises(ValueError, match="duplicate"):
            list(_dense_batches(_table(rows), _NMOL, _NBP, 2))

    def test_nan_mod_qual_is_not_a_duplicate(self):
        rows = _ROWS + [(2, 2, np.nan)]
        dense = _collect(_dense_batches(_table(rows), _NMOL, _NBP, 2))
        assert np.isnan(dense[2]).all()

    def test_empty_table(self):
        rows = pd.DataFrame(
            {
                "mol_index": np.array([], dtype=np.int64),
                "pos": np.array([], dtype=np.int64),
                "mod_qual": np.array([], dtype=np.float64),
            }
        )
        dense = _collect(_dense_batches(rows, _NMOL, _NBP, 2))
        assert dense.shape == (_NMOL, _NBP)
        assert np.isnan(dense).all()

    def test_does_not_modify_input(self):
        df = _table(_ROWS)
        before = df.copy()
        list(_dense_batches(df, _NMOL, _NBP, 2))
        pd.testing.assert_frame_equal(df, before)


class TestIterReadSites:
    def _batches(self, exp, *, batch_size=2, mask_name=None, span="full"):
        ctx, assay = _read_args()
        return list(
            _iter_read_sites(
                exp,
                "chr1",
                "test",
                ctx,
                assay,
                batch_size=batch_size,
                mask_name=mask_name,
                span=span,
            )
        )

    def _masked_experiment(self):
        exp = _experiment()
        keep = np.array([True, False, True, True, False])
        exp.analysis["chr1"]["test_dm"] = pd.DataFrame({"keep": keep})
        return exp, keep

    def test_scores_match_to_dense(self):
        exp = _experiment()
        dense = exp.to_dense("chr1").to_numpy()
        q = np.concatenate([b.q for b in self._batches(exp)], axis=0)
        called = np.concatenate([b.called for b in self._batches(exp)], axis=0)
        np.testing.assert_array_equal(called, ~np.isnan(dense))
        np.testing.assert_array_equal(q[called], dense[called])

    @pytest.mark.parametrize("span", ["full", "calls"])
    def test_masked_molecules_read_no_site(self, span):
        exp, keep = self._masked_experiment()
        batches = self._batches(exp, mask_name="dm", span=span)
        q = np.concatenate([b.q for b in batches], axis=0)
        read = np.concatenate([b.read for b in batches], axis=0)
        called = np.concatenate([b.called for b in batches], axis=0)
        assert np.isnan(q[~keep]).all()
        assert not read[~keep].any()
        assert not called[~keep].any()

    @pytest.mark.parametrize("span", ["full", "calls"])
    def test_kept_molecules_ignore_mask(self, span):
        exp, keep = self._masked_experiment()
        masked = self._batches(exp, mask_name="dm", span=span)
        plain = self._batches(exp, span=span)
        q_masked = np.concatenate([b.q for b in masked], axis=0)
        q_plain = np.concatenate([b.q for b in plain], axis=0)
        np.testing.assert_array_equal(q_masked[keep], q_plain[keep])

    def test_masked_molecules_are_not_blank(self):
        exp, keep = self._masked_experiment()
        batches = self._batches(exp, mask_name="dm")
        blank = np.concatenate([b.blank for b in batches])
        assert not blank[~keep].any()

    def test_missing_mask_raises(self):
        with pytest.raises(KeyError):
            self._batches(_experiment(), mask_name="nonexistent")

    def test_duplicate_rows_raise(self):
        exp = _experiment(rows=_ROWS + [(0, 0, 0.99)])
        with pytest.raises(ValueError, match="duplicate"):
            self._batches(exp)

    def test_creates_no_scratch_files(self):
        exp = _experiment()
        tmp_dir = Path(exp.resolve_tmp_dir())
        before = set(tmp_dir.glob("*.h5"))
        self._batches(exp)
        assert set(tmp_dir.glob("*.h5")) == before
