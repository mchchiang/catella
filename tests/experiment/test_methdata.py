# test_methdata.py

import warnings
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest

from nucmc.experiment.methdata import (
    LazyRawDataMap, MethPrintData, MethPrintExperiment)
from nucmc.h5_array import H5Array


def _make_raw(chrom, nmol, nbp, seed, refseq=None):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(nmol):
        npos = rng.integers(max(1, nbp // 2), nbp + 1)
        positions = rng.choice(nbp, size=npos, replace=False)
        for p in positions:
            rows.append((m, int(p), "+", float(rng.random()), 0))
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "strand",
                                     "mod_qual", "mod_code"])
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)
    return MethPrintData._create(
        chrom=chrom, nbp=nbp, refseq=refseq,
        test_mol_id=mol_id, test_data=df,
        meth_mol_id=None, meth_data=None,
        unmeth_mol_id=None, unmeth_data=None)


def _make_multi_chrom_experiment_file(tmp_path, chroms, nmol=3, nbp=6):
    raw = {c: _make_raw(c, nmol, nbp, seed=i)
          for i, c in enumerate(chroms)}
    exp = MethPrintExperiment._create(_raw_data=raw)
    path = tmp_path / "experiment.h5"
    exp.save(path)
    return path


class TestLazyRawDataMap:
    def test_load_does_not_read_any_chromosome_eagerly(self, tmp_path):
        path = _make_multi_chrom_experiment_file(
            tmp_path, ["chr1", "chr2", "chr3"])
        with patch.object(MethPrintData, "_load",
                          wraps=MethPrintData._load) as spy:
            exp = MethPrintExperiment.load(path)
            assert spy.call_count == 0
            assert set(exp.chroms) == {"chr1", "chr2", "chr3"}
            assert spy.call_count == 0

    def test_access_triggers_exactly_one_load_per_chromosome(
            self, tmp_path):
        path = _make_multi_chrom_experiment_file(tmp_path, ["chr1", "chr2"])
        with patch.object(MethPrintData, "_load",
                          wraps=MethPrintData._load) as spy:
            exp = MethPrintExperiment.load(path, max_cached_chroms=2)
            raw1 = exp.raw["chr1"]
            assert spy.call_count == 1

            # Repeated access to the same chromosome is a cache hit
            raw1_again = exp.raw["chr1"]
            assert spy.call_count == 1
            assert raw1_again is raw1

            exp.raw["chr2"]
            assert spy.call_count == 2

    def test_lru_eviction_bounds_cached_chromosomes(self, tmp_path):
        path = _make_multi_chrom_experiment_file(
            tmp_path, ["chr1", "chr2", "chr3"])
        with patch.object(MethPrintData, "_load",
                          wraps=MethPrintData._load) as spy:
            exp = MethPrintExperiment.load(path, max_cached_chroms=1)
            exp.raw["chr1"]
            exp.raw["chr2"]
            assert spy.call_count == 2

            # chr1 was evicted when chr2 was loaded (max_cached_chroms=1)
            exp.raw["chr1"]
            assert spy.call_count == 3

    def test_default_max_cached_is_one(self, tmp_path):
        path = _make_multi_chrom_experiment_file(tmp_path, ["chr1", "chr2"])
        exp = MethPrintExperiment.load(path)
        raw_map = exp._raw_data
        assert isinstance(raw_map, LazyRawDataMap)

        exp.raw["chr1"]
        exp.raw["chr2"]
        assert len(raw_map._cache) == 1


class TestChromSelection:
    def test_selects_subset_of_chromosomes(self, tmp_path):
        chroms = ["chr1", "chr2", "chr3"]
        raw = {c: _make_raw(c, nmol=2, nbp=6, seed=i)
              for i, c in enumerate(chroms)}
        exp = MethPrintExperiment._create(_raw_data=raw)
        exp.analysis["chr2"]["stat"] = pd.DataFrame({"x": [1, 2]})
        path = tmp_path / "experiment.h5"
        exp.save(path)

        loaded = MethPrintExperiment.load(path, chroms=["chr1", "chr3"])
        assert set(loaded.chroms) == {"chr1", "chr3"}
        assert set(loaded.analysis.keys()) == {"chr1", "chr3"}

    def test_unknown_chromosome_raises(self, tmp_path):
        path = _make_multi_chrom_experiment_file(tmp_path, ["chr1", "chr2"])
        with pytest.raises(ValueError):
            MethPrintExperiment.load(path, chroms=["chrX"])


class TestRefseq:
    def test_refseq_roundtrips_through_save_load(self, tmp_path):
        raw = {"chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0,
                                 refseq="ACGTAC")}
        exp = MethPrintExperiment._create(_raw_data=raw)
        path = tmp_path / "experiment.h5"
        exp.save(path)

        loaded = MethPrintExperiment.load(path)
        assert loaded.raw["chr1"].refseq == "ACGTAC"

    def test_refseq_none_roundtrips_to_none(self, tmp_path):
        raw = {"chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0)}
        exp = MethPrintExperiment._create(_raw_data=raw)
        path = tmp_path / "experiment.h5"
        exp.save(path)

        with h5py.File(path, "r") as h5stream:
            gmeta = h5stream["raw_data"]["chr1"]["metadata"]
            assert "refseq" not in gmeta

        loaded = MethPrintExperiment.load(path)
        assert loaded.raw["chr1"].refseq is None


def _make_raw_exact(chrom, nbp, nmol, rows, which="test"):
    """Build a MethPrintData with an exact, deterministic long-format
    table for `which` ("test", "meth", or "unmeth"), leaving the other
    two kinds as None. `rows` is a list of (mol_index, pos, mod_qual)."""
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "mod_qual"])
    df["strand"] = "+"
    df["mod_code"] = 0
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)
    kwargs = dict(chrom=chrom, nbp=nbp,
                 test_mol_id=None, test_data=None,
                 meth_mol_id=None, meth_data=None,
                 unmeth_mol_id=None, unmeth_data=None)
    kwargs[f"{which}_mol_id"] = mol_id
    kwargs[f"{which}_data"] = df
    return MethPrintData._create(**kwargs)


def _manual_pivot(rows, mol_ids, nbp):
    """Reference pivot: dense (len(mol_ids), nbp) array, NaN for any
    (mol_index, pos) not present in `rows`."""
    out = np.full((len(mol_ids), nbp), np.nan)
    lookup = {(m, p): v for m, p, v in rows}
    for i, m in enumerate(mol_ids):
        for p in range(nbp):
            if (m, p) in lookup:
                out[i, p] = lookup[(m, p)]
    return out


# nmol=4, nbp=3. Molecule 2 has zero observed calls (all-NaN row).
# Molecule 0 is missing pos 1. Molecule 3 has only one observed call.
_TO_DENSE_NBP = 3
_TO_DENSE_NMOL = 4
_TO_DENSE_ROWS = [
    (0, 0, 0.1), (0, 2, 0.3),
    (1, 0, 0.4), (1, 1, 0.5), (1, 2, 0.6),
    (3, 1, 0.9),
]


class TestToDense:
    def _experiment(self, which="test"):
        raw = _make_raw_exact("chr1", _TO_DENSE_NBP, _TO_DENSE_NMOL,
                              _TO_DENSE_ROWS, which=which)
        return MethPrintExperiment._create(_raw_data={"chr1": raw})

    @pytest.mark.parametrize("as_h5array", [True, False])
    def test_dtype_must_be_floating(self, as_h5array):
        exp = self._experiment()
        with pytest.raises(ValueError):
            exp.to_dense("chr1", dtype=np.int64, as_h5array=as_h5array)

    def test_missing_which_raises(self):
        exp = self._experiment(which="test")
        with pytest.raises(ValueError):
            exp.to_dense("chr1", which="meth")

    def test_duplicate_entries_raise(self):
        rows = _TO_DENSE_ROWS + [(0, 0, 0.99)]  # duplicate (0, 0)
        raw = _make_raw_exact("chr1", _TO_DENSE_NBP, _TO_DENSE_NMOL, rows)
        exp = MethPrintExperiment._create(_raw_data={"chr1": raw})
        with pytest.raises(ValueError):
            exp.to_dense("chr1")

    @pytest.mark.parametrize("batch_size", [1, 2, 100])
    def test_h5array_full_matches_manual_pivot(self, batch_size):
        exp = self._experiment()
        out = exp.to_dense("chr1", batch_size=batch_size)
        assert isinstance(out, H5Array)
        expected = _manual_pivot(_TO_DENSE_ROWS, range(_TO_DENSE_NMOL),
                                 _TO_DENSE_NBP)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert isinstance(out.index, pd.RangeIndex)

    def test_h5array_subset_unsorted_matches_manual_pivot_and_index(self):
        exp = self._experiment()
        mols = [3, 0]  # unsorted, non-contiguous subset
        out = exp.to_dense("chr1", mols=mols, batch_size=1)
        assert isinstance(out, H5Array)
        expected = _manual_pivot(_TO_DENSE_ROWS, mols, _TO_DENSE_NBP)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert list(out.index) == mols

    def test_as_h5array_defaults_true_even_with_mols(self):
        exp = self._experiment()
        out = exp.to_dense("chr1", mols=[0, 1])
        assert isinstance(out, H5Array)

    def test_dtype_is_honored_for_h5array(self):
        exp = self._experiment()
        out = exp.to_dense("chr1", dtype=np.float32)
        assert out.dtype == np.float32

    def test_dataframe_subset_matches_manual_pivot(self):
        exp = self._experiment()
        mols = [2, 1]  # includes the zero-call molecule
        out = exp.to_dense("chr1", mols=mols, as_h5array=False)
        assert isinstance(out, pd.DataFrame)
        expected = _manual_pivot(_TO_DENSE_ROWS, mols, _TO_DENSE_NBP)
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert list(out.index) == mols

    def test_dataframe_scalar_mols(self):
        exp = self._experiment()
        out = exp.to_dense("chr1", mols=2, as_h5array=False)
        assert list(out.index) == [2]
        assert np.isnan(out.to_numpy()).all()

    def test_dataframe_full_matches_manual_pivot(self):
        exp = self._experiment()
        out = exp.to_dense("chr1", as_h5array=False)
        assert isinstance(out, pd.DataFrame)
        expected = _manual_pivot(_TO_DENSE_ROWS, range(_TO_DENSE_NMOL),
                                 _TO_DENSE_NBP)
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_dtype_is_honored_for_dataframe(self):
        exp = self._experiment()
        out = exp.to_dense("chr1", as_h5array=False, dtype=np.float32)
        assert out.to_numpy().dtype == np.float32

    @pytest.mark.parametrize("which", ["meth", "unmeth"])
    def test_which_selects_right_table(self, which):
        exp = self._experiment(which=which)
        out = exp.to_dense("chr1", which=which)
        expected = _manual_pivot(_TO_DENSE_ROWS, range(_TO_DENSE_NMOL),
                                 _TO_DENSE_NBP)
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_warns_when_dataframe_exceeds_threshold(self, monkeypatch):
        import nucmc.experiment.methdata as methdata_mod
        monkeypatch.setattr(methdata_mod, "_DENSE_WARN_ROWS", 2)
        exp = self._experiment()
        with pytest.warns(UserWarning):
            exp.to_dense("chr1", as_h5array=False)

    def test_no_warning_below_threshold(self, monkeypatch):
        import nucmc.experiment.methdata as methdata_mod
        monkeypatch.setattr(methdata_mod, "_DENSE_WARN_ROWS", 100)
        exp = self._experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.to_dense("chr1", as_h5array=False)

    def test_no_warning_for_h5array_regardless_of_size(self, monkeypatch):
        import nucmc.experiment.methdata as methdata_mod
        monkeypatch.setattr(methdata_mod, "_DENSE_WARN_ROWS", 1)
        exp = self._experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.to_dense("chr1", as_h5array=True)

    @pytest.mark.parametrize("as_h5array", [True, False])
    def test_mask_name_nans_out_masked_rows(self, as_h5array):
        exp = self._experiment()
        # Molecules 1 and 3 dropped; 0 and 2 kept.
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]})
        out = exp.to_dense("chr1", as_h5array=as_h5array,
                          mask_name="dropout_mask")
        arr = out.to_numpy()
        assert np.isnan(arr[1]).all()
        assert np.isnan(arr[3]).all()
        expected = _manual_pivot(_TO_DENSE_ROWS, [0, 2], _TO_DENSE_NBP)
        np.testing.assert_array_equal(arr[[0, 2]], expected)

    def test_no_mask_name_unaffected(self):
        exp = self._experiment()
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]})
        out = exp.to_dense("chr1", as_h5array=False)
        expected = _manual_pivot(_TO_DENSE_ROWS, range(_TO_DENSE_NMOL),
                                 _TO_DENSE_NBP)
        np.testing.assert_array_equal(out.to_numpy(), expected)
