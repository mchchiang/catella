# test_methdata.py

import warnings
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
from catella.experiment.methdata import (
    LazyRawDataMap,
    MethPrintData,
    MethPrintExperiment,
    SourceData,
)
from catella.h5_array import H5Array


def _make_raw(region, nmol, nbp, seed, refseq=None):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(nmol):
        npos = rng.integers(max(1, nbp // 2), nbp + 1)
        positions = rng.choice(nbp, size=npos, replace=False)
        for p in positions:
            rows.append((m, int(p), "+", float(rng.random()), 0))
    df = pd.DataFrame(
        rows, columns=["mol_index", "pos", "strand", "mod_qual", "mod_code"]
    )
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)
    return MethPrintData._create(
        region=region,
        nbp=nbp,
        refseq=refseq,
        test_mol_id=mol_id,
        test_data=df,
        meth_mol_id=None,
        meth_data=None,
        unmeth_mol_id=None,
        unmeth_data=None,
    )


def _make_multi_region_experiment_file(tmp_path, regions, nmol=3, nbp=6):
    raw = {c: _make_raw(c, nmol, nbp, seed=i) for i, c in enumerate(regions)}
    exp = MethPrintExperiment._create(_raw_data=raw)
    path = tmp_path / "experiment.h5"
    exp.save(path)
    return path


class TestLazyRawDataMap:
    def test_load_does_not_read_any_region_eagerly(self, tmp_path):
        path = _make_multi_region_experiment_file(
            tmp_path, ["chr1", "chr2", "chr3"]
        )
        with patch.object(
            MethPrintData, "_load", wraps=MethPrintData._load
        ) as spy:
            exp = MethPrintExperiment.load(path)
            assert spy.call_count == 0
            assert set(exp.regions) == {"chr1", "chr2", "chr3"}
            assert spy.call_count == 0

    def test_access_triggers_exactly_one_load_per_region(self, tmp_path):
        path = _make_multi_region_experiment_file(tmp_path, ["chr1", "chr2"])
        with patch.object(
            MethPrintData, "_load", wraps=MethPrintData._load
        ) as spy:
            exp = MethPrintExperiment.load(path, max_cached_regions=2)
            raw1 = exp.raw["chr1"]
            assert spy.call_count == 1

            # Repeated access to the same region is a cache hit
            raw1_again = exp.raw["chr1"]
            assert spy.call_count == 1
            assert raw1_again is raw1

            exp.raw["chr2"]
            assert spy.call_count == 2

    def test_lru_eviction_bounds_cached_regions(self, tmp_path):
        path = _make_multi_region_experiment_file(
            tmp_path, ["chr1", "chr2", "chr3"]
        )
        with patch.object(
            MethPrintData, "_load", wraps=MethPrintData._load
        ) as spy:
            exp = MethPrintExperiment.load(path, max_cached_regions=1)
            exp.raw["chr1"]
            exp.raw["chr2"]
            assert spy.call_count == 2

            # chr1 was evicted when chr2 was loaded (max_cached_regions=1)
            exp.raw["chr1"]
            assert spy.call_count == 3

    def test_default_max_cached_is_one(self, tmp_path):
        path = _make_multi_region_experiment_file(tmp_path, ["chr1", "chr2"])
        exp = MethPrintExperiment.load(path)
        raw_map = exp._raw_data
        assert isinstance(raw_map, LazyRawDataMap)

        exp.raw["chr1"]
        exp.raw["chr2"]
        assert len(raw_map._cache) == 1


class TestRegionSelection:
    def test_selects_subset_of_regions(self, tmp_path):
        regions = ["chr1", "chr2", "chr3"]
        raw = {
            c: _make_raw(c, nmol=2, nbp=6, seed=i)
            for i, c in enumerate(regions)
        }
        exp = MethPrintExperiment._create(_raw_data=raw)
        exp.analysis["chr2"]["stat"] = pd.DataFrame({"x": [1, 2]})
        path = tmp_path / "experiment.h5"
        exp.save(path)

        loaded = MethPrintExperiment.load(path, regions=["chr1", "chr3"])
        assert set(loaded.regions) == {"chr1", "chr3"}
        assert set(loaded.analysis.keys()) == {"chr1", "chr3"}

    def test_unknown_region_raises(self, tmp_path):
        path = _make_multi_region_experiment_file(tmp_path, ["chr1", "chr2"])
        with pytest.raises(ValueError):
            MethPrintExperiment.load(path, regions=["chrX"])


class TestSaveToExistingFile:
    def test_new_experiment_overwrites_stale_raw_data(self, tmp_path):
        # Regression: save() used to skip writing raw data whenever
        # the destination already had *a* raw_data group, even from
        # an unrelated experiment.
        old = MethPrintExperiment._create(
            _raw_data={"chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0)}
        )
        path = tmp_path / "experiment.h5"
        old.save(path)

        new = MethPrintExperiment._create(
            _raw_data={"chr1": _make_raw("chr1", nmol=3, nbp=6, seed=1)}
        )
        assert new._exp_file is None
        new.save(path)

        loaded = MethPrintExperiment.load(path)
        assert list(loaded.raw["chr1"].test.mol_id) == list(
            new._raw_data["chr1"].test.mol_id
        )

    def test_resaving_same_file_skips_rewriting_raw_data(self, tmp_path):
        exp = MethPrintExperiment._create(
            _raw_data={"chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0)}
        )
        path = tmp_path / "experiment.h5"
        exp.save(path)

        with patch.object(
            MethPrintData, "_save", wraps=MethPrintData._save
        ) as spy:
            exp.save(path)
            assert spy.call_count == 0


class TestSaveInPlace:
    @pytest.fixture
    def path(self, tmp_path):
        exp = MethPrintExperiment._create(
            _raw_data={"chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0)}
        )
        arr = H5Array.create((2, 3), dir=tmp_path)
        arr.write_batch(0, 2, np.arange(6.0).reshape(2, 3))
        exp._analysis["chr1"]["sig"] = arr
        path = tmp_path / "experiment.h5"
        exp.save(path)
        return path

    def test_resave_keeps_entries_and_size(self, path):
        size = path.stat().st_size
        loaded = MethPrintExperiment.load(path)
        loaded.save()
        assert path.stat().st_size < 1.5 * size
        reloaded = MethPrintExperiment.load(path)
        np.testing.assert_array_equal(
            reloaded.analysis["chr1"]["sig"].to_numpy(),
            np.arange(6.0).reshape(2, 3),
        )

    def test_held_reference_usable_after_save(self, path):
        loaded = MethPrintExperiment.load(path)
        ref = loaded.analysis["chr1"]["sig"]
        loaded.save(path)
        np.testing.assert_array_equal(
            ref.to_numpy(), np.arange(6.0).reshape(2, 3)
        )

    def test_interrupted_save_keeps_previous_analysis(self, path):
        loaded = MethPrintExperiment.load(path)
        loaded._analysis["chr1"]["df"] = pd.DataFrame({"v": [1.0]})
        with (
            patch(
                "catella._analysis_io._link_or_write",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            loaded.save()
        reloaded = MethPrintExperiment.load(path)
        assert set(reloaded.analysis["chr1"]) == {"sig"}
        assert loaded.analysis["chr1"]["sig"].shape == (2, 3)

    def test_save_to_other_existing_file(self, path, tmp_path):
        other = tmp_path / "other.h5"
        exp = MethPrintExperiment._create(
            _raw_data={"chr2": _make_raw("chr2", nmol=2, nbp=6, seed=1)}
        )
        exp.save(other)
        loaded = MethPrintExperiment.load(path)
        loaded.save(other)
        reloaded = MethPrintExperiment.load(other)
        assert reloaded.regions == ("chr1",)
        np.testing.assert_array_equal(
            reloaded.analysis["chr1"]["sig"].to_numpy(),
            np.arange(6.0).reshape(2, 3),
        )

    def test_subset_load_keeps_other_regions_analysis(self, tmp_path):
        raw = {
            c: _make_raw(c, nmol=2, nbp=6, seed=i)
            for i, c in enumerate(["chr1", "chr2"])
        }
        exp = MethPrintExperiment._create(_raw_data=raw)
        exp._analysis["chr2"]["df2"] = pd.DataFrame({"v": [2.0]})
        path = tmp_path / "multi.h5"
        exp.save(path)

        subset = MethPrintExperiment.load(path, regions=["chr1"])
        subset._analysis["chr1"]["df1"] = pd.DataFrame({"v": [1.0]})
        subset.save()

        full = MethPrintExperiment.load(path)
        assert set(full.analysis["chr1"]) == {"df1"}
        pd.testing.assert_frame_equal(
            full.analysis["chr2"]["df2"], pd.DataFrame({"v": [2.0]})
        )

    def test_save_to_other_file_drops_its_regions(self, path, tmp_path):
        other = tmp_path / "other.h5"
        exp = MethPrintExperiment._create(
            _raw_data={"chr2": _make_raw("chr2", nmol=2, nbp=6, seed=1)}
        )
        exp._analysis["chr2"]["df2"] = pd.DataFrame({"v": [2.0]})
        exp.save(other)
        MethPrintExperiment.load(path).save(other)
        with h5py.File(other, "r") as f:
            assert set(f["analysis"]) == {"chr1"}

    def test_file_open_elsewhere_raises(self, path):
        loaded = MethPrintExperiment.load(path)
        other = MethPrintExperiment.load(path)
        with pytest.raises(OSError, match="open elsewhere"):
            loaded.save()
        other.analysis["chr1"]["sig"].close()


class TestLabeledAnalysisLoad:
    @pytest.fixture
    def path(self, tmp_path):
        exp = MethPrintExperiment._create(
            _raw_data={"chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0)}
        )
        labels = {"index": ["m0", "m1"], "columns": [5, 6, 7]}
        exp._analysis["chr1"]["sig"] = H5Array.create(
            (2, 3), dir=tmp_path, **labels
        )
        exp._global_analysis["sig"] = H5Array.create(
            (2, 3), dir=tmp_path, **labels
        )
        path = tmp_path / "experiment.h5"
        exp.save(path)
        return path

    def test_region_labels_not_loaded_as_entries(self, path):
        loaded = MethPrintExperiment.load(path)
        assert set(loaded._analysis["chr1"]) == {"sig"}
        assert list(loaded._analysis["chr1"]["sig"].index) == ["m0", "m1"]
        assert list(loaded._analysis["chr1"]["sig"].columns) == [5, 6, 7]

    def test_global_labels_not_loaded_as_entries(self, path):
        loaded = MethPrintExperiment.load(path)
        assert set(loaded._global_analysis) == {"sig"}
        assert list(loaded._global_analysis["sig"].index) == ["m0", "m1"]


class TestRefseq:
    def test_refseq_roundtrips_through_save_load(self, tmp_path):
        raw = {
            "chr1": _make_raw("chr1", nmol=2, nbp=6, seed=0, refseq="ACGTAC")
        }
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


def _make_raw_exact(region, nbp, nmol, rows, which="test"):
    """Build a MethPrintData with an exact, deterministic long-format
    table for `which` ("test", "meth", or "unmeth"), leaving the other
    two kinds as None. `rows` is a list of (mol_index, pos, mod_qual)."""
    df = pd.DataFrame(rows, columns=["mol_index", "pos", "mod_qual"])
    df["strand"] = "+"
    df["mod_code"] = 0
    mol_id = np.array([f"mol{m}" for m in range(nmol)], dtype=object)
    kwargs = {
        "region": region,
        "nbp": nbp,
        "test_mol_id": None,
        "test_data": None,
        "meth_mol_id": None,
        "meth_data": None,
        "unmeth_mol_id": None,
        "unmeth_data": None,
    }
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
    (0, 0, 0.1),
    (0, 2, 0.3),
    (1, 0, 0.4),
    (1, 1, 0.5),
    (1, 2, 0.6),
    (3, 1, 0.9),
]


class TestToDense:
    def _experiment(self, which="test"):
        raw = _make_raw_exact(
            "chr1", _TO_DENSE_NBP, _TO_DENSE_NMOL, _TO_DENSE_ROWS, which=which
        )
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
        expected = _manual_pivot(
            _TO_DENSE_ROWS, range(_TO_DENSE_NMOL), _TO_DENSE_NBP
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)
        assert isinstance(out.index, pd.RangeIndex)

    def test_nonpositive_batch_size_raises(self):
        exp = self._experiment()
        with pytest.raises(ValueError, match="batch_size must be positive"):
            exp.to_dense("chr1", batch_size=-5)

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
        expected = _manual_pivot(
            _TO_DENSE_ROWS, range(_TO_DENSE_NMOL), _TO_DENSE_NBP
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_dtype_is_honored_for_dataframe(self):
        exp = self._experiment()
        out = exp.to_dense("chr1", as_h5array=False, dtype=np.float32)
        assert out.to_numpy().dtype == np.float32

    @pytest.mark.parametrize("which", ["meth", "unmeth"])
    def test_which_selects_right_table(self, which):
        exp = self._experiment(which=which)
        out = exp.to_dense("chr1", which=which)
        expected = _manual_pivot(
            _TO_DENSE_ROWS, range(_TO_DENSE_NMOL), _TO_DENSE_NBP
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)

    def test_warns_when_dataframe_exceeds_threshold(self, monkeypatch):
        import catella.experiment.methdata as methdata_mod

        monkeypatch.setattr(methdata_mod, "_DENSE_WARN_BYTES", 1)
        exp = self._experiment()
        with pytest.warns(UserWarning, match="GB of memory"):
            exp.to_dense("chr1", as_h5array=False)

    def test_no_warning_below_threshold(self, monkeypatch):
        import catella.experiment.methdata as methdata_mod

        monkeypatch.setattr(methdata_mod, "_DENSE_WARN_BYTES", 10**9)
        exp = self._experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.to_dense("chr1", as_h5array=False)

    def test_no_warning_for_h5array_regardless_of_size(self, monkeypatch):
        import catella.experiment.methdata as methdata_mod

        monkeypatch.setattr(methdata_mod, "_DENSE_WARN_BYTES", 1)
        exp = self._experiment()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            exp.to_dense("chr1", as_h5array=True)

    @pytest.mark.parametrize("as_h5array", [True, False])
    def test_mask_name_nans_out_masked_rows(self, as_h5array):
        exp = self._experiment()
        # Molecules 1 and 3 dropped; 0 and 2 kept.
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]}
        )
        out = exp.to_dense(
            "chr1", as_h5array=as_h5array, mask_name="dropout_mask"
        )
        arr = out.to_numpy()
        assert np.isnan(arr[1]).all()
        assert np.isnan(arr[3]).all()
        expected = _manual_pivot(_TO_DENSE_ROWS, [0, 2], _TO_DENSE_NBP)
        np.testing.assert_array_equal(arr[[0, 2]], expected)

    def test_no_mask_name_unaffected(self):
        exp = self._experiment()
        exp.analysis["chr1"]["test_dropout_mask"] = pd.DataFrame(
            {"keep": [True, False, True, False]}
        )
        out = exp.to_dense("chr1", as_h5array=False)
        expected = _manual_pivot(
            _TO_DENSE_ROWS, range(_TO_DENSE_NMOL), _TO_DENSE_NBP
        )
        np.testing.assert_array_equal(out.to_numpy(), expected)


def _calls(nmol):
    return pd.DataFrame(
        {
            "mol_index": np.arange(nmol),
            "pos": np.zeros(nmol, dtype=int),
            "strand": ["+"] * nmol,
            "mod_qual": np.full(nmol, 0.5),
            "mod_code": ["a"] * nmol,
        }
    )


def _ids(nmol):
    return np.array([f"m{i}" for i in range(nmol)], dtype=object)


class TestSourceData:
    def test_nmol_and_fields(self):
        src = SourceData._create(
            mol_id=_ids(2),
            data=_calls(2),
            mol_start=np.array([0, 3]),
            mol_end=np.array([5, 9]),
        )
        assert src.nmol == 2
        assert src.mol_end.tolist() == [5, 9]
        assert src.gaps is None

    def test_arrays_are_read_only(self):
        src = SourceData._create(mol_id=_ids(1), data=_calls(1))
        with pytest.raises(ValueError):
            src.mol_id[0] = "x"

    def test_extent_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="mol_start has length 1"):
            SourceData._create(
                name="test",
                mol_id=_ids(2),
                data=_calls(2),
                mol_start=np.array([0]),
                mol_end=np.array([5, 9]),
            )

    def test_gaps_without_offset_raises(self):
        with pytest.raises(ValueError, match="both be given"):
            SourceData._create(
                mol_id=_ids(1),
                data=_calls(1),
                gaps=np.zeros((0, 2), dtype=np.int64),
            )

    def test_gap_offset_must_fit(self):
        with pytest.raises(ValueError, match="gap_offset must have length 3"):
            SourceData._create(
                mol_id=_ids(2),
                data=_calls(2),
                gaps=np.array([[1, 3]]),
                gap_offset=np.array([0, 1]),
            )

    def test_direct_instantiation_raises(self):
        with pytest.raises(TypeError, match="built by MethPrintData"):
            SourceData(mol_id=_ids(1), data=_calls(1))


class TestMethPrintDataSources:
    def _raw(self, **controls):
        kwargs = {"test_mol_id": _ids(2), "test_data": _calls(2)}
        for name in controls:
            kwargs[f"{name}_mol_id"] = _ids(1)
            kwargs[f"{name}_data"] = _calls(1)
        return MethPrintData._create(region="r", nbp=10, **kwargs)

    def test_test_only(self):
        raw = self._raw()
        assert list(raw.sources) == ["test"]
        assert raw.unmeth is None
        assert raw.meth is None
        assert raw.test.nmol == 2

    def test_canonical_order(self):
        raw = self._raw(meth=True, unmeth=True)
        assert list(raw.sources) == ["test", "unmeth", "meth"]
        assert raw.sources["meth"].nmol == 1

    def test_missing_control_is_key_error(self):
        with pytest.raises(KeyError):
            self._raw().sources["meth"]

    def test_sources_is_read_only(self):
        raw = self._raw()
        with pytest.raises(TypeError):
            raw.sources["meth"] = raw.test

    def test_save_load_round_trip(self, tmp_path):
        raw = self._raw(unmeth=True)
        with h5py.File(tmp_path / "f.h5", "w") as h5:
            raw._save(h5.create_group("raw_data"))
        with h5py.File(tmp_path / "f.h5", "r") as h5:
            loaded = MethPrintData._load(h5["raw_data"], "r")
        assert list(loaded.sources) == ["test", "unmeth"]
        assert loaded.test.mol_id.tolist() == ["m0", "m1"]
        pd.testing.assert_frame_equal(loaded.unmeth.data, raw.unmeth.data)
