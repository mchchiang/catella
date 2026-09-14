# test_results.py

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from catella_cpp import Dump

from catella.h5_array import H5Array
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import SimDataset


def _make_settings(**overrides):
    defaults = dict(nucbp=147, llink=20, elink=1.0, mu=-1.0, start_temp=1.0,
                     end_temp=0.1, cool_option="linear", nsweep=5,
                     print_freq=1, emax=5.0)
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_meth_prob(nmol=3, nbp=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((nmol, nbp))


def _make_dataset(tmp_path, nmol=2, nsim=1, seed=1, settings_overrides=None,
                  **run_overrides):
    meth_prob = _make_meth_prob(nmol=nmol)
    manager = SimManager(nworker=1, verbose=False)
    settings = _make_settings(**(settings_overrides or {}))
    return manager.run(chroms="chr1", nsim=nsim, settings=settings,
                       meth_prob=meth_prob, out_dir=tmp_path / "dataset",
                       seed=seed, **run_overrides)


class TestOutTypeSeedPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=2, seed=42)
        loaded = SimDataset.load(dataset._dataset_file)

        assert loaded.out_type == dataset.out_type
        assert loaded.seed == dataset.seed
        assert (loaded.seed_table["chr1"] == dataset.seed_table["chr1"]).all()

    def test_legacy_dataset_defaults_to_none(self, tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"], nmol={"chr1": 1}, nsim=1, nbp={"chr1": 50},
            settings=_make_settings(), out_dir=tmp_path / "legacy")
        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)

        assert loaded.out_type is None
        assert loaded.seed is None
        assert loaded.seed_table is None


class TestSaveOverwriteCleanup:
    def test_tmp_file_removed_on_keyboard_interrupt(self, tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"], nmol={"chr1": 1}, nsim=1, nbp={"chr1": 50},
            settings=_make_settings(), out_dir=tmp_path / "ds")
        dataset.save()
        path = Path(dataset._dataset_file)
        # An analysis entry backed by the destination file itself
        # forces save() onto its tmp-then-replace overwrite path.
        dataset._global_analysis["dummy"] = H5Array.create((2, 2), path=path)

        with patch("catella.simulation.results.os.replace",
                   side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                dataset.save(path, overwrite=True)

        leaked = list(path.parent.glob(f"{path.name}.tmp*"))
        assert not leaked


class TestExtractMissingMolecules:
    def test_fully_missing_molecule_is_none_with_agg_func(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nsim=2)
        for run in range(dataset.nsim):
            dataset.sim_file("chr1", 1, run).unlink()

        def count_agg(chrom, mol, raw_data):
            return sum(len(x) for x in raw_data)

        time = dataset.raw["chr1", 0, 0].time[-1]
        results = dataset.extract(time=time, obs="position",
                                  agg_func=count_agg)

        assert results["chr1"][1] is None
        assert results["chr1"][0] is not None
        assert results["chr1"][2] is not None

    def test_no_missing_molecules_when_all_files_present(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nsim=1)

        def count_agg(chrom, mol, raw_data):
            return sum(len(x) for x in raw_data)

        time = dataset.raw["chr1", 0, 0].time[-1]
        results = dataset.extract(time=time, obs="position",
                                  agg_func=count_agg)

        assert all(v is not None for v in results["chr1"])


class TestFindIncompleteRuns:
    def test_missing_run_detected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=1)
        dataset.sim_file("chr1", 1, 0).unlink()

        incomplete = dataset.find_incomplete_runs()
        assert incomplete["missing"] == [("chr1", 1, 0)]
        assert incomplete["corrupted"] == []
        assert incomplete["truncated"] == []

    def test_corrupted_run_detected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=1)
        dataset.sim_file("chr1", 1, 0).write_bytes(b"not valid hdf5")

        incomplete = dataset.find_incomplete_runs()
        assert incomplete["corrupted"] == [("chr1", 1, 0)]
        assert incomplete["missing"] == []
        assert incomplete["truncated"] == []

    def test_truncated_run_detected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=1, nsim=1,
                                settings_overrides=dict(nsweep=10,
                                                        print_freq=1))
        sim_file = dataset.sim_file("chr1", 0, 0)
        with h5py.File(sim_file, "a") as f:
            # Truncate the recorded trajectory below the expected length
            # (nsweep // print_freq + 1 = 11 frames). All frame-aligned
            # datasets must shrink together for the file to stay readable
            # (only shorter, not corrupted).
            g = f["data"]
            for name in ("time", "energy", "temp"):
                data = g[name][:3]
                del g[name]
                g.create_dataset(name, data=data)
            del g["position_offset"]
            g.create_dataset("position_offset",
                             data=np.zeros(4, dtype=np.uint64))

        incomplete = dataset.find_incomplete_runs()
        assert incomplete["truncated"] == [("chr1", 0, 0)]
        assert incomplete["missing"] == []
        assert incomplete["corrupted"] == []

    def test_complete_run_not_reported(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=1, nsim=1)
        incomplete = dataset.find_incomplete_runs()
        assert incomplete == {"missing": [], "corrupted": [], "truncated": []}

    def test_nworker_parallel_matches_serial(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=4, nsim=2)
        for chrom, mol, run in [("chr1", 0, 0), ("chr1", 2, 1)]:
            dataset.sim_file(chrom, mol, run).unlink()

        serial = dataset.find_incomplete_runs(nworker=1)
        parallel = dataset.find_incomplete_runs(nworker=4)

        assert set(serial["missing"]) == set(parallel["missing"])
        assert set(serial["corrupted"]) == set(parallel["corrupted"])
        assert set(serial["truncated"]) == set(parallel["truncated"])
