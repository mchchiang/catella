# test_results.py

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
from catella.h5_array import H5Array
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager
from catella.simulation.results import (
    SimDataset,
    SimTMData,
    _consolidate_tm_samples,
    _read_tm_sample,
    _write_tm_file,
    _write_tm_samples,
)


def _make_settings(**overrides):
    defaults = {
        "nucbp": 147,
        "llink": 20,
        "elink": 1.0,
        "mu": -1.0,
        "start_temp": 1.0,
        "end_temp": 0.1,
        "cool_option": "linear",
        "nsweep": 5,
        "print_freq": 1,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_meth_prob(nmol=3, nbp=50, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((nmol, nbp))


def _make_dataset(
    tmp_path, nmol=2, nsim=1, seed=1, settings_overrides=None, **run_overrides
):
    meth_prob = _make_meth_prob(nmol=nmol)
    manager = SimManager(nworker=1, verbose=False)
    settings = _make_settings(**(settings_overrides or {}))
    return manager.run(
        chroms="chr1",
        nsim=nsim,
        settings=settings,
        meth_prob=meth_prob,
        out_dir=tmp_path / "dataset",
        seed=seed,
        **run_overrides,
    )


class TestOutTypeSeedPersistence:
    def test_save_load_roundtrip(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=2, seed=42)
        loaded = SimDataset.load(dataset._dataset_file)

        assert loaded.out_type == dataset.out_type
        assert loaded.seed == dataset.seed
        assert (loaded.seed_table["chr1"] == dataset.seed_table["chr1"]).all()

    def test_legacy_dataset_defaults_to_none(self, tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "legacy",
        )
        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)

        assert loaded.out_type is None
        assert loaded.seed is None
        assert loaded.seed_table is None


class TestSaveOverwriteCleanup:
    def test_tmp_file_removed_on_keyboard_interrupt(self, tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "ds",
        )
        dataset.save()
        path = Path(dataset._dataset_file)
        # An analysis entry backed by the destination file itself
        # forces save() onto its tmp-then-replace overwrite path.
        dataset._global_analysis["dummy"] = H5Array.create((2, 2), path=path)

        with (
            patch(
                "catella.simulation.results.os.replace",
                side_effect=KeyboardInterrupt,
            ),
            pytest.raises(KeyboardInterrupt),
        ):
            dataset.save(path, overwrite=True)

        leaked = list(path.parent.glob(f"{path.name}.tmp*"))
        assert not leaked


class TestExtractMissingMolecules:
    def test_fully_missing_molecule_is_none_with_agg_func(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nsim=2)
        for run in range(dataset.nsim):
            dataset.mcmc_file("chr1", 1, run).unlink()

        def count_agg(chrom, mol, raw_data):
            return sum(len(x) for x in raw_data)

        time = dataset.raw["chr1", 0, 0].time[-1]
        results = dataset.extract(
            time=time, obs="position", agg_func=count_agg
        )

        assert results["chr1"][1] is None
        assert results["chr1"][0] is not None
        assert results["chr1"][2] is not None

    def test_no_missing_molecules_when_all_files_present(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=3, nsim=1)

        def count_agg(chrom, mol, raw_data):
            return sum(len(x) for x in raw_data)

        time = dataset.raw["chr1", 0, 0].time[-1]
        results = dataset.extract(
            time=time, obs="position", agg_func=count_agg
        )

        assert all(v is not None for v in results["chr1"])


class TestFindIncompleteRuns:
    def test_missing_run_detected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=1)
        dataset.mcmc_file("chr1", 1, 0).unlink()

        incomplete = dataset.find_incomplete_runs()
        assert incomplete["missing"] == [("chr1", 1, 0)]
        assert incomplete["corrupted"] == []
        assert incomplete["truncated"] == []

    def test_corrupted_run_detected(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=1)
        dataset.mcmc_file("chr1", 1, 0).write_bytes(b"not valid hdf5")

        incomplete = dataset.find_incomplete_runs()
        assert incomplete["corrupted"] == [("chr1", 1, 0)]
        assert incomplete["missing"] == []
        assert incomplete["truncated"] == []

    def test_truncated_run_detected(self, tmp_path):
        dataset = _make_dataset(
            tmp_path,
            nmol=1,
            nsim=1,
            settings_overrides={"nsweep": 10, "print_freq": 1},
        )
        sim_file = dataset.mcmc_file("chr1", 0, 0)
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
            g.create_dataset(
                "position_offset", data=np.zeros(4, dtype=np.uint64)
            )

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
            dataset.mcmc_file(chrom, mol, run).unlink()

        serial = dataset.find_incomplete_runs(nworker=1)
        parallel = dataset.find_incomplete_runs(nworker=4)

        assert set(serial["missing"]) == set(parallel["missing"])
        assert set(serial["corrupted"]) == set(parallel["corrupted"])
        assert set(serial["truncated"]) == set(parallel["truncated"])


class TestSimTMDataRoundtrip:
    def test_write_and_load(self, tmp_path):
        out_file = tmp_path / "sample.h5"
        _write_tm_file(
            out_file,
            nucbp=147,
            nbp=400,
            llink=20,
            elink=1.0,
            mu=-1.0,
            seed=7,
            position=np.array([10, 200, 350]),
            energy=-3.5,
            logz=12.34,
        )
        data = SimTMData.load(out_file)
        assert data.nucbp == 147
        assert data.nbp == 400
        assert data.llink == 20
        assert data.elink == pytest.approx(1.0)
        assert data.mu == pytest.approx(-1.0)
        assert data.seed == 7
        np.testing.assert_array_equal(data.position, [10, 200, 350])
        assert data.energy == pytest.approx(-3.5)
        assert data.logz == pytest.approx(12.34)

    def test_requires_internal_flag(self):
        with pytest.raises(TypeError):
            SimTMData(nucbp=1, nbp=1, llink=1, elink=1.0, mu=1.0, seed=1)


class TestConsolidateTMSamples:
    def test_merge_and_read_back(self, tmp_path):
        positions_list = [[5, 20], [100], [7, 8, 9]]
        sample_files = []
        for i, pos in enumerate(positions_list):
            f = tmp_path / f"sample_{i}.h5"
            _write_tm_file(
                f,
                nucbp=2,
                nbp=50,
                llink=3,
                elink=1.0,
                mu=0.5,
                seed=100 + i,
                position=np.array(pos),
                energy=float(i),
                logz=9.87,
            )
            sample_files.append(f)

        combined = tmp_path / "combined.h5"
        _consolidate_tm_samples(sample_files, combined)

        for i, pos in enumerate(positions_list):
            sample = _read_tm_sample(combined, i)
            np.testing.assert_array_equal(sample.position, pos)
            assert sample.energy == pytest.approx(float(i))
            assert sample.seed == 100 + i
            assert sample.logz == pytest.approx(9.87)
            assert sample.nucbp == 2
            assert sample.nbp == 50

    def test_rejects_mismatched_params(self, tmp_path):
        f1 = tmp_path / "a.h5"
        f2 = tmp_path / "b.h5"
        _write_tm_file(
            f1, nucbp=2, nbp=50, llink=3, elink=1.0, mu=0.5, seed=1,
            position=np.array([1]), energy=0.0, logz=1.0,
        )
        _write_tm_file(
            f2, nucbp=3, nbp=50, llink=3, elink=1.0, mu=0.5, seed=2,
            position=np.array([2]), energy=0.0, logz=1.0,
        )
        with pytest.raises(ValueError):
            _consolidate_tm_samples([f1, f2], tmp_path / "combined.h5")

    def test_requires_at_least_one_input(self, tmp_path):
        with pytest.raises(ValueError):
            _consolidate_tm_samples([], tmp_path / "combined.h5")


class TestWriteTMSamples:
    def test_write_and_read_back(self, tmp_path):
        out_file = tmp_path / "mol.h5"
        positions = [np.array([5, 20]), np.array([], dtype=int), np.array([7])]
        _write_tm_samples(
            out_file,
            nucbp=2,
            nbp=50,
            llink=3,
            elink=1.0,
            mu=0.5,
            logz=9.87,
            positions=positions,
            energies=[0.0, 1.0, 2.0],
            seeds=[100, 101, 102],
        )
        for i, pos in enumerate(positions):
            sample = _read_tm_sample(out_file, i)
            np.testing.assert_array_equal(sample.position, pos)
            assert sample.energy == pytest.approx(float(i))
            assert sample.seed == 100 + i
            assert sample.logz == pytest.approx(9.87)
            assert sample.nucbp == 2
            assert sample.nbp == 50
            assert sample.llink == 3
            assert sample.elink == pytest.approx(1.0)
            assert sample.mu == pytest.approx(0.5)

    def test_zero_samples(self, tmp_path):
        out_file = tmp_path / "mol.h5"
        _write_tm_samples(
            out_file,
            nucbp=2,
            nbp=50,
            llink=3,
            elink=1.0,
            mu=0.5,
            logz=1.0,
            positions=[],
            energies=[],
            seeds=[],
        )
        with h5py.File(out_file, "r") as f:
            assert len(f["data"]["energy"]) == 0

    def test_rejects_mismatched_lengths(self, tmp_path):
        with pytest.raises(ValueError):
            _write_tm_samples(
                tmp_path / "mol.h5",
                nucbp=2,
                nbp=50,
                llink=3,
                elink=1.0,
                mu=0.5,
                logz=1.0,
                positions=[np.array([1])],
                energies=[0.0, 1.0],
                seeds=[1],
            )


class TestTMDataset:
    def test_create_with_zero_nsim(self, tmp_path):
        dataset = SimDataset.create(
            chroms="chr1",
            nmol=2,
            nsim=0,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "tm_dataset",
            method="tm",
        )
        assert dataset.method == "tm"
        assert dataset.nsim == 0
        with pytest.raises(ValueError):
            dataset.raw["chr1", 0, 0]

    def test_mcmc_requires_positive_nsim(self, tmp_path):
        with pytest.raises(ValueError):
            SimDataset.create(
                chroms="chr1",
                nmol=2,
                nsim=0,
                nbp={"chr1": 50},
                settings=_make_settings(),
                out_dir=tmp_path / "bad_dataset",
                method="mcmc",
            )

    def test_invalid_method_rejected(self, tmp_path):
        with pytest.raises(ValueError):
            SimDataset.create(
                chroms="chr1",
                nmol=1,
                nsim=1,
                nbp={"chr1": 50},
                settings=_make_settings(),
                out_dir=tmp_path / "bad_method",
                method="not_a_method",
            )

    def _generate_tm_samples(self, dataset, chrom, mol, positions):
        """Stand in for Stage 6's generation+consolidation, for testing
        the data layer independently of SimManager dispatch."""
        sample_files = []
        for run, pos in enumerate(positions):
            f = dataset.mcmc_file(chrom, mol, run)
            f.parent.mkdir(parents=True, exist_ok=True)
            _write_tm_file(
                f,
                nucbp=dataset.settings["nucbp"],
                nbp=dataset.nbp[chrom],
                llink=dataset.settings["llink"],
                elink=dataset.settings["elink"],
                mu=dataset.settings["mu"],
                seed=run + 1,
                position=np.array(pos),
                energy=float(run),
                logz=5.0,
            )
            sample_files.append(f)
        combined = dataset.tm_file(chrom, mol)
        combined.parent.mkdir(parents=True, exist_ok=True)
        _consolidate_tm_samples(sample_files, combined)

    def test_raw_access_after_generating_samples(self, tmp_path):
        dataset = SimDataset.create(
            chroms="chr1",
            nmol=2,
            nsim=3,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "tm_dataset2",
            method="tm",
        )
        self._generate_tm_samples(
            dataset, "chr1", 0, [[2], [4], [6]]
        )

        sample = dataset.raw["chr1", 0, 1]
        assert isinstance(sample, SimTMData)
        assert sample.energy == pytest.approx(1.0)
        np.testing.assert_array_equal(sample.position, [4])

        with pytest.raises(ValueError):
            dataset.raw["chr1", 1, 0]  # molecule 1 has no combined file

    def test_classify_run_tm(self, tmp_path):
        dataset = SimDataset.create(
            chroms="chr1",
            nmol=1,
            nsim=2,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "tm_dataset3",
            method="tm",
        )
        incomplete = dataset.find_incomplete_runs()
        assert set(incomplete["missing"]) == {("chr1", 0, 0), ("chr1", 0, 1)}

        # Only 1 of the 2 requested samples was actually generated.
        self._generate_tm_samples(dataset, "chr1", 0, [[1]])

        incomplete = dataset.find_incomplete_runs()
        assert ("chr1", 0, 0) not in incomplete["missing"]
        assert ("chr1", 0, 1) in incomplete["missing"]

    def test_corrupted_combined_file(self, tmp_path):
        dataset = SimDataset.create(
            chroms="chr1",
            nmol=1,
            nsim=1,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "tm_dataset4",
            method="tm",
        )
        combined = dataset.tm_file("chr1", 0)
        combined.parent.mkdir(parents=True, exist_ok=True)
        combined.write_bytes(b"not valid hdf5")

        incomplete = dataset.find_incomplete_runs()
        assert incomplete["corrupted"] == [("chr1", 0, 0)]

    def test_method_and_nsim_persist_through_save_load(self, tmp_path):
        dataset = SimDataset.create(
            chroms="chr1",
            nmol=1,
            nsim=0,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "tm_dataset5",
            method="tm",
        )
        dataset.save()
        loaded = SimDataset.load(dataset._dataset_file)
        assert loaded.method == "tm"
        assert loaded.nsim == 0

        # Simulate generating samples once, after initial (empty) save.
        self._generate_tm_samples(dataset, "chr1", 0, [[1], [2]])
        dataset._nsim = 2
        dataset.save()
        reloaded = SimDataset.load(dataset._dataset_file)
        assert reloaded.nsim == 2
        assert reloaded.raw["chr1", 0, 0].energy == pytest.approx(0.0)

    def test_legacy_dataset_defaults_to_mcmc(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=1, nsim=1)
        dataset.save()
        with h5py.File(dataset._dataset_file, "a") as f:
            del f["metadata"].attrs["method"]
        loaded = SimDataset.load(dataset._dataset_file)
        assert loaded.method == "mcmc"
