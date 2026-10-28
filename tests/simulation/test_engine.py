# test_engine.py

from pathlib import Path

import h5py
import numpy as np
import pytest
from catella.simulation.config import SimSettings
from catella.simulation.engine import MCMCJob, SimManager, TMJob
from catella.simulation.results import SimDataset, _read_tm_sample
from catella_cpp import Dump, NucPosTM
from scipy.special import logit


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


def _make_meth_prob(nmol=3, nbp=400, seed=0):
    rng = np.random.default_rng(seed)
    return rng.random((nmol, nbp))


def _make_sim_run(out_file, **overrides):
    seq_energy = np.asarray(
        _make_meth_prob(nmol=1)[0], dtype=np.float64
    ).tobytes()
    defaults = {
        "chrom": "chr1",
        "mol": 0,
        "run": 0,
        "seed": 1,
        "seq_energy": seq_energy,
        "out_type": Dump.OutputType.All,
        "out_file": out_file,
        "settings": _make_settings(),
    }
    defaults.update(overrides)
    return MCMCJob(**defaults)


class TestSimManagerInit:
    def test_invalid_nworker_raises(self):
        with pytest.raises(ValueError):
            SimManager(nworker=0)

    def test_default_construction(self):
        manager = SimManager()
        assert manager.nworker == 1
        assert manager.verbose is True

    def test_no_longer_accepts_mp_context(self):
        with pytest.raises(TypeError):
            SimManager(mp_context="spawn")


class TestRunJob:
    def test_success_returns_true(self, tmp_path):
        p = _make_sim_run(tmp_path / "out.h5")
        chrom, mol, run, success, err_msg = SimManager._run_mcmc_job(p)
        assert (chrom, mol, run) == ("chr1", 0, 0)
        assert success is True
        assert err_msg is None

    def test_failure_returns_false_with_message(self, tmp_path):
        # An embedded null byte in the path makes mkdir fail, so the
        # job should report failure instead of raising.
        bad_path = Path(str(tmp_path) + "/bad\x00dir/out.h5")
        p = _make_sim_run(bad_path)
        chrom, mol, run, success, err_msg = SimManager._run_mcmc_job(p)
        assert (chrom, mol, run) == ("chr1", 0, 0)
        assert success is False
        assert err_msg


class TestSimManagerRun:
    def test_serial_run_produces_dataset(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=3, nbp=200)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "serial",
            seed=1,
        )
        assert dataset.nmol["chr1"] == 3

    def test_multiple_out_types_combine_correctly(self, tmp_path):
        # Regression test: combining more than one out_type previously
        # always raised ValueError (the validation checked the resolved
        # enum value against a dict of string keys, which never matches),
        # and even past that, Dump.OutputType is a scoped C++ enum class
        # that pybind11 doesn't bind '|' for directly.
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "multi",
            seed=1,
            out_types=["energy", "position"],
        )
        sim_file = dataset.mcmc_file("chr1", 0, 0)
        with h5py.File(sim_file, "r") as f:
            keys = set(f["data"].keys())
        assert "energy" in keys
        assert {"position_flat", "position_offset"} <= keys

    def test_parallel_run_produces_same_shape_dataset(self, tmp_path):
        # Regression test: parallel dispatch previously hardcoded a
        # 'fork' multiprocessing context, which raised NameError on
        # macOS (missing 'os' import) and duplicated the parent
        # process's memory into every worker. It now always uses
        # 'spawn' via the stdlib multiprocessing module.
        meth_prob = _make_meth_prob(nmol=3, nbp=200)
        manager = SimManager(nworker=2, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "parallel",
            seed=1,
        )
        assert dataset.nmol["chr1"] == 3

    def test_failed_job_does_not_abort_the_batch(
        self, tmp_path, monkeypatch, capsys
    ):
        # Regression test: `if not result:` on the always-truthy result
        # tuple used to make failed jobs silently disappear. A failure
        # should now (a) be reported to the console and (b) not stop the
        # rest of the batch from completing. `_run_mcmc_job` is monkeypatched
        # (rather than relying on a genuine backend failure) so this test
        # exercises the batch's error-handling regardless of what the C++
        # backend happens to reject.
        calls = []

        def fake_run_mcmc_job(p):
            calls.append(p)
            if len(calls) == 1:
                return (p.chrom, p.mol, p.run, False, "synthetic failure")
            return (p.chrom, p.mol, p.run, True, None)

        monkeypatch.setattr(
            SimManager, "_run_mcmc_job", staticmethod(fake_run_mcmc_job)
        )

        meth_prob = _make_meth_prob(nmol=2, nbp=200)
        manager = SimManager(nworker=1, verbose=True)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "fail",
            seed=1,
        )

        assert len(calls) == 2
        assert dataset.nmol["chr1"] == 2
        captured = capsys.readouterr()
        assert "Simulation failed" in captured.out

    def test_persists_out_type_seed_and_seed_table(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "persist",
            seed=7,
        )

        assert dataset.out_type == Dump.OutputType.All
        assert dataset.seed == 7
        assert dataset.seed_table["chr1"].shape == (2, 2)

        sim_file = dataset.mcmc_file("chr1", 1, 0)
        with h5py.File(sim_file, "r") as f:
            recorded_seed = int(f["params"].attrs["seed"])
        assert int(dataset.seed_table["chr1"][1, 0]) == recorded_seed

    def test_median_eseq_mu_reflected_in_persisted_settings(self, tmp_path):
        # Regression test: use_median_eseq_mu used to adjust mu only via a
        # per-job MCMCJob.override dict, so dataset.settings.mu silently
        # kept showing the pre-adjustment value. It's now applied via
        # dataclasses.replace() before the dataset is created, so the
        # persisted settings reflect the mu actually used.
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "medianmu",
            seed=1,
            use_median_eseq_mu=True,
        )
        assert dataset.settings["mu"] == np.median(dataset.eseq["chr1"])

    def test_median_eseq_mu_ignores_nan(self, tmp_path):
        # meth_prob commonly has trailing NaN columns (left-aligned
        # window edge from compute_model_prob/compute_empirical_prob)
        # or all-NaN dropout rows. Those must not turn mu (and hence
        # the whole simulation's energy) into NaN.
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        meth_prob[:, -5:] = np.nan
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "mediannan",
            seed=1,
            use_median_eseq_mu=True,
        )
        assert not np.isnan(dataset.settings["mu"])
        assert dataset.settings["mu"] == np.nanmedian(dataset.eseq["chr1"])

    def test_median_eseq_mu_without_store_eseq(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        stored = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "stored",
            seed=1,
            use_median_eseq_mu=True,
        )
        unstored = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "unstored",
            seed=1,
            store_eseq=False,
            use_median_eseq_mu=True,
        )
        assert unstored.settings["mu"] == stored.settings["mu"]
        assert not unstored.has_eseq

    def test_same_seed_gives_same_seed_table_regardless_of_mols(
        self, tmp_path
    ):
        # The seed table is order-independent: a triplet's seed doesn't
        # depend on which other molecules were selected via 'mols'.
        meth_prob = _make_meth_prob(nmol=3, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        full = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "full",
            seed=3,
        )
        subset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "subset",
            seed=3,
            mols=slice(2, 3),
        )
        assert (full.seed_table["chr1"] == subset.seed_table["chr1"]).all()

    def test_broken_pool_is_reported_and_reraised(self, tmp_path, capsys):
        # Regression test for the sliding-window dispatch: if a future's
        # result() raises for a reason _run_mcmc_job's own try/except can't
        # catch (e.g. the worker process itself died), that must be
        # reported distinctly (including the status of every run that was
        # still in flight) and re-raised rather than silently hanging or
        # being swallowed. A genuine worker crash is hard to trigger
        # portably, so an unpicklable settings field is used instead: it
        # makes ProcessPoolExecutor.submit()'s payload fail to serialize,
        # which surfaces via future.result() the same way a broken pool
        # would, without needing to actually kill a subprocess. Exercised
        # directly against _dispatch() (rather than the full run()) so it
        # doesn't also need a dataset.save()-writable settings object.
        class Unpicklable:
            def __reduce__(self):
                raise TypeError("cannot pickle this")

        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 10},
            settings=_make_settings(),
            out_dir=tmp_path / "broken",
            out_type=Dump.OutputType.All,
        )

        bad_run = _make_sim_run(
            dataset.mcmc_file("chr1", 0, 0), settings=Unpicklable()
        )
        manager = SimManager(nworker=2, verbose=False)

        with pytest.raises(TypeError, match="cannot pickle"):
            manager._dispatch(
                dataset, iter([bad_run]), 1, SimManager._run_mcmc_job
            )

        captured = capsys.readouterr()
        assert "Worker pool crashed" in captured.out
        assert "chr1" in captured.out
        assert "missing" in captured.out


class TestSimManagerRerun:
    def test_only_none_fixes_missing_run(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr1",
            seed=5,
        )

        target_file = dataset.mcmc_file("chr1", 1, 0)
        original_seed = int(dataset.seed_table["chr1"][1, 0])
        target_file.unlink()
        assert dataset.find_incomplete_runs()["missing"] == [("chr1", 1, 0)]

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        incomplete = dataset.find_incomplete_runs()
        assert incomplete == {"missing": [], "corrupted": [], "truncated": []}
        with h5py.File(target_file, "r") as f:
            assert int(f["params"].attrs["seed"]) == original_seed

    def test_dataset_accepted_positionally(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr_pos",
            seed=5,
        )
        dataset.mcmc_file("chr1", 1, 0).unlink()

        manager.rerun(dataset, meth_prob=meth_prob)

        incomplete = dataset.find_incomplete_runs()
        assert incomplete == {"missing": [], "corrupted": [], "truncated": []}

    def test_explicit_only_reruns_exactly_that_target(
        self, tmp_path, monkeypatch
    ):
        meth_prob = _make_meth_prob(nmol=2, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr2",
            seed=5,
        )

        calls = []
        real_run_mcmc_job = SimManager._run_mcmc_job

        def spy_run_mcmc_job(p):
            calls.append((p.chrom, p.mol, p.run))
            return real_run_mcmc_job(p)

        monkeypatch.setattr(
            SimManager, "_run_mcmc_job", staticmethod(spy_run_mcmc_job)
        )

        manager.rerun(
            dataset=dataset, meth_prob=meth_prob, only=[("chr1", 0, 0)]
        )
        assert calls == [("chr1", 0, 0)]

    def test_legacy_dataset_requires_explicit_out_type_and_seed(
        self, tmp_path
    ):
        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 50},
            settings=_make_settings(),
            out_dir=tmp_path / "legacy",
        )
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)

        with pytest.raises(ValueError, match="out_type"):
            manager.rerun(
                dataset=dataset, meth_prob=meth_prob, only=[("chr1", 0, 0)]
            )

        with pytest.raises(ValueError, match="seed"):
            manager.rerun(
                dataset=dataset,
                meth_prob=meth_prob,
                only=[("chr1", 0, 0)],
                out_type=Dump.OutputType.All,
            )

    def test_explicit_seed_override_differs_from_reproduction(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr3",
            seed=5,
        )
        original_seed = int(dataset.seed_table["chr1"][0, 0])
        target_file = dataset.mcmc_file("chr1", 0, 0)

        manager.rerun(
            dataset=dataset,
            meth_prob=meth_prob,
            only=[("chr1", 0, 0)],
            seed=999,
        )

        with h5py.File(target_file, "r") as f:
            reran_seed = int(f["params"].attrs["seed"])
        assert reran_seed != original_seed

    def test_rerun_overwrites_corrupted_file(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=50)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=1,
            settings=_make_settings(nsweep=10),
            meth_prob=meth_prob,
            out_dir=tmp_path / "rr4",
            seed=5,
        )
        target_file = dataset.mcmc_file("chr1", 0, 0)
        target_file.write_bytes(b"not a valid hdf5 file")
        assert dataset.find_incomplete_runs()["corrupted"] == [("chr1", 0, 0)]

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }


def _make_tm_settings(**overrides):
    defaults = {
        "nucbp": 3,
        "llink": 2,
        "elink": 1.0,
        "mu": 0.5,
        "end_temp": 1.0,
        "emax": 5.0,
    }
    defaults.update(overrides)
    return SimSettings(**defaults)


def _make_tm_job(out_file, seeds=(1, 2, 3), nbp=20, **overrides):
    seq_energy = np.asarray(
        logit(_make_meth_prob(nmol=1, nbp=nbp)[0]), dtype=np.float64
    ).tobytes()
    defaults = {
        "chrom": "chr1",
        "mol": 0,
        "seeds": seeds,
        "seq_energy": seq_energy,
        "out_file": out_file,
        "settings": _make_tm_settings(),
    }
    defaults.update(overrides)
    return TMJob(**defaults)


class TestRunTMJob:
    def test_success_writes_all_samples(self, tmp_path):
        out_file = tmp_path / "sub" / "mol.h5"
        p = _make_tm_job(out_file, seeds=(4, 5, 6))
        result = SimManager._run_tm_job(p)
        assert result == ("chr1", 0, None, True, None)
        for run, s in enumerate((4, 5, 6)):
            sample = _read_tm_sample(out_file, run)
            assert sample.seed == s
            assert sample.nucbp == 3
            assert sample.nbp == 20

    def test_samples_match_fresh_solve_per_seed(self, tmp_path):
        out_file = tmp_path / "mol.h5"
        p = _make_tm_job(out_file, seeds=(11, 12, 13))
        SimManager._run_tm_job(p)
        settings = p.settings
        eseq = np.frombuffer(p.seq_energy, dtype=np.float64).tolist()
        for run, s in enumerate(p.seeds):
            tm = NucPosTM(
                settings.nucbp,
                len(eseq),
                settings.llink,
                settings.elink,
                settings.mu,
                s,
            )
            tm.setEnergy(eseq, settings.emax)
            tm.solve(settings.end_temp)
            position, energy = tm.sampleConfig()
            sample = _read_tm_sample(out_file, run)
            np.testing.assert_array_equal(sample.position, position)
            assert sample.energy == energy
            assert sample.logz == tm.getLogZ()

    def test_failure_reports_error_and_writes_nothing(self, tmp_path):
        out_file = tmp_path / "mol.h5"
        p = _make_tm_job(out_file, settings=_make_tm_settings(end_temp=0.0))
        chrom, mol, run, success, err_msg = SimManager._run_tm_job(p)
        assert (chrom, mol, run) == ("chr1", 0, None)
        assert success is False
        assert err_msg
        assert not out_file.exists()


class TestSimManagerRunTM:
    def test_zero_nsim_creates_dataset_with_no_samples(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=0,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm0",
            seed=1,
        )
        assert dataset.method == "tm"
        assert dataset.nsim == 0
        with pytest.raises(ValueError):
            dataset.raw["chr1", 0, 0]
        assert not dataset.tm_file("chr1", 0).exists()

    def test_samples_use_seed_table(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=3,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_seeds",
            seed=1,
        )
        for mol in range(2):
            for run in range(3):
                sample = dataset.raw["chr1", mol, run]
                assert sample.seed == dataset.seed_table["chr1"][mol, run]

    def test_parallel_matches_serial(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=3, nbp=30)
        datasets = [
            SimManager(nworker=nworker, verbose=False).run(
                chroms="chr1",
                nsim=2,
                method="tm",
                settings=_make_tm_settings(),
                meth_prob=meth_prob,
                out_dir=tmp_path / f"tm_par{nworker}",
                seed=1,
            )
            for nworker in (1, 2)
        ]
        for mol in range(3):
            for run in range(2):
                a = datasets[0].raw["chr1", mol, run]
                b = datasets[1].raw["chr1", mol, run]
                np.testing.assert_array_equal(a.position, b.position)
                assert a.energy == b.energy

    def test_generates_and_consolidates_samples(self, tmp_path):
        from catella.simulation.results import SimTMData

        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=3,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm1",
            seed=1,
        )
        assert dataset.method == "tm"
        assert dataset.nsim == 3

        sample = dataset.raw["chr1", 0, 1]
        assert isinstance(sample, SimTMData)
        assert sample.nucbp == 3
        assert sample.nbp == 30

        # Temp per-run files should be cleaned up after consolidation.
        for run in range(3):
            assert not dataset.mcmc_file("chr1", 0, run).exists()
        assert dataset.tm_file("chr1", 0).exists()

        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }

    def test_reload_preserves_method_and_samples(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm2",
            seed=1,
        )
        loaded = SimDataset.load(dataset._dataset_file)
        assert loaded.method == "tm"
        assert loaded.nsim == 2
        assert loaded.raw["chr1", 0, 0].nucbp == 3


class TestRerunTM:
    def test_rejects_triplets_in_only(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=1, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_rerun1",
            seed=1,
        )
        with pytest.raises(ValueError):
            manager.rerun(
                dataset=dataset,
                meth_prob=meth_prob,
                only=[("chr1", 0, 0)],
            )

    def test_regenerates_missing_molecule(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_rerun2",
            seed=1,
        )
        dataset.tm_file("chr1", 1).unlink()
        assert dataset.find_incomplete_runs()["missing"] != []

        manager.rerun(dataset=dataset, meth_prob=meth_prob)

        assert dataset.find_incomplete_runs() == {
            "missing": [],
            "corrupted": [],
            "truncated": [],
        }
        assert dataset.raw["chr1", 1, 0] is not None

    def test_only_pairs_regenerate_specific_molecule(self, tmp_path):
        meth_prob = _make_meth_prob(nmol=2, nbp=30)
        manager = SimManager(nworker=1, verbose=False)
        dataset = manager.run(
            chroms="chr1",
            nsim=2,
            method="tm",
            settings=_make_tm_settings(),
            meth_prob=meth_prob,
            out_dir=tmp_path / "tm_rerun3",
            seed=1,
        )
        old_energy = dataset.raw["chr1", 0, 0].energy
        dataset.tm_file("chr1", 0).unlink()

        manager.rerun(
            dataset=dataset, meth_prob=meth_prob, only=[("chr1", 0)]
        )

        # Reproduced with the same seed_table, so the exact sample
        # should match what was there before.
        assert dataset.raw["chr1", 0, 0].energy == pytest.approx(
            old_energy
        )
