# test_jobs.py

import numpy as np
import pytest
from catella.simulation._jobs import (
    MCMCJob,
    TMJob,
    _check_seed_source,
    _compute_eseq,
    _eseq_bytes,
    _iter_mcmc_jobs,
    _iter_selected,
    _iter_tm_jobs,
    _job_seed,
    _mcmc_rerun_targets,
    _median_eseq_mu,
    _resolve_master_seed,
    _run_seeds,
    _seed_table,
    _tm_rerun_targets,
)
from catella.simulation.config import SimSettings
from catella.simulation.results import SimDataset
from catella_cpp import Dump, NucPosMCMC
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


def _make_dataset(tmp_path, *, nmol=3, nsim=2, method="mcmc", seed=7):
    chroms = ["chr1", "chr2"]
    nmol_map = {chrom: nmol for chrom in chroms}
    return SimDataset.create(
        chroms=chroms,
        nmol=nmol_map,
        nsim=nsim,
        nbp={chrom: 30 for chrom in chroms},
        settings=_make_settings(),
        out_dir=tmp_path / "ds",
        method=method,
        seed=seed,
        seed_table=_seed_table(seed, chroms, nmol_map, nsim),
    )


def _meth_prob(nmol=3, nbp=30):
    rng = np.random.default_rng(0)
    return {
        "chr1": rng.uniform(0.1, 0.9, (nmol, nbp)),
        "chr2": rng.uniform(0.1, 0.9, (nmol, nbp)),
    }


class TestJobSeed:
    def test_deterministic(self):
        assert _job_seed(5, 0, 1, 2) == _job_seed(5, 0, 1, 2)

    @pytest.mark.parametrize(
        "other", [(6, 0, 1, 2), (5, 1, 1, 2), (5, 0, 2, 2), (5, 0, 1, 3)]
    )
    def test_depends_on_each_input(self, other):
        assert _job_seed(5, 0, 1, 2) != _job_seed(*other)

    def test_returns_python_int(self):
        assert type(_job_seed(5, 0, 1, 2)) is int


class TestResolveMasterSeed:
    def test_given_seed_returned(self):
        assert _resolve_master_seed(123) == 123

    def test_none_draws_seed_in_range(self):
        seed = _resolve_master_seed(None)
        assert isinstance(seed, int)
        assert 0 <= seed < 2**63


class TestSeedTable:
    def test_shape_dtype_and_values(self):
        nmol = {"a": 2, "b": 3}
        table = _seed_table(9, ["a", "b"], nmol, 4)
        assert list(table) == ["a", "b"]
        for chrom_idx, chrom in enumerate(["a", "b"]):
            arr = table[chrom]
            assert arr.shape == (nmol[chrom], 4)
            assert arr.dtype == np.uint32
            for mol in range(nmol[chrom]):
                for run in range(4):
                    assert arr[mol, run] == _job_seed(9, chrom_idx, mol, run)

    def test_zero_runs(self):
        table = _seed_table(9, ["a"], {"a": 2}, 0)
        assert table["a"].shape == (2, 0)


class TestCheckSeedSource:
    def test_passes_with_seed_table(self, tmp_path):
        _check_seed_source(_make_dataset(tmp_path), None)

    def test_passes_with_new_seed(self, tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 30},
            settings=_make_settings(),
            out_dir=tmp_path / "legacy",
        )
        _check_seed_source(dataset, 3)

    def test_raises_without_seed_table(self, tmp_path):
        dataset = SimDataset.create(
            chroms=["chr1"],
            nmol={"chr1": 1},
            nsim=1,
            nbp={"chr1": 30},
            settings=_make_settings(),
            out_dir=tmp_path / "legacy",
        )
        with pytest.raises(ValueError, match="seed"):
            _check_seed_source(dataset, None)


class TestRunSeeds:
    def test_stored_seeds(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        seeds = _run_seeds(dataset, "chr2", 1, [1, 0])
        table = dataset.seed_table["chr2"]
        assert seeds == (int(table[1, 1]), int(table[1, 0]))

    def test_new_master_seed(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        seeds = _run_seeds(dataset, "chr2", 1, [0, 1], seed=42)
        assert seeds == (_job_seed(42, 1, 1, 0), _job_seed(42, 1, 1, 1))


class TestEseqBytes:
    def test_round_trip(self):
        row = np.array([0.2, 0.5, 0.8])
        out = np.frombuffer(_eseq_bytes(row), dtype=np.float64)
        np.testing.assert_allclose(out, logit(row))


class TestComputeEseq:
    def test_matches_model_per_molecule(self):
        settings = _make_settings()
        meth_prob = _meth_prob(nmol=2)
        eseq = _compute_eseq(settings, meth_prob, ["chr2"])
        assert list(eseq) == ["chr2"]
        assert eseq["chr2"].shape == meth_prob["chr2"].shape
        model = NucPosMCMC(147, 30, 20, 1.0, -1.0, 0)
        model.setEnergy(logit(meth_prob["chr2"][1]), 5.0)
        np.testing.assert_allclose(eseq["chr2"][1], model.getSeqEnergy())


class TestMedianEseqMu:
    def test_mean_of_chrom_medians(self):
        eseq = {
            "a": np.array([[1.0, 2.0, 9.0]]),
            "b": np.array([[np.nan, 4.0, 6.0]]),
        }
        assert _median_eseq_mu(eseq) == pytest.approx(3.5)


class TestIterSelected:
    def test_selections_per_chrom(self):
        out = list(
            _iter_selected(
                ["a", "b"],
                {"a": 4, "b": 3},
                {"a": slice(1, 3), "b": [2, 0]},
            )
        )
        assert out == [("a", 1), ("a", 2), ("b", 2), ("b", 0)]

    def test_single_index(self):
        out = list(_iter_selected(["a"], {"a": 4}, {"a": 3}))
        assert out == [("a", 3)]

    def test_yields_python_ints(self):
        out = list(_iter_selected(["a"], {"a": 2}, {"a": slice(None)}))
        assert all(type(mol) is int for _, mol in out)


class TestIterMCMCJobs:
    def test_job_fields(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        settings = _make_settings(llink={"chr1": 10, "chr2": 15})
        meth_prob = _meth_prob()
        targets = [("chr1", 0, 1), ("chr2", 2, 0)]
        jobs = list(
            _iter_mcmc_jobs(
                dataset, settings, meth_prob, targets, Dump.OutputType.All
            )
        )
        assert [(j.chrom, j.mol, j.run) for j in jobs] == targets
        for job, (chrom, mol, run) in zip(jobs, targets, strict=True):
            assert isinstance(job, MCMCJob)
            assert job.seed == int(dataset.seed_table[chrom][mol, run])
            assert job.settings == settings.for_chrom(chrom)
            assert job.out_type == Dump.OutputType.All
            assert job.out_file == dataset.mcmc_file(chrom, mol, run)
            assert job.seq_energy == _eseq_bytes(meth_prob[chrom][mol, :])

    def test_new_master_seed(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        jobs = list(
            _iter_mcmc_jobs(
                dataset,
                _make_settings(),
                _meth_prob(),
                [("chr2", 1, 1)],
                Dump.OutputType.Energy,
                seed=11,
            )
        )
        assert jobs[0].seed == _job_seed(11, 1, 1, 1)

    def test_lazy(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        gen = _iter_mcmc_jobs(
            dataset,
            _make_settings(),
            _meth_prob(),
            iter([("chr1", 0, 0)]),
            Dump.OutputType.All,
        )
        assert next(gen).mol == 0
        with pytest.raises(StopIteration):
            next(gen)


class TestIterTMJobs:
    def test_job_fields(self, tmp_path):
        dataset = _make_dataset(tmp_path, nsim=3, method="tm")
        settings = _make_settings()
        meth_prob = _meth_prob()
        jobs = list(
            _iter_tm_jobs(dataset, settings, meth_prob, [("chr2", 1)])
        )
        assert len(jobs) == 1
        job = jobs[0]
        assert isinstance(job, TMJob)
        assert (job.chrom, job.mol) == ("chr2", 1)
        expected = tuple(int(s) for s in dataset.seed_table["chr2"][1, :])
        assert job.seeds == expected
        assert job.out_file == dataset.tm_file("chr2", 1)
        assert job.settings == settings.for_chrom("chr2")
        assert job.seq_energy == _eseq_bytes(meth_prob["chr2"][1, :])

    def test_new_master_seed(self, tmp_path):
        dataset = _make_dataset(tmp_path, nsim=2, method="tm")
        jobs = list(
            _iter_tm_jobs(
                dataset, _make_settings(), _meth_prob(), [("chr1", 2)], 5
            )
        )
        assert jobs[0].seeds == (_job_seed(5, 0, 2, 0), _job_seed(5, 0, 2, 1))


class TestMCMCRerunTargets:
    def test_only_returned_as_list(self, tmp_path):
        dataset = _make_dataset(tmp_path)
        only = (t for t in [("chr1", 0, 1)])
        assert _mcmc_rerun_targets(dataset, only) == [("chr1", 0, 1)]

    def test_none_uses_incomplete_runs(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=2)
        targets = _mcmc_rerun_targets(dataset, None)
        expected = {
            (chrom, mol, run)
            for chrom in ("chr1", "chr2")
            for mol in range(2)
            for run in range(2)
        }
        assert set(targets) == expected
        assert len(targets) == len(expected)


class TestTMRerunTargets:
    def test_only_pairs_returned(self, tmp_path):
        dataset = _make_dataset(tmp_path, method="tm")
        only = [("chr1", 0), ("chr2", 2)]
        assert _tm_rerun_targets(dataset, only) == only

    def test_triplet_raises(self, tmp_path):
        dataset = _make_dataset(tmp_path, method="tm")
        with pytest.raises(ValueError):
            _tm_rerun_targets(dataset, [("chr1", 0, 0)])

    def test_none_uses_distinct_sorted_molecules(self, tmp_path):
        dataset = _make_dataset(tmp_path, nmol=2, nsim=3, method="tm")
        targets = _tm_rerun_targets(dataset, None)
        assert targets == [("chr1", 0), ("chr1", 1), ("chr2", 0), ("chr2", 1)]
