# test_methdata.py

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from nucmc.experiment.methdata import (
    LazyRawDataMap, MethPrintData, MethPrintExperiment)


def _make_raw(chrom, nmol, nbp, seed):
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
        chrom=chrom, nbp=nbp, test_mol_id=mol_id, test_data=df,
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
