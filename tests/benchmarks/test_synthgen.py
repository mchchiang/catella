# test_synthgen.py

import numpy as np
import pandas as pd
import pytest
import synthgen


class TestPlaceNucleosomes:
    def test_columns(self):
        nucs = synthgen.place_nucleosomes(2000, 3, seed=0)
        assert list(nucs.columns) == ["mol_id", "chrom", "start", "end"]

    def test_lengths_and_bounds(self):
        nucs = synthgen.place_nucleosomes(2000, 5, lnuc=147, seed=1)
        assert (nucs["end"] - nucs["start"] == 147).all()
        assert (nucs["start"] >= 0).all()
        assert (nucs["end"] <= 2000).all()

    def test_no_overlaps(self):
        nucs = synthgen.place_nucleosomes(5000, 10, seed=2)
        for _, grp in nucs.groupby("mol_id"):
            starts = grp["start"].to_numpy()
            ends = grp["end"].to_numpy()
            assert (np.diff(starts) > 0).all()
            assert (starts[1:] >= ends[:-1]).all()

    def test_molecule_count_and_chrom(self):
        nucs = synthgen.place_nucleosomes(3000, 4, chrom="seg", seed=3)
        assert nucs["mol_id"].nunique() == 4
        assert (nucs["chrom"] == "seg").all()

    def test_zero_linker_packs_densely(self):
        nucs = synthgen.place_nucleosomes(
            1470, 1, lnuc=147, linker_mean=0.0, seed=0
        )
        assert len(nucs) == 10
        assert nucs["start"].tolist() == list(range(0, 1470, 147))

    def test_seed_reproducible(self):
        a = synthgen.place_nucleosomes(2000, 3, seed=7)
        b = synthgen.place_nucleosomes(2000, 3, seed=7)
        pd.testing.assert_frame_equal(a, b)

    def test_molecule_shorter_than_nucleosome(self):
        nucs = synthgen.place_nucleosomes(100, 2, lnuc=147, seed=0)
        assert len(nucs) == 0

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"nbp": 0, "nmol": 1},
            {"nbp": 100, "nmol": 0},
            {"nbp": 100, "nmol": 1, "lnuc": 0},
            {"nbp": 100, "nmol": 1, "linker_mean": -1.0},
        ],
    )
    def test_invalid_args(self, kwargs):
        with pytest.raises(ValueError):
            synthgen.place_nucleosomes(**kwargs)


class TestWriteTruthBed:
    def test_roundtrip(self, tmp_path):
        nucs = synthgen.place_nucleosomes(2000, 2, seed=0)
        path = synthgen.write_truth_bed(nucs, tmp_path / "truth.bed")
        back = pd.read_csv(
            path, sep="\t", header=None,
            names=["chrom", "start", "end", "mol_id"],
        )
        pd.testing.assert_frame_equal(
            back[["mol_id", "chrom", "start", "end"]], nucs
        )
