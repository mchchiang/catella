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


class TestRandomSequence:
    def test_length_and_alphabet(self):
        seq = synthgen.random_sequence(500, seed=0)
        assert len(seq) == 500
        assert set(seq) <= set("ACGT")

    def test_seed_reproducible(self):
        a = synthgen.random_sequence(100, seed=3)
        b = synthgen.random_sequence(100, seed=3)
        assert a == b

    def test_invalid_nbp(self):
        with pytest.raises(ValueError):
            synthgen.random_sequence(0)


class TestEligibleSites:
    @pytest.mark.parametrize(
        "mtase, strand, expected",
        [
            ("A", "+", [0, 4]),
            ("A", "-", [5, 6]),
            ("CG", "+", [2]),
            ("CG", "-", [3]),
            ("GC", "+", [8]),
            ("GC", "-", [7]),
        ],
    )
    def test_positions(self, mtase, strand, expected):
        seq = "ACCGATTGC"
        pos = synthgen.eligible_sites(seq, mtase, strand)
        assert pos.tolist() == expected

    def test_lowercase_input(self):
        pos = synthgen.eligible_sites("acga", "A", "+")
        assert pos.tolist() == [0, 3]

    def test_invalid_mtase(self):
        with pytest.raises(ValueError):
            synthgen.eligible_sites("ACGT", "TA", "+")

    def test_invalid_strand(self):
        with pytest.raises(ValueError):
            synthgen.eligible_sites("ACGT", "A", ".")


class TestSimulateCalls:
    def _data(self, nbp=3000, nmol=20, seed=0):
        seq = synthgen.random_sequence(nbp, seed=seed)
        nucs = synthgen.place_nucleosomes(nbp, nmol, seed=seed)
        return seq, nucs

    def test_columns(self):
        seq, nucs = self._data()
        calls = synthgen.simulate_calls(seq, nucs, seed=0)
        assert list(calls.columns) == [
            "mol_id", "chrom", "pos", "strand", "mod_qual"
        ]

    def test_one_strand_per_molecule(self):
        seq, nucs = self._data()
        calls = synthgen.simulate_calls(seq, nucs, seed=0)
        assert (calls.groupby("mol_id")["strand"].nunique() == 1).all()
        assert set(calls["strand"]) <= {"+", "-"}

    @pytest.mark.parametrize("mtase", ["A", "CG", "GC"])
    def test_positions_are_eligible(self, mtase):
        seq, nucs = self._data()
        calls = synthgen.simulate_calls(seq, nucs, mtase=mtase, seed=1)
        for (_, strand), grp in calls.groupby(["mol_id", "strand"]):
            expected = synthgen.eligible_sites(seq, mtase, strand)
            assert grp["pos"].tolist() == expected.tolist()

    def test_mod_qual_range(self):
        seq, nucs = self._data()
        calls = synthgen.simulate_calls(seq, nucs, seed=0)
        assert ((calls["mod_qual"] >= 0) & (calls["mod_qual"] < 1)).all()

    def test_linker_methylated_more_than_nucleosome(self):
        seq, nucs = self._data(nmol=50)
        calls = synthgen.simulate_calls(
            seq, nucs, p_linker=0.8, p_nuc=0.1, seed=2
        )
        inside = np.zeros(len(calls), dtype=bool)
        for mol_id, grp in nucs.groupby("mol_id"):
            sel = (calls["mol_id"] == mol_id).to_numpy()
            pos = calls.loc[sel, "pos"].to_numpy()
            hit = np.zeros(len(pos), dtype=bool)
            for start, end in zip(grp["start"], grp["end"]):
                hit |= (pos >= start) & (pos < end)
            inside[sel] = hit
        meth = calls["mod_qual"].to_numpy() >= 0.5
        assert meth[inside].mean() == pytest.approx(0.1, abs=0.03)
        assert meth[~inside].mean() == pytest.approx(0.8, abs=0.03)

    def test_extreme_probabilities(self):
        seq, nucs = self._data()
        calls = synthgen.simulate_calls(
            seq, nucs, p_linker=1.0, p_nuc=1.0, seed=0
        )
        assert (calls["mod_qual"] >= 0.5).all()

    def test_mol_ids_adds_nucleosome_free_molecule(self):
        seq, nucs = self._data(nmol=2)
        calls = synthgen.simulate_calls(
            seq, nucs, ["mol0", "mol1", "free"], p_linker=1.0, seed=0
        )
        assert set(calls["mol_id"]) == {"mol0", "mol1", "free"}
        free = calls[calls["mol_id"] == "free"]
        assert (free["mod_qual"] >= 0.5).all()

    def test_chrom(self):
        seq, nucs = self._data(nmol=2)
        calls = synthgen.simulate_calls(seq, nucs, chrom="seg", seed=0)
        assert (calls["chrom"] == "seg").all()

    def test_seed_reproducible(self):
        seq, nucs = self._data()
        a = synthgen.simulate_calls(seq, nucs, seed=5)
        b = synthgen.simulate_calls(seq, nucs, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_nucleosome_out_of_bounds(self):
        nucs = pd.DataFrame(
            {"mol_id": ["m"], "chrom": ["c"], "start": [0], "end": [200]}
        )
        with pytest.raises(ValueError):
            synthgen.simulate_calls("ACGT" * 10, nucs)

    @pytest.mark.parametrize(
        "kwargs", [{"p_linker": 1.5}, {"p_nuc": -0.1}]
    )
    def test_invalid_probabilities(self, kwargs):
        seq, nucs = self._data(nmol=1)
        with pytest.raises(ValueError):
            synthgen.simulate_calls(seq, nucs, **kwargs)


class TestWriteChromsize:
    def test_contents(self, tmp_path):
        path = synthgen.write_chromsize(
            {"chr1": 1000, "chr2": 50}, tmp_path / "x.size"
        )
        assert path.read_text() == "chrom\tlength\nchr1\t1000\nchr2\t50\n"


class TestWriteFasta:
    def test_contents(self, tmp_path):
        path = synthgen.write_fasta(
            {"a": "ACGTA", "b": "GG"}, tmp_path / "x.fa", width=2
        )
        assert path.read_text() == ">a\nAC\nGT\nA\n>b\nGG\n"

    def test_invalid_width(self, tmp_path):
        with pytest.raises(ValueError):
            synthgen.write_fasta({"a": "A"}, tmp_path / "x.fa", width=0)


class TestWriteModkitTsv:
    def _calls(self):
        seq = synthgen.random_sequence(1000, seed=0)
        nucs = synthgen.place_nucleosomes(1000, 3, seed=0)
        return synthgen.simulate_calls(seq, nucs, seed=0)

    @pytest.mark.parametrize("mtase, code", [("A", "a"), ("CG", "m")])
    def test_roundtrip(self, tmp_path, mtase, code):
        calls = self._calls()
        path = synthgen.write_modkit_tsv(
            calls, tmp_path / "x.tsv", mtase=mtase
        )
        back = pd.read_csv(path, sep="\t")
        assert list(back.columns) == [
            "read_id",
            "ref_position",
            "chrom",
            "ref_strand",
            "mod_qual",
            "mod_code",
        ]
        assert (back["mod_code"] == code).all()
        assert back["read_id"].tolist() == calls["mol_id"].tolist()
        assert back["ref_position"].tolist() == calls["pos"].tolist()
        assert back["ref_strand"].tolist() == calls["strand"].tolist()
        np.testing.assert_allclose(back["mod_qual"], calls["mod_qual"])

    def test_invalid_mtase(self, tmp_path):
        with pytest.raises(ValueError):
            synthgen.write_modkit_tsv(
                self._calls(), tmp_path / "x.tsv", mtase="TA"
            )
