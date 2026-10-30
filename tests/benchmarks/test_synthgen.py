# test_synthgen.py

import math

import numpy as np
import pandas as pd
import pytest
import synthgen
from typer.testing import CliRunner

runner = CliRunner()


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
            {"nbp": 100, "nmol": 1, "lnuc_range": (0, 10)},
            {"nbp": 100, "nmol": 1, "lnuc_range": (20, 10)},
        ],
    )
    def test_invalid_args(self, kwargs):
        with pytest.raises(ValueError):
            synthgen.place_nucleosomes(**kwargs)

    def test_lnuc_range_lengths(self):
        nucs = synthgen.place_nucleosomes(
            20000, 5, lnuc_range=(120, 160), seed=0
        )
        lengths = nucs["end"] - nucs["start"]
        assert lengths.min() >= 120
        assert lengths.max() <= 160
        assert lengths.nunique() > 10
        for _, grp in nucs.groupby("mol_id"):
            starts = grp["start"].to_numpy()
            ends = grp["end"].to_numpy()
            assert (starts[1:] >= ends[:-1]).all()
        assert (nucs["end"] <= 20000).all()

    def test_lnuc_range_single_value(self):
        nucs = synthgen.place_nucleosomes(
            3000, 3, lnuc_range=(100, 100), seed=0
        )
        assert (nucs["end"] - nucs["start"] == 100).all()


class TestWriteTruthBed:
    def test_roundtrip(self, tmp_path):
        nucs = synthgen.place_nucleosomes(2000, 2, seed=0)
        path = synthgen.write_truth_bed(nucs, tmp_path / "truth.bed")
        assert path.read_text().splitlines()[0] == "#chrom\tstart\tend\tmol_id"
        back = pd.read_csv(
            path,
            sep="\t",
            header=None,
            comment="#",
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

    @pytest.mark.parametrize("mtase", ["A", "CG", "GC"])
    def test_double_strand_reads_both_strands(self, mtase):
        seq, nucs = self._data(nmol=5)
        calls = synthgen.simulate_calls(
            seq, nucs, mtase=mtase, double_strand=True, seed=0
        )
        for _, grp in calls.groupby("mol_id"):
            assert (np.diff(grp["pos"].to_numpy()) > 0).all()
            for strand in ("+", "-"):
                pos = grp.loc[grp["strand"] == strand, "pos"]
                expected = synthgen.eligible_sites(seq, mtase, strand)
                assert pos.tolist() == expected.tolist()

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
        assert ((calls["mod_qual"] >= 0) & (calls["mod_qual"] <= 1)).all()

    def test_linker_methylated_more_than_nucleosome(self):
        seq, nucs = self._data(nmol=50)
        calls = synthgen.simulate_calls(
            seq, nucs, p_linker=0.8, p_nuc=0.1, signal_sep=10.0, seed=2
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
            seq, nucs, p_linker=1.0, p_nuc=1.0, signal_sep=10.0, seed=0
        )
        assert (calls["mod_qual"] >= 0.5).all()

    def test_mol_ids_adds_nucleosome_free_molecule(self):
        seq, nucs = self._data(nmol=2)
        calls = synthgen.simulate_calls(
            seq,
            nucs,
            ["mol0", "mol1", "free"],
            p_linker=1.0,
            signal_sep=10.0,
            seed=0,
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
        "kwargs",
        [
            {"p_linker": 1.5},
            {"p_nuc": -0.1},
            {"signal_sep": 0.0},
            {"signal_sep": -1.0},
            {"report_sep": 0.0},
        ],
    )
    def test_invalid_args(self, kwargs):
        seq, nucs = self._data(nmol=1)
        with pytest.raises(ValueError):
            synthgen.simulate_calls(seq, nucs, **kwargs)

    def _free_quals(self, p_linker, signal_sep, seed=0, report_sep=None):
        seq = synthgen.random_sequence(5000, seed=0)
        nucs = pd.DataFrame(columns=["mol_id", "chrom", "start", "end"])
        mol_ids = [f"m{i}" for i in range(40)]
        calls = synthgen.simulate_calls(
            seq,
            nucs,
            mol_ids,
            p_linker=p_linker,
            signal_sep=signal_sep,
            report_sep=report_sep,
            seed=seed,
        )
        return calls["mod_qual"].to_numpy()

    def test_report_sep_keeps_thresholded_calls(self):
        calibrated = self._free_quals(0.5, 1.5, seed=3)
        overconfident = self._free_quals(0.5, 1.5, seed=3, report_sep=3.0)
        np.testing.assert_array_equal(calibrated >= 0.5, overconfident >= 0.5)
        assert not np.allclose(calibrated, overconfident)

    def test_overconfident_scores(self):
        meth_q = self._free_quals(1.0, 1.5, seed=1, report_sep=3.0)
        unmeth_q = self._free_quals(0.0, 1.5, seed=2, report_sep=3.0)
        n_meth = ((meth_q > 0.75) & (meth_q < 0.85)).sum()
        n_unmeth = ((unmeth_q > 0.75) & (unmeth_q < 0.85)).sum()
        frac = n_meth / (n_meth + n_unmeth)
        assert frac == pytest.approx(2 / 3, abs=0.04)

    @pytest.mark.parametrize("signal_sep", [1.0, 2.0])
    def test_wrong_side_rate(self, signal_sep):
        expected = 0.5 * math.erfc(signal_sep / math.sqrt(2))
        meth_q = self._free_quals(1.0, signal_sep, seed=1)
        unmeth_q = self._free_quals(0.0, signal_sep, seed=2)
        assert (meth_q < 0.5).mean() == pytest.approx(expected, abs=0.005)
        assert (unmeth_q >= 0.5).mean() == pytest.approx(expected, abs=0.005)

    def test_calibrated(self):
        meth_q = self._free_quals(1.0, 1.5, seed=1)
        unmeth_q = self._free_quals(0.0, 1.5, seed=2)
        n_meth = ((meth_q > 0.75) & (meth_q < 0.85)).sum()
        n_unmeth = ((unmeth_q > 0.75) & (unmeth_q < 0.85)).sum()
        assert n_meth / (n_meth + n_unmeth) == pytest.approx(0.8, abs=0.03)


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


def _revcomp(seq):
    return seq[::-1].translate(str.maketrans("ACGT", "TGCA"))


def _read_sam(path):
    header, reads = [], {}
    for line in path.read_text().splitlines():
        if line.startswith("@"):
            header.append(line)
            continue
        f = line.split("\t")
        tags = dict(t.split(":", 2)[::2] for t in f[11:])
        reads[f[0]] = {
            "flag": int(f[1]),
            "chrom": f[2],
            "pos": int(f[3]),
            "cigar": f[5],
            "seq": f[9],
            "mm": tags["MM"],
            "ml": tags["ML"],
        }
    return header, reads


def _decode(read):
    seq = read["seq"]
    rev = read["flag"] & 16
    read_seq = _revcomp(seq) if rev else seq
    ml_all = [int(v) for v in read["ml"].split(",")[1:]]
    entries, used = [], 0
    for entry in read["mm"].rstrip(";").split(";"):
        mod, *skips = entry.split(",")
        cand = [i for i, b in enumerate(read_seq) if b == mod[0]]
        k, idx = -1, []
        for s in skips:
            k += int(s) + 1
            idx.append(cand[k])
        ref = [len(seq) - 1 - i if rev else i for i in idx]
        entries.append((mod, ref, ml_all[used : used + len(idx)]))
        used += len(idx)
    assert used == len(ml_all)
    return entries


class TestWriteSam:
    def _calls(self, mtase="A", nmol=6, seed=0):
        seq = synthgen.random_sequence(1500, seed=seed)
        nucs = synthgen.place_nucleosomes(1500, nmol, seed=seed)
        calls = synthgen.simulate_calls(seq, nucs, mtase=mtase, seed=seed)
        return seq, calls

    def test_header(self, tmp_path):
        seq, calls = self._calls()
        path = synthgen.write_sam(calls, {"chr1": seq}, tmp_path / "x.sam")
        header, _ = _read_sam(path)
        assert header[0].startswith("@HD")
        assert "@SQ\tSN:chr1\tLN:1500" in header

    def test_one_read_per_molecule(self, tmp_path):
        seq, calls = self._calls()
        path = synthgen.write_sam(calls, {"chr1": seq}, tmp_path / "x.sam")
        _, reads = _read_sam(path)
        assert set(reads) == set(calls["mol_id"])
        for mol_id, read in reads.items():
            strand = calls.loc[calls["mol_id"] == mol_id, "strand"].iloc[0]
            assert read["flag"] == (0 if strand == "+" else 16)
            assert read["chrom"] == "chr1"
            assert read["pos"] == 1
            assert read["cigar"] == "1500M"
            assert read["seq"] == seq

    @pytest.mark.parametrize(
        "mtase, mod", [("A", "A+a?"), ("CG", "C+m?"), ("GC", "C+m?")]
    )
    def test_mm_ml_roundtrip(self, tmp_path, mtase, mod):
        seq, calls = self._calls(mtase=mtase)
        path = synthgen.write_sam(
            calls, {"chr1": seq}, tmp_path / "x.sam", mtase=mtase
        )
        _, reads = _read_sam(path)
        for mol_id, grp in calls.groupby("mol_id"):
            [(got_mod, ref, ml)] = _decode(reads[mol_id])
            assert got_mod == mod
            expected = dict(zip(grp["pos"], grp["mod_qual"]))
            assert sorted(ref) == sorted(expected)
            want = [min(int(expected[p] * 256), 255) for p in ref]
            assert ml == want

    def test_sparse_calls_have_skips(self, tmp_path):
        seq, calls = self._calls(mtase="A")
        rng = np.random.default_rng(0)
        sub = calls[rng.random(len(calls)) < 0.3].reset_index(drop=True)
        path = synthgen.write_sam(sub, {"chr1": seq}, tmp_path / "x.sam")
        _, reads = _read_sam(path)
        assert any(
            int(s) > 0
            for r in reads.values()
            for s in r["mm"].rstrip(";").split(",")[1:]
        )
        for mol_id, grp in sub.groupby("mol_id"):
            [(_, ref, ml)] = _decode(reads[mol_id])
            assert sorted(ref) == sorted(grp["pos"])
            assert len(ml) == len(grp)

    @pytest.mark.parametrize(
        "mtase, plus_mod, minus_mod",
        [
            ("A", "A+a?", "T-a?"),
            ("CG", "C+m?", "G-m?"),
            ("GC", "C+m?", "G-m?"),
        ],
    )
    def test_double_strand_roundtrip(
        self, tmp_path, mtase, plus_mod, minus_mod
    ):
        seq = synthgen.random_sequence(1500, seed=0)
        nucs = synthgen.place_nucleosomes(1500, 4, seed=0)
        calls = synthgen.simulate_calls(
            seq, nucs, mtase=mtase, double_strand=True, seed=0
        )
        path = synthgen.write_sam(
            calls, {"chr1": seq}, tmp_path / "x.sam", mtase=mtase
        )
        _, reads = _read_sam(path)
        for mol_id, grp in calls.groupby("mol_id"):
            read = reads[mol_id]
            assert read["flag"] == 0
            decoded = {mod: (ref, ml) for mod, ref, ml in _decode(read)}
            assert set(decoded) == {plus_mod, minus_mod}
            for mod, strand in ((plus_mod, "+"), (minus_mod, "-")):
                sub = grp[grp["strand"] == strand]
                expected = dict(zip(sub["pos"], sub["mod_qual"]))
                ref, ml = decoded[mod]
                assert sorted(ref) == sorted(expected)
                want = [min(int(expected[p] * 256), 255) for p in ref]
                assert ml == want

    def test_invalid_mtase(self, tmp_path):
        seq, calls = self._calls()
        with pytest.raises(ValueError):
            synthgen.write_sam(
                calls, {"chr1": seq}, tmp_path / "x.sam", mtase="TA"
            )


class TestMakeDataset:
    def test_files_written(self, tmp_path):
        paths = synthgen.make_dataset(
            tmp_path / "out", prefix="p", nbp=1000, nmol=5, seed=0
        )
        assert set(paths) == {"chromsize", "fasta", "modkit", "sam", "truth"}
        for path in paths.values():
            assert path.exists()
            assert path.name.startswith("p.")

    def test_consistent_outputs(self, tmp_path):
        paths = synthgen.make_dataset(
            tmp_path, nbp=1000, nmol=5, chrom="seg", seed=0
        )
        tsv = pd.read_csv(paths["modkit"], sep="\t")
        _, reads = _read_sam(paths["sam"])
        truth = pd.read_csv(
            paths["truth"], sep="\t", header=None, comment="#"
        )
        fasta = paths["fasta"].read_text().splitlines()
        assert paths["chromsize"].read_text() == "chrom\tlength\nseg\t1000\n"
        assert fasta[0] == ">seg"
        assert len("".join(fasta[1:])) == 1000
        assert set(reads) == set(tsv["read_id"])
        assert set(truth[3]) <= set(tsv["read_id"])
        assert (tsv["chrom"] == "seg").all()

    def test_user_nucs(self, tmp_path):
        nucs = pd.DataFrame(
            {"mol_id": ["a", "b"], "start": [100, 300], "end": [247, 447]}
        )
        paths = synthgen.make_dataset(tmp_path, nbp=600, nucs=nucs, seed=0)
        truth = pd.read_csv(
            paths["truth"], sep="\t", header=None, comment="#"
        )
        assert truth.values.tolist() == [
            ["chr1", 100, 247, "a"],
            ["chr1", 300, 447, "b"],
        ]
        tsv = pd.read_csv(paths["modkit"], sep="\t")
        assert set(tsv["read_id"]) == {"a", "b"}

    def test_controls(self, tmp_path):
        paths = synthgen.make_dataset(
            tmp_path, nbp=5000, nmol=20, controls=True, seed=0
        )
        keys = {"meth_modkit", "meth_sam", "unmeth_modkit", "unmeth_sam"}
        assert keys <= set(paths)
        err = 0.5 * math.erfc(2.0 / math.sqrt(2))
        for name, rate in (("meth", 0.7), ("unmeth", 0.0)):
            tsv = pd.read_csv(paths[f"{name}_modkit"], sep="\t")
            assert set(tsv["read_id"]) == {f"{name}{i}" for i in range(20)}
            _, reads = _read_sam(paths[f"{name}_sam"])
            assert set(reads) == set(tsv["read_id"])
            called = (tsv["mod_qual"] >= 0.5).mean()
            expected = rate * (1 - err) + (1 - rate) * err
            assert called == pytest.approx(expected, abs=0.02)

    def test_no_controls_by_default(self, tmp_path):
        paths = synthgen.make_dataset(tmp_path, nbp=800, nmol=3, seed=0)
        assert not any(key.startswith(("meth", "unmeth")) for key in paths)

    def test_double_strand(self, tmp_path):
        paths = synthgen.make_dataset(
            tmp_path, nbp=800, nmol=3, double_strand=True, seed=0
        )
        tsv = pd.read_csv(paths["modkit"], sep="\t")
        assert (tsv.groupby("read_id")["ref_strand"].nunique() == 2).all()

    def test_seed_reproducible(self, tmp_path):
        a = synthgen.make_dataset(tmp_path / "a", nbp=800, nmol=3, seed=1)
        b = synthgen.make_dataset(tmp_path / "b", nbp=800, nmol=3, seed=1)
        for key in a:
            assert a[key].read_text() == b[key].read_text()


class TestCli:
    def test_writes_files(self, tmp_path):
        result = runner.invoke(
            synthgen.app,
            [str(tmp_path), "--nbp", "800", "--nmol", "3", "--seed", "0"],
        )
        assert result.exit_code == 0, result.output
        for ext in ("size", "fa", "tsv", "sam", "truth.bed"):
            assert (tmp_path / f"synth.{ext}").exists()

    def test_lnuc_range_option(self, tmp_path):
        result = runner.invoke(
            synthgen.app,
            [
                str(tmp_path),
                "--nbp",
                "3000",
                "--nmol",
                "3",
                "--lnuc-range",
                "120",
                "160",
            ],
        )
        assert result.exit_code == 0, result.output
        truth = pd.read_csv(
            tmp_path / "synth.truth.bed", sep="\t", header=None, comment="#"
        )
        lengths = truth[2] - truth[1]
        assert lengths.between(120, 160).all()
        assert lengths.nunique() > 1

    def test_controls_and_double_strand_flags(self, tmp_path):
        result = runner.invoke(
            synthgen.app,
            [
                str(tmp_path),
                "--nbp",
                "800",
                "--nmol",
                "3",
                "--controls",
                "--double-strand",
            ],
        )
        assert result.exit_code == 0, result.output
        for name in ("meth", "unmeth"):
            for ext in ("tsv", "sam"):
                assert (tmp_path / f"synth.{name}.{ext}").exists()

    def test_nucs_bed(self, tmp_path):
        bed = tmp_path / "in.bed"
        bed.write_text("chr1\t50\t197\tm1\n")
        result = runner.invoke(
            synthgen.app,
            [str(tmp_path / "out"), "--nbp", "400", "--nucs", str(bed)],
        )
        assert result.exit_code == 0, result.output
        truth = (tmp_path / "out" / "synth.truth.bed").read_text()
        assert truth == "#chrom\tstart\tend\tmol_id\nchr1\t50\t197\tm1\n"

    def test_nucs_bed_with_header(self, tmp_path):
        bed = tmp_path / "in.bed"
        bed.write_text("#chrom\tstart\tend\tmol_id\nchr1\t50\t197\tm1\n")
        result = runner.invoke(
            synthgen.app,
            [str(tmp_path / "out"), "--nbp", "400", "--nucs", str(bed)],
        )
        assert result.exit_code == 0, result.output
        truth = (tmp_path / "out" / "synth.truth.bed").read_text()
        assert truth.splitlines()[1:] == ["chr1\t50\t197\tm1"]
