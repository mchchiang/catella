# test_filter_dropout.py

import numpy as np
import pytest

from nucmc.experiment.methdata import MethPrintExperiment


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write("read_id\tref_position\tchrom\tref_strand\tmod_qual\t"
                "mod_code\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        for chrom, length in sizes.items():
            f.write(f"{chrom}\t{length}\n")


def _write_fasta(path, records):
    with open(path, "w") as f:
        for chrom, seq in records.items():
            f.write(f">{chrom}\n{seq}\n")


def _load(tmp_path, refseq, rows, *, mtase, chrom="chr1"):
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {chrom: len(refseq)})
    fasta = tmp_path / "ref.fa"
    _write_fasta(fasta, {chrom: refseq})
    test_file = tmp_path / "test.tsv"
    _write_tsv(test_file, rows)
    return MethPrintExperiment.load_raw(
        chromsize=chromsize, test_file=test_file, fasta_file=fasta,
        mtase=mtase)


class TestValidation:
    def test_missing_mtase_raises(self, tmp_path):
        exp = _load(tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")],
                   mtase=None)
        with pytest.raises(ValueError):
            exp.filter_dropout()
        exp.close()

    def test_missing_refseq_raises(self, tmp_path):
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 2})
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, [("m0", 0, "chr1", "+", 0.5, "a")])
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase="A")
        with pytest.raises(ValueError):
            exp.filter_dropout()
        exp.close()

    def test_bad_threshold_raises(self, tmp_path):
        exp = _load(tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")],
                   mtase="A")
        with pytest.raises(ValueError):
            exp.filter_dropout(threshold=1.5)
        exp.close()

    def test_bad_unmapped_strand_raises(self, tmp_path):
        exp = _load(tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")],
                   mtase="A")
        with pytest.raises(ValueError):
            exp.filter_dropout(unmapped_strand="bogus")
        exp.close()

    def test_bad_method_raises(self, tmp_path):
        exp = _load(tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")],
                   mtase="A")
        with pytest.raises(ValueError):
            exp.filter_dropout(method="bogus")
        exp.close()

    def test_missing_channel_raises(self, tmp_path):
        exp = _load(tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")],
                   mtase="A")
        with pytest.raises(ValueError):
            exp.filter_dropout(which="meth")
        exp.close()


class TestAContext:
    # refseq "AT": pos0=A, pos1=T.
    def test_strand_aware_coverage(self, tmp_path):
        rows = [
            ("plus_at_A", 0, "chr1", "+", 0.5, "a"),
            ("plus_at_T", 1, "chr1", "+", 0.5, "a"),
            ("minus_at_T", 1, "chr1", "-", 0.5, "a"),
            ("minus_at_A", 0, "chr1", "-", 0.5, "a"),
        ]
        exp = _load(tmp_path, "AT", rows, mtase="A")
        exp.filter_dropout(which="test", threshold=0.2)
        mol_id = exp.raw["chr1"].test_mol_id
        mask = exp.analysis["chr1"]["test_dropout_mask"]
        keep = dict(zip(mol_id, mask["keep"]))
        assert keep["plus_at_A"] == True
        assert keep["plus_at_T"] == False
        assert keep["minus_at_T"] == True
        assert keep["minus_at_A"] == False
        exp.close()


class TestDinucleotideContext:
    # refseq "CG": CG+ -> idx0 (the C); CG- -> idx1.
    def test_cg_strand_offset(self, tmp_path):
        rows = [
            ("plus_at_0", 0, "chr1", "+", 0.5, "a"),
            ("plus_at_1", 1, "chr1", "+", 0.5, "a"),
            ("minus_at_1", 1, "chr1", "-", 0.5, "a"),
            ("minus_at_0", 0, "chr1", "-", 0.5, "a"),
        ]
        exp = _load(tmp_path, "CG", rows, mtase="CG")
        exp.filter_dropout(which="test", threshold=0.2)
        mol_id = exp.raw["chr1"].test_mol_id
        keep = dict(zip(mol_id,
                        exp.analysis["chr1"]["test_dropout_mask"]["keep"]))
        assert keep["plus_at_0"] == True
        assert keep["plus_at_1"] == False
        assert keep["minus_at_1"] == True
        assert keep["minus_at_0"] == False
        exp.close()

    # refseq "GC": GC+ -> idx1 (the C); GC- -> idx0.
    def test_gc_strand_offset(self, tmp_path):
        rows = [
            ("plus_at_1", 1, "chr1", "+", 0.5, "a"),
            ("plus_at_0", 0, "chr1", "+", 0.5, "a"),
            ("minus_at_0", 0, "chr1", "-", 0.5, "a"),
            ("minus_at_1", 1, "chr1", "-", 0.5, "a"),
        ]
        exp = _load(tmp_path, "GC", rows, mtase="GC")
        exp.filter_dropout(which="test", threshold=0.2)
        mol_id = exp.raw["chr1"].test_mol_id
        keep = dict(zip(mol_id,
                        exp.analysis["chr1"]["test_dropout_mask"]["keep"]))
        assert keep["plus_at_1"] == True
        assert keep["plus_at_0"] == False
        assert keep["minus_at_0"] == True
        assert keep["minus_at_1"] == False
        exp.close()


class TestMultiLabel:
    # refseq "AGCGA": GC match at (1,2), CG match at (2,3), sharing idx2.
    def test_gcg_shared_measurement_counts_for_both_labels(self, tmp_path):
        exp = _load(tmp_path, "AGCGA",
                   [("m0", 2, "chr1", "+", 0.5, "a")], mtase=["CG", "GC"])
        exp.filter_dropout(which="test", threshold=0.2, method="separate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert keep == [True]
        exp.close()

    def test_separate_requires_every_label(self, tmp_path):
        # 10 A-sites (idx0-9), 1 CG-site (idx10='C' of "CG" at end).
        refseq = "AAAAAAAAAACG"
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(9)]
        exp = _load(tmp_path, refseq, rows, mtase=["A", "CG"])
        exp.filter_dropout(which="test", threshold=0.2, method="separate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        # A: 9/10 covered (dropout 0.1, passes). CG: 0/1 covered
        # (dropout 1.0, fails) -> overall False under "separate".
        assert keep == [False]
        exp.close()

    def test_aggregate_pools_across_labels(self, tmp_path):
        refseq = "AAAAAAAAAACG"
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(9)]
        exp = _load(tmp_path, refseq, rows, mtase=["A", "CG"])
        exp.filter_dropout(which="test", threshold=0.2, method="aggregate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        # pooled: covered=9, total=11 -> dropout_frac = 2/11 = 0.182 <= 0.2
        assert keep == [True]
        exp.close()

    def test_aggregate_is_not_a_plain_mean(self, tmp_path):
        # A label has many more sites than CG, so aggregate should be
        # dominated by A's coverage, not a 50/50 average of the two
        # labels' fractions.
        refseq = "AAAAAAAAAACG"  # 10 A-sites, 1 CG-site
        # Cover ALL 10 A-sites but 0/1 CG-sites.
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(10)]
        exp = _load(tmp_path, refseq, rows, mtase=["A", "CG"])
        exp.filter_dropout(which="test", threshold=0.2, method="aggregate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        # plain mean of fractions: (0 + 1.0)/2 = 0.5 -> would fail 0.2.
        # pooled: covered=10, total=11 -> dropout_frac=1/11=0.0909 <= 0.2
        assert keep == [True]
        exp.close()


class TestUnmappedStrand:
    # refseq "AT": A+ mask=idx0, A- mask=idx1. molecule on '.' strand
    # with signal at idx1 (a T).
    @pytest.mark.parametrize("mode,expected", [
        ("union", False),   # union total=2, covered=1 -> 0.5 > 0.2
        ("drop", False),    # covered forced 0 -> 1.0 > 0.2
        ("+", False),       # + mask doesn't include idx1 -> 1.0 > 0.2
        ("-", True),        # - mask includes idx1 -> 0.0 <= 0.2
    ])
    def test_modes(self, tmp_path, mode, expected):
        exp = _load(tmp_path, "AT",
                   [("m0", 1, "chr1", ".", 0.5, "a")], mtase="A")
        exp.filter_dropout(which="test", threshold=0.2,
                          unmapped_strand=mode)
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert keep == [expected]
        exp.close()


class TestWhichAndMaskName:
    def test_which_none_covers_all_present_channels(self, tmp_path):
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 2})
        fasta = tmp_path / "ref.fa"
        _write_fasta(fasta, {"chr1": "AT"})
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, [("m0", 0, "chr1", "+", 0.5, "a")])
        meth_file = tmp_path / "meth.tsv"
        _write_tsv(meth_file, [("m0", 0, "chr1", "+", 0.9, "a")])

        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, meth_file=meth_file,
            fasta_file=fasta, mtase="A")
        exp.filter_dropout(which=None, threshold=0.2)
        assert "test_dropout_mask" in exp.analysis["chr1"]
        assert "meth_dropout_mask" in exp.analysis["chr1"]
        assert "unmeth_dropout_mask" not in exp.analysis["chr1"]
        exp.close()

    def test_custom_mask_name(self, tmp_path):
        exp = _load(tmp_path, "AT",
                   [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A")
        exp.filter_dropout(which="test", threshold=0.2, mask_name="qc")
        assert "test_qc" in exp.analysis["chr1"]
        assert "test_dropout_mask" not in exp.analysis["chr1"]
        exp.close()
