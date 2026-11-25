# test_filter_dropout.py

import numpy as np
import pytest
from catella.experiment.methdata import MethPrintExperiment


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write(
            "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"
        )
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)


def _write_chromsize(path, sizes):
    with open(path, "w") as f:
        f.write("chrom\tlength\n")
        f.writelines(f"{chrom}\t{length}\n" for chrom, length in sizes.items())


def _write_fasta(path, records):
    with open(path, "w") as f:
        f.writelines(f">{region}\n{seq}\n" for region, seq in records.items())


def _load(
    tmp_path,
    refseq,
    rows,
    *,
    mtase,
    region="chr1",
    ignore_strand=False,
    wrap=False,
):
    chromsize = tmp_path / "sizes.tsv"
    _write_chromsize(chromsize, {region: len(refseq)})
    fasta = tmp_path / "ref.fa"
    _write_fasta(fasta, {region: refseq})
    test_file = tmp_path / "test.tsv"
    _write_tsv(test_file, rows)
    return MethPrintExperiment.load_raw(
        chromsize=chromsize,
        test_file=test_file,
        fasta_file=fasta,
        mtase=mtase,
        ignore_strand=ignore_strand,
        wrap=wrap,
    )


class TestValidation:
    def test_missing_mtase_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase=None
        )
        with pytest.raises(ValueError):
            exp.filter_dropout()
        exp.close()

    def test_missing_refseq_raises(self, tmp_path):
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 2})
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, [("m0", 0, "chr1", "+", 0.5, "a")])
        exp = MethPrintExperiment.load_raw(
            chromsize=chromsize, test_file=test_file, mtase="A"
        )
        with pytest.raises(ValueError):
            exp.filter_dropout()
        exp.close()

    def test_bad_thres_max_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        with pytest.raises(ValueError):
            exp.filter_dropout(thres_max=1.5)
        exp.close()

    def test_bad_thres_min_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        with pytest.raises(ValueError):
            exp.filter_dropout(thres_min=-0.1)
        exp.close()

    def test_thres_min_above_thres_max_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        with pytest.raises(ValueError):
            exp.filter_dropout(thres_min=0.5, thres_max=0.2)
        exp.close()

    def test_bad_unmapped_strand_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        with pytest.raises(ValueError):
            exp.filter_dropout(unmapped_strand="bogus")
        exp.close()

    def test_bad_method_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        with pytest.raises(ValueError):
            exp.filter_dropout(method="bogus")
        exp.close()

    def test_missing_channel_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
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
        exp.filter_dropout(which="test", thres_max=0.2)
        mol_id = exp.raw["chr1"].test.mol_id
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
            ("plus_at_0", 0, "chr1", "+", 0.5, "m"),
            ("plus_at_1", 1, "chr1", "+", 0.5, "m"),
            ("minus_at_1", 1, "chr1", "-", 0.5, "m"),
            ("minus_at_0", 0, "chr1", "-", 0.5, "m"),
        ]
        exp = _load(tmp_path, "CG", rows, mtase="CG")
        exp.filter_dropout(which="test", thres_max=0.2)
        mol_id = exp.raw["chr1"].test.mol_id
        keep = dict(
            zip(mol_id, exp.analysis["chr1"]["test_dropout_mask"]["keep"])
        )
        assert keep["plus_at_0"] == True
        assert keep["plus_at_1"] == False
        assert keep["minus_at_1"] == True
        assert keep["minus_at_0"] == False
        exp.close()

    # refseq "GC": GC+ -> idx1 (the C); GC- -> idx0.
    def test_gc_strand_offset(self, tmp_path):
        rows = [
            ("plus_at_1", 1, "chr1", "+", 0.5, "m"),
            ("plus_at_0", 0, "chr1", "+", 0.5, "m"),
            ("minus_at_0", 0, "chr1", "-", 0.5, "m"),
            ("minus_at_1", 1, "chr1", "-", 0.5, "m"),
        ]
        exp = _load(tmp_path, "GC", rows, mtase="GC")
        exp.filter_dropout(which="test", thres_max=0.2)
        mol_id = exp.raw["chr1"].test.mol_id
        keep = dict(
            zip(mol_id, exp.analysis["chr1"]["test_dropout_mask"]["keep"])
        )
        assert keep["plus_at_1"] == True
        assert keep["plus_at_0"] == False
        assert keep["minus_at_0"] == True
        assert keep["minus_at_1"] == False
        exp.close()


class TestMultiLabel:
    # refseq "AGCGA": GC match at (1,2), CG match at (2,3), sharing idx2.
    def test_gcg_shared_measurement_counts_for_both_labels(self, tmp_path):
        exp = _load(
            tmp_path,
            "AGCGA",
            [("m0", 2, "chr1", "+", 0.5, "m")],
            mtase=["CG", "GC"],
        )
        exp.filter_dropout(which="test", thres_max=0.2, method="separate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert keep == [True]
        exp.close()

    def test_separate_requires_every_label(self, tmp_path):
        # 10 A-sites (idx0-9), 1 CG-site (idx10='C' of "CG" at end).
        refseq = "AAAAAAAAAACG"
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(9)]
        exp = _load(tmp_path, refseq, rows, mtase=["A", "CG"])
        exp.filter_dropout(which="test", thres_max=0.2, method="separate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        # A: 9/10 covered (dropout 0.1, passes). CG: 0/1 covered
        # (dropout 1.0, fails) -> overall False under "separate".
        assert keep == [False]
        exp.close()

    def test_aggregate_pools_across_labels(self, tmp_path):
        refseq = "AAAAAAAAAACG"
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(9)]
        exp = _load(tmp_path, refseq, rows, mtase=["A", "CG"])
        exp.filter_dropout(which="test", thres_max=0.2, method="aggregate")
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
        exp.filter_dropout(which="test", thres_max=0.2, method="aggregate")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        # plain mean of fractions: (0 + 1.0)/2 = 0.5 -> would fail 0.2.
        # pooled: covered=10, total=11 -> dropout_frac=1/11=0.0909 <= 0.2
        assert keep == [True]
        exp.close()


class TestThresholdRange:
    # refseq "N" + "A"*8: pos0='N' is not methylatable (registers a
    # molecule without covering any A-site); pos1-8 are the 8 A-sites
    # used to vary each molecule's dropout fraction. 8 sites (a power
    # of two) keeps dropout fractions exact in binary floating point.
    def _make(self, tmp_path):
        refseq = "N" + "A" * 8
        rows = (
            [("mol_none", 0, "chr1", "+", 0.5, "a")]
            + [("mol_mid", i, "chr1", "+", 0.5, "a") for i in range(5)]
            + [("mol_high", i, "chr1", "+", 0.5, "a") for i in range(8)]
            + [("mol_full", i, "chr1", "+", 0.5, "a") for i in range(9)]
        )
        return _load(tmp_path, refseq, rows, mtase="A")

    def test_range_keeps_only_middle_dropout(self, tmp_path):
        exp = self._make(tmp_path)
        # dropout fractions: mol_none=1.0, mol_mid=0.5, mol_high=0.125,
        # mol_full=0.0
        exp.filter_dropout(which="test", thres_min=0.2, thres_max=0.8)
        mol_id = exp.raw["chr1"].test.mol_id
        keep = dict(
            zip(mol_id, exp.analysis["chr1"]["test_dropout_mask"]["keep"])
        )
        assert keep["mol_none"] == False
        assert keep["mol_mid"] == True
        assert keep["mol_high"] == False
        assert keep["mol_full"] == False
        exp.close()

    def test_range_bounds_are_inclusive(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(which="test", thres_min=0.125, thres_max=0.5)
        mol_id = exp.raw["chr1"].test.mol_id
        keep = dict(
            zip(mol_id, exp.analysis["chr1"]["test_dropout_mask"]["keep"])
        )
        assert keep["mol_mid"] == True
        assert keep["mol_high"] == True
        assert keep["mol_none"] == False
        assert keep["mol_full"] == False
        exp.close()


class TestUnmappedStrand:
    # refseq "AT": A+ mask=idx0, A- mask=idx1. molecule on '.' strand
    # with signal at idx1 (a T).
    @pytest.mark.parametrize(
        "mode,expected",
        [
            ("union", False),  # union total=2, covered=1 -> 0.5 > 0.2
            ("drop", False),  # covered forced 0 -> 1.0 > 0.2
            ("+", False),  # + mask doesn't include idx1 -> 1.0 > 0.2
            ("-", True),  # - mask includes idx1 -> 0.0 <= 0.2
        ],
    )
    def test_modes(self, tmp_path, mode, expected):
        exp = _load(
            tmp_path, "AT", [("m0", 1, "chr1", ".", 0.5, "a")], mtase="A"
        )
        exp.filter_dropout(which="test", thres_max=0.2, unmapped_strand=mode)
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert keep == [expected]
        exp.close()


class TestIgnoreStrand:
    # refseq "AT": A+ mask=idx0, A- mask=idx1. molecule recorded on
    # '+' strand with signal only at idx1 (a T, valid on '-').
    def test_default_uses_recorded_strand(self, tmp_path):
        exp = _load(
            tmp_path, "AT", [("m0", 1, "chr1", "+", 0.5, "a")], mtase="A"
        )
        exp.filter_dropout(which="test", thres_max=0.2)
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert keep == [False]  # + mask excludes idx1 -> uncovered
        exp.close()

    def test_ignore_strand_lets_unmapped_strand_govern(self, tmp_path):
        exp = _load(
            tmp_path,
            "AT",
            [("m0", 1, "chr1", "+", 0.5, "a")],
            mtase="A",
            ignore_strand=True,
        )
        exp.filter_dropout(which="test", thres_max=0.2, unmapped_strand="-")
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert keep == [True]  # '-' mask includes idx1 -> covered
        exp.close()

    def test_dropout_fractions_matches_filter_dropout(self, tmp_path):
        exp = _load(
            tmp_path,
            "AT",
            [("m0", 1, "chr1", "+", 0.5, "a")],
            mtase="A",
            ignore_strand=True,
        )
        fracs = exp.dropout_fractions(unmapped_strand="-")
        assert fracs[("chr1", "test", "A")] == pytest.approx([0.0])
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
            chromsize=chromsize,
            test_file=test_file,
            meth_file=meth_file,
            fasta_file=fasta,
            mtase="A",
        )
        exp.filter_dropout(which=None, thres_max=0.2)
        assert "test_dropout_mask" in exp.analysis["chr1"]
        assert "meth_dropout_mask" in exp.analysis["chr1"]
        assert "unmeth_dropout_mask" not in exp.analysis["chr1"]
        exp.close()

    def test_custom_mask_name(self, tmp_path):
        exp = _load(
            tmp_path, "AT", [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        exp.filter_dropout(which="test", thres_max=0.2, mask_name="qc")
        assert "test_qc" in exp.analysis["chr1"]
        assert "test_dropout_mask" not in exp.analysis["chr1"]
        exp.close()


class TestSaveLoadRoundTrip:
    def test_mask_survives_save_and_load(self, tmp_path):
        # Regression test: the mask's 'keep' column must round-trip
        # through save()/load() with correct bool semantics. A bool
        # column stored via h5_utils comes back as literal "True"/
        # "False" text (not bool), and casting that text to bool makes
        # every entry truthy -- silently turning the mask into a
        # no-op. filter_dropout stores 'keep' as int specifically to
        # avoid this.
        rows = [
            ("plus_at_A", 0, "chr1", "+", 0.5, "a"),
            ("plus_at_T", 1, "chr1", "+", 0.5, "a"),
        ]
        exp = _load(tmp_path, "AT", rows, mtase="A")
        exp.filter_dropout(which="test", thres_max=0.2)
        before = exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()

        exp_file = tmp_path / "exp.h5"
        exp.save(exp_file)
        exp.close()

        exp2 = MethPrintExperiment.load(exp_file)
        after = exp2.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()
        assert [bool(v) for v in after] == [bool(v) for v in before]
        assert [bool(v) for v in after] == [True, False]

        dense = exp2.to_dense(
            "chr1", which="test", as_h5array=False, mask_name="dropout_mask"
        )
        assert not np.isnan(dense.loc[0]).all()
        assert np.isnan(dense.loc[1]).all()
        exp2.close()


class TestChannelSubset:
    # refseq "AGCG": A-site at idx0; CG match at (2,3), C at idx2.
    # m0 covers only the A-site.
    def _make(self, tmp_path):
        return _load(
            tmp_path,
            "AGCG",
            [("m0", 0, "chr1", "+", 0.5, "a")],
            mtase=["A", "CG"],
        )

    def test_unassayed_channel_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.filter_dropout(channels=["GC"])
        exp.close()

    def test_unknown_channel_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.filter_dropout(channels=["bogus"])
        exp.close()

    def test_subset_narrows_evaluated_channels(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(thres_max=0.0, mask_name="full")
        assert exp.analysis["chr1"]["test_full"]["keep"].tolist() == [False]
        exp.filter_dropout(channels=["A"], thres_max=0.0, mask_name="a_only")
        assert exp.analysis["chr1"]["test_a_only"]["keep"].tolist() == [True]
        exp.close()

    def test_default_none_is_one_channel_per_mtase(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(
            channels=["A", "CG"], thres_max=0.0, mask_name="explicit"
        )
        exp.filter_dropout(thres_max=0.0, mask_name="implicit")
        explicit = exp.analysis["chr1"]["test_explicit"]["keep"].tolist()
        implicit = exp.analysis["chr1"]["test_implicit"]["keep"].tolist()
        assert explicit == implicit == [False]
        exp.close()


class TestRegionsSubset:
    # chr1 refseq "AAAA" (4 A-sites, fully covered by m0); chr2 has no
    # refseq at all, so touching it raises -- proves a restricted
    # `regions` never evaluates the excluded region.
    def _make(self, tmp_path):
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(4)] + [
            ("m0", 0, "chr2", "+", 0.5, "a")
        ]
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, rows)
        chromsize = tmp_path / "sizes.tsv"
        _write_chromsize(chromsize, {"chr1": 4, "chr2": 4})
        fasta = tmp_path / "ref.fa"
        _write_fasta(fasta, {"chr1": "AAAA"})
        return MethPrintExperiment.load_raw(
            chromsize=chromsize,
            test_file=test_file,
            fasta_file=fasta,
            mtase="A",
        )

    def test_unknown_region_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.filter_dropout(regions=["bogus"])
        exp.close()

    def test_default_none_touches_every_region_and_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.filter_dropout()
        exp.close()

    def test_filter_dropout_restricts_to_given_regions(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(regions=["chr1"], thres_max=0.0)
        assert "test_dropout_mask" in exp.analysis["chr1"]
        assert "test_dropout_mask" not in exp.analysis["chr2"]
        exp.close()

    def test_dropout_fractions_restricts_to_given_regions(self, tmp_path):
        exp = self._make(tmp_path)
        fractions = exp.dropout_fractions(regions=["chr1"])
        assert {key[0] for key in fractions} == {"chr1"}
        exp.close()


class TestDropoutFractions:
    # refseq "AAAAAAAAAACG": 10 A-sites (idx0-9), 1 CG-site (idx10=C).
    # m0 covers 9/10 A-sites (idx0-8) and 0/1 CG-site; m1 covers none.
    def _make(self, tmp_path):
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(9)]
        return _load(tmp_path, "AAAAAAAAAACG", rows, mtase=["A", "CG"])

    def test_keys_and_values(self, tmp_path):
        exp = self._make(tmp_path)
        fracs = exp.dropout_fractions()
        assert set(fracs) == {
            ("chr1", "test", "A"),
            ("chr1", "test", "CG"),
            ("chr1", "test", "all"),
        }
        assert fracs[("chr1", "test", "A")] == pytest.approx([0.1])
        assert fracs[("chr1", "test", "CG")] == pytest.approx([1.0])
        assert fracs[("chr1", "test", "all")] == pytest.approx(
            [1.0 - 9 / 11]
        )
        exp.close()

    def test_channel_subset_no_all_key(self, tmp_path):
        exp = self._make(tmp_path)
        fracs = exp.dropout_fractions(channels=["A"])
        assert set(fracs) == {("chr1", "test", "A")}
        exp.close()

    def test_unassayed_channel_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.dropout_fractions(channels=["GC"])
        exp.close()

    def test_both_c_mtases_default_unmerged(self, tmp_path):
        # "AGCAACGAGCG", + strand: A at 0,3,4,7; GCH at 2; HCG at 5;
        # GCG at 9. m0 covers 0, 2, 9.
        rows = [
            ("m0", 0, "chr1", "+", 0.5, "a"),
            ("m0", 2, "chr1", "+", 0.5, "m"),
            ("m0", 9, "chr1", "+", 0.5, "m"),
        ]
        exp = _load(
            tmp_path, "AGCAACGAGCG", rows, mtase=["A", "CG", "GC"]
        )
        fracs = exp.dropout_fractions()
        assert {key[2] for key in fracs} == {
            "A", "GCH", "HCG", "GCG", "all"
        }
        assert fracs[("chr1", "test", "A")] == pytest.approx([0.75])
        assert fracs[("chr1", "test", "GCH")] == pytest.approx([0.0])
        assert fracs[("chr1", "test", "HCG")] == pytest.approx([1.0])
        assert fracs[("chr1", "test", "GCG")] == pytest.approx([0.0])
        assert fracs[("chr1", "test", "all")] == pytest.approx(
            [1.0 - 3 / 7]
        )
        exp.close()

    def test_no_mtase_raises(self, tmp_path):
        exp = _load(
            tmp_path, "AAAA", [("m0", 0, "chr1", "+", 0.5, "a")], mtase=None
        )
        with pytest.raises(ValueError):
            exp.dropout_fractions()
        exp.close()

    def test_missing_source_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.dropout_fractions(which="meth")
        exp.close()


class TestWrap:
    def _make(self, tmp_path):
        rows = [("m0", i, "chr1", "+", 0.5, "a") for i in range(5, 10)]
        return _load(tmp_path, "A" * 10, rows, mtase="A", wrap=True)

    def test_dropout_fractions_on_folded_sites(self, tmp_path):
        exp = self._make(tmp_path)
        fracs = exp.dropout_fractions()
        assert fracs[("chr1", "test", "A")] == pytest.approx([0.0])
        exp.close()

    def test_filter_dropout_keeps_fully_covered(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(thres_max=0.1)
        keep = exp.analysis["chr1"]["test_dropout_mask"]["keep"]
        assert keep.tolist() == [1]
        exp.close()

    def test_bad_unmapped_strand_raises(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(ValueError):
            exp.dropout_fractions(unmapped_strand="bogus")
        exp.close()


class TestIsKept:
    def _make(self, tmp_path):
        return TestThresholdRange()._make(tmp_path)

    def test_matches_stored_flags(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(thres_min=0.2, thres_max=0.8)
        got = exp.is_kept("chr1")
        assert got.dtype == bool
        stored = exp.analysis["chr1"]["test_dropout_mask"]["keep"]
        np.testing.assert_array_equal(got, stored.to_numpy().astype(bool))
        mol_id = list(exp.raw["chr1"].test.mol_id)
        assert got[mol_id.index("mol_mid")]
        assert not got[mol_id.index("mol_full")]
        exp.close()

    def test_custom_mask_name(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout(thres_max=0.2, mask_name="strict")
        exp.filter_dropout(thres_max=1.0)
        strict = exp.is_kept("chr1", mask_name="strict")
        loose = exp.is_kept("chr1")
        assert strict.sum() < loose.sum()
        exp.close()

    def test_none_keeps_every_molecule(self, tmp_path):
        exp = self._make(tmp_path)
        got = exp.is_kept("chr1", mask_name=None)
        assert got.dtype == bool
        assert got.tolist() == [True] * len(exp.raw["chr1"].test.mol_id)
        exp.close()

    def test_missing_flags_raise(self, tmp_path):
        exp = self._make(tmp_path)
        with pytest.raises(KeyError, match="run filter_dropout"):
            exp.is_kept("chr1")
        exp.close()

    def test_missing_source_flags_raise(self, tmp_path):
        exp = self._make(tmp_path)
        exp.filter_dropout()
        with pytest.raises(KeyError, match="meth_dropout_mask"):
            exp.is_kept("chr1", which="meth")
        exp.close()


def _partial_rows():
    return [("m0", p, "chr1", "+", 0.5, "a") for p in range(10)] + [
        ("m1", p, "chr1", "+", 0.5, "a") for p in range(20)
    ]


class TestSpan:
    def _keep(self, exp):
        return exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist()

    def test_full_penalizes_partial_read(self, tmp_path):
        exp = _load(tmp_path, "A" * 20, _partial_rows(), mtase="A")
        exp.filter_dropout(thres_max=0.2)
        assert self._keep(exp) == [0, 1]
        exp.close()

    @pytest.mark.parametrize("span", ["read", "calls"])
    def test_partial_read_kept(self, tmp_path, span):
        exp = _load(tmp_path, "A" * 20, _partial_rows(), mtase="A")
        exp.filter_dropout(thres_max=0.2, span=span)
        assert self._keep(exp) == [1, 1]
        exp.close()

    def test_fractions_follow_span(self, tmp_path):
        exp = _load(tmp_path, "A" * 20, _partial_rows(), mtase="A")
        full = exp.dropout_fractions(which="test")[("chr1", "test", "A")]
        read = exp.dropout_fractions(which="test", span="read")
        assert full.tolist() == [0.5, 0.0]
        assert read[("chr1", "test", "A")].tolist() == [0.0, 0.0]
        exp.close()

    def test_invalid_span_raises(self, tmp_path):
        exp = _load(tmp_path, "A" * 20, _partial_rows(), mtase="A")
        with pytest.raises(ValueError, match="span must be one of"):
            exp.filter_dropout(span="bogus")
        exp.close()

    def test_read_without_extents_raises(self, tmp_path):
        exp = _load(
            tmp_path, "A" * 20, _partial_rows(), mtase="A", wrap=True
        )
        with pytest.raises(ValueError, match="requires read extents"):
            exp.filter_dropout(span="read")
        exp.close()


class TestNoSites:
    def test_channel_without_sites_is_nan(self, tmp_path):
        exp = _load(
            tmp_path,
            "AGCGA",
            [("m0", 2, "chr1", "+", 0.5, "m")],
            mtase=["CG", "GC"],
        )
        fracs = exp.dropout_fractions(which="test")
        assert np.isnan(fracs[("chr1", "test", "GCH")]).all()
        assert fracs[("chr1", "test", "GCG")].tolist() == [0.0]
        exp.close()

    def test_region_without_sites_warns_and_drops(self, tmp_path):
        exp = _load(
            tmp_path, "C" * 10, [("m0", 0, "chr1", "+", 0.5, "a")], mtase="A"
        )
        with pytest.warns(UserWarning, match="no sites of channel"):
            exp.filter_dropout()
        assert exp.analysis["chr1"]["test_dropout_mask"]["keep"].tolist() == [
            0
        ]
        exp.close()
