# test_qc.py

import numpy as np
import pytest
from catella.experiment.methdata import MethPrintExperiment

SEQ = "AAAAATTTTT"


def _write_tsv(path, rows):
    with open(path, "w") as f:
        f.write(
            "read_id\tref_position\tchrom\tref_strand\tmod_qual\tmod_code\n"
        )
        f.writelines("\t".join(str(x) for x in r) + "\n" for r in rows)


def _mol(name, positions, strand, q):
    return [(name, p, "chr1", strand, q, "a") for p in positions]


def _load(
    tmp_path,
    test,
    *,
    meth=None,
    unmeth=None,
    seq=SEQ,
    mtase="A",
    wrap=False,
    ignore_strand=False,
):
    sizes = tmp_path / "sizes.tsv"
    sizes.write_text(f"chrom\tlength\nchr1\t{len(seq)}\n")
    fasta = tmp_path / "ref.fa"
    fasta.write_text(f">chr1\n{seq}\n")
    files = {}
    for name, rows in (("test", test), ("meth", meth), ("unmeth", unmeth)):
        if rows is not None:
            files[f"{name}_file"] = tmp_path / f"{name}.tsv"
            _write_tsv(files[f"{name}_file"], rows)
    return MethPrintExperiment.load_raw(
        chromsize=sizes,
        fasta_file=fasta,
        mtase=mtase,
        wrap=wrap,
        ignore_strand=ignore_strand,
        **files,
    )


def _full(q_plus, q_minus):
    return _mol("p", range(5), "+", q_plus) + _mol(
        "m", range(5, 10), "-", q_minus
    )


@pytest.fixture
def exp(tmp_path):
    e = _load(
        tmp_path,
        _full(0.9, 0.1),
        meth=_full(0.9, 0.9),
        unmeth=_full(0.1, 0.1),
    )
    yield e
    e.close()


class TestColumns:
    def test_single_channel_has_no_all_row(self, exp):
        df = exp.qc()
        assert df["channel"].tolist() == ["A"]
        assert df["region"].tolist() == ["chr1"]

    def test_site_and_molecule_counts(self, exp):
        row = exp.qc().iloc[0]
        assert row["nsite_plus"] == 5
        assert row["nsite_minus"] == 5
        assert row["nmol"] == 2

    def test_strand_balance(self, exp):
        row = exp.qc().iloc[0]
        for src in ("test", "meth", "unmeth"):
            assert row[f"frac_plus_{src}"] == pytest.approx(0.5)
            assert row[f"frac_minus_{src}"] == pytest.approx(0.5)
            assert row[f"frac_unmapped_{src}"] == 0.0
            assert row[f"frac_masked_{src}"] == 0.0

    def test_rates(self, exp):
        row = exp.qc().iloc[0]
        assert row["meth_frac_test"] == pytest.approx(0.5)
        assert row["meth_frac_unmeth"] == pytest.approx(0.1)
        assert row["meth_frac_meth"] == pytest.approx(0.9)
        assert row["accessible_frac"] == pytest.approx(0.5)

    def test_full_coverage_has_zero_dropout(self, exp):
        row = exp.qc().iloc[0]
        for src in ("test", "meth", "unmeth"):
            assert row[f"dropout_{src}_p50"] == 0.0


class TestDropout:
    def test_percentile_columns(self, tmp_path):
        test = _mol("p", range(2), "+", 0.5) + _mol(
            "m", range(5, 10), "-", 0.5
        )
        e = _load(tmp_path, test)
        row = e.qc(percentiles=[0, 100]).iloc[0]
        assert row["dropout_test_p0"] == pytest.approx(0.0)
        assert row["dropout_test_p100"] == pytest.approx(0.6)
        assert "dropout_test_p50" not in row
        e.close()

    def test_unmapped_molecules_left_out_of_dropout(self, tmp_path):
        test = _full(0.5, 0.5) + _mol("u", [0], ".", 0.5)
        e = _load(tmp_path, test)
        row = e.qc(percentiles=[100]).iloc[0]
        assert row["frac_unmapped_test"] == pytest.approx(1 / 3)
        assert row["dropout_test_p100"] == 0.0
        e.close()

    def test_wrap_folds_sites(self, tmp_path):
        test = _mol("p", range(5, 10), "+", 0.5)
        e = _load(tmp_path, test, seq="A" * 10, wrap=True)
        row = e.qc().iloc[0]
        assert row["nsite_plus"] == 5
        assert row["dropout_test_p50"] == 0.0
        e.close()


class TestMissingControls:
    def test_test_only_drops_control_columns(self, tmp_path):
        e = _load(tmp_path, _full(0.5, 0.5))
        df = e.qc()
        assert df["meth_frac_test"].iloc[0] == pytest.approx(0.5)
        for col in ("meth_frac_unmeth", "meth_frac_meth", "accessible_frac"):
            assert col not in df
        assert not any(c.endswith(("_meth", "_unmeth")) for c in df)
        assert not any("_meth_" in c or "_unmeth_" in c for c in df)
        e.close()

    def test_unmeth_only_has_unmeth_rate(self, tmp_path):
        e = _load(tmp_path, _full(0.5, 0.5), unmeth=_full(0.1, 0.1))
        df = e.qc()
        assert df["meth_frac_unmeth"].iloc[0] == pytest.approx(0.1)
        assert "meth_frac_meth" not in df
        assert "accessible_frac" not in df
        e.close()


class TestChannels:
    def test_all_row_pools_channels(self, tmp_path):
        seq = "AGCATTTTTT"
        test = _mol("p", [0, 3], "+", 0.5)
        e = _load(tmp_path, test, seq=seq, mtase=["A", "GC"])
        df = e.qc().set_index("channel")
        assert list(df.index) == ["A", "GCH", "GCG", "all"]
        assert df.loc["all", "nsite_plus"] == sum(
            df.loc[c, "nsite_plus"] for c in ("A", "GCH", "GCG")
        )
        e.close()

    def test_all_row_has_no_rates(self, tmp_path):
        seq = "AGCATTTTTT"
        test = _mol("p", [0, 3], "+", 0.5)
        ctrl = _mol("c", [0, 3], "+", 0.5)
        e = _load(
            tmp_path, test, meth=ctrl, unmeth=ctrl, seq=seq, mtase=["A", "GC"]
        )
        df = e.qc().set_index("channel")
        for col in ("meth_frac_test", "meth_frac_unmeth"):
            assert np.isnan(df.loc["all", col])
            assert not np.isnan(df.loc["A", col])
        e.close()

    def test_merged_gc_row(self, tmp_path):
        seq = "AGCGCAGCTT"
        test = [("p", 2, "chr1", "+", 0.5, "m")]
        e = _load(tmp_path, test, seq=seq, mtase="GC")
        sep = e.qc(channels=["GCH", "GCG"]).set_index("channel")
        merged = e.qc(channels="GC", overwrite=True)
        assert merged["channel"].tolist() == ["GC"]
        assert merged["nsite_plus"].iloc[0] == (
            sep.loc["GCH", "nsite_plus"] + sep.loc["GCG", "nsite_plus"]
        )
        e.close()


class TestIgnoreStrand:
    def test_single_nsite_and_no_strand_columns(self, tmp_path):
        e = _load(tmp_path, _full(0.5, 0.5), ignore_strand=True)
        df = e.qc()
        assert df["nsite"].iloc[0] == 10
        assert "nsite_plus" not in df
        assert not any(c.startswith(("frac_plus", "frac_minus")) for c in df)
        assert not any(c.startswith("frac_unmapped") for c in df)
        e.close()


class TestMask:
    def test_mask_applies_to_all_columns(self, tmp_path):
        test = _mol("p", range(2), "+", 0.9) + _mol(
            "m", range(5, 10), "-", 0.1
        )
        e = _load(tmp_path, test)
        e.filter_dropout(thres_max=0.5)
        row = e.qc(mask_name="dropout_mask", percentiles=[100]).iloc[0]
        assert row["nmol"] == 1
        assert row["frac_masked_test"] == pytest.approx(0.5)
        assert row["frac_plus_test"] == 0.0
        assert row["dropout_test_p100"] == 0.0
        e.close()

    def test_unknown_mask_raises(self, exp):
        with pytest.raises(KeyError):
            exp.qc(mask_name="bogus")


class TestStorage:
    def test_stored_in_global_analysis(self, exp):
        df = exp.qc()
        assert exp.global_analysis["qc"].equals(df)

    def test_existing_table_raises_without_overwrite(self, exp):
        exp.qc()
        with pytest.raises(ValueError):
            exp.qc()

    def test_overwrite_replaces_table(self, exp):
        exp.qc()
        df = exp.qc(percentiles=[25], overwrite=True)
        assert "dropout_test_p25" in exp.global_analysis["qc"]
        assert exp.global_analysis["qc"].equals(df)

    def test_survives_save_load(self, exp, tmp_path):
        df = exp.qc()
        path = tmp_path / "exp.h5"
        exp.save(path)
        loaded = MethPrintExperiment.load(path)
        got = loaded.global_analysis["qc"]
        assert got["channel"].tolist() == df["channel"].tolist()
        np.testing.assert_allclose(
            got["accessible_frac"].to_numpy(),
            df["accessible_frac"].to_numpy(),
        )
        loaded.close()


class TestValidation:
    @pytest.mark.parametrize("percentiles", [[], [-1], [101]])
    def test_bad_percentiles_raise(self, exp, percentiles):
        with pytest.raises(ValueError):
            exp.qc(percentiles=percentiles)

    def test_unknown_channel_raises(self, exp):
        with pytest.raises(ValueError):
            exp.qc(channels="GCH")

    def test_unknown_region_raises(self, exp):
        with pytest.raises(ValueError):
            exp.qc(regions=["bogus"])

    def test_missing_refseq_raises(self, tmp_path):
        sizes = tmp_path / "sizes.tsv"
        sizes.write_text("chrom\tlength\nchr1\t10\n")
        test_file = tmp_path / "test.tsv"
        _write_tsv(test_file, _full(0.5, 0.5))
        e = MethPrintExperiment.load_raw(
            chromsize=sizes, test_file=test_file, mtase="A"
        )
        with pytest.raises(ValueError):
            e.qc()
        e.close()
