# test_dropout_masks.py

import numpy as np
import pytest
from catella.experiment._contexts import CG, GC, GCG, GCH, HCG, A
from catella.experiment._dropout import _channel_site_masks


def _idx(mask):
    return np.flatnonzero(mask).tolist()


class TestChannelSiteMasks:
    def test_keys_are_channel_names(self):
        masks = _channel_site_masks("ACGTGCA", (A, GCH, HCG, GCG))
        assert list(masks) == ["A", "GCH", "HCG", "GCG"]

    def test_masks_have_sequence_length(self):
        seq = "ACGTGCAGCGC"
        for plus, minus in _channel_site_masks(seq, (A, GCH)).values():
            assert plus.shape == (len(seq),)
            assert minus.shape == (len(seq),)
            assert plus.dtype == bool

    def test_a_channel_plus_is_a_minus_is_t(self):
        seq = "AATGCTTA"
        plus, minus = _channel_site_masks(seq, (A,))["A"]
        assert _idx(plus) == [0, 1, 7]
        assert _idx(minus) == [2, 5, 6]

    def test_gch_sites(self):
        plus, minus = _channel_site_masks("AGCA", (GCH,))["GCH"]
        assert _idx(plus) == [2]
        assert _idx(minus) == [1]

    def test_hcg_sites(self):
        plus, minus = _channel_site_masks("ACGA", (HCG,))["HCG"]
        assert _idx(plus) == [1]
        assert _idx(minus) == [2]

    def test_gcg_and_gch_split(self):
        masks = _channel_site_masks("GCGC", (GCH, GCG))
        assert _idx(masks["GCG"][0]) == [1]
        assert _idx(masks["GCG"][1]) == [2]
        assert _idx(masks["GCH"][0]) == [3]
        assert _idx(masks["GCH"][1]) == [0]

    def test_channels_are_disjoint(self):
        rng = np.random.default_rng(0)
        seq = "".join(rng.choice(list("ACGT"), 500))
        masks = _channel_site_masks(seq, (A, GCH, HCG, GCG))
        for s in (0, 1):
            total = sum(m[s].astype(int) for m in masks.values())
            assert total.max() <= 1

    @pytest.mark.parametrize(
        ("merged", "parts"), [(GC, ("GCH", "GCG")), (CG, ("HCG", "GCG"))]
    )
    def test_merged_channel_pools_parts(self, merged, parts):
        rng = np.random.default_rng(1)
        seq = "".join(rng.choice(list("ACGT"), 500))
        name = "GC" if merged == GC else "CG"
        got = _channel_site_masks(seq, (merged,))[name]
        sep = _channel_site_masks(seq, (GCH, HCG, GCG))
        for s in (0, 1):
            expected = sep[parts[0]][s] | sep[parts[1]][s]
            np.testing.assert_array_equal(got[s], expected)

    def test_merged_gc_has_equal_strand_counts(self):
        rng = np.random.default_rng(2)
        seq = "".join(rng.choice(list("ACGT"), 500))
        plus, minus = _channel_site_masks(seq, (GC,))["GC"]
        assert plus.sum() == minus.sum()
