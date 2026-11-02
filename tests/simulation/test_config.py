# test_config.py

import json

import pytest
import yaml
from catella.simulation.config import SimSettings


def _values(**overrides):
    values = {
        "nucbp": 20,
        "llink": 7,
        "elink": 25.0,
        "mu": 0.0,
        "end_temp": 1.0,
        "emax": 10.0,
    }
    values.update(overrides)
    return values


class TestForChrom:
    def test_integer_llink_applies_to_every_chromosome(self):
        settings = SimSettings(**_values())
        assert settings.for_chrom("a").llink == 7
        assert settings.for_chrom("b").llink == 7

    def test_mapping_gives_chromosome_value(self):
        settings = SimSettings(**_values(llink={"a": 3, "b": 0}))
        assert settings.for_chrom("a").llink == 3
        assert settings.for_chrom("b").llink == 0

    def test_mapping_result_is_integer(self):
        settings = SimSettings(**_values(llink={"a": 3}))
        assert type(settings.for_chrom("a").llink) is int

    def test_other_fields_are_kept(self):
        settings = SimSettings(**_values(llink={"a": 3}, mu=1.5))
        assert settings.for_chrom("a").mu == 1.5

    def test_original_is_unchanged(self):
        llink = {"a": 3}
        settings = SimSettings(**_values(llink=llink))
        settings.for_chrom("a")
        assert settings.llink == llink

    def test_missing_chromosome_raises(self):
        settings = SimSettings(**_values(llink={"a": 3}))
        with pytest.raises(KeyError, match="no value for chromosome 'b'"):
            settings.for_chrom("b")


class TestLoad:
    @pytest.mark.parametrize(
        ("name", "dump"),
        [("settings.json", json.dumps), ("settings.yml", yaml.safe_dump)],
    )
    def test_loads_mapping(self, tmp_path, name, dump):
        path = tmp_path / name
        path.write_text(dump(_values(llink={"a": 3, "b": 0})))
        assert SimSettings.load(path).llink == {"a": 3, "b": 0}

    def test_numeric_yaml_chromosome_names_become_strings(self, tmp_path):
        path = tmp_path / "settings.yml"
        path.write_text(yaml.safe_dump(_values(llink={601: 25})))
        assert SimSettings.load(path).for_chrom("601").llink == 25

    def test_integer_llink_is_unchanged(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps(_values()))
        assert SimSettings.load(path).llink == 7
