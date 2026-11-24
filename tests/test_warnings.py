# test_warnings.py

from types import SimpleNamespace

import catella
import numpy as np
import pytest
from catella import utils
from catella._warnings import _warn
from catella.simulation.config import SimSettings
from catella.simulation.engine import SimManager


def _llink_exp(linker=10, nmol=50, nbp=200):
    energy = np.full((nmol, nbp), 5.0)
    energy[:, 0 : nbp - 4 : 5 + linker] = -20.0
    prob = 1.0 / (1.0 + np.exp(-energy))
    return SimpleNamespace(
        regions=("chr1",), analysis={"chr1": {"meth_prob": prob}}
    )


def _llink_settings():
    return SimSettings(
        lnuc=5, llink=99, elink=25.0, mu=0.0, end_temp=1.0, emax=100.0
    )


class TestWarn:
    def test_default_category_is_user_warning(self):
        with pytest.warns(UserWarning, match="hello") as rec:
            _warn("hello")
        assert rec[0].category is UserWarning

    def test_custom_category(self):
        with pytest.warns(RuntimeWarning, match="hello"):
            _warn("hello", RuntimeWarning)

    def test_attributed_to_caller(self):
        with pytest.warns(UserWarning) as rec:
            _warn("hello")
        assert rec[0].filename == __file__


class TestAttribution:
    def test_api_estimate_llink(self):
        with pytest.warns(UserWarning, match="largest candidate") as rec:
            catella.estimate_llink(
                _llink_exp(), _llink_settings(), llinks=[0, 5]
            )
        assert rec[0].filename == __file__

    def test_sim_manager_estimate_llink(self):
        with pytest.warns(UserWarning, match="largest candidate") as rec:
            SimManager().estimate_llink(
                _llink_exp(), _llink_settings(), llinks=[0, 5]
            )
        assert rec[0].filename == __file__

    def test_compute_linkage(self, monkeypatch):
        monkeypatch.setattr(utils, "_CLUSTER_WARN_ROWS", 3)
        matrix = np.random.default_rng(0).random((5, 4))
        with pytest.warns(UserWarning, match="clustering 5 rows") as rec:
            utils.compute_linkage(matrix)
        assert rec[0].filename == __file__
