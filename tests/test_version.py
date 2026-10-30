from importlib.metadata import version

import catella


def test_version_matches_metadata():
    assert catella.__version__ == version("catella")
