# _warnings.py

import os
import warnings

_PKG_DIR = os.path.dirname(os.path.abspath(__file__)) + os.sep


def _warn(message, category=UserWarning):
    """
    Issue a warning attributed to the first frame outside catella.

    A fixed `stacklevel` is wrong when the same code is reached through
    call paths of different depths (e.g. `catella.estimate_llink` and
    `SimManager.estimate_llink`), so frames inside the package are
    skipped instead.
    """
    warnings.warn(message, category, skip_file_prefixes=(_PKG_DIR,))
