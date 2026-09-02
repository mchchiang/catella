# __init__.py

import warnings

# catella_cpp doesn't declare free-threading support, so Python 3.13+
# auto-enables the GIL on import and warns; suppress the warning.
warnings.filterwarnings(
    "ignore", message=".*global interpreter lock.*",
    category=RuntimeWarning,
)

from catella.api import *

__all__ = ["preprocess", "run", "analyze", "downsample", "sort_by_linkage",
           "plot_occup", "plot_nuc_pos", "plot_energy", "plot_methmap"]
