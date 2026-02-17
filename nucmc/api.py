# api.py

# High-level interface for processing fiber-seq data and running simulations

from collections.abc import Iterable, Mapping
from pathlib import Path
import numpy as np
import pandas as pd
from .prep import FiberSeqAnalysis
from .sim import SimManager
from .seq_data import FiberSeqExperiment
from .sim_data import SimDataset
from .util import IndexType

def preprocess(*, binsize : int,
               chromsize : str,
               test_file : str | Path,
               out_path : str | Path | None = None,
               unmeth_file : str | Path | None = None,
               meth_file : str | Path | None = None,
               wrap : bool = False,
               colidx : Iterable | None = None) -> FiberSeqExperiment:

    # Load the raw data (generated from modkit)
    exp = FiberSeqExperiment.load_raw(chromsize=chromsize,
                                      test_file=test_file,
                                      unmeth_file=unmeth_file,
                                      meth_file=meth_file,
                                      wrap=wrap,
                                      colidx=colidx)

    # Normalize the data as required
    ana = FiberSeqAnalysis()
    ana.normalize(binsize, exp)
    
    # Save the results
    if (out_path is not None):
        exp.save(out_path)

    return exp


def run(*, nucbp : int,
        llink : int,
        mu : float,
        chroms : str | Iterable[str],
        nsim : int,
        nsweep : int,
        start_temp : float,
        end_temp : float,
        ninc_temp : int,
        print_freq : int,
        seed : int,
        meth : np.ndarray | Mapping[str,np.ndarray|pd.DataFrame],
        out_types : str | Path,
        out_path : str | Path,
        nworker : int = 1,
        mols : IndexType | Mapping[str,IndexType] = slice(None),
        extract_final_pos : bool = True) -> SimDataset:

    manager = SimManager(nucbp, llink, mu)
    dataset = manager.run(chroms, nsim, nsweep, start_temp, end_temp,
                          ninc_temp, print_freq, seed, meth, out_types,
                          out_path, nworker=nworker, mols=mols)
    return dataset

def occupancy():
    pass
