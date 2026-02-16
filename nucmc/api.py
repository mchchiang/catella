# api.py

from collections.abc import Iterable, Mapping
from typing import List, Dict
from pathlib import Path
import numpy as np
from .prep import FiberSeqAnalysis
from .sim import SimManager
from .seq_data import FiberSeqExperiment
from .sim_data import SimDataset
from .util import IndexType

def preprocess(binsize : int,
               chromsize : str,
               test_file : str| Path,
               out_path : str | Path = None,
               unmeth_file : str = None,
               meth_file : str = None,
               wrap : bool = False,
               colidx : List = None) -> FiberSeqExperiment:

    # Load the raw data (generated from modkit)
    exp = FiberSeqExperiment.load_raw(chromsize, test_file, unmeth_file,
                                      meth_file, wrap=wrap, colidx=colidx)

    # Normalize the data as required
    ana = FiberSeqAnalysis()
    ana.normalize(binsize, exp)
    
    # Save the results
    if (out_path is not None):
        exp.save(out_path)

    return exp


def run(nucbp : int,
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
        meth : np.ndarray | Mapping[str,np.ndarray],
        out_types : str | Path,
        out_path : str | Path,
        nproc : int = 1,
        mols : IndexType | Mapping[str,IndexType] = slice(None),
        extract_final_pos : bool = True) -> SimDataset:

    manager = SimManager(nucbp, llink, mu)
    dataset = manager.run(chroms, nsim, nsweep, start_temp, end_temp,
                          ninc_temp, print_freq, seed, meth, out_types,
                          out_path, mols)

    # Store the final positions of the nucleosomes as a separate analysis
    if extract_final_pos:
        pos = dataset.position(chroms=chroms, time_idx=-1)
        
    return dataset
