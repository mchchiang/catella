# cli.py

import nucmc
import typer
import pandas as pd
import numpy as np
from pathlib import Path
from typer import Option
from typing import Annotated, Type, TypeVar, Callable, Optional
app = typer.Typer(help="Nucleosome Positioning Monte Carlo Simulation Suite")

# Helper method for parsing a list of items for an option
T = TypeVar("T")
def csv_parser(target_type: Type[T]) -> Callable[[str], list[T]]:
    def parser(value: str) -> list[T]:
        if not value: return []
        # Split by comma, strip whitespace, and convert to target type
        return [target_type(item.strip()) for item in value.split(",")]
    return parser

@app.command()
def preprocess(
    chromsize: Annotated[Path, typer.Argument(help="Chromosome sizes file",
                                              exists=True, file_okay=True,
                                              dir_okay=False, readable=True)],
    test_file: Annotated[Path, typer.Argument(help="Primary methylation file",
                                              exists=True, file_okay=True,
                                              dir_okay=False, readable=True)],
    out_file: Annotated[Path, typer.Argument(help="Path to save results",
                                             file_okay=True, dir_okay=False)],
    unmeth_file: Annotated[
        Optional[Path], typer.Option(help="Unmethylated control file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    meth_file: Annotated[
        Optional[Path], typer.Option(help="Methylated control file",
                                     exists=True, file_okay=True,
                                     dir_okay=False, readable=True)] = None,
    binsize: Annotated[int, typer.Option(help="Window size (bp)")] = 147,
    wrap: Annotated[
        bool, typer.Option(help="Wrap relative to center")] = False,
    colidx: Annotated[
        Optional[list[int]], 
        typer.Option(parser=csv_parser(int), help="ModKit column indices")
    ] = None,
    clip_low: Annotated[
        Optional[float], typer.Option(help="Lower clip percentile")] = 0.1,
    clip_high: Annotated[
        Optional[float], typer.Option(help="Upper clip percentile")] = 99.9,
    norm_by_strand: Annotated[
        Optional[bool], typer.Option(help="Normalize by strand")] = False):
    """
    Preprocess raw methylation data.
    """
    return nucmc.preprocess(chromsize=chromsize, test_file=test_file,
                            out_file=out_file, unmeth_file=unmeth_file,
                            meth_file=meth_file, binsize=binsize, wrap=wrap,
                            colidx=colidx, clip_low=clip_low,
                            clip_high=clip_high, norm_by_strand=norm_by_strand)
    

@app.command()
def run(
    chroms: Annotated[
        list[str], typer.Option(parser=csv_parser(str),
                                help="Chromosome names")],
    nsim: Annotated[int, typer.Option(help="Simulations per molecule")],
    settings: Annotated[
        Path, typer.Option(help="Simulation configuration file directory",
                           exists=True, file_okay=True, dir_okay=False,
                           readable=True)],
    exp_file: Annotated[
        Path, typer.Option(help="Experiment HDF5 file directory",
                           exists=True, file_okay=True, dir_okay=False,
                           readable=True)],
    out_path: Annotated[Path, typer.Option(help="Output directory",
                                           exists=False, file_okay=False,
                                           dir_okay=True, readable=True)],
    out_types: Annotated[
        Optional[list[str]], typer.Option(parser=csv_parser(str),
                                          help="energy,position...")] = "all",
    seed: Annotated[Optional[int], typer.Option(help="Random seed")] = None,
    nworker: Annotated[
        Optional[int], typer.Option(help="Number of processes")] = 1,
    store_eseq: Annotated[
        Optional[bool], typer.Option(help="Store sequence-specific energy "
                                     "landscape")] = True,
    use_zero_point_mu: Annotated[
        Optional[bool], typer.Option(help="Shift mu to median of sequence-"
                                     "specific energy")] = False,
    verbose: Annotated[
        Optional[bool], typer.Option(help="Print progress")] = True):
    """
    Execute a parallelized methylation simulation.
    """
    exp_data = MethPrintExperiment.load(exp_file)
    meth_prob = {chrom:exp_data.analysis[chrom]["meth_prob"]
                 for chrom in exp_data.chroms}
    return nucmc.run(
        chroms=chroms,
        nsim=nsim,
        settings=settings,
        meth_prob=meth_prob,
        out_path=out_path,
        out_types=out_types,
        seed=seed,
        nworker=nworker,
        store_eseq=store_eseq,
        use_zero_point_mu=use_zero_point_mu,
        verbose=verbose
    )


if __name__ == "__main__":
    app()
