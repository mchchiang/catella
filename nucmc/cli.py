# cli.py

import typer
import nucmc
from typing import Optional
app = typer.Typer(help="Nucleosome Positioning Monte Carlo Simulation Suite")

@app.command()
def preprocess(
        binsize : int,
        chromsize : str,
        test_file : str,
        out_path : str,
        unmeth_file : Optional[str] = None,
        meth_file : Optional[str] = None,
        wrap : bool = False,
        colidx : Optional[str] = None):
    if  (colidx is not None):
        colidx = [int(s) for s in colidx.split(",")]
    nucmc.preprocess(binsize, chromsize, test_file, out_path, unmeth_file,
                     meth_file, wrap, colidx)
    
@app.command()
def run(
        nucbp,
        llink,
        mu,
        chroms,
        nsim,
        nsweep,
        start_temp,
        end_temp,
        ninc_temp,
        print_freq,
        seed,
        meth,
        out_types,
        out_path):
    nucmc.run(nucbp, llink, mu, chroms, nsim, nsweep, start_temp, end_temp,
              ninc_temp, print_freq, seed, meth, out_types, out_path)

if __name__ == "__main__":
    app()
