Example Pipeline
==================

This page gives the full list of commands to go from raw methylation
footprinting data to nucleosome positioning predictions, with minimal
commentary. See :doc:`tutorials/tut01_meth_prob` and
:doc:`tutorials/tut02_run_sim` for the detailed walkthrough.

.. code-block:: python

   from pathlib import Path
   import catella
   from catella.mapping import CoordsTransform
   from catella.experiment.methdata import MethPrintExperiment
   from catella.simulation.engine import SimSettings

   # --- Tutorial 1: raw data to normalized methylation probability ---

   # Load the raw data
   exp_dir = Path("example/tut01_meth_prob/experiment")
   raw_dir = exp_dir/"raw_data"
   chromsize = raw_dir/"segments.size"
   fasta_file = raw_dir/"segments.fa"
   test_file = raw_dir/"chromatin.tsv.gz"
   unmeth_file = raw_dir/"unmethylated.tsv.gz"
   meth_file = raw_dir/"methylated.tsv.gz"

   exp_data = catella.load_raw(chromsize=chromsize,
                               test_file=test_file,
                               unmeth_file=unmeth_file,
                               meth_file=meth_file,
                               fasta_file=fasta_file,
                               mtase=["A", "GC"],
                               wrap=True,
                               nworker=3)

   # QC: plot raw methylation scores sorted by similarity, and dropout ECDF
   for which in ("test", "unmeth", "meth"):
       link = catella.sort_by_linkage(exp=exp_data, raw_which=which,
                                      fill_nan=0.0)
       catella.plot_meth_prob(exp=exp_data, chrom="601", raw_which=which,
                              link_mat=link["601"], vmin=0.0, vmax=1.0)
   catella.plot_dropout_ecdf(exp=exp_data, chroms="601")

   # Model-based normalized methylation probability
   exp_data = catella.compute_model_prob(exp=exp_data,
                                         prob_name="meth_prob_model")
   link = catella.sort_by_linkage(exp=exp_data, data_name="meth_prob_model",
                                  fill_nan=0.0)
   meth_prob = exp_data.analysis["601"]["meth_prob_model"]
   catella.plot_meth_prob(meth_prob, link_mat=link["601"],
                          vmin=0.0, vmax=1.0)
   catella.plot_meth_energy(meth_prob, link_mat=link["601"])

   # Optional: center-align the left-aligned probability windows
   lnuc = 147
   coords = CoordsTransform(lnuc=lnuc)
   for c in exp_data.chroms:
       exp_data.analysis[c]["meth_prob_model_centered"] = \
           coords.left_to_center_aligned(
	       exp_data.analysis[c]["meth_prob_model"])

   # Empirical normalized methylation probability (cross-check)
   exp_data = catella.compute_empirical_prob(exp=exp_data,
                                             prob_name="meth_prob_empirical")
   link = catella.sort_by_linkage(exp=exp_data, chroms="601",
                                  data_name="meth_prob_empirical",
                                  fill_nan=0.0)
   catella.plot_meth_prob(
       data=exp_data.analysis["601"]["meth_prob_empirical"],
       link_mat=link["601"], vmin=0.0, vmax=1.0)
   catella.plot_meth_energy(
       data=exp_data.analysis["601"]["meth_prob_empirical"],
       link_mat=link["601"])

   # Save processed experiment data
   exp_h5 = exp_dir/"analysis.h5"
   exp_data.save(exp_h5)

   # --- Tutorial 2: methylation probability to nucleosome positioning ---

   exp_data = MethPrintExperiment.load(exp_h5)

   start_temp = catella.estimate_start_temp(exp_data,
                                            prob_name="meth_prob_model",
                                            percentile=99.0)

   # Set simulation parameters
   settings = SimSettings(nucbp=147,
                          llink=50,
			  elink=25.0,
			  mu=0.0,
			  start_temp=start_temp,
			  end_temp=1.0,
			  cool_option="geometric",
			  nsweep=20000,
			  print_freq=100,
			  emax=25.0)

   # Extract methylation probabilities needed for running the simulations
   meth_prob = {"601": exp_data.analysis["601"]["meth_prob_model"]}

   # Set up the simulation directory - which must not exist before
   sim_dir = Path("example/tut02_run_sim/simulation/result/")

   # Run the simulations
   sim_data = catella.run(chroms="601",
                          nsim=5,
                          seed=2346,
                          meth_prob=meth_prob,
                          settings=settings,
                          out_dir=sim_dir,
                          nworker=20)

   # Sanity checks on one MCMC run
   from catella.simulation.plot import SimPlot

   simplot = SimPlot()
   simplot.plot_energy(sim_data, chrom="601", mol=0, run=0)
   simplot.plot_samples(sim_data, chrom="601", mol=0, plot_eseq=True)

   # Analyze and save results
   catella.compute_occup(sim_data)
   link = catella.sort_by_linkage(dataset=sim_data, data_name="occup")
   catella.plot_occup(sim_data, chrom="601", link_mat=link["601"],
                      plot_eseq=True)
   sim_data.save()
