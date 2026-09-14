Input and Output Data Formats
===============================

This page documents the file formats catella reads (raw methylation
footprinting data) and writes (HDF5-backed experiment and simulation
results). It is a reference page; for a walkthrough of how these files
are produced and consumed in practice, see the tutorials.

Input data
-----------

Chromosome-size file
~~~~~~~~~~~~~~~~~~~~~~

A tab-separated file with two columns, ``chrom`` and ``length`` (bp).
A header row matching ``chrom\tlength`` exactly is optional; if the
first line does not match, the file is read headerless and the two
columns are assigned positionally. Chromosome names must be unique.

Example (``segments.size``)::

   chrom	length
   OCT4	5000
   GREB1	5443

This file is required by ``load_raw`` (and the ``catella load_raw``
CLI command), and every ``chrom`` referenced in the methylation call
files must appear here (rows for chromosomes absent from this file are
silently dropped).

Methylation call files (ModKit output)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``load_raw`` takes one required file (``test_file``) and two optional
control files (``unmeth_file``, ``meth_file``, see
:ref:`data-format-model-params`). Each is a tab-delimited text file,
optionally gzip-compressed, produced by ``modkit extract full``, e.g.:

.. code-block:: bash

   modkit extract full --motif A 0 --motif CG 0 --motif GC 1 \
       --mapped --ignore-implicit --ref genome.fa input.bam -

Keep ModKit's header row (do not pass ``--no-header``) unless you
supply ``colidx`` to select columns positionally. Only six of ModKit's
output columns are read:

.. list-table::
   :header-rows: 1
   :widths: 18 18 12 52

   * - ModKit column
     - catella name
     - dtype
     - Meaning
   * - ``read_id``
     - ``mol_id``
     - string
     - Read/molecule identifier.
   * - ``ref_position``
     - ``upos``
     - int64
     - 0-based aligned reference position.
   * - ``chrom``
     - ``chrom``
     - string
     - Aligned contig/segment name; must match the chromosome-size
       file.
   * - ``ref_strand``
     - ``strand``
     - string
     - ``+``, ``-``, or ``.``.
   * - ``mod_qual``
     - ``mod_qual``
     - float64
     - Modification-call confidence score; this is :math:`q_x` in the
       :doc:`model` page.
   * - ``mod_code``
     - ``mod_code``
     - string
     - Base-modification code from the MM tag (e.g. ``a`` for 6mA,
       ``m`` for 5mC, ``-`` for canonical).

If ``colidx`` is given (a list of six integers, in the order of the
table above), the file is assumed headerless and those column indices
are used instead of ModKit's column names.

.. warning::

   ModKit emits both 5mC (``m``) and 5hmC (``h``) rows for the same
   cytosine by default. catella only allows one probability per base,
   so filter one out before passing the file to catella, e.g.:

   .. code-block:: bash

      awk '$14 != "h"'

``unmeth_file`` and ``meth_file`` (when given) must cover exactly the
same set of chromosomes as ``test_file``, or ``load_raw`` raises
``ValueError``. If both ``mtase`` and ``fasta_file`` are given, rows
are additionally filtered for strand/reference-base consistency with
the requested methyltransferase context(s).

Reference FASTA
~~~~~~~~~~~~~~~~~

A multi-FASTA file (``fasta_file``), one record per chromosome. Each
record's id must match a ``chrom`` in the chromosome-size file, and
its sequence length must exactly equal that chromosome's ``length``
(``ValueError`` otherwise). Read via ``pyfaidx``, so a ``.fai`` index
is expected alongside it (built automatically if missing).

``fasta_file`` is optional for ``load_raw``/``compute_empirical_prob``,
but required by ``compute_model_prob``, which needs sequence context
to classify assayable sites into methylation channels.

.. _data-format-model-params:

Methyltransferase (``mtase``) channels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``mtase`` is not a file but a parameter constraining how calls are
parsed: one or more of ``"A"`` (any-context adenine, e.g. EcoGII m6A),
``"CG"`` (CpG, e.g. M.SssI), or ``"GC"`` (GpC, e.g. M.CviPI). These
correspond to the assayable-site channels A/HCG/GCH/GCG described in
the :doc:`model` page.

Simulation settings file
~~~~~~~~~~~~~~~~~~~~~~~~~~

``catella run`` (and the ``SimSettings`` API) takes a settings file in
JSON or YAML, a flat mapping whose keys must match ``SimSettings``
fields exactly (unknown keys raise ``TypeError``):

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - ``nucbp``
     - int
     - Base pairs occupied by one bound nucleosome (e.g. 147).
   * - ``llink``
     - int
     - Linker DNA length (bp) between nucleosomes.
   * - ``mu``
     - float
     - Chemical potential (:math:`k_BT`); energy gained per bound
       nucleosome.
   * - ``start_temp``
     - float
     - Annealing start temperature.
   * - ``end_temp``
     - float
     - Annealing end temperature (must be at most ``start_temp``).
   * - ``cool_option``
     - str
     - ``"linear"``, ``"geometric"``, or ``"constant"``.
   * - ``nsweep``
     - int
     - Number of Monte Carlo sweeps.
   * - ``print_freq``
     - int
     - Sweep interval at which frames are recorded.
   * - ``emax``
     - float
     - Maximum methylation energy cost (:math:`k_BT`).

Methylation probability array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The array passed to ``catella.run`` as ``meth_prob`` (or read from
``exp_data.analysis[chrom]["meth_prob"]`` when using the CLI) is a
NumPy array (single chromosome) or a mapping of chromosome name to
array/DataFrame, of shape ``(n_molecules, nbp)``, with values in
``[0, 1]``. As throughout catella, the coordinate convention is
left-aligned: a footprint or window's position is its leftmost bp.

Output data
------------

Storage primitives
~~~~~~~~~~~~~~~~~~~~

All HDF5 output builds on two primitives:

``H5Array``
   A dense 2D numeric matrix (default dtype ``float64``, gzip+shuffle
   compressed), stored as a dataset ``data`` with optional sibling
   datasets ``<name>__index`` and ``<name>__columns`` for row/column
   labels (defaulting to a plain range index if absent). Used for
   large per-molecule, per-position matrices such as ``meth_prob`` and
   ``occup``.

``DataFrameMap`` / ``save_df``
   A small tabular result, stored under its own HDF5 sub-group with
   ``_column_order`` (the original column order), a ``num_values``
   dataset (numeric columns stacked, float64) with parallel
   ``num_names``, and one ``str_<colname>`` dataset per non-numeric
   column. Used for one-row-per-position or one-row-per-molecule
   summaries such as ``<name>_theta`` or ``mean_nnuc``.

Experiment HDF5 file (``analysis.h5``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Written by ``MethPrintExperiment.save()`` (or the ``catella
load_raw``/``compute_*_prob`` CLI commands via ``--out_file``). Two
top-level attributes, ``mtase`` (comma-joined channel labels) and
``wrap`` (bool), plus:

.. code-block:: text

   /raw_data/<chrom>/metadata      group; attrs: chrom, nbp;
                                    optional "refseq" dataset (str)
   /raw_data/<chrom>/data/
       test_mol_id                 string dataset, one entry per
                                    test-channel molecule
       test_data                   save_df sub-group: columns
                                    mol_index, pos, strand, mod_qual,
                                    mod_code
       unmeth_mol_id, unmeth_data   same shape, optional
       meth_mol_id, meth_data       same shape, optional
   /analysis/<chrom>/<key>          H5Array or save_df sub-group
   /global_analysis/<key>           H5Array or save_df sub-group

Raw per-source tables (``test_data``/``meth_data``/``unmeth_data``)
have columns ``mol_index`` (row position in ``*_mol_id``), ``pos``
(bp position, folded if ``wrap``), ``strand``, ``mod_qual`` (the
confidence score :math:`q_x`), and ``mod_code``.

Common ``analysis[chrom]`` keys produced by the processing pipeline:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Contents
   * - ``test_smoothed``, ``meth_smoothed``, ``unmeth_smoothed``
     - Rolling-window-averaged raw signal, ``(n_mol, nbp)``
       ``H5Array``, left-aligned with the trailing edge filled per
       ``fill_edge``.
   * - ``meth_prob`` (or a user-chosen ``prob_name``)
     - Final normalized methylation probability matrix, ``(n_mol,
       nbp)``, values in ``[0, 1]`` (``NaN`` at un-computable trailing
       positions or dropout-masked rows).
   * - ``<prob_name>_theta``
     - DataFrame, one row per bp position: ``theta_prot``,
       ``theta_acc``, ``informative`` (or, with
       ``norm_by_strand=True``, the same columns split into
       ``_pos``/``_neg`` variants) — the calibrated per-position
       protected/accessible call rates used by ``compute_model_prob``.
   * - ``<source>_<mask_name>`` (default ``dropout_mask``)
     - DataFrame with a boolean ``keep`` column, one row per molecule,
       from ``filter_dropout``.
   * - ``<data_name>_sorted``, ``<data_name>_linkage``
     - Outputs of ``sort_by_linkage``.

Common ``global_analysis`` keys:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Contents
   * - ``<smoothed_name>_params``
     - DataFrame (1 row): ``lnuc``, ``nan_method``, ``fill_edge``,
       ``mask_name``.
   * - ``<prob_name>_eta``
     - DataFrame (1 row): ``eta_M6A``, ``eta_GCH``, ``eta_HCG``,
       ``eta_GCG`` — per-channel crosstalk correction factors
       (``compute_model_prob`` only).
   * - ``<prob_name>_rho`` (if ``store_rho=True``)
     - DataFrame: ``lag``, ``rho_<channel>`` per auto-estimated
       channel.
   * - ``<prob_name>_params``
     - DataFrame (1 row) recording every ``compute_model_prob`` kwarg
       (``pi0``, ``eta_max_lag``, ``nu``, ``rho_leak``, ``min_gap``,
       ``lnuc``, ``n_min``, ``max_iters``, ``init_prot``,
       ``init_acc``, ``tol``, ``fill_edge``, ``norm_by_strand``,
       ``mask_name``).
   * - ``<prob_name>_calib``
     - DataFrame, one row per chromosome (and strand, if
       ``norm_by_strand``): ``chrom``, ``has_controls``,
       ``frac_informative``, and, on the no-controls EM path, also
       ``iters``, ``log_likelihood``, ``frac_protected``, and
       ``theta_prot_<channel>``/``theta_acc_<channel>`` per channel.

Simulation output
~~~~~~~~~~~~~~~~~~~

A simulation run (``catella.run``, ``SimManager``) writes a directory
that must not already exist:

.. code-block:: text

   out_dir/
     <dataset_name>.h5     default "results.h5"; dataset metadata
                            and analysis (see below)
     raw_data/              one HDF5 file per (chrom, molecule, run),
                             sharded into subdirectories of
                             max_mols_per_dir molecules each

Each per-run raw file (``SimData``, written by the C++ engine) has:

.. code-block:: text

   /params   group (attrs only): nucbp, nbp, llink, mu, seed
   /data     group
       time              1D array, recorded sweep/time points
       energy            1D float array, total energy at each point
       temp              1D float array, temperature at each point
       position_flat     1D array, concatenated nucleosome start
                          positions across all recorded frames
       position_offset   1D int array (len(time) + 1), CSR-style
                          offsets into position_flat delimiting each
                          frame's nucleosome list

Which of ``energy``/``position_flat``+``position_offset``/``temp`` are
actually written is controlled by ``out_type`` (``"energy"``,
``"position"``, ``"temp"``, or ``"all"``); missing ones load as
``np.nan``.

The dataset-level file (``<dataset_name>.h5``, default
``results.h5``, written by ``SimDataset.save``):

.. code-block:: text

   /metadata
       attrs: raw_dir, nsim, out_type, seed
       chroms              string dataset, chromosome names
       nmol                int array, molecules per chromosome
       nbp                 int array, bp length per chromosome
       settings/           group; attrs = every SimSettings field
       eseq/<chrom>        (optional, store_eseq=True) H5Array,
                            (n_mol, nbp), sequence-specific binding
                            energy landscape
       seed_table/<chrom>  (optional) uint32 array, (nmol, nsim),
                            per-(molecule, run) seed for reproducing
                            an individual run
   /analysis/<chrom>/<key>   H5Array or DataFrameMap entry
   /global_analysis/<key>    H5Array or DataFrameMap entry

Common ``analysis[chrom]`` keys produced by ``catella.analyze``:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Contents
   * - ``occup`` (default name)
     - ``H5Array`` float64, ``(n_mol, nbp)``: fraction of ``nsim``
       runs where a base pair is nucleosome-occupied at the analyzed
       time point (the final frame by default). Molecules with no
       simulation file on disk are all-``NaN`` rows.
   * - ``mean_nnuc`` (default name)
     - DataFrame, one row per molecule: mean number of bound
       nucleosomes across ``nsim`` runs at the analyzed time.
   * - ``<...>_sorted``, ``<...>_linkage``
     - Outputs of ``sort_by_linkage``.

Canonical project layout
~~~~~~~~~~~~~~~~~~~~~~~~~~

The tutorials and CLI establish this directory convention for a
project combining an experiment and its simulations:

.. code-block:: text

   <project>/
     experiment/
       raw_data/
         segments.size          chromosome-size file
         segments.fa(.fai)      reference fasta
         chromatin.tsv.gz       test_file
         methylated.tsv.gz      meth_file (control)
         unmethylated.tsv.gz    unmeth_file (control)
       analysis.h5              MethPrintExperiment.save() output
     simulation/
       results.h5                SimDataset output (default name)
       raw_data/                  per-run SimData files (sharded)

``MethPrintExperiment.save``/``SimDataset.save`` (and their CLI
wrappers) support ``--out_file``/``--overwrite``: omit ``out_file`` to
overwrite the source file in place, or give a path (``--overwrite``
is then only needed if it collides with an ``H5Array`` already backing
loaded analysis data).
