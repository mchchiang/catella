Examples
========

Loading a Dataset
-----------------

.. code-block:: python

    from nucmc.sim_data import SimDataset

    dataset = SimDataset("results/")
    sim = dataset.raw["chr1", 0, 5]


Access Analysis
---------------

.. code-block:: python

    from nucmc.sim import Sim
		
    energy = sim.analysis["energy"]
    print(energy.head())
