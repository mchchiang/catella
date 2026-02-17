Theory
======

Model Overview
--------------

The system consists of a one-dimensional lattice of length :math:`L`,
representing DNA base-pair positions. Nucleosomes occupy a footprint
of size :math:`w`.

The configuration of the system at time :math:`t` is described by
the occupancy field:

.. math::
   :label: occupancy

   n_i(t) \in \{0,1\}, \quad i = 1, \dots, L

where :math:`n_i = 1` indicates that site :math:`i` is covered
by a nucleosome.

Energy Function
---------------

The total energy of a configuration is defined as:

.. math::
   :label: energy

   E = \sum_{i=1}^{L} V_i n_i
       + \sum_{i<j} U(|i-j|) n_i n_j

where:

- :math:`V_i` is the sequence-dependent potential
- :math:`U(r)` is an interaction potential

Metropolis Sampling Dynamics
-------------------

We evolve the system using a Metropolis Monte Carlo scheme.

For a proposed move with energy change :math:`\Delta E`,
the acceptance probability is:

.. math::
   :label: metropolis

   P_{\mathrm{acc}} =
   \min \left( 1, e^{-\beta \Delta E} \right)

where:

.. math::

   \beta = \frac{1}{k_B T}

The algorithm ensures detailed balance with respect to the
Boltzmann distribution:

.. math::
   :label: boltzmann

   P(\{n_i\}) \propto e^{-\beta E}

Mean Occupancy
--------------

The mean occupancy profile is defined as:

.. math::
   :label: mean_occ

   \langle n_i \rangle
   =
   \frac{1}{T}
   \sum_{t=1}^{T}
   n_i(t)

This quantity corresponds to the time-averaged coverage
at base-pair :math:`i`.

Relationship to Output
----------------------

In the implementation:

- :math:`\langle n_i \rangle` is stored in
  ``analysis["occupancy"]``
- the energy trajectory
  :math:`E(t)` is stored in
  ``analysis["energy"]``

Equation :eq:`metropolis` is implemented in the C++ core
of the simulation engine.
