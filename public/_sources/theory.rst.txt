Simulation Model
================

We model the DNA fiber as a one-dimensional (1D) lattice array given by
coordinates :math:`x \in \{0,\ldots,L-1\}`, where :math:`L` is the total
length of the DNA fiber (in units of bp). Within the simulations, nucleosomes
of size :math:`\ell_{\text{nuc}}` are considered.

The Energy of the System
------------------------

The effective Hamiltonian of the system has three contributions:

.. math::

   H = E - \mu N \equiv E_{\text{seq}} + E_{\text{rep}} -\mu N \;,

where :math:`E_{\text{seq}}` is an energy associated with the methylation
footprinting signal (see below), and :math:`E_{\text{rep}}` is a pairwise
repulsion energy due to the excluded volume of the nucleosomes (i.e., they
cannot overlap). The last term is a chemical energy associated with a
non-sequence-specific nucleosomal binding energy to the DNA fiber, with
:math:`\mu` being the chemical potential (note that a positive :math:`\mu`
means it is favourable for nucleosomes to bind to the fiber) and :math:`N`
the total number of nucleosomes on the fiber.

To obtain :math:`E_{\text{seq}}(x)`, we first compute the probability
:math:`p(x)` of a nucleosome binding to the DNA fiber at :math:`x` based on the
methylation footprinting data. If control data are available (i.e.,
footprinting data for a fully methylated and unmethylated fiber), we define
a normalized methylation score :math:`M(x)` as follows:

.. math::

   M(x) = \frac{\langle S_{\text{test}} \rangle_{\text{bin}}(x) -
   \langle S_{\text{unmeth}} \rangle_{\text{bin}}(x)}
   {\langle S_{\text{meth}} \rangle_{\text{bin}}(x) -
   \langle S_{\text{unmeth}} \rangle_{\text{bin}}(x)} \;,

where :math:`S_{\text{test}}`, :math:`S_{\text{meth}}`, and
:math:`S_{\text{unmeth}}` are the methylation signal strengths as computed
by ModKit (i.e., the value from the `mod_qual` column; defined for each bp)
for the test condition (chromatinized DNA), a fully methylated fiber, and an
unmethylated DNA fiber, respectively. Here, the operation
:math:`\langle\cdot\rangle_{\text{bin}}` denotes that we have first smoothed
the signal by doing a rolling average with a bin size of
:math:`\ell_{\text{bin}}`. Typically, we take :math:`\ell_{\text{bin}} = 147`
bp, which is the typically length of DNA wrapped around a nucleosome [#f1]_. If
control datasets are unavailable, we set

.. math::

   M(x) = \langle S_{\text{test}} \rangle_{\text{bin}}(x) \;.

To convert :math:`M(x)` into a formal probability that lies between 0 and 1, we
rescale the score linearly, setting those ranking below 1% to 0 and those
ranking above 99% to 1, i.e.,

.. math::
   p(x) = 1-\frac{\widetilde{M(x)} - M_{p=0.01}}{M_{p=0.99}-M_{p=0.01}} \;,

where

.. math::
   \widetilde{M(x)} = \text{min}[\text{max}[M(x),M_{p=0.01}],M_{p=0.99}] \;.
   
Finally, we convert :math:`p(x)` to the energy assuming it follows a
Boltzmann distribution:

.. math::
   E_{\text{seq}} = \text{min}[-k_BT\log p(x), E_{\text{seq}}^{\text{max}}] \;,

where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature of
the system, and :math:`E_{\text{seq}}^{\text{max}}` is a threshold maximum
energy that the user can specify to avoid :math:`E_{\text{seq}} \to \infty`.

To model the pairwise repulsion between two nucleosomes (say :math:`i` and
:math:`j`), we consider the following potential:

.. math::

   E_{\text{rep}}^{ij}(x_{ij}) =
   \begin{cases}
     \infty & \text{if } x_{ij} < \ell_{\text{nuc}} \\
     k_BT\left[\left(\frac{\ell_{\text{link}}}{x_{ij}-\ell_{\text{nuc}}}\right)^{12} - 2\left(\frac{\ell_{\text{link}}}{x_{ij}-\ell_{\text{nuc}}}\right)^6 + 1\right] & \text{if } \ell_{\text{nuc}} \leq x_{ij} \leq \ell_{\text{nuc}}+\ell_{\text{link}} \\
     0 & \text{otherwise} \;,
   \end{cases}

where :math:`x_{ij} = |x_i-x_j|` is the separation between the nucleosomes,
:math:`\ell_{\text{nuc}} = 147` bp, and :math:`\ell_{\text{link}}` is the
typical length of the linker DNA between nucleosomes (set to 50 bp). Under
the condition :math:`\ell_{\text{link}} < \ell_{\text{nuc}}`, only nearest
neighbour interactions are effective, and so

.. math::

   E_{\text{rep}} = \sum_{i=0}^{N-2}E_{\text{rep}}^{i,i+1}

for :math:`N \geq 2` and 0 otherwise.

Monte Carlo Sampling Moves
--------------------------

Since we allow the number of nucleosomes to vary, the statistical ensemble
sampled is the **grand canonical ensemble**. In thermal equilibrium, the
probability of observing a configuration :math:`\mathcal{C}` is

.. math::

   P(\mathcal{C}) = \frac{1}{\mathcal{Z}}\exp[-\beta H(\mathcal{C})] \;,

where :math:`\beta = 1/(k_BT)` and :math:`\mathcal{Z}` is the partition
function. We perform one of the following three moves in each step:

1. **Shifting a nucleosome**
   We randomly select the :math:`i`-th nucleosome and move its position
   :math:`x_i \to x_i' = x_i \pm 1` bp. The acceptance probability is:

   .. math::

      p(x_i\to x_i') = \min\left[1,\exp[-\beta(E(x_i')-E(x_i))]\right] \;.

2. **Adding a nucleosome**
   We randomly select a position :math:`x`. The probability of accepting a new
   nucleosome at that position is:

   .. math::

      p(\text{add at } x) = \min\left[1,\frac{L}{(N+1)\ell_{\text{nuc}}}\exp[\beta\mu-\beta\Delta E_{\text{add}}(x)]\right] \;,

3. **Removing a nucleosome**
   We randomly select and remove the :math:`i`-th nucleosome. The acceptance
   probability is:

   .. math::

      p(\text{remove } i) = \min\left[1,\frac{N\ell_{\text{nuc}}}{L}\exp[-\beta\mu-\beta\Delta E_{\text{remove}}(x_i)]\right] \;.

These moves follow the **Metropolis algorithm**, ensuring detailed balance and
thermal equilibrium.

..
   Observables
   --------------

   The mean occupancy profile is defined as follows:

   .. math::
      :label: mean_occ

      \langle n(x) \rangle = \frac{1}{T}\sum_{t=1}^{T} n(x,t)

   This quantity corresponds to the time-averaged coverage at base-pair :math:`x`.


.. rubric:: Footnotes

.. [#f1] We perform the rolling average in a left-aligned manner, so
	 :math:`\langle S \rangle_{\text{bin}}(0) = \frac{1}{\ell_{\text{bin}}}
	 \sum_{x=0}^{\ell_{\text{bin}}-1}S(x)`. For
	 :math:`x \geq L-\ell_{\text{bin}}`, we set the score to be same as
	 :math:`\langle S \rangle_{\text{bin}}(L-\ell_{\text{bin}}-1)`. This
	 has no effect as the nucleosome cannot "fall off" the fiber.
