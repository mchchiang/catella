Monte Carlo Simulations of Nucleosome Positions
===============================================

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
footprinting signal (related to a sequence-specific nucleosomal binding
energy), and :math:`E_{\text{rep}}` is a pairwise repulsion energy due to the
excluded volume of the nucleosomes (i.e., they cannot overlap). The last term
is a chemical energy associated with a non-sequence-specific nucleosomal
binding energy to the DNA fiber, with :math:`\mu` being the chemical potential
(note that a positive :math:`\mu` means it is favourable for nucleosomes to
bind to the fiber) and :math:`N` the total number of nucleosomes on the fiber.

More specifically, we define :math:`E_{\text{seq}}` via the normalized
methylation score :math:`M(x)`, which is given by

.. math::

   M(x) = \frac{\langle S_{\text{test}} \rangle_{\text{bin}}(x)-\langle S_{\text{unmeth}} \rangle_{\text{bin}}(x)}{\langle S_{\text{meth}} \rangle_{\text{bin}}(x)-\langle S_{\text{unmeth}} \rangle_{\text{bin}}(x)} \;,

where :math:`S_{\text{test}}` is the methylation score (defined for each bp)
for the test condition (e.g., chromatinized DNA), while :math:`S_{\text{meth}}`
and :math:`S_{\text{unmeth}}` are the methylation scores for the cases of a
fully methylated and unmethylated DNA fiber, respectively. 

Here, the operation :math:`\langle\cdot\rangle_{\text{bin}}` denotes that we
have first smoothed the signal by doing a rolling average with a bin size of
:math:`\ell_{\text{bin}}`. Typically, we take :math:`\ell_{\text{bin}} = 147`
bp, which is the typically length of DNA wrapped around a nucleosome [#f1]_. 

With a set of :math:`N` nucleosomes occupying positions
:math:`x_0, ..., x_{N-1}`, the energy associated with the methylation
footprinting data is

.. math::

   E_{\text{seq}} = k_BT\sum_{i=0}^{N-1}\widetilde{M}(x) \equiv k_BT\sum_{i=0}^{N-1} \left[M(x_i)-\langle M \rangle \right] \;,

where :math:`k_B` is the Boltzmann constant, :math:`T` is the temperature of
the system, and :math:`\langle M \rangle` is the averaged normalized
methylation score over all molecules sequenced for the test condition (i.e.,
an ensemble average).

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
