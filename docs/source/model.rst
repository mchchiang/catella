Nucleosome Occupancy Model
===========================

Introduction
--------------

In this document, we explore how to use methylation footprinting data to
infer nucleosome positioning via statistical modeling. The input of the
model is the methylation footprinting dataset, such as that from
fiber-seq, where we have a table of :math:`N_{\text{mol}}` molecules with
length :math:`L` base pair (bp), storing the confidence score in calling
individual bases methylated. The output of the model is a set of likely
nucleosome positions for each molecule based on the methylation data. We
use Bayesian inference and biophysical modeling to sample the nucleosome
positions by running Monte Carlo simulations.

Model
-------

We consider a DNA fiber of length :math:`L`, with position along the
fiber described by the discrete coordinate :math:`x \in
\{0,1,\dots,L-1\}`. Along the fiber, we place :math:`N` nucleosomes, each
leaving a footprint of :math:`\ell_{\text{nuc}} = 147` bp, and their
left-aligned positions form the configuration set

.. math::

   \mathcal{C} = \{x_1 < x_2 < \dots < x_N\} \;,

where :math:`|x_{i+1}-x_i| \geq \ell_{\text{nuc}}`.

Our idea is to start with an initial configuration :math:`\mathcal{C}_0`
(e.g., a set of randomly positioned nucleosomes) and use Bayesian
inference to "improve" the configuration given the methylation
footprinting data. In other words, we want to sample the *posterior*
probability :math:`P(\mathcal{C}|\text{data})` based on the *likelihood*
of observing the data given the current configuration
:math:`P(\text{data}|\mathcal{C})` and the *prior* probability of the
configuration :math:`P(\mathcal{C})`. By Bayes' theorem, we have

.. math::

   P(\mathcal{C}|\text{data}) =
   \frac{P(\text{data}|\mathcal{C})\,P(\mathcal{C})}{P(\text{data})} \;.

Here, :math:`P(\text{data})` is the *evidence* of observing the
methylation footprinting data. To make the connection with statistical
mechanics, we take the logarithm and exponentiate the right-hand side
(RHS) of the equation to rewrite the posterior as a Boltzmann
distribution:

.. math::

   P(\mathcal{C}|\text{data}) =
   \frac{1}{Z}\exp[-\beta H(\mathcal{C},\vec{\theta})] \;,

where :math:`\beta = 1/(k_BT)` is the inverse temperature, and
:math:`\vec{\theta} = (\theta_1,\theta_2,\dots)` is a vector of any
additional model parameters we use to link between the configuration and
methylation data. Moreover, one can identify the normalization factor
(i.e., partition function) :math:`Z` being proportional to the evidence
:math:`P(\text{data})` and the effective Hamiltonian taking the form

.. math::

   \beta H(\mathcal{C},\vec{\theta}) = -\left[\log
   P(\text{data}|\mathcal{C},\vec{\theta}) + \log P(\mathcal{C}) +
   \text{const.}\right] \;.

This link between Bayes' theorem and statistical mechanics provides the
foundation of our sampling framework. We seek to find an ensemble of
nucleosome configurations that satisfies the posterior, and we have just
demonstrated that this is equivalent to sampling a Boltzmann distribution
with an effective Hamiltonian, which can be done via Monte Carlo
simulations. We will now discuss in turn how we model the prior
:math:`P(\mathcal{C})` and the likelihood
:math:`P(\text{data}|\mathcal{C})`, before explaining in more detail the
sampling algorithm.

Prior probability -- the biophysical model of a nucleosome array
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The prior probability describes the chance of observing a particular
configuration assuming we have no knowledge of the experimental data. We
build this probability based on the biophysics and thermodynamics of how
nucleosomes interact with the DNA fiber. Since nucleosomes can bind and
be removed from the fiber, the natural statistical ensemble here is the
grand canonical ensemble, and so the prior is given by

.. math::

   P(\mathcal{C}) = \frac{1}{Z_0}\,\exp[-\beta H_0(\mathcal{C})] \;,

where :math:`Z_0 = \sum_{\mathcal{C}}\exp[-\beta H_0(\mathcal{C})]`, and
:math:`H_0` is the baseline Hamiltonian of the system and has two
contributions:

.. math::

   H_0(\mathcal{C}) = E^{(\text{rep})}(\mathcal{C}) - \mu N \;.

Here, the first term :math:`E^{(\text{rep})}(\mathcal{C})` represents the
steric repulsion between nucleosomes (i.e., their footprints on the DNA
cannot overlap), and the second term takes into account the changes in
the number of nucleosomes along the fiber, tuned by the chemical
potential :math:`\mu` (note that a positive :math:`\mu` means it is
favourable for nucleosomes to bind to the fiber). For the pairwise
repulsion between nucleosomes (say :math:`i` and :math:`j`), we consider
a hard core potential with a Weeks-Chandler-Andersen (WCA) drop-off to
model the linker length:

.. math::

   E^{(\text{rep})}(x_{ij}) =
   \begin{cases}
     \infty & \text{if } x_{ij} < \ell_{\text{nuc}} \\
     \epsilon_{\text{link}}\left[\left(\frac{\ell_{\text{link}}}
     {x_{ij}-\ell_{\text{nuc}}+1}\right)^6 - 1\right]^2 &
     \text{if } \ell_{\text{nuc}} \leq x_{ij} <
     \ell_{\text{nuc}}+\ell_{\text{link}} \\
     0 & \text{otherwise} \;,
   \end{cases}

where :math:`x_{ij} = |x_i-x_j|` is the separation between the
nucleosomes, :math:`\ell_{\text{nuc}} = 147` bp, :math:`\ell_{\text{link}}`
is the typical length of the linker DNA between nucleosomes, and
:math:`\epsilon_{\text{link}}` is an energy scale that sets the strength
of the repulsion. With this repulsion potential, only nearest-neighbour
interactions are effective and so

.. math::

   E^{(\text{rep})} = \sum_{i=1}^{N-1}E^{(\text{rep})}(x_{i+1},x_i)

for :math:`N \geq 2` and 0 otherwise.

Emission model data likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To determine the likelihood of observing the data for a particular
configuration, we need to build a model relating methylation calling and
nucleosome occupancy, while accounting for the fact that there could be
false positives in the calling. We note that not all sites are assayable
(i.e., some bases cannot be methylated), and those that are may still not
emit any signal by the base caller. We denote :math:`\mathcal{A}` as the
set of sites that are assayable and have emitted a call, and the model
only considers these sites. For each :math:`x \in \mathcal{A}` along the
fiber, there are three key observables:

.. math::

   \sigma_x \rightarrow m_x \rightarrow s_x \;.

Here, :math:`\sigma_x` is the nucleosome occupancy state, indicating
whether a nucleosome is covering the site (:math:`\sigma_x = 1`) or not
(:math:`\sigma_x = 0`). Next, :math:`m_x` is the methylation state,
representing whether the DNA methyltransferase (MTase) has marked the
site (:math:`m_x = 1`) or not (:math:`m_x = 0`), which depends on
:math:`\sigma_x`. Finally, :math:`s_x` is the actual signal (e.g., ionic
current from nanopore sequencing) output by the base caller based on the
methylation state :math:`m_x` of the site. Both :math:`\sigma_x` and
:math:`m_x` are hidden to us, and we only observe their impact via
:math:`s_x`.

With these observables, we can define the conditional
:math:`p(s_x|\sigma_x)` -- the probability of seeing the data signal
:math:`s_x` given the nucleosome state :math:`\sigma_x`. The overall data
likelihood is then the product of this conditional across all
:math:`x \in \mathcal{A}`, i.e.,

.. math::

   P(\text{data}|\mathcal{C},\vec{\theta}) = \prod_{x\in\mathcal{A}}
   p(s_x|\sigma_x(\mathcal{C})) \;.

In doing so, we have made the assumption that the methylation calling of
individual sites are independent. This is a simplification that we
revisit below, as MTase processivity correlates nearby events. We
introduce two model parameters :math:`\vec{\theta} = (\theta_P,\theta_A)`
to help us calculate the conditional :math:`p(s_x|\sigma_x)`. We let
:math:`\theta_P` be the probability that a site :math:`x \in \mathcal{A}`
is called methylated when protected (i.e., occupied by a nucleosome), and
:math:`\theta_A` the same when the site is accessible, with
:math:`\theta_A \gg \theta_P`. It should be noted that :math:`\theta_P`
is not exactly the false positive rate (FPR) as nucleosomes can "breathe"
and be partially unwrapped, and so :math:`\theta_P` is typically higher
than the FPR. Letting

.. math::

   \theta_{\sigma_x} = \sigma_x\theta_P + (1-\sigma_x)\theta_A \;,

we can now write down the conditional :math:`p(s_x|\sigma_x)` as follows:

.. math::

   p(s_x|\sigma_x) &= \sum_{m_x\in\{0,1\}}p(s_x|m_x)\,p(m_x|\sigma_x) \\
   &= p(s_x|m_x = 1)\,\theta_{\sigma_x} +
   p(s_x|m_x=0)\,(1-\theta_{\sigma_x}) \;.

Hence, the overall data likelihood can be written as

.. math::

   P(\text{data}|\mathcal{C},\vec{\theta}) = \prod_{x\in\mathcal{A}}
   \left[p(s_x|m_x = 1)\,\theta_{\sigma_x} +
   p(s_x|m_x=0)\,(1-\theta_{\sigma_x})\right] \;.

When sampling the posterior, it is more useful to compare the data
likelihood of a given configuration with that for a reference state,
which we take to be the null configuration :math:`\emptyset` where there
is no nucleosome occupying the fiber. In this null configuration, the
likelihood :math:`P(\text{data}|\emptyset,\vec{\theta})` is the same as
above but with :math:`\theta_{\sigma_x} = \theta_A`, as all sites are
accessible. Therefore, the ratio of the likelihoods becomes

.. math::

   \frac{P(\text{data}|\mathcal{C},\vec{\theta})}
   {P(\text{data}|\emptyset,\vec{\theta})} =
   \prod_{\substack{x\in\mathcal{A} \\ \sigma_x =1}}
   \frac{p(s_x|m_x = 1)\,\theta_P + p(s_x|m_x=0)\,(1-\theta_P)}
   {p(s_x|m_x = 1)\,\theta_A + p(s_x|m_x=0)\,(1-\theta_A)} \;,

where we have used the fact that the numerator cancels with the
denominator for sites that are accessible (:math:`\sigma_x = 0`).
Defining

.. math::

   L_x = \frac{p(s_x|m_x=1)}{p(s_x|m_x=0)} \;,

we can then rewrite the ratio as

.. math::

   \frac{P(\text{data}|\mathcal{C},\vec{\theta})}
   {P(\text{data}|\emptyset,\vec{\theta})} =
   \prod_{\substack{x\in\mathcal{A} \\ \sigma_x =1}}
   \frac{L_x\theta_P + 1-\theta_P}{L_x\theta_A + 1-\theta_A} \equiv
   \prod_{\substack{x\in\mathcal{A} \\ \sigma_x =1}}
   \exp(\lambda_x(\vec{\theta})) \;.

While both :math:`p(s_x|m_x=1)` and :math:`p(s_x|m_x=0)` in :math:`L_x`
are quantities we cannot directly measure, we do however know
:math:`q_x = p(m_x=1|s_x)`, which is the confidence score (ranging
between 0 and 1) the base caller outputs when calling a site methylated
given the raw signal it has seen (this is the `mod_qual` field when
extracting methylation data from a BAM file using ModKit). Assuming a
prior probability :math:`\pi_0 = p(m_x=1)` that the site :math:`x` is
methylated, we relate :math:`q_x` to the unknowns above using Bayes'
theorem:

.. math::

   q_x = p(m_x=1|s_x) = \frac{\pi_0\,p(s_x|m_x=1)}
   {\pi_0\,p(s_x|m_x=1) + (1-\pi_0)\,p(s_x|m_x=0)} \;.

Considering the expression for :math:`1-q_x` and after some
rearrangement, we find

.. math::

   L_x(q_x,\pi_0) = \frac{p(s_x|m_x=1)}{p(s_x|m_x=0)} =
   \frac{q_x}{1-q_x}\cdot\frac{1-\pi_0}{\pi_0}\;.

We note that the prior :math:`\pi_0` is based on the base caller's
training and is not directly reported. Lacking this information, we
adopt the default uninformative choice :math:`\pi_0 = 1/2`, giving
:math:`L_x = q_x/(1-q_x)` (this can however be adjusted in the model via
the `pi0` argument). Finally, putting everything together, we find the
logarithm of the likelihood ratio to be

.. math::

   \log\left(\frac{P(\text{data}|\mathcal{C},\vec{\theta})}
   {P(\text{data}|\emptyset,\vec{\theta})}\right) \equiv
   \Lambda(\mathcal{C},\vec{\theta}) =
   \sum_{\substack{x\in\mathcal{A} \\ \sigma_x =1}} \lambda_x(\vec{\theta})
   = \sum_{i=1}^{N} \lambda_i(\vec{\theta})\;,

where we have regrouped the sum to be done by nucleosome (rather than by
sites) and defined

.. math::

   \lambda_i(\vec{\theta}) = \sum_{\substack{x\in\mathcal{A} \\
   x \in [x_i,x_i+\ell_{\text{nuc}})}}
   \log\left(\frac{L_x(q_x,\pi_0)\,\theta_P + 1-\theta_P}
   {L_x(q_x,\pi_0)\,\theta_A + 1-\theta_A}\right) \;.

This is helpful as it shows that the log-likelihood ratio enters into the
Hamiltonian as a one-body interaction that can be pre-computed before
running the simulations, and the full Hamiltonian now reads as

.. math::

   \beta H(\mathcal{C},\vec{\theta}) = -\Lambda(\mathcal{C},\vec{\theta}) +
   \beta \left[E^{(\text{rep})}(\mathcal{C}) - \mu N\right] + \text{const.}
   \;,

where the constant at the end now includes the term
:math:`\log P(\text{data}|\emptyset,\vec{\theta})` that is irrelevant for
sampling. What remains to be done is estimating the parameters
:math:`\vec{\theta} = (\theta_P,\theta_A)` that we have introduced here.
This can be carried out via two approaches: (i) performing control
experiments on bare (non-chromatinized) DNA fibers and (ii) estimating
these parameters by maximizing the data likelihood. The main advantage of
(i) is that it allows us to account for sequence biases and introduce
spatial variation in the parameters :math:`\vec{\theta} =
\vec{\theta}(x)`, increasing the sensitivity of the model to the data.
However, control experiments are not practical for cases such as probing
nucleosome positioning genome-wide, where one would have to resort to
approach (ii). We now discuss these two approaches in turn.

Estimating model parameters with control experiments
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We perform two control experiments to estimate :math:`\vec{\theta}`:

(a) An experiment with a saturated level of methyltransferase (MTase) so
    that all assayable sites are methylated.
(b) An experiment without MTase so that all assayable sites remain
    unmethylated.

The first experiment allows us to estimate :math:`\theta_A` -- the
probability of an assayable site being accessible and called methylated.
This is because we have a bare DNA fiber where all bases are accessible,
and the assayable ones are all methylated due to the saturated level of
MTase. Let :math:`N_x` be the number of molecules at an assayable site
:math:`x` that have emitted a signal and :math:`q_{x,\alpha}^{(Y)}` be
the confidence score of the methylation calling at site :math:`x` for
molecule :math:`\alpha` in experiment :math:`Y \in \{a,b\}`. We write the
estimator of :math:`\theta_A` as

.. math::

   \hat{\theta}_A(x) &= \frac{\sum_{\alpha=1}^{N_{\text{mol}}}
   q_{x,\alpha}^{(a)} + \nu \left\langle\hat{\theta}_{A}(x)
   \right\rangle}{N_x + \nu} \;, \\
   &\quad\text{with}\quad
   \left\langle\hat{\theta}_A(x)\right\rangle =
   \frac{1}{N_{\text{site}}}\sum_{x\in\mathcal{A}}
   \sum_{\alpha=1}^{N_{\text{mol}}}\frac{q_{x,\alpha}^{(a)}}{N_x} \;,

where :math:`N_{\text{site}}` is the total number of assayable sites
along the fiber. We have introduced a regularization component in this
fraction so that when the number of molecules is low, :math:`\theta_A(x)`
tends towards its mean across all :math:`x\in\mathcal{A}`, i.e.,
:math:`\left\langle\theta_A(x)\right\rangle`. This tendency is controlled
by :math:`\nu`, the number of molecules present before variation in
:math:`\theta_A` across :math:`x` becomes favoured.

The second experiment tells us about the false positive rate (FPR).
Since there is no MTase present, none of the sites should have been
called methylated; yet, the base caller may still emit a call for some
sites by chance, producing false positives. The FPR can be estimated in
the same manner as for :math:`\theta_A`, leading to

.. math::

   \text{FPR}(x) &= \frac{\sum_{\alpha=1}^{N_{\text{mol}}}
   q_{x,\alpha}^{(b)} + \nu \left\langle\text{FPR}(x)\right\rangle}
   {N_x + \nu} \;, \\
   &\quad\text{with}\quad
   \left\langle\text{FPR}(x)\right\rangle =
   \frac{1}{N_{\text{site}}}\sum_{x\in\mathcal{A}}
   \sum_{\alpha=1}^{N_{\text{mol}}}\frac{q_{x,\alpha}^{(b)}}{N_x}\;.

The FPR is closely related to :math:`\theta_P`, the probability that an
assayable site protected by a nucleosome is called methylated. This is
because the setup of this experiment can be thought of as an
approximation of the condition where all sites are covered by
nucleosomes and thus cannot be methylated. Nevertheless, the two
conditions are not exactly equivalent as nucleosomes can breathe, and the
partial unwrapping allows some "leakage" for methylation. As such, we
expect :math:`\theta_P > \text{FPR}` and estimate the former as

.. math::

   \hat{\theta}_P(x) = \text{FPR}(x) + \rho\,
   [\hat{\theta}_A(x) - \text{FPR}(x)] \;,

where :math:`\rho` describes the level of leakage of methylation signal.
The default values used in the simulation framework for :math:`\nu` and
:math:`\rho` are 10 and 0.1, respectively.

Estimating model parameters by maximizing data likelihood
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As mentioned above, it is often not feasible to perform control
experiments. In this scenario, we try to determine a set of parameters
that maximize the data likelihood instead. To make progress, we divide
the DNA into non-overlapping bins, where each bin has the size of a
nucleosome :math:`\ell_{\text{nuc}}`. We choose non-overlapping bins
instead of rolling-window bins because the latter inflates the number of
observations and makes the estimated parameters look more certain than
they actually are. We pool together all molecules so that the total
number of bins :math:`N_{\text{bin}}` is the product of the number of
molecules :math:`N_{\text{mol}}` and the maximum number of bins per
molecule. Each bin :math:`w = 1,2,\dots,N_{\text{bin}}` is described by
three attributes: (i) the number of assayable sites with an emitted
signal :math:`M_w`, (ii) the "number" of which are likely to be
methylated :math:`k_w = \sum_{x\in\text{bin } w} q_x` (note that
:math:`k_w \in \mathbb{R}` and not :math:`\mathbb{N}`, since
:math:`q_x \in \mathbb{R}`), and (iii) the nucleosome occupancy state
:math:`z_w \in \{0,1\}` that indicates whether a nucleosome has covered
the bin (:math:`z_w = 1`) or not (:math:`z_w = 0`); like
:math:`\sigma_x`, :math:`z_w` is hidden and cannot be measured directly.

We model :math:`k_w` as drawn from a two-component mixture, protected
(:math:`z_w=1`, call rate :math:`\theta_P`) versus accessible
(:math:`z_w=0`, call rate :math:`\theta_A`), and fit the mixing weight
:math:`\pi = p(z_w=1)` together with :math:`\vec{\theta}=(\theta_P,
\theta_A)` by maximizing the resulting log-likelihood over all bins.
Because the occupancy state :math:`z_w` is hidden, this maximization
cannot be done in closed form; we use the standard
expectation-maximization (EM) algorithm to do so iteratively, and we do
not reproduce that derivation here. At convergence, the EM algorithm
yields the responsibility :math:`\gamma_w`, i.e., the posterior
probability that bin :math:`w` is protected given the fitted parameters,
from which the parameter estimates follow as weighted averages,

.. math::

   \hat{\pi} = \frac{1}{N_{\text{bin}}}\sum_w \gamma_w \;,\qquad
   \hat{\theta}_P = \frac{\sum_w \gamma_wk_w}{\sum_w \gamma_wM_w} \;,\qquad
   \hat{\theta}_A = \frac{\sum_w (1-\gamma_w)k_w}
   {\sum_w (1-\gamma_w)M_w} \;.

Because the EM process does not itself distinguish which of the two
mixture components is "protected", it can converge on a set of
parameters where :math:`\hat{\theta}_P > \hat{\theta}_A`; in that case,
we swap the values of the two estimators and set
:math:`\hat{\pi} \to 1-\hat{\pi}`. In the package, this fit is controlled
by `max_iters` and `tol` (the EM iteration cap and convergence
tolerance), `init_prot`/`init_acc` (its initial guesses for
:math:`\theta_P`/:math:`\theta_A`), and `n_min` (the minimum number of
context-eligible sites a bin needs to be used in the fit).

Handling data with multiple methyltransferase channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The theory presented thus far focuses on the methylation data from a
single type of methyltransferase (MTase). Nevertheless, it is readily
generalizable to multiple types of MTases. Indeed, experiments often use
more than one kind of MTase to improve signal coverage. The common
exogenous DNA MTases used are EcoGII [methyl-6-adenine (m6A)-specific],
M.CviPI [GpC-5-methyl-cytosine (5mGpC)-specific], and M.SssI
[CpG-5-methyl-cytosine (5mCpG)-specific]. To accommodate these different
MTases, we extend the model to have multiple assayable sets
:math:`\{\mathcal{A}_c\}`, where :math:`c\in\mathcal{M} =
\{\text{A},\text{HCG},\text{GCH},\text{GCG}\}` are the methylation
channels (H stands for any base except guanine). We need three separate
channels for GpC and CpG methylation because GCG are ambiguous sites that
can be targeted by both GpC and CpG MTases, and the kinetics there are
likely to be different from stand-alone GpC or CpG sites. Every reference
coordinate :math:`x` belongs to at most one assayable set and these
assayable sets do not overlap (i.e., :math:`\mathcal{A}_c \cap
\mathcal{A}_{c'} = \emptyset` for :math:`c\neq c'`), so GCG sites are not
double-counted in GCH or HCG channels.

Assuming that the methylation kinetics for these four channels are
independent, the log-likelihood ratio :math:`\Lambda` can be partitioned
by channel. Denoting :math:`\vec{\theta}_c = (\theta_{P,c},\theta_{A,c})`
as the model parameters for channel :math:`c \in \mathcal{M}`, the
log-likelihood ratio now reads as

.. math::

   \Lambda(\mathcal{C},\{\vec{\theta}_c\}) =
   \sum_i\sum_{c\in\mathcal{M}}\lambda_{i,c}(\vec{\theta}_c) \;,

where

.. math::

   \lambda_{i,c}(\vec{\theta}_c) = \sum_{\substack{x\in\mathcal{A}_c \\
   x \in [x_i,x_i+\ell_{\text{nuc}})}}
   \log\left(\frac{L_x(q_{x,c},\pi_{0,c})\,\theta_{P,c} + 1-\theta_{P,c}}
   {L_x(q_{x,c},\pi_{0,c})\,\theta_{A,c} + 1-\theta_{A,c}}\right) \;,

with :math:`q_{x,c}` being the confidence score in calling a site
:math:`x \in \mathcal{A}_c` methylated in channel :math:`c` and
:math:`\pi_{0,c}` the prior that :math:`x` is methylated in that channel
(for simplicity, the package sets :math:`\pi_{0,c} = \pi_0\,\forall c`).
Estimating the parameters :math:`\vec{\theta}_c` for each channel can be
done in the same way as before, both for the case with control
experiments and the case without.

We should also be aware of the following details when considering
different MTase channels. First, with *in vivo* data, both the HCG and
GCG channels are contaminated by endogenous methylation at CpG (i.e.,
5mCpG), so they should not be used as a readout for accessibility.
Second, if an MTase channel is unused in a given experiment,
:math:`\theta_{A,c} \approx \theta_{P,c}` and :math:`\lambda_{x,c}
\approx 0`, so the theory automatically excludes its contribution to
updating the posterior. Finally, we evaluate methylation contexts in a
strand-specific manner, and both strands contribute. In practice, the
assayable sets :math:`\{\mathcal{A}_c\}` and the confidence scores
:math:`q_{x,c}` are taken from ModKit's output, filtering out
canonical-call rows (`mod_code` of '-') so that only sites with an
emitted modification call populate :math:`\mathcal{A}_c`. ModKit
searches both strands and reports the strand for each call, so sites
on either strand contribute to their respective channels without
further processing.

Accounting for correlation in methyltransferase activity between proximal sites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Throughout this section we have made the assumption that the methylation
activity at individual assayable sites is independent. This is clearly
an oversimplification, as nearby sites can experience crosstalk. For
instance, CpG and GpC sites are palindromic, so the methylation signal at
the cytosine on one strand will be highly correlated with that on the
opposite strand just one base apart. This oversampling inflates the
log-likelihood ratio, and we correct for this heuristically by
introducing a weighting factor :math:`\eta`, i.e.,

.. math::

   \Lambda(\mathcal{C},\{\vec{\theta}_c\}) \to
   \eta\,\Lambda(\mathcal{C},\{\vec{\theta}_c\}) \;,

and one can think of :math:`\eta` as a dial that controls the importance
of the data relative to our prior model. By default, the package sets
:math:`\eta = 1`.

While :math:`\eta=1` is a reasonable default in the absence of further
information, it can also be estimated directly from the data by
quantifying how strongly :math:`\lambda_x` is autocorrelated along the
fiber. If assayable sites were independent, :math:`\text{Var}(\Lambda)`
would simply be :math:`\sum_{x\in\mathcal A}\text{Var}(\lambda_x)`;
correlation between nearby sites inflates this variance, and
:math:`\eta` is chosen to undo that inflation. Treating
:math:`\{\lambda_x\}_{x\in\mathcal A}` as an (approximately) stationary
sequence along the fiber with common variance
:math:`\sigma^2=\text{Var}(\lambda_x)`, the variance of the sum is

.. math::

   \text{Var}(\Lambda) = \sum_{x,x'\in\mathcal A}
   \text{Cov}(\lambda_x,\lambda_{x'}) = n\sigma^2\left[1+2\sum_{k=1}^{n-1}
   \left(1-\frac{k}{n}\right)\rho(k)\right] \;,

where :math:`n=|\mathcal A|` is the number of assayable, called sites,
and

.. math::

   \rho(k) = \frac{\displaystyle\sum_{\substack{x,x+k\in\mathcal A}}
   (\lambda_x-\bar\lambda)(\lambda_{x+k}-\bar\lambda)}
   {\displaystyle\sum_{x\in\mathcal A}(\lambda_x-\bar\lambda)^2}

is the (sample) lag-:math:`k` autocorrelation of :math:`\lambda_x`, with
:math:`\bar\lambda = n^{-1}\sum_{x\in\mathcal A}\lambda_x` and the
numerator summed over pairs of assayable sites exactly :math:`k` bp
apart. The bracketed factor above is the familiar variance-inflation
factor from correlated sampling, and it reduces the effective number of
independent sites from :math:`n` to :math:`n_\text{eff}=n/V`, with
:math:`V` the bracketed term. Choosing

.. math::

   \eta = \frac1V = \left[1+2\sum_{k=1}^{n-1}\left(1-\frac{k}{n}\right)
   \rho(k)\right]^{-1}

therefore rescales :math:`\Lambda` so that its variance matches that of a
sum over :math:`n_\text{eff}` independent sites, correcting for the
oversampling described above. In practice, :math:`\rho(k)` is estimated
only up to a small cutoff lag (`eta_max_lag`) beyond which it is
indistinguishable from sampling noise, since crosstalk is expected to
decay quickly with separation (e.g., the palindromic CpG/GpC correlation
above spans just :math:`k=1` bp). When multiple MTase channels are in
use, the same estimator is applied separately to each channel's sequence
:math:`\lambda_{x,c}`, giving a channel-specific :math:`\eta_c` (by
default, estimated from the methylated control when available, or from
the test data otherwise; a value can also be pinned per channel via the
`eta` argument).

An empirical alternative to the likelihood
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide here an alternative approach to estimate the likelihood that
does not involve statistical modeling, which may not be suitable for
conditions where single-molecule footprinting data are not available.
One such scenario is when there are only bulk accessibility data, such as
those based on MNase-seq.

For a molecule :math:`\alpha`, we start from the raw footprinting signal
:math:`S_{x,\alpha}` (e.g., the confidence score :math:`q_{x,\alpha}`
used above) and smooth it with a left-aligned rolling average of bin
size :math:`\ell_\text{bin}`:

.. math::

   \widetilde S_{x,\alpha} = \frac{1}{\ell_\text{bin}}
   \sum_{i=0}^{\ell_\text{bin}-1} S_{x+i,\alpha} \;,

so that :math:`\widetilde S_{x,\alpha}` depends only on sites
:math:`x,x+1,\dots,x+\ell_\text{bin}-1`; we typically take
:math:`\ell_\text{bin}=\ell_{\text{nuc}}=147` bp [#f1]_.

Averaging :math:`\widetilde S_{x,\alpha}` over all molecules of a given
experimental condition :math:`Y\in\{\text{test},\text{meth},
\text{unmeth}\}` (cf. the saturated-MTase and no-MTase control
experiments above) gives :math:`\widetilde S_x^{(Y)} =
\big(N_\text{mol}^{(Y)}\big)^{-1}\sum_\alpha \widetilde
S_{x,\alpha}^{(Y)}`. When both controls are available, we normalize the
(molecule-level) test signal relative to them,

.. math::

   M_{x,\alpha} = \frac{\widetilde S_{x,\alpha}^{(\text{test})} -
   \widetilde S_x^{(\text{unmeth})}}{\widetilde S_x^{(\text{meth})} -
   \widetilde S_x^{(\text{unmeth})}} \;,

so that :math:`M_{x,\alpha}` tends towards 0 where the test molecule
resembles the unmethylated (protected) control and towards 1 where it
resembles the fully methylated (accessible) control. If no control
experiments are available, we instead set :math:`M_{x,\alpha} =
\widetilde S_{x,\alpha}^{(\text{test})}`.

To convert :math:`M_{x,\alpha}` into a formal probability
:math:`p^{(\text{meth})}_{x,\alpha}\in[0,1]` that site :math:`x` is
methylated (i.e., accessible) on molecule :math:`\alpha`, we linearly
rescale :math:`M_{x,\alpha}` against two percentile bounds, clipping
values that fall outside of them:

.. math::

   p^{(\text{meth})}_{x,\alpha} = \text{clip}\!\left[
   \frac{M_{x,\alpha} - M^{(\text{unmeth})}_{p=0.1\%}}
   {M^{(\text{meth})}_{p=99.9\%} - M^{(\text{unmeth})}_{p=0.1\%}},
   \,0,\,1\right] \;,

where :math:`M^{(\text{unmeth})}_{p=0.1\%}` is the 0.1th percentile and
:math:`M^{(\text{meth})}_{p=99.9\%}` the 99.9th percentile (the default
choice used in the package) of the normalized score :math:`M_{x,\alpha}`,
each estimated once per fiber, but deliberately from different sources:
the lower bound :math:`M^{(\text{unmeth})}_{p=0.1\%}` is taken over a
random subsample of molecules from the unmeth control channel alone, and
the upper bound :math:`M^{(\text{meth})}_{p=99.9\%}` over a random
subsample from the meth control channel alone, so that each bound is
anchored to the condition it is meant to represent rather than to the
(possibly differently distributed) test signal. Both percentiles are
pooled over all sites :math:`x` within their respective sample rather
than estimated separately at each site. When no controls are available,
both bounds are instead estimated from the test channel itself, i.e.,
:math:`M^{(\text{unmeth})}_{p=0.1\%}\to M^{(\text{test})}_{p=0.1\%}` and
:math:`M^{(\text{meth})}_{p=99.9\%}\to M^{(\text{test})}_{p=99.9\%}`.

As above, we identify the complementary probability
:math:`p^{(\text{seq})}_{x,\alpha} = 1-p^{(\text{meth})}_{x,\alpha}` as
the (empirical) probability that :math:`x` is occupied by a nucleosome on
molecule :math:`\alpha`. To turn :math:`p^{(\text{seq})}` into an energy,
we treat the site as a two-state system -- occupied, with probability
:math:`p^{(\text{seq})}`, versus accessible, with probability
:math:`1-p^{(\text{seq})}=p^{(\text{meth})}` -- and take their relative
Boltzmann weight to define an energy difference :math:`E^{(\text{seq})}`
(in units of :math:`k_BT`, i.e., setting :math:`\beta=1`):

.. math::

   \frac{p^{(\text{seq})}_{x,\alpha}}{1-p^{(\text{seq})}_{x,\alpha}} =
   \exp[-E^{(\text{seq})}_{x,\alpha}] \;,

having set the accessible state as the reference (zero-energy) state.
Inverting this, and clipping the result to a maximum magnitude
:math:`E_\text{max}` to prevent it from diverging as
:math:`p^{(\text{meth})}\to\{0,1\}`, we arrive at

.. math::

   E^{(\text{seq})}_{x,\alpha} &= \text{clip}\!\left[ -\log
   \frac{p^{(\text{seq})}_{x,\alpha}}{1-p^{(\text{seq})}_{x,\alpha}},
   \,-E_\text{max},\,E_\text{max}\right] \\
   &= \text{clip}\!\left[\log
   \frac{p^{(\text{meth})}_{x,\alpha}} {1-p^{(\text{meth})}_{x,\alpha}},
   \,-E_\text{max},\,E_\text{max}\right] \;,

i.e., :math:`E^{(\text{seq})}` is simply the (clipped) log-odds, or
logit, of :math:`p^{(\text{meth})}`. This differs from a naive Boltzmann
inversion :math:`-k_BT\log p^{(\text{seq})}_x` in that it is referenced
against the complementary (accessible) state rather than against a
hypothetical zero-energy vacuum, which keeps :math:`E^{(\text{seq})}`
finite and antisymmetric about
:math:`p^{(\text{meth})}=p^{(\text{seq})}=1/2` (where it is favourable
neither to bind nor not to bind); the two forms would coincide only in
the limit :math:`p^{(\text{meth})}\ll1`. Like the statistical model
above, this construction is applied independently to each molecule
:math:`\alpha`, so every molecule is assigned its own empirical energy
landscape :math:`E^{(\text{seq})}_{x,\alpha}` prior to running its Monte
Carlo simulation.

The full Hamiltonian
~~~~~~~~~~~~~~~~~~~~~~

We can now combine the prior above with either likelihood model
discussed above (the statistical, emission-based likelihood, or the
empirical alternative) into a single effective Hamiltonian for use in
the Monte Carlo sampling. Recalling that the posterior takes the
Boltzmann form :math:`P(\mathcal C|\text{data})\propto
\exp[-\beta H(\mathcal C)]`, and that the repulsion and
chemical-potential terms of the prior enter :math:`H` unchanged, we
write

.. math::

   H(\mathcal C) = E^{(\text{data})}(\mathcal C) +
   E^{(\text{rep})}(\mathcal C) - \mu N \;,

where :math:`E^{(\text{data})}(\mathcal C)` depends on which likelihood model
is used:

.. math::

   E^{(\text{data})}(\mathcal C) =
   \begin{cases}
     \displaystyle -k_BT\,\Lambda(\mathcal C,\vec\theta) =
     -k_BT\sum_{i=1}^N \lambda_i(\vec\theta) &
     \text{(statistical model)} \;,\\[2ex]
     \displaystyle \sum_{i=1}^N E^{(\text{seq})}_{x_i,\alpha} &
     \text{(empirical alternative)} \;.
   \end{cases}

In both cases, :math:`E^{(\text{data})}` is a sum of one-body contributions,
one per nucleosome, that depend only on that nucleosome's own position
:math:`x_i` and can therefore be precomputed for a given molecule before
the simulation starts: in the statistical model, the contribution
:math:`\lambda_i(\vec\theta)` sums the site-level log-odds over the
nucleosome's entire footprint :math:`[x_i,x_i+\ell_{\text{nuc}})`,
whereas in the empirical alternative, the contribution
:math:`E^{(\text{seq})}_{x_i,\alpha}` is read off directly from the
smoothed and normalized signal at the nucleosome's left edge
:math:`x_i` alone.

In fact, both routes only ever enter the Hamiltonian by way of a
methylation probability :math:`p^{(\text{meth})}_{x,\alpha}`, which is
then turned into an energy by the identical clipped log-odds transform
above -- what differs between the two likelihood models is only how
:math:`p^{(\text{meth})}_{x,\alpha}` itself is obtained. In the empirical
alternative, :math:`p^{(\text{meth})}_{x,\alpha}` is the smoothed,
control-normalized, percentile-clipped score above. In the statistical
model, we instead read it off the emission model: generalizing the
one-body term :math:`\lambda_i(\vec\theta)` -- there defined only at an
existing nucleosome's index :math:`i`, summing the site-level log-odds
over its footprint :math:`[x_i,x_i+\ell_{\text{nuc}})` -- to an arbitrary
candidate position :math:`x`,

.. math::

   \lambda(x;\vec\theta) = \sum_{\substack{x'\in\mathcal{A} \\
   x' \in [x,x+\ell_{\text{nuc}})}}
   \log\left(\frac{L_{x'}(q_{x'},\pi_0)\,\theta_P + 1-\theta_P}
   {L_{x'}(q_{x'},\pi_0)\,\theta_A + 1-\theta_A}\right) \;,

so that :math:`\lambda_i(\vec\theta) \equiv \lambda(x_i;\vec\theta)`, we
take

.. math::

   p^{(\text{meth})}_{x,\alpha} = \frac{1}{1+\exp[\lambda(x;\vec\theta)]}
   \;,

i.e., the emission model's log-odds ratio :math:`\lambda(x;\vec\theta)`
is itself treated as a log-odds and inverted into a probability, exactly
as :math:`M_{x,\alpha}` is rescaled into a probability in the empirical
route. Substituting this into the clipped log-odds transform above and
using :math:`\log[p/(1-p)] = -\lambda(x;\vec\theta)` for
:math:`p=p^{(\text{meth})}_{x,\alpha}`, we find

.. math::

   E^{(\text{seq})}_{x,\alpha} = \text{clip}\!\left[-\lambda(x;\vec\theta),
   \,-E_\text{max},\,E_\text{max}\right] \qquad
   \text{(statistical model)} \;.

Comparing this with the Hamiltonian :math:`\beta H =
-\Lambda(\mathcal C,\vec\theta)+\beta[E^{(\text{rep})}(\mathcal C)-\mu N]+
\text{const.}` above, we see that using the emission-model likelihood is,
up to the additive constant already noted there, the same as taking
:math:`E^{(\text{data})}(\mathcal C) = -k_BT\,\Lambda(\mathcal C,\vec\theta) =
-k_BT\sum_{i=1}^N\lambda_i(\vec\theta)` -- except that each one-body term
is now clipped to :math:`\pm E_\text{max}` before being summed, precisely
as in the empirical alternative. This detail was left implicit above:
without it, an occasional site with an extreme log-odds ratio
:math:`\lambda(x;\vec\theta)` (e.g., from a run of highly confident
calls) could otherwise dominate the Hamiltonian and destabilize the
sampling, so both likelihood routes cap their one-body contribution the
same way before it enters :math:`E^{(\text{data})}(\mathcal C)`.

In both cases, :math:`E^{(\text{data})}` is therefore a sum of
identically-structured one-body contributions, one per nucleosome, that
depend only on that nucleosome's own position :math:`x_i` (and the
choice of likelihood model used to obtain :math:`p^{(\text{meth})}_{x,
\alpha}` there) and can be precomputed for a given molecule before the
simulation starts. This is the Hamiltonian we sample in the Monte Carlo
simulations described next.

Sampling the posterior
-----------------------

Having assembled the full Hamiltonian :math:`H(\mathcal C)` above, we now
turn to how nucleosome configurations are sampled from the posterior.
Since we allow the number of nucleosomes :math:`N` to vary, the natural
statistical ensemble is the **grand canonical ensemble**, in which, at
thermal equilibrium, a configuration :math:`\mathcal C` is observed with
probability

.. math::

   P(\mathcal C) = \frac1Z \exp[-\beta H(\mathcal C)] \;,

:math:`Z` being the partition function and :math:`\beta=1/(k_BT)`. We
discuss two approaches for sampling this distribution: (1) running a
Markov Chain Monte Carlo procedure, or (2) using the transfer matrix
method to compute :math:`Z` exactly and derive the relevant marginal
statistics directly, without drawing any samples at all.

Markov Chain Monte Carlo
~~~~~~~~~~~~~~~~~~~~~~~~~

We sample this distribution with a Markov Chain Monte Carlo (MCMC)
procedure, performing one of the following three moves at each step.

1. **Shift a nucleosome.** We randomly select the :math:`i`-th
   nucleosome and propose to move it by one bp, :math:`x_i\to
   x_i'=x_i\pm1`. The move is accepted with probability

   .. math::

      p(x_i\to x_i') = \min\Big\{1, \exp[-\beta(\Delta E^{(\text{data})}+
      \Delta E^{(\text{rep})})]\Big\} \;,

   where :math:`\Delta E^{(\text{data})}` and
   :math:`\Delta E^{(\text{rep})}` are the changes in the data and
   repulsion energies caused by the move (the number of nucleosomes,
   and hence the :math:`-\mu N` term, is unchanged).

2. **Add a nucleosome.** We randomly select a candidate position
   :math:`x` from the :math:`n_\text{pos}=L-\ell_{\text{nuc}}+1` possible
   (left-aligned) positions on the fiber [#f2]_. The move is accepted
   with probability

   .. math::

      p(\text{add at }x) = \min\left\{1,\frac{n_\text{pos}}{N+1}
      \exp[\beta\mu-\beta\Delta E^{(\text{add})}(x)]\right\} \;,

   where :math:`\Delta E^{(\text{add})}(x) = E^{(\text{data})}(x)+
   \Delta E^{(\text{rep})}` is the change in the data and repulsion energies
   from placing a nucleosome at :math:`x`, and the prefactor
   :math:`n_\text{pos}/(N+1)` corrects for the asymmetry between
   proposing a new position out of :math:`n_\text{pos}` choices and
   proposing its removal out of the resulting :math:`N+1` nucleosomes.

3. **Remove a nucleosome.** We randomly select the :math:`i`-th
   nucleosome (out of :math:`N`) for removal. The move is accepted with
   probability

   .. math::

      p(\text{remove }i) = \min\left\{1,\frac{N}{n_\text{pos}}
      \exp[-\beta\mu-\beta\Delta E^{(\text{remove})}(x_i)]\right\} \;,

   where :math:`\Delta E^{(\text{remove})}(x_i) = -E^{(\text{data})}(x_i) +
   \Delta E^{(\text{rep})}` is the change in the data and repulsion energies
   from removing the nucleosome at :math:`x_i`.

These moves satisfy detailed balance with respect to :math:`P(\mathcal
C)` and together define a **Metropolis algorithm** that drives the
ensemble of nucleosome configurations towards thermal equilibrium. In
practice, we run the chain under a temperature annealing schedule, with
:math:`\beta^{-1}` decreasing from a starting value to a final target
value -- linearly, geometrically, or held constant -- over the course of
the simulation, which helps the chain avoid metastable configurations
before it settles into equilibrium at the target temperature.

Transfer matrix method
~~~~~~~~~~~~~~~~~~~~~~~

Due to the unique structure of the Hamiltonian, the transfer matrix
method provides an alternative to MCMC for obtaining marginal statistics
of the posterior, by computing the partition function :math:`Z` directly
rather than sampling from it. Writing the Hamiltonian explicitly in
terms of its one-body and nearest-neighbour pairwise terms,

.. math::

   H(\mathcal C) = \sum_{i=1}^N E^{(\text{data})}(x_i) +
   \sum_{i=1}^{N-1} E^{(\text{rep})}(x_i,x_{i+1}) - \mu N \;,

with :math:`E^{(\text{rep})}(x_i,x_{i+1})` the pairwise repulsion
recalled above, the brute-force approach to computing :math:`Z` is to
enumerate every legal configuration and sum its Boltzmann weight
directly,

.. math::

   Z = \sum_{N=0}^\infty \sum_{\substack{0\leq x_1 < x_2 < \cdots <
   x_N < n_\text{pos} \\ x_{i+1}-x_i\geq\ell_{\text{nuc}}\,\forall i}}
   \exp\Big\{-\beta\Big[\sum_{i=1}^N E^{(\text{data})}(x_i) +
   \sum_{i=1}^{N-1}E^{(\text{rep})}(x_i,x_{i+1}) - \mu N\Big]\Big\} \;.

This sum ranges over an exponentially large collection of increasing
subsequences [a combinatorial object whose size grows like
:math:`\binom{n_\text{pos}}{N}`, summed over :math:`N`], which is
infeasible to enumerate for a fiber of any realistic length. MCMC
sidesteps this difficulty by sampling rather than summing.

The restrictive, nearest-neighbour structure of the Hamiltonian
simplifies the problem instead. Every term in the exponent is either a
function of a single :math:`x_i` (:math:`E^{(\text{data})}`) or of a
single adjacent pair (:math:`E^{(\text{rep})}`), with nothing coupling
:math:`x_i` to :math:`x_j` for :math:`|i-j|\geq2`. The summand above
therefore factorizes into a chain of local factors, one per nucleosome,
with each depending only on that nucleosome and its immediate
predecessor to the left,

.. math::

   \exp[-\beta H(\mathcal C)] = \prod_{i=2}^N g(x_i|x_{i-1})\cdot
   g(x_1) \;,

where

.. math::

   \begin{split}
   g(x_i|x_{i-1}) &= \exp\big\{-\beta\big[E^{(\text{data})}(x_i) +
   E^{(\text{rep})}(x_{i-1},x_i) - \mu\big]\big\} \;, \\
   g(x_1) &= \exp\{-\beta[E^{(\text{data})}(x_1)-\mu]\} \;.
   \end{split}

This factorization mirrors that of the 1D Ising model (or any 1D model
with only nearest-neighbour interactions), and a factorizable summand is
exactly the condition under which the transfer matrix method applies.
Unlike spins in the Ising model, however, the number of nucleosomes is
not conserved and the number of factors varies with :math:`N`; the
system is also heterogeneous, since individual factors differ through
the data term. Both of these make the transfer matrix method non-trivial
to carry out here.

Memory state and the transfer matrix
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To simplify the summing process, we factorize :math:`Z` in units that
remain constant (with no dependence on :math:`N`): the possible start
sites for placing nucleosomes, whose total :math:`n_\text{pos} =
L-\ell_{\text{nuc}}+1` is fixed. At each site, one faces a binary
decision: place a new nucleosome there or not. Suppose this has already
been decided for sites :math:`0` to :math:`x-1`; the outcome at
:math:`x` then rests on (i) the distance from the most recent nucleosome
on the left, and (ii) the data-derived energy :math:`E^{(\text{data})}
(x)` for adding a nucleosome at :math:`x`. For (i), it is forbidden to
add a nucleosome if that distance is smaller than
:math:`\ell_{\text{nuc}}`. Crucially, the interaction
:math:`E^{(\text{rep})}` with the previous nucleosome has a *finite*
range :math:`W\equiv\ell_{\text{nuc}}+\ell_{\text{link}}`: at and beyond
:math:`W`, :math:`E^{(\text{rep})}=0` regardless of the exact distance.
This lets us perform the sum by tracking only the (capped) distance from
the previous nucleosome (located at :math:`y`) as we step along the
fiber, via a memory state

.. math::

   k_x = \min(x-y,W) \;, \qquad k_x\in\{1,\dots,W\} \;.

The state :math:`k_x` is a sufficient statistic for the configurational
history to the left of :math:`x`: whether a nucleosome can be added at
:math:`x`, and the cost :math:`E^{(\text{rep})}` of doing so, depend on
the previous nucleosome only via :math:`k_x`. Any two partial
configurations with the same :math:`k_x` therefore have identical
possible futures, and their partial Boltzmann weights can be merged
into a single number -- this Markov property is what lets the sum over
exponentially many configurations collapse into :math:`W` running
totals. We define the *forward* vector :math:`\vec v(x) =
(v_1(x),\dots,v_W(x))^\mathsf T`, representing the partial sums of the
partition function up to position :math:`x`, with

.. math::

   \begin{split}
   v_k(x) &= \sum_{\substack{\mathcal C'\subset\{0,\dots,x-1\}\,
   \text{valid} \\ k_x(\mathcal C')=k}} \\
   &\quad \times \exp\Big\{-\beta\Big[\sum_i E^{(\text{data})}(x_i) +
   \sum_i E^{(\text{rep})}(x_i,x_{i+1}) - \mu|\mathcal C'|\Big]\Big\} \;,
   \end{split}

where :math:`\mathcal C'` ranges over valid configurations of
nucleosome starting positions that give state :math:`k` at :math:`x`.
To reach the full sum :math:`Z`, we need rules for obtaining
:math:`\vec v(x+1)` from :math:`\vec v(x)`; there are two "summing
moves" we can perform at :math:`x`:

1. **Skip.** No new nucleosome is added. This shifts the memory state
   from :math:`k` to :math:`\min(k+1,W)` for the next step at
   :math:`x+1`; note that :math:`k=W` is a recurring state, since
   :math:`E^{(\text{rep})}=0` there.
2. **Add.** A new nucleosome is placed at :math:`x`. This is only
   possible if :math:`k\geq\ell_{\text{nuc}}`, and it adds a new
   Boltzmann factor :math:`w(x,k)` (depending on :math:`E^{(\text{rep})}`
   and :math:`E^{(\text{data})}`) to the sum, resetting the memory state
   to :math:`k=1` at :math:`x+1`.

These moves define the transfer matrix :math:`\vec T(x)` (of dimension
:math:`W\times W`) that advances :math:`\vec v(x)` to
:math:`\vec v(x+1)`,

.. math::

   \vec v(x+1) = \vec T(x)\cdot\vec v(x) \;, \qquad
   x=0,\dots,n_\text{pos}-1 \;,

with matrix elements

.. math::

   T_{k'k}(x) = \begin{cases}
     \delta_{k',\min(k+1,W)} & \text{if } 1\leq k<\ell_{\text{nuc}} \\
     \delta_{k',\min(k+1,W)}+\delta_{k',1}\,w(x,k) &
     \text{if } \ell_{\text{nuc}}\leq k\leq W \;,
   \end{cases}

where :math:`\delta_{x,y}` is the Kronecker delta and

.. math::

   w(x,k) = \exp\{-\beta[E^{(\text{data})}(x)+E^{(\text{rep})}(k)-\mu]\}
   \;.

As an example, take :math:`\ell_{\text{nuc}}=3` and
:math:`\ell_{\text{link}}=2`, so :math:`W=5` and
:math:`k\in\{1,2,3,4,5\}`. Here, :math:`k=1,2` represent the excluded
state (no new nucleosome can be added), :math:`k=3,4` the interacting
state (a nucleosome can be added but linker repulsion still applies),
and :math:`k=5` the recurring state (no repulsion). The transfer matrix
takes the form

.. math::

   T(x) = \begin{pmatrix}
     0 & 0 & w(x,3) & w(x,4) & w(x,5) \\
     1 & 0 & 0 & 0 & 0 \\
     0 & 1 & 0 & 0 & 0 \\
     0 & 0 & 1 & 0 & 0 \\
     0 & 0 & 0 & 1 & 1
   \end{pmatrix} \;.

The first row collects every way of placing a nucleosome at :math:`x`
(from any state that allows it); the subdiagonal 1's advance the
current state :math:`k` by one when nothing is placed; and the
trailing 1 in the bottom-right entry is the recurring state's
self-loop. The matrix is sparse, with at most two nonzero entries per
row except the first, which collects all possible additions -- so
applying :math:`\vec T` to :math:`\vec v` costs :math:`O(W)` rather
than :math:`O(W^2)`.

To start the sum, we need an initial state :math:`\vec v(0)`. At the
left of the fiber, where no nucleosome could have been placed
beforehand, we must be in state :math:`k=W` (with
:math:`E^{(\text{rep})}=0`), so :math:`\vec v(0)=\vec e_W=
(0,\dots,1)^\mathsf T`. Since :math:`\vec v(n_\text{pos})` contains the
total weights for all configurations ending in each memory state, the
partition function is simply the sum over these weights,

.. math::

   \begin{split}
   Z &= \sum_{k=1}^W v_k(n_\text{pos}) \\
   &= \mathbf 1\cdot\vec T(n_\text{pos}-1)\cdot\vec T(n_\text{pos}-2)
   \cdots\vec T(1)\cdot\vec T(0)\cdot\vec e_W \;, \qquad
   \mathbf 1=(1,\dots,1)^\mathsf T \;.
   \end{split}

This expression is the basis of how :math:`Z` is computed in the
package, needed for the marginals derived below. In practice, the sum
is performed via dynamic programming with dynamic rescaling to ensure
numerical stability, since multiplying many exponential factors
together is susceptible to underflow or overflow:

1. Compute the raw product :math:`\vec v(x+1) = \vec T(x)\cdot
   \tilde{\vec v}(x)` [:math:`\vec v(0)=\tilde{\vec v}(0)` is already
   normalized].
2. Rescale the product by a normalization factor :math:`c(x+1) =
   \sum_k v_k(x+1)`, i.e., :math:`\tilde{\vec v}(x+1) =
   \vec v(x+1)/c(x+1)`.
3. Add :math:`\log c(x+1)` to an accumulated sum (started with
   :math:`\log c(0)=0`), and repeat step (1) for :math:`\vec v(x+2)`
   using the rescaled factor :math:`\tilde{\vec v}(x+1)`.

We stop once :math:`x=n_\text{pos}` is reached, obtaining the last
rescaled factor :math:`c(n_\text{pos})`, and the logarithm of the
partition function is then

.. math::

   \log Z = \sum_{x=0}^{n_\text{pos}} \log c(x) \;.

Obtaining marginals: occupancy profile and mean number of nucleosomes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With :math:`Z` (or its logarithm) computed, we can derive marginal
statistics, such as the nucleosome occupancy profile, without running
any simulations. We introduce the *backward* vector :math:`\vec u(x) =
(u_1(x),\dots,u_W(x))^\mathsf T`, with :math:`u_k(x)` the partial
Boltzmann weight of all valid completions of the fiber from :math:`x`
onward given memory state :math:`k` at :math:`x`. By this definition,
:math:`\vec v(x)` and :math:`\vec u(x)` obey

.. math::

   Z = \vec u(x)\cdot\vec v(x) \;, \qquad x = 0,\dots,n_\text{pos} \;.

Applying this at :math:`x` and :math:`x+1` and using
:math:`\vec v(x+1) = \vec T(x)\cdot\vec v(x)` gives
:math:`\vec u(x)\cdot\vec v(x) = \vec u(x+1)\cdot\vec T(x)\cdot
\vec v(x)`. Because :math:`\vec u(x)` depends only on the fiber from
:math:`x` onward and not on the configuration to the left, this holds
for any :math:`\vec v(x)` (and so component-wise), giving the backward
recursion

.. math::

   \vec u(x)^\mathsf T = \vec u(x+1)\cdot\vec T(x) \qquad\text{or}
   \qquad \vec u(x) = \vec T(x)^\mathsf T\cdot\vec u(x+1) \;,

started at :math:`x=n_\text{pos}` with :math:`\vec u(n_\text{pos})=
\mathbf 1` (since :math:`Z=\vec u(n_\text{pos})\cdot\vec v(n_\text{pos})
= \mathbf 1\cdot\vec v(n_\text{pos})`).

With the backward vector defined, we can compute the probability that a
nucleosome starts at :math:`x`. For this, the memory state at :math:`x`
must lie in :math:`k\in[\ell_{\text{nuc}},W]` (otherwise addition is
prohibited), and the state at :math:`x+1` resets to :math:`k=1`, so
that

.. math::

   P(\text{start at }x) = \frac{u_1(x+1)}{Z}
   \sum_{k=\ell_{\text{nuc}}}^W w(x,k)\,v_k(x) \;,

weighting each possible start state by its Boltzmann factor :math:`w`.
An alternative derivation treats the chemical potential as
site-dependent, :math:`\mu_x` at :math:`x` (recovering the above at
:math:`\mu_x=\mu`): :math:`P(\text{start at }x)` is then the Boltzmann
weight of all configurations carrying the factor
:math:`\exp(\beta\mu_x)`, proportional to :math:`\partial Z/\partial
\mu_x` and normalized by :math:`Z`,

.. math::

   P(\text{start at }x) = \frac1Z\cdot\frac1\beta
   \frac{\partial Z}{\partial\mu_x} = \frac1\beta
   \frac{\partial\log Z}{\partial\mu_x} \;.

Only :math:`T_{k'k}(x)` depends on :math:`\mu_x`, and only for
:math:`k\geq\ell_{\text{nuc}}`, where :math:`\partial T_{k'k}(x)/
\partial\mu_x = \delta_{k',1}\,\beta w(x,k)`; substituting this into
:math:`\partial Z/\partial\mu_x = \vec u(x+1)\cdot[\partial\vec T(x)/
\partial\mu_x]\cdot\vec v(x)` recovers exactly the expression above for
:math:`P(\text{start at }x)`.

To find the mean number of nucleosomes :math:`\langle N\rangle` along
the fiber, let :math:`n_x=1` if a nucleosome starts at :math:`x` (and
:math:`0` otherwise), so :math:`N=\sum_x n_x`. Taking the ensemble
average, and since :math:`\langle n_x\rangle = P(\text{start at }x)`,

.. math::

   \langle N\rangle = \sum_x\langle n_x\rangle =
   \sum_x P(\text{start at }x) = \sum_x\frac1\beta
   \frac{\partial\log Z}{\partial\mu_x} \overset{\mu_x=\mu}{=}
   \frac1\beta\frac{\partial\log Z}{\partial\mu} \;.

For the occupancy profile, recall the indicator :math:`\sigma_y` from
above, which is unity when a nucleosome covers site :math:`y` and zero
otherwise. Summing all :math:`n_x` whose nucleosome footprint can cover
site :math:`y` gives

.. math::

   \sigma_y = \sum_{x=\max(0,y-\ell_{\text{nuc}}+1)}^
   {\min(y,n_\text{pos}-1)} n_x \;.

Because nucleosomes cannot overlap, at most one nucleosome can start
within this window, so :math:`\sigma_y\in\{0,1\}` strictly. The
probability of nucleosome occupancy at :math:`y` is therefore

.. math::

   \begin{split}
   P(\text{occupied at }y) &= \langle\sigma_y\rangle \\
   &= \sum_{x=\max(0,y-\ell_{\text{nuc}}+1)}^{\min(y,n_\text{pos}-1)}
   \langle n_x\rangle \\
   &= \sum_{x=\max(0,y-\ell_{\text{nuc}}+1)}^{\min(y,n_\text{pos}-1)}
   P(\text{start at }x) \;.
   \end{split}

We can also quantify the uncertainty in these observables via their
variance. For the number of nucleosomes,

.. math::

   \text{Var}(N) = \langle N^2\rangle - \langle N\rangle^2 =
   \frac1{\beta^2}\left[\frac1Z\frac{\partial^2Z}{\partial\mu^2} -
   \frac1{Z^2}\left(\frac{\partial Z}{\partial\mu}\right)^2\right] =
   \frac1{\beta^2}\frac{\partial^2\log Z}{\partial\mu^2} \;,

which we evaluate by central finite difference,

.. math::

   \frac{\partial^2\log Z}{\partial\mu^2} \approx
   \frac1{\Delta\mu^2}\left[\log Z(\mu+\Delta\mu) +
   \log Z(\mu-\Delta\mu) - 2\log Z(\mu)\right] \;.

The variance of nucleosome occupancy is simpler still. Since
:math:`\sigma_y^2=\sigma_y`,

.. math::

   \begin{split}
   \text{Var}(\sigma_y) &= \langle\sigma_y^2\rangle -
   \langle\sigma_y\rangle^2 \\
   &= \langle\sigma_y\rangle\left(1-\langle\sigma_y\rangle\right) \\
   &= P(\text{occupied at }y)\cdot[1-P(\text{occupied at }y)] \;,
   \end{split}

so the occupancy profile computed above can be reused directly for
this calculation.

Exact configuration sampling
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In addition to marginal statistics, we can draw independent
configurations :math:`\mathcal C` that sample the posterior via
"stochastic traceback." Starting from :math:`x=0` with state
:math:`k=W`, at each site we compute the two branch weights implied by
the transfer matrix,

.. math::

   \pi_\text{add} = w(x,k)\,u_1(x+1) \;, \qquad
   \pi_\text{skip} = u_{\min(k+1,W)}(x+1) \;,

and add a nucleosome with probability

.. math::

   P(\text{add at }x) = \begin{cases}
     0 & \text{if } k<\ell_{\text{nuc}} \\
     \dfrac{\pi_\text{add}}{\pi_\text{add}+\pi_\text{skip}} &
     \text{otherwise} \;.
   \end{cases}

We then update :math:`k\to1` if a nucleosome is added, or
:math:`k\to\min(k+1,W)` otherwise, and advance to site :math:`x+1`. A
single pass over the fiber yields one exact sample. Unlike MCMC
samples, successive draws are independent by construction (no
autocorrelation, no burn-in, and no thinning), since each is a fresh
traceback through the precomputed :math:`\vec u(x)`.

.. rubric:: Footnotes

.. [#f1] Missing calls are tolerated within the averaging window itself
	 (any window containing at least one valid site yields a defined
	 average). A gap wide enough to leave :math:`\widetilde
	 S_{x,\alpha}` undefined over an interior stretch (bounded by
	 valid smoothed values on both sides) is filled with the
	 neutral, no-evidence probability of 0.5, while the
	 :math:`\ell_\text{bin}-1` trailing sites, where the window runs
	 off the fiber, are handled separately and by default left
	 undefined (a fixed value can be substituted instead).

.. [#f2] This is the exact count of valid left-aligned starting
	 positions on a fiber of length :math:`L`; a common
	 approximation is to instead take :math:`n_\text{pos}\approx
	 L/\ell_{\text{nuc}}`, the typical number of non-overlapping
	 nucleosomes that fit on the fiber.
