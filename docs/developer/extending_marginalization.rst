Extending marginalization
=========================

:func:`~pymc_extras.marginal.marginalize` is built on a small set of
composable pieces, so new kinds of marginalization (a new conjugate pair, a
new approximation) can be added without touching the core machinery. This
page explains how the pipeline works and what you need to implement.

All code lives under ``pymc_extras/model/marginal/``.

How marginalization works
-------------------------

``marginalize`` operates on the model's
:class:`~pytensor.graph.fg.FunctionGraph` representation (obtained via :func:`pymc.model.fgraph.fgraph_from_model`) in
two stages:

1. **Marking.** For each variable to marginalize, the subgraph connecting it
   to its dependent RVs is wrapped behind a ``MarginalSubgraph`` marker
   ``Op`` (in ``rewrites.py``). The marker delimits the variable's Markov
   blanket: its children (the dependent RVs) are the subgraph outputs, and
   its parents together with the children's other parents form the boundary
   inputs. Given the boundary, the marginalized variable is conditionally
   independent of the rest of the model, so rewrites can reason about the
   marker locally. The marker itself is type-agnostic — it knows nothing
   about distributions.

2. **Resolution.** The ``marginal_rewrites_db`` (a pytensor
   ``EquilibriumDB``) is run on the graph.
   Each registered rewrite inspects ``MarginalSubgraph`` nodes and, when it
   recognizes a pattern it can handle (e.g. a finite discrete variable, or a
   Normal whose dependent is also Normal), replaces the marker with a typed
   ``MarginalRV``. If no rewrite claims a marker, ``marginalize`` raises
   ``NotImplementedError``.

A ``MarginalRV`` (in ``distributions/core.py``) is an
:class:`~pytensor.compile.builders.OpFromGraph` that is also a PyMC
``MeasurableOp``. Its inner graph is the original *generative* subgraph —
it still draws the marginalized variable and the dependents given it, so
forward sampling (``pm.sample_prior_predictive``) works unchanged. What makes
it "marginal" is its logp implementation:

* Each ``MarginalRV`` subclass registers a ``_logprob`` dispatch that returns
  the **marginal** logp of the dependent values, with the marginalized
  variable integrated/summed out.
* Optionally, it also registers ``marginalized_conditional``, which builds
  ``p(marginalized | dependents)``. This is what powers
  :func:`~pymc_extras.marginal.conditional` and
  :func:`~pymc_extras.marginal.recover`.

:func:`~pymc_extras.marginal.unmarginalize` is fully generic: it just inlines
the ``OpFromGraph`` and restores the marginalized variable as a free RV, so
new marginalizations get it for free.

Adding a new marginalization
----------------------------

The example below adds Gamma-Poisson marginalization for the simplest
possible case — a *scalar* Gamma that is *directly* the rate of a single
Poisson:

.. math::

   z \sim \text{Gamma}(\alpha, \beta), \quad y \sim \text{Poisson}(z)

The marginal of :math:`y` is
:math:`\text{NegativeBinomial}(\alpha, \beta / (\beta + 1))` and the
conditional is the conjugate posterior
:math:`z \mid y \sim \text{Gamma}(\alpha + y, \beta + 1)`.

1. **Subclass** ``MarginalRV``. The inner graph outputs are laid out as
   ``[marginalized_rv, *dependent_rvs, *rng_updates]``:

   .. code-block:: python

      from pymc_extras.model.marginal.distributions.core import MarginalRV


      class GammaPoissonMarginalRV(MarginalRV):
          """Marginalized Gamma-Poisson pair."""

          def __init__(self, *args, marginalized_dims, **kwargs):
              self.marginalized_dims = marginalized_dims
              self.n_dependent_rvs = 1
              super().__init__(*args, **kwargs)

2. **Write the rewrite** that recognizes the pattern. It tracks
   ``MarginalSubgraph`` and uses ``extract_marginal_subgraph`` to get the
   subgraph's inputs/outputs (RNG updates included). Make the pattern match
   as restrictive as needed to keep the implementation simple — here we
   require a scalar Gamma used directly as the Poisson rate, which sidesteps
   transformed parameters and batch-dimension bookkeeping entirely. Return
   ``None`` whenever the pattern does not apply, so other rewrites get a
   chance:

   .. code-block:: python

      from pymc.distributions import Gamma, Poisson
      from pytensor.graph import node_rewriter

      from pymc_extras.model.marginal.rewrites import (
          MarginalSubgraph,
          extract_marginal_subgraph,
          marginal_rewrites_db,
      )


      @node_rewriter(tracks=[MarginalSubgraph])
      def gamma_poisson_marginal(fgraph, node):
          if node.op.n_dependent_rvs != 1:
              return None

          inputs, outputs = extract_marginal_subgraph(node)
          marginalized_rv, dependent_rv = outputs[:2]

          if not (
              isinstance(marginalized_rv.owner.op, Gamma)
              and isinstance(dependent_rv.owner.op, Poisson)
              and marginalized_rv.type.ndim == 0
          ):
              return None

          [poisson_mu] = dependent_rv.owner.op.dist_params(dependent_rv.owner)
          if poisson_mu is not marginalized_rv:
              return None

          typed_op = GammaPoissonMarginalRV(
              inputs=inputs,
              outputs=outputs,
              marginalized_dims=node.op.marginalized_dims,
          )
          new_outputs = typed_op(*inputs)
          return list(new_outputs)[: len(node.outputs)]


      marginal_rewrites_db.register(
          "gamma_poisson_marginal", gamma_poisson_marginal, "basic"
      )

   Because the database is an ``EquilibriumDB``, rewrites run in no
   particular order and repeatedly until the graph stabilizes. Be
   conservative in what you match, and decline (``return None``) rather than
   raise when unsure — raising is reserved for patterns that are recognizably
   yours but unsupported (see ``finite_discrete_marginal`` for an example).

3. **Register the marginal logp.** Use ``inline_ofg_outputs`` to recover the
   inner generative graph expressed over the node's actual inputs, extract
   the parameters, and return the logp of the dependent values with the
   marginalized variable integrated out. Note that ``dist_params`` returns
   the *backend* parametrization — for Gamma that is ``(alpha, scale)``, not
   ``(alpha, beta)``:

   .. code-block:: python

      from pymc import NegativeBinomial
      from pymc.logprob.abstract import _logprob
      from pymc.logprob.basic import logp

      from pymc_extras.model.marginal.distributions.core import inline_ofg_outputs


      @_logprob.register(GammaPoissonMarginalRV)
      def gamma_poisson_marginal_logp(op, values, *inputs, **kwargs):
          [value] = values
          marginalized_rv, _ = inline_ofg_outputs(op, inputs)[:2]
          alpha, scale = marginalized_rv.owner.op.dist_params(marginalized_rv.owner)
          beta = 1 / scale
          return logp(NegativeBinomial.dist(n=alpha, p=beta / (beta + 1)), value)

4. **(Optional) Register the conditional** to support
   :func:`~pymc_extras.marginal.conditional` and
   :func:`~pymc_extras.marginal.recover`. It returns a
   ``(sample_graph, dep_dummies)`` pair: a random variable distributed as
   ``p(marginalized | dependents)``, and placeholder tensors standing in for
   the dependent values (the caller substitutes the actual model variables or
   observed data; see the docstring of ``marginalized_conditional`` in
   ``distributions/core.py`` for the full contract):

   .. code-block:: python

      from pymc_extras.model.marginal.distributions.core import marginalized_conditional


      @marginalized_conditional.register(GammaPoissonMarginalRV)
      def gamma_poisson_conditional(op, node):
          fgraph = op.fgraph.clone()
          marginalized, dependent = fgraph.outputs[:2]
          alpha, scale = marginalized.owner.op.dist_params(marginalized.owner)
          beta = 1 / scale

          dep_dummy = dependent.type()
          return Gamma.dist(alpha + dep_dummy, beta + 1), [dep_dummy]

   Without this registration everything else works;
   ``conditional``/``recover`` will raise for your variable.

That's the whole extension:

.. code-block:: python

   import pymc as pm
   from pymc_extras.marginal import conditional, marginalize

   with pm.Model() as m:
       z = pm.Gamma("z", 2.0, 3.0)
       y = pm.Poisson("y", mu=z, observed=4)

   marg_m = marginalize(m, [z])  # y is now NegativeBinomial under the hood
   cond_m = conditional(marg_m)  # z is back, as Gamma(2 + 4, 3 + 1)

For a real-world template handling broadcasting and parameter transformations
see ``NormalNormalMarginalRV`` (``distributions/normal.py``).

Other things you get (or must check) for free
---------------------------------------------

* **Support points** for initialization are derived generically from the
  inner graph (``_support_point_marginal_rv`` in ``distributions/core.py``).
* **Nested marginalization** (marginalizing a variable that depends on an
  already-marginalized one) is handled by ``unwrap_inner_marginal_rv``,
  which inlines the existing ``MarginalRV`` and re-marks both subgraphs.
  Your rewrite only ever sees plain ``MarginalSubgraph`` markers.
* **Batched dependencies**: if your marginalization needs to reason about
  how batch dimensions of the marginalized variable map onto the dependents
  (as the finite discrete case does), use
  ``subgraph_batch_dim_connection`` from ``graph_analysis.py``.

Tests live in ``tests/model/marginal/``; ``test_distributions.py`` and
``test_rewrites.py`` show the expected coverage for a new marginalization:
the marginal logp against a reference, forward sampling, and round-trips
through ``unmarginalize``/``recover``.
