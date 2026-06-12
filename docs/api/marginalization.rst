Marginalization
===============

Model transformations that integrate variables out of a model, and recover
them afterwards. Marginalizing discrete variables allows sampling with
gradient-based samplers like NUTS; marginalizing conjugate pairs or using the
Laplace approximation reduces the dimensionality of the posterior.

``marginalize`` returns a model where the requested variables no longer
appear, but the remaining variables keep their original joint distribution
(exactly, or approximately when using the Laplace approximation).
``unmarginalize`` undoes the transformation, and ``conditional`` /
``recover`` reintroduce the marginalized variables conditioned on the
posterior of the remaining ones.

.. currentmodule:: pymc_extras.marginal
.. autosummary::
   :toctree: ../generated/

   marginalize
   unmarginalize
   conditional
   recover

The set of supported marginalizations is extensible; see
:doc:`../developer/extending_marginalization`.
