Inference
=========

Fitting methods beyond ``pm.sample``: optimization-based point estimates
(``find_MAP``), Gaussian approximations (Laplace, INLA), and fast variational
methods (Pathfinder, DADVI). ``fit`` is a single entry point that dispatches
to these by name.

.. currentmodule:: pymc_extras.inference
.. autosummary::
   :toctree: ../generated/

   fit
   find_MAP
   fit_laplace
   fit_pathfinder
   fit_dadvi
   fit_INLA
