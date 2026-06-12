Reparametrization
=================

Automatic reparametrization of hierarchical models. VIP (variationally
inferred parametrization) makes the choice between centered and non-centered
parametrizations continuous and learns the best setting per variable, instead
of leaving it as a manual, all-or-nothing decision.

.. currentmodule:: pymc_extras.model.transforms
.. autosummary::
   :toctree: ../generated/

   autoreparam.vip_reparametrize
   autoreparam.VIP
