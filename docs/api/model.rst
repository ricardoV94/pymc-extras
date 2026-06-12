Model building
==============

Tools for defining models. ``as_model`` turns a function with PyMC
statements into a reusable model factory, and ``ModelBuilder`` is a base
class for packaging a model behind a scikit-learn-like ``fit``/``predict``
interface, with saving and loading included.

.. currentmodule:: pymc_extras
.. autosummary::
   :toctree: ../generated/

   as_model
   model_builder.ModelBuilder
