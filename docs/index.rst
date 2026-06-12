PyMC Extras
===================================================
|Tests|
|Coverage|
|Black|


.. |Tests| image:: https://github.com/pymc-devs/pymc-extras/actions/workflows/test.yml/badge.svg
    :target: https://github.com/pymc-devs/pymc-extras

.. |Coverage| image:: https://codecov.io/gh/pymc-devs/pymc-extras/branch/main/graph/badge.svg?token=ZqH0KCLKAE
    :target: https://codecov.io/gh/pymc-devs/pymc-extras


PyMC Extras extends `PyMC <https://www.pymc.io>`_ with additional
distributions, inference methods, and model transformations. It is maintained
by the PyMC team and hosts functionality that is too specialized for the core
library, but useful enough that you shouldn't have to write it yourself.

What's inside
=============

* :doc:`Automatic marginalization <api/marginalization>`: exact for finite
  discrete and conjugate variables, approximate via the Laplace approximation.
* :doc:`Alternative inference methods <api/inference>`: Pathfinder, DADVI,
  INLA, Laplace approximation, and better MAP estimation.
* :doc:`Statespace models <api/statespace>`: SARIMAX, VARMAX, ETS, and
  structural time series with Kalman filtering.
* :doc:`Additional distributions <api/distributions>` such as
  ``DiscreteMarkovChain``, ``GeneralizedPoisson``, and ``GenExtreme``.
* :doc:`Model building tools <api/model>` like the ``as_model`` decorator and
  the ``ModelBuilder`` base class.

See the full :doc:`api_reference` for everything else.

Installation
============

To install the latest release on `PyPI <https://pypi.org/project/pymc-extras/>`_, you can use a package manager like pip:

.. code-block:: bash

   pip install pymc-extras

For the development version, you can install directly from GitHub:

.. code-block:: bash

  pip install git+https://github.com/pymc-devs/pymc-extras.git

Contributing
============
We welcome contributions from interested individuals or groups! For information
about contributing to PyMC Extras check out our instructions, policies, and
guidelines `here <https://github.com/pymc-devs/pymc-extras/blob/main/CONTRIBUTING.md>`_.
If you want to extend the internals (e.g. add a new marginalization), start
with the :doc:`developer/index`.

Contributors
============
See the `GitHub contributor page <https://github.com/pymc-devs/pymc-extras/graphs/contributors>`_.

.. toctree::
   :hidden:

   api_reference
   developer/index
