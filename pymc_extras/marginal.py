"""Public namespace for marginalization utilities.

The implementation lives in :mod:`pymc_extras.model.marginal`; this module
re-exports the public API under the shorter ``pymc_extras.marginal`` path.
"""

from pymc_extras.model.marginal.model import (
    conditional,
    marginalize,
    recover,
    unmarginalize,
)

__all__ = ["conditional", "marginalize", "recover", "unmarginalize"]
