"""Marginalizing a variable that nothing depends on.

``int p(f) df == 1``, so removing such a variable leaves the model density
unchanged. That makes this the degenerate case of marginalization rather than an
error: no conjugacy is required, and it holds for any distribution, not just
Gaussian ones.

The subtlety is recovery. A `MarginalRV` is found again by walking the model
fgraph, which only reaches nodes referenced from its outputs. With dependents,
the node is held there by them; with none, nothing holds it and the node would
be pruned, leaving `conditional` and `unmarginalize` no way back. So the
marginalized output is anchored with a `ModelNamed` wrapper -- reachable and
named, but not a free RV, which would put it back in the sampler.
"""

from pytensor.graph import node_rewriter

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)
from pymc_extras.model.marginal.rewrites import (
    MarginalSubgraph,
    extract_marginal_subgraph,
    marginal_ir_rewrites_db,
)


class TrivialMarginalRV(MarginalRV):
    """A marginalized variable with no dependent RVs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, n_dependent_rvs=0, **kwargs)


# No `_logprob` is registered: with no dependent RVs there are no value
# variables referring to this op, so the logp machinery never dispatches on it.
# That is the correct contribution -- the factor integrates to one.


@marginalized_conditional.register(TrivialMarginalRV)
def trivial_marginalized_conditional(op, inputs, dep_rvs):
    """The conditional of a variable nothing observed is its prior."""
    return inline_ofg_outputs(op, inputs)[0]


@node_rewriter(tracks=[MarginalSubgraph])
def trivial_marginal_rewrite(fgraph, node):
    op = node.op
    if op.n_dependent_rvs != 0:
        return None

    inputs, outputs = extract_marginal_subgraph(node)
    typed_op = TrivialMarginalRV(
        inputs=inputs,
        outputs=outputs,
        marginalized_name=op.marginalized_name,
        marginalized_dims=op.marginalized_dims,
    )
    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_ir_rewrites_db.register("trivial_marginal", trivial_marginal_rewrite, "basic")
