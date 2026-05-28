from collections.abc import Sequence
from functools import singledispatch

from pymc.distributions.distribution import _support_point, support_point
from pymc.logprob.abstract import MeasurableOp
from pytensor.compile.builders import OpFromGraph
from pytensor.graph import FunctionGraph
from pytensor.graph.basic import Variable
from pytensor.graph.replace import graph_replace
from pytensor.tensor.random.type import RandomType


def inline_ofg_outputs(op: OpFromGraph, inputs: Sequence[Variable]) -> tuple[Variable]:
    """Inline the inner graph (outputs) of an OpFromGraph Op.

    Whereas `OpFromGraph` "wraps" a graph inside a single Op, this function "unwraps"
    the inner graph.
    """
    return graph_replace(
        op.inner_outputs,
        replace=tuple(zip(op.inner_inputs, inputs)),
        strict=False,
    )


class MarginalRV(OpFromGraph, MeasurableOp):
    """Base class for supported MarginalRVs."""


@_support_point.register(MarginalRV)
def _support_point_marginal_rv(op, rv, *inputs):
    outputs = rv.owner.outputs

    fgraph = op.fgraph.clone()
    inner_inputs = fgraph.inputs
    inner_outputs = fgraph.outputs
    del op

    inner_rv = inner_outputs[outputs.index(rv)]
    marginalized_inner_rv, *other_dependent_inner_rvs = (
        out for out in inner_outputs if out is not inner_rv and not isinstance(out.type, RandomType)
    )

    marginalized_inner_rv_dummy = marginalized_inner_rv.clone()
    inner_to_dummy_replacements = []
    dummy_to_outer_replacements = []
    for other_inner_rv in other_dependent_inner_rvs:
        dummy = other_inner_rv.clone()
        inner_to_dummy_replacements.append((other_inner_rv, dummy))
        dummy_to_outer_replacements.append((dummy, outputs[inner_outputs.index(other_inner_rv)]))

    fgraph.replace(marginalized_inner_rv, marginalized_inner_rv_dummy, import_missing=True)
    fgraph.replace_all(tuple(inner_to_dummy_replacements), import_missing=True)

    inner_rv_support_point = support_point(inner_rv)
    marginalized_inner_rv_support_point = support_point(marginalized_inner_rv)

    fgraph = FunctionGraph(outputs=[inner_rv_support_point], clone=False)
    fgraph.replace(
        marginalized_inner_rv_dummy, marginalized_inner_rv_support_point, import_missing=True
    )
    fgraph.replace_all(tuple(zip(inner_inputs, inputs)), import_missing=True)
    fgraph.replace_all(tuple(dummy_to_outer_replacements), import_missing=True)

    [rv_support_point] = fgraph.outputs
    return rv_support_point


@singledispatch
def marginalized_conditional(op, node):
    """Build the conditional distribution of a marginalized variable given its dependents.

    Dispatches on the MarginalRV op type.

    The inner graph of a MarginalRV is generative: it draws the marginalized
    variable and then the dependents given it, factoring as
    ``p(marginalized | inputs) * p(dependents | marginalized, inputs)``.
    This function returns the reverse factor
    ``p(marginalized | dependents, inputs)``, where the dependents are given
    values rather than random draws: a Categorical over the enumerated domain
    weighted by the joint logp for finite discrete marginals, the conjugate
    posterior Normal for Normal-Normal.

    Returns ``(sample_graph, dep_dummies)`` where *sample_graph* is a random
    variable distributed as ``p(marginalized | dependents, inputs)``,
    expressed over the op's ``inner_inputs``, and *dep_dummies* are
    placeholder tensors standing in for the dependent values. The caller
    replaces the dummies with the actual model variables (or observed data)
    and the inner inputs with the node's inputs.
    """
    raise NotImplementedError(
        f"Cannot recover marginalized variable with distribution {type(op).__name__}"
    )
