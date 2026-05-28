from pymc.distributions import Bernoulli, Categorical, DiscreteUniform
from pymc.model.fgraph import model_free_rv
from pymc.pytensorf import collect_default_updates
from pytensor.compile import SharedVariable
from pytensor.graph import Apply, Op, node_rewriter
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.db import EquilibriumDB
from pytensor.graph.traversal import graph_inputs

from pymc_extras.distributions.timeseries import DiscreteMarkovChain
from pymc_extras.model.marginal.distributions.core import MarginalRV, inline_ofg_outputs
from pymc_extras.model.marginal.distributions.enumerable import (
    MarginalDiscreteMarkovChainRV,
    MarginalFiniteDiscreteRV,
)
from pymc_extras.model.marginal.distributions.laplace import MarginalLaplaceRV
from pymc_extras.model.marginal.graph_analysis import subgraph_batch_dim_connection


class MarginalSubgraphBase(Op):
    """Base for flat IR markers representing marginalized subgraphs.

    Inputs: [*subgraph_outputs, *boundary_vars]
    Outputs: [marginalized_rv, *dependent_rvs]

    The marker delimits the Markov blanket of the marginalized RV: the
    dependent RVs are its children, and the boundary contains its parents
    and the children's other parents. Given the boundary, the marginalized
    RV is conditionally independent of the rest of the model, so rewrites
    can resolve the marker locally.

    The actual subgraph lives in the fgraph between the boundary vars
    and the subgraph outputs. At rewrite time, the subgraph is cloned
    out of the fgraph to build the OpFromGraph (MarginalRV subclass).
    RNG updates are discovered at clone time, not stored on the marker.
    """

    def __init__(self, n_dependent_rvs, marginalized_dims, output_types):
        self.n_dependent_rvs = n_dependent_rvs
        self.marginalized_dims = marginalized_dims
        self.output_types = output_types
        super().__init__()

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def make_node(self, *inputs):
        outputs = [t() for t in self.output_types]
        return Apply(self, list(inputs), outputs)

    @property
    def n_subgraph_outputs(self):
        return 1 + self.n_dependent_rvs

    def split_node_inputs(self, node):
        """Split node.inputs into (subgraph_outputs, boundary)."""
        n = self.n_subgraph_outputs
        return list(node.inputs[:n]), list(node.inputs[n:])

    def perform(self, node, inputs, outputs):
        raise NotImplementedError("MarginalSubgraph should be resolved by rewrites")


class MarginalSubgraph(MarginalSubgraphBase):
    """Ready-to-resolve marginalized subgraph marker."""


class DeferredMarginalSubgraph(MarginalSubgraphBase):
    """Marginalized subgraph whose inner deps are not yet resolved.

    Some dependent RVs come from unresolved MarginalSubgraph nodes.
    Type-specific rewrites (finite_discrete_marginal, etc.) track
    MarginalSubgraph and won't match this class. Once the inner
    MarginalSubgraph nodes are resolved by the EquilibriumDB,
    resolve_deferred_marginal_subgraph converts this to a plain
    MarginalSubgraph so those rewrites can fire.
    """


class LaplaceMarginalSubgraph(MarginalSubgraphBase):
    """Marginalized subgraph to be resolved via Laplace approximation.

    Created when the user calls ``marginalize(..., use_laplace=True)``.
    The precision matrix Q of the marginalized variable is appended as the
    last boundary input; the minimizer options are stored on the marker and
    forwarded to the MarginalLaplaceRV.
    """

    def __init__(
        self,
        *args,
        minimizer_seed: int,
        minimizer_kwargs: dict = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-8}},
        **kwargs,
    ):
        self.minimizer_seed = minimizer_seed
        self.minimizer_kwargs = minimizer_kwargs
        super().__init__(*args, **kwargs)


def extract_marginal_subgraph(node):
    """Extract inputs/outputs from a MarginalSubgraph node for building an OpFromGraph.

    ModelValuedVar nodes inside the subgraph were already unwrapped by
    _unwrap_subgraph_model_vars during replace_marginal_subgraph. RNG
    updates are discovered here. The OpFromGraph constructor handles cloning.

    Returns (inputs, outputs) where outputs = [marginalized_rv, *deps, *rng_updates].
    """
    subgraph_outputs, boundary = node.op.split_node_inputs(node)

    n_rvs = 1 + node.op.n_dependent_rvs
    rng_updates = collect_default_updates(
        subgraph_outputs[:n_rvs], inputs=boundary, must_be_shared=False
    )

    outputs = subgraph_outputs + list(rng_updates.values())
    return boundary, outputs


@node_rewriter(tracks=[MarginalRV])
def local_unmarginalize(fgraph, node):
    all_outputs = inline_ofg_outputs(node.op, node.inputs)
    n_dep = node.op.n_dependent_rvs
    unmarginalized_rv = all_outputs[0]
    dependent_rvs = list(all_outputs[1 : 1 + n_dep])
    rngs = list(all_outputs[1 + n_dep :])

    value = unmarginalized_rv.clone()
    fgraph.add_input(value)
    transform = None
    unmarginalized_free_rv = model_free_rv(
        unmarginalized_rv, value, transform, *node.op.marginalized_dims
    )

    dependent_rvs = graph_replace(dependent_rvs, {unmarginalized_rv: unmarginalized_free_rv})

    return [unmarginalized_free_rv, *dependent_rvs, *rngs]


marginal_rewrites_db = EquilibriumDB()
marginal_rewrites_db.name = "marginal_rewrites_db"


@node_rewriter(tracks=[MarginalSubgraph])
def finite_discrete_marginal(fgraph, node):
    op = node.op
    n_dep = op.n_dependent_rvs

    inputs, outputs = extract_marginal_subgraph(node)
    marginalized_rv = outputs[0]

    marginalized_rv_op = marginalized_rv.owner.op
    if not isinstance(
        marginalized_rv_op, Bernoulli | Categorical | DiscreteUniform | DiscreteMarkovChain
    ):
        return None

    if isinstance(marginalized_rv_op, DiscreteMarkovChain):
        if marginalized_rv_op.n_lags > 1:
            raise NotImplementedError(
                "Marginalization for DiscreteMarkovChain with n_lags > 1 is not supported"
            )
        if marginalized_rv.owner.inputs[0].type.ndim > 2:
            raise NotImplementedError(
                "Marginalization for DiscreteMarkovChain with non-matrix transition probability "
                "is not supported"
            )

    try:
        dependent_rvs_dim_connections = subgraph_batch_dim_connection(
            marginalized_rv, outputs[1 : 1 + n_dep]
        )
    except (ValueError, NotImplementedError) as e:
        raise type(e)(
            "The graph between the marginalized and dependent RVs cannot be marginalized efficiently. "
            "You can try splitting the marginalized RV into separate components and marginalizing "
            f"them separately. {e}"
        ) from e

    if isinstance(marginalized_rv_op, DiscreteMarkovChain):
        constructor = MarginalDiscreteMarkovChainRV
    else:
        constructor = MarginalFiniteDiscreteRV

    typed_op = constructor(
        inputs=inputs,
        outputs=outputs,
        dims_connections=dependent_rvs_dim_connections,
        marginalized_dims=op.marginalized_dims,
        n_dependent_rvs=n_dep,
    )

    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_rewrites_db.register("finite_discrete_marginal", finite_discrete_marginal, "basic")


@node_rewriter(tracks=[LaplaceMarginalSubgraph])
def laplace_marginal(fgraph, node):
    op = node.op

    # Q was appended as the last boundary input and is kept as a dummy input
    # of the OpFromGraph (popped again by the logp implementation)
    inputs, outputs = extract_marginal_subgraph(node)

    typed_op = MarginalLaplaceRV(
        inputs=inputs,
        outputs=outputs,
        marginalized_dims=op.marginalized_dims,
        n_dependent_rvs=op.n_dependent_rvs,
        minimizer_seed=op.minimizer_seed,
        minimizer_kwargs=op.minimizer_kwargs,
    )

    new_outputs = typed_op(*inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs[: len(node.outputs)]


marginal_rewrites_db.register("laplace_marginal", laplace_marginal, "basic")


@node_rewriter(tracks=[MarginalSubgraph])
def unwrap_inner_marginal_rv(fgraph, node):
    """Unwrap a MarginalRV inside a MarginalSubgraph's subgraph.

    When a variable absorbed by a prior marginalize() call is re-marginalized,
    its raw RV comes from a MarginalRV (OpFromGraph). This rewrite inlines that
    MarginalRV and rebuilds as nested MarginalSubgraph markers that the
    type-specific rewrites can handle.
    """
    subgraph_outputs, boundary = node.op.split_node_inputs(node)
    marginalized_rv = subgraph_outputs[0]
    outer_dep_outputs = subgraph_outputs[1:]

    if not (marginalized_rv.owner and isinstance(marginalized_rv.owner.op, MarginalRV)):
        return None

    marg_rv_node = marginalized_rv.owner
    marg_rv_op = marg_rv_node.op

    # Inline the MarginalRV to get raw variables
    inlined = inline_ofg_outputs(marg_rv_op, marg_rv_node.inputs)
    inlined_marginalized = inlined[0]
    inlined_deps = list(inlined[1 : 1 + marg_rv_op.n_dependent_rvs])

    # Map MFD dep outputs → inlined raw variables
    target_idx = list(marg_rv_node.outputs).index(marginalized_rv) - 1
    target_inlined = inlined_deps[target_idx]
    deps_inlined = [
        inlined_deps[list(marg_rv_node.outputs).index(d) - 1] for d in outer_dep_outputs
    ]

    def _shared_boundary(outputs, base_boundary):
        return base_boundary + [
            inp
            for inp in graph_inputs(outputs, blockers=base_boundary)
            if isinstance(inp, SharedVariable) and inp not in base_boundary
        ]

    # Inner MS: marginalize the target variable (e.g. sub_idx), deps are outer deps
    # Compute boundary from scratch — only shared vars actually used by this subgraph.
    # Block inlined_marginalized so idx's RNG doesn't leak into the inner boundary.
    inner_subgraph = [target_inlined, *deps_inlined]
    inner_boundary = _shared_boundary(inner_subgraph, [inlined_marginalized])
    inner_ms = MarginalSubgraph(
        n_dependent_rvs=len(deps_inlined),
        marginalized_dims=node.op.marginalized_dims,
        output_types=[o.type for o in inner_subgraph],
    )
    inner_outs = inner_ms(*(inner_subgraph + inner_boundary))
    if not isinstance(inner_outs, list):
        inner_outs = list(inner_outs)

    # Outer DeferredMS: marginalize the previously-marginalized variable (e.g. idx)
    # Use original boundary (not inner_boundary) so inlined_marginalized stays internal
    outer_subgraph = [inlined_marginalized, *inner_outs[1:]]
    outer_boundary = _shared_boundary(outer_subgraph, list(boundary))
    outer_ms = DeferredMarginalSubgraph(
        n_dependent_rvs=len(deps_inlined),
        marginalized_dims=marg_rv_op.marginalized_dims,
        output_types=[o.type for o in outer_subgraph],
    )
    outer_outs = outer_ms(*(outer_subgraph + outer_boundary))
    if not isinstance(outer_outs, list):
        outer_outs = list(outer_outs)

    return outer_outs[: len(node.outputs)]


marginal_rewrites_db.register(
    "unwrap_inner_marginal_rv", unwrap_inner_marginal_rv, "basic", "unwrap"
)


@node_rewriter(tracks=[DeferredMarginalSubgraph])
def resolve_deferred_marginal_subgraph(fgraph, node):
    """Convert DeferredMarginalSubgraph to MarginalSubgraph once inner deps are resolved.

    The EquilibriumDB resolves inner MarginalSubgraph nodes first (they live in
    the same fgraph). Once none of this node's inputs come from a
    MarginalSubgraph, this rewrite promotes it to a plain MarginalSubgraph
    so the type-specific rewrites can fire.
    """
    for inp in node.inputs:
        if inp.owner is not None and isinstance(inp.owner.op, MarginalSubgraphBase):
            return None

    op = node.op
    resolved_op = MarginalSubgraph(
        n_dependent_rvs=op.n_dependent_rvs,
        marginalized_dims=op.marginalized_dims,
        output_types=op.output_types,
    )
    new_outputs = resolved_op(*node.inputs)
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)
    return new_outputs


marginal_rewrites_db.register(
    "resolve_deferred_marginal_subgraph",
    resolve_deferred_marginal_subgraph,
    "basic",
)
