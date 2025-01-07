from pymc import Normal
from pymc.logprob.rewriting import measurable_ir_rewrites_db
from pymc.pytensorf import constant_fold
from pytensor import clone_replace, graph_replace
from pytensor.graph import node_rewriter, ancestors
import pytensor.tensor as pt
from pymc_extras.model.marginal.distributions import MarginalRV, inline_ofg_outputs


@node_rewriter(tracks=[MarginalRV])
def normal_normal_marginal(fgraph, node):
    if type(node.op) is not MarginalRV:
        # Already not a raw MarginalRV
        return

    if len(node.outputs) != 2 * 2:
        # More than two dependent variables
        return

    marginalized_rv, dependent_rv, *_ = node.op.fgraph.outputs
    if not (
        isinstance(marginalized_rv.owner.op, Normal)
        and isinstance(marginalized_rv.owner.op, Normal)
    ):
        return

    mu_dependent_rv, sigma_dependent_rv = dependent_rv.owner.op.dist_params(dependent_rv.owner)
    mu_marginalized_rv, sigma_marginalized_rv = marginalized_rv.owner.op.dist_params(marginalized_rv.owner)

    if marginalized_rv in ancestors([sigma_dependent_rv]):
        return

    # Check that we have mu = marginalized_rv + offset
    if not mu_dependent_rv is marginalized_rv:
        add_node = mu_dependent_rv.owner
        if not (add_node and add_node.op == pt.add and len(add_node.inputs) == 2):
            return
        a, b = add_node.inputs
        if a is marginalized_rv:
            if marginalized_rv in ancestors([b]):
                # The marginalized_rv shows up in both branches of the addition
                return
        elif b is marginalized_rv:
            if marginalized_rv in ancestors([a]):
                # The marginalized_rv shows up in both branches of the addition
                return
        else:
            # There's a more complicated function between the marginalized_rv and the mean of the dependent_rv
            return


    # Replace reference to marginalized RV by its mean (possibly broadcasted):
    if marginalized_rv.type.broadcastable != mu_marginalized_rv.type.broadcastable:
        mu_marginalized_rv = pt.broadcast_to(
            mu_marginalized_rv,
            constant_fold(marginalized_rv.shape, raise_not_constant=False)
        )
    rng_dependent_rv = dependent_rv.owner.op.rng_param(dependent_rv.owner)
    size_dependent_rv = dependent_rv.owner.op.size_param(dependent_rv.owner)

    new_mu =  clone_replace(mu_dependent_rv, {marginalized_rv: mu_marginalized_rv})
    new_sigma = pt.sqrt(sigma_dependent_rv ** 2 + sigma_marginalized_rv ** 2)
    new_rv = Normal.dist(mu=new_mu, sigma=new_sigma, size=size_dependent_rv, rng=rng_dependent_rv)

    # Replace inner inputs by outer inputs
    new_rv = graph_replace(
        new_rv,
        replace=tuple(zip(node.op.inner_inputs, node.inputs)),
        strict=False,
    )
    return {node.outputs[1]: new_rv}



measurable_ir_rewrites_db.register(
    normal_normal_marginal.__name__, normal_normal_marginal, "basic", "marginal"
)