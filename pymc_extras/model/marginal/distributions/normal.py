from pymc import Normal
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import get_symbolic_rv_shapes
from pytensor.graph.replace import graph_replace
from pytensor.tensor import broadcast_to, constant, sqrt

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    inline_ofg_outputs,
    marginalized_conditional,
)


class NormalNormalMarginalRV(MarginalRV):
    """Marginalized Normal-Normal conjugate pair.

    Inner graph: [marginalized_normal, dependent_normal, *rng_updates]
    """

    def __init__(self, *args, marginalized_dims, **kwargs):
        self.marginalized_dims = marginalized_dims
        self.n_dependent_rvs = 1
        super().__init__(*args, **kwargs)


@_logprob.register(NormalNormalMarginalRV)
def normal_normal_marginal_rv_logp(op: NormalNormalMarginalRV, values, *inputs, **kwargs):
    [value] = values

    all_outputs = inline_ofg_outputs(op, inputs)
    marginalized_rv = all_outputs[0]
    dependent_rv = all_outputs[1]

    mu_m, sigma_m = marginalized_rv.owner.op.dist_params(marginalized_rv.owner)
    mu_d, sigma_d = dependent_rv.owner.op.dist_params(dependent_rv.owner)

    if marginalized_rv.type.broadcastable != mu_m.type.broadcastable:
        mu_m = broadcast_to(mu_m, get_symbolic_rv_shapes([marginalized_rv.shape])[0])

    new_mu = graph_replace(mu_d, {marginalized_rv: mu_m})
    new_sigma = sqrt(sigma_d**2 + sigma_m**2)
    return logp(Normal.dist(mu=new_mu, sigma=new_sigma), value)


@marginalized_conditional.register(NormalNormalMarginalRV)
def normal_normal_conditional(op, node):
    fgraph = op.fgraph.clone()
    marginalized, inner_dependent = fgraph.outputs[:2]

    mu_m, sigma_m = marginalized.owner.op.dist_params(marginalized.owner)
    mu_d, sigma_d = inner_dependent.owner.op.dist_params(inner_dependent.owner)

    dep_dummy = inner_dependent.type()

    offset = graph_replace(mu_d, {marginalized: constant(0, dtype=marginalized.type.dtype)})

    precision_m = 1 / sigma_m**2
    precision_d = 1 / sigma_d**2
    posterior_precision = precision_m + precision_d
    posterior_sigma = sqrt(1 / posterior_precision)
    posterior_mu = (mu_m * precision_m + (dep_dummy - offset) * precision_d) / posterior_precision

    return Normal.dist(mu=posterior_mu, sigma=posterior_sigma), [dep_dummy]
