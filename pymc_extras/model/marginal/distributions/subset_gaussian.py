"""Marginalizing the coordinates of a Gaussian latent that nothing reads.

`marginalize` removes a whole random variable. This removes a *sub-block* of
one: the trailing rows of a packed `MvNormal` that no dependent variable slices
out. Their factor integrates to one, so the posterior over the rows that remain
is unchanged and no conjugacy is required.

This is the only `MarginalRV` whose marginalized quantity is part of a variable
rather than a whole one, so the op carries the partition as a boolean mask and
its logp is the marginal over the kept block:

    f ~ MvNormal(m, K)   partitioned into  f_o = f[keep], f_u = f[~keep]
    f_o ~ MvNormal(m_o, K_oo)
    f_u | f_o ~ MvNormal(m_u + K_uo K_oo^-1 (f_o - m_o), K_uu - K_uo K_oo^-1 K_ou)

That conditional is the textbook Gaussian conditional, i.e. exactly what
`pymc_extras.gp.project` and `conditional_covariance` compute. Here it lives
inside a generic op instead of a GP-specific helper, and because it travels with
the op, `conditional` recovers the dropped block on its own -- there is no
opportunity to pair a posterior mean with the wrong covariance and understate
the spread.

Recovering the block is *not* prediction at new inputs: the partition is fixed
when the model is built. Predicting elsewhere still means evaluating the kernel
at inputs the model never saw, which is what `project` is for.
"""

import numpy as np
import pytensor.tensor as pt

from pymc import MvNormal
from pymc.logprob.abstract import _logprob
from pymc.logprob.basic import logp
from pymc.pytensorf import collect_default_updates

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    marginalized_conditional,
)

JITTER = 1e-8

__all__ = ["SubsetMarginalRV", "build_subset_marginal"]


class SubsetMarginalRV(MarginalRV):
    """An `MvNormal` with the coordinates outside ``keep_mask`` marginalized.

    Outputs are ``[f_unobserved, f_observed, *rng_updates]``, following the
    ``[marginalized_rv, *dependent_rvs]`` convention: the kept block is treated
    as the single dependent, since it is what the rest of the model reads.

    The partition is a boolean mask rather than a split point, so the dropped
    coordinates need not be contiguous or trailing.
    """

    def __init__(self, *args, keep_mask, **kwargs):
        self.keep_mask = np.asarray(keep_mask, dtype=bool)
        super().__init__(*args, n_dependent_rvs=1, **kwargs)


def _partition(op, inputs):
    """``(m_o, m_u, K_oo, K_uo, K_uu)`` from the op's own inputs."""
    _rng, mu, cov = inputs
    keep = np.flatnonzero(op.keep_mask)
    drop = np.flatnonzero(~op.keep_mask)
    mu = pt.atleast_1d(pt.broadcast_to(mu, (cov.shape[0],)))
    return (
        mu[keep],
        mu[drop],
        cov[np.ix_(keep, keep)],
        cov[np.ix_(drop, keep)],
        cov[np.ix_(drop, drop)],
    )


def build_subset_marginal(rv, keep_mask, marginalized_name: str, marginalized_dims=()):
    """Wrap an `MvNormal` node into a `SubsetMarginalRV`.

    Returns ``(unobserved_out, observed_out)``. Both come from one draw of the
    joint, so the generative semantics are unchanged.
    """
    rng, _size, mu, cov = rv.owner.inputs
    keep_mask = np.asarray(keep_mask, dtype=bool)
    keep = np.flatnonzero(keep_mask)
    drop = np.flatnonzero(~keep_mask)

    rng_i, mu_i, cov_i = rng.type(), mu.type(), cov.type()
    full = MvNormal.dist(mu=mu_i, cov=cov_i, rng=rng_i)
    updates = collect_default_updates([full], inputs=[rng_i, mu_i, cov_i], must_be_shared=False)

    op = SubsetMarginalRV(
        inputs=[rng_i, mu_i, cov_i],
        outputs=[full[drop], full[keep], *updates.values()],
        keep_mask=keep_mask,
        marginalized_name=marginalized_name,
        marginalized_dims=marginalized_dims,
    )
    outputs = op(rng, mu, cov)
    if not isinstance(outputs, list):
        outputs = list(outputs)
    return outputs[0], outputs[1]


@_logprob.register(SubsetMarginalRV)
def subset_marginal_logp(op, values, *inputs, **kwargs):
    """The marginal over the kept block; the dropped rows integrate to one."""
    [value] = values
    m_o, _m_u, K_oo, _K_uo, _K_uu = _partition(op, inputs)
    return logp(MvNormal.dist(mu=m_o, cov=K_oo), pt.atleast_1d(value))


@marginalized_conditional.register(SubsetMarginalRV)
def subset_marginalized_conditional(op, inputs, dep_rvs):
    """``f_u | f_o``, the Gaussian conditional of the dropped block."""
    [f_o] = dep_rvs
    m_o, m_u, K_oo, K_uo, K_uu = _partition(op, inputs)

    K_oo = K_oo + JITTER * pt.eye(K_oo.shape[0])
    L = pt.linalg.cholesky(K_oo)
    # V.T @ V == K_uo K_oo^-1 K_ou, and A == K_uo K_oo^-1
    V = pt.linalg.solve_triangular(L, K_uo.T, lower=True)
    alpha = pt.linalg.solve_triangular(L, pt.atleast_1d(f_o) - m_o, lower=True)

    post_mu = m_u + V.T @ alpha
    post_cov = K_uu - V.T @ V
    post_cov = 0.5 * (post_cov + post_cov.T) + JITTER * pt.eye(post_cov.shape[0])
    return MvNormal.dist(mu=post_mu, cov=post_cov)
