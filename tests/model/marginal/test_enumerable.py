import numpy as np
import pymc as pm
import scipy.special
import scipy.stats

from pymc.logprob.abstract import _logprob
from pymc.pytensorf import collect_default_updates
from pytensor import tensor as pt

from pymc_extras.marginal import marginalize
from pymc_extras.model.marginal.distributions import MarginalFiniteDiscreteRV


def test_marginalized_bernoulli_logp():
    """Test logp of IR TestFiniteMarginalDiscreteRV directly"""
    mu = pt.vector("mu")

    idx = pm.Bernoulli.dist(0.7, name="idx")
    y = pm.Normal.dist(mu=mu[idx], sigma=1.0, name="y")
    # The inner RVs draw from shared RNGs, which the OpFromGraph requires as
    # explicit inputs (with their updates as extra outputs).
    updates = collect_default_updates([idx, y])
    rngs, rng_updates = list(updates.keys()), list(updates.values())
    marginal_rv_node = MarginalFiniteDiscreteRV(
        [mu, *rngs],
        [idx, y, *rng_updates],
        n_dependent_rvs=1,
        dims_connections=(((),),),
        marginalized_name="idx",
        marginalized_dims=(),
    )(mu, *rngs)[0].owner

    y_vv = y.clone()
    (logp,) = _logprob(
        marginal_rv_node.op,
        (y_vv,),
        *marginal_rv_node.inputs,
    )

    ref_logp = pm.logp(pm.NormalMixture.dist(w=[0.3, 0.7], mu=mu, sigma=1.0), y_vv)
    np.testing.assert_almost_equal(
        logp.eval({mu: [-1, 1], y_vv: 2}),
        ref_logp.eval({mu: [-1, 1], y_vv: 2}),
    )


def test_multivariate_dependent_with_extra_batch_dim():
    """A dependent whose own logp collapses its core dims, plus a dim to marginalize over.

    ``dims_connections`` is indexed against the dependent RV, but MvNormal's logp has already
    dropped ``c``, so the axes left to reduce have to be renumbered against the logp.
    """
    with pm.Model() as m:
        idx = pm.Categorical("idx", p=[0.3, 0.7], shape=(3,))
        mu = np.zeros((5, 3, 4)) + idx[None, :, None] * 2.0
        pm.MvNormal("y", mu=mu, chol=np.eye(4), shape=(5, 3, 4))

    logp = marginalize(m, ["idx"]).compile_logp()({"y": np.zeros((5, 3, 4))})

    # idx is shared across the 5 obs, so each trial marginalizes over one idx draw
    expected = 3 * scipy.special.logsumexp(
        [
            np.log(w)
            + 5
            * scipy.stats.multivariate_normal.logpdf(np.zeros(4), np.full(4, 2.0 * k), np.eye(4))
            for k, w in [(0, 0.3), (1, 0.7)]
        ]
    )
    np.testing.assert_allclose(logp, expected)
