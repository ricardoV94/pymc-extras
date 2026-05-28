import numpy as np
import pandas as pd
import pymc as pm
import pytest

from pymc import Model
from pymc.distributions.transforms import ordered
from scipy.special import logsumexp

from pymc_extras.marginal import marginalize


@pytest.fixture
def disaster_model():
    # fmt: off
    disaster_data = pd.Series(
        [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
         3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
         2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
         1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
         0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
         3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
         0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1]
    )
    # fmt: on
    years = np.arange(1851, 1962)

    with Model() as disaster_model:
        switchpoint = pm.DiscreteUniform("switchpoint", lower=years.min(), upper=years.max())
        early_rate = pm.Exponential("early_rate", 1.0)
        late_rate = pm.Exponential("late_rate", 1.0)
        rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)
        with pytest.warns(Warning):
            disasters = pm.Poisson("disasters", rate, observed=disaster_data)

    return disaster_model, years


def test_change_point_model(disaster_model):
    m, years = disaster_model

    ip = m.initial_point()
    ip["late_rate_log__"] += 1.0  # Make early and endpoint ip different

    ip.pop("switchpoint")
    ref_logp_fn = m.compile_logp(
        [m["switchpoint"], m["disasters_observed"], m["disasters_unobserved"]]
    )
    ref_logp = logsumexp([ref_logp_fn({**ip, **{"switchpoint": year}}) for year in years])

    marginal_m = marginalize(m, m["switchpoint"])

    with pytest.warns(UserWarning, match="There are multiple dependent variables"):
        marginal_m_logp = marginal_m.compile_logp(
            [marginal_m["disasters_observed"], marginal_m["disasters_unobserved"]]
        )(ip)
    np.testing.assert_almost_equal(marginal_m_logp, ref_logp)


@pytest.mark.slow
def test_change_point_model_sampling(disaster_model):
    m, _ = disaster_model

    rng = np.random.default_rng(211)

    with m:
        before_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(sample=("draw", "chain"))

    marginal_m = marginalize(m, "switchpoint")

    with marginal_m:
        with pytest.warns(UserWarning, match="There are multiple dependent variables"):
            after_marg = pm.sample(chains=2, random_seed=rng).posterior.stack(
                sample=("draw", "chain")
            )

    np.testing.assert_allclose(
        before_marg["early_rate"].mean(), after_marg["early_rate"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        before_marg["late_rate"].mean(), after_marg["late_rate"].mean(), rtol=1e-2
    )
    np.testing.assert_allclose(
        before_marg["disasters_unobserved"].mean(),
        after_marg["disasters_unobserved"].mean(),
        rtol=1e-2,
    )


@pytest.mark.parametrize("univariate", (True, False))
def test_vector_univariate_mixture(univariate):
    with Model() as m:
        idx = pm.Bernoulli("idx", p=0.5, shape=(2,) if univariate else ())

        def dist(idx, size):
            return pm.math.switch(
                pm.math.eq(idx, 0),
                pm.Normal.dist([-10, -10], 1),
                pm.Normal.dist([10, 10], 1),
            )

        pm.CustomDist("norm", idx, dist=dist)

    marginal_m = marginalize(m, idx)
    logp_fn = marginal_m.compile_logp()

    if univariate:
        with pm.Model() as ref_m:
            pm.NormalMixture("norm", w=[0.5, 0.5], mu=[[-10, 10], [-10, 10]], shape=(2,))
    else:
        with pm.Model() as ref_m:
            pm.Mixture(
                "norm",
                w=[0.5, 0.5],
                comp_dists=[
                    pm.MvNormal.dist([-10, -10], np.eye(2)),
                    pm.MvNormal.dist([10, 10], np.eye(2)),
                ],
                shape=(2,),
            )
    ref_logp_fn = ref_m.compile_logp()

    for test_value in (
        [-10, -10],
        [10, 10],
        [-10, 10],
        [-10, 10],
    ):
        pt = {"norm": test_value}
        np.testing.assert_allclose(logp_fn(pt), ref_logp_fn(pt))


def test_k_censored_clusters_model():
    data = np.array([[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]])
    nobs = data.shape[0]
    n_clusters = 5

    def build_model(build_batched: bool) -> Model:
        coords = {
            "cluster": range(n_clusters),
            "ndim": ("x", "y"),
            "obs": range(nobs),
        }
        with Model(coords=coords) as m:
            if build_batched:
                idx = pm.Categorical("idx", p=np.ones(n_clusters) / n_clusters, dims=["obs"])
            else:
                idx = pm.math.stack(
                    [
                        pm.Categorical(f"idx_{i}", p=np.ones(n_clusters) / n_clusters)
                        for i in range(nobs)
                    ]
                )

            mu_x = pm.Normal(
                "mu_x",
                dims=["cluster"],
                transform=ordered,
            )
            mu_y = pm.Normal("mu_y", dims=["cluster"])
            mu = pm.math.stack([mu_x, mu_y], axis=-1)  # (cluster, ndim)
            mu_indexed = mu[idx, :]

            sigma = pm.HalfNormal("sigma")

            y = pm.Censored(
                "y",
                dist=pm.Normal.dist(mu_indexed, sigma),
                lower=-3,
                upper=3,
                observed=data,
                dims=["obs", "ndim"],
            )

        return m

    m = marginalize(build_model(build_batched=True), "idx")
    m.set_initval(m["mu_x"], np.linspace(-1, 1, n_clusters))

    ref_m = marginalize(build_model(build_batched=False), [f"idx_{i}" for i in range(nobs)])
    test_point = m.initial_point()
    np.testing.assert_almost_equal(
        m.compile_logp()(test_point),
        ref_m.compile_logp()(test_point),
    )


def test_mutable_indexing_jax_backend():
    pytest.importorskip("jax")
    from pymc.sampling.jax import get_jaxified_logp

    with Model() as model:
        data = pm.Data("data", np.zeros(10))

        cat_effect = pm.Normal("cat_effect", sigma=1, shape=5)
        cat_effect_idx = pm.Data("cat_effect_idx", np.array([0, 1] * 5))

        is_outlier = pm.Bernoulli("is_outlier", 0.4, shape=10)
        pm.LogNormal("y", mu=cat_effect[cat_effect_idx], sigma=1 + is_outlier, observed=data)
    marginal_model = marginalize(model, ["is_outlier"])
    get_jaxified_logp(marginal_model)


def test_numpyro_compat():
    pytest.importorskip("numpyro")

    with pm.Model() as m:
        p_outlier = pm.Beta("p_outlier", 1, 1)
        is_outlier = pm.Bernoulli("is_outlier", p=p_outlier, shape=(10,))
        sigma = pm.Exponential("sigma", 1, shape=(2,))
        pm.Normal("y_hat", mu=0, sigma=sigma[is_outlier])

    with marginalize(m, [is_outlier]):
        pm.sample(nuts_sampler="numpyro", chains=1, tune=1, draws=1)
