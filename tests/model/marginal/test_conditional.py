import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy

from arviz_base import from_dict
from pymc.model.transform.conditioning import remove_value_transforms
from scipy.special import logsumexp

from pymc_extras.marginal import conditional, marginalize, recover, unmarginalize


def compute_conditional_logprob(cond_model, var_name, domain, point):
    """Compute log P(var=k | data, params) for each k in domain.

    This is the pattern users would follow to evaluate conditional
    log-probabilities from a model returned by ``conditional()``.

    Parameters
    ----------
    cond_model : pm.Model
        Model returned by ``conditional()``.
    var_name : str
        Name of the recovered variable.
    domain : array-like
        Domain values to evaluate.
    point : dict
        Values for all other variables in the model.

    Returns
    -------
    logps : array
        Log-probabilities for each domain value.
    """
    logp_fn = cond_model.compile_logp(vars=[cond_model[var_name]])
    return np.array([logp_fn({**point, var_name: k}) for k in domain])


def build_normal_chain_model():
    with pm.Model() as m:
        mu = pm.Normal("mu", 0, 10)
        x = pm.Normal("x", mu=mu, sigma=3.0)
        pm.Normal("y", mu=x + 1.5, sigma=4.0)
    return m


def build_bernoulli_model():
    with pm.Model() as m:
        idx = pm.Bernoulli("idx", p=0.75)
        pm.Normal("y", mu=idx, sigma=2.0)
    return m


@pytest.mark.parametrize(
    "k_dist, probs, mu_vals",
    [
        pytest.param(
            lambda: pm.Bernoulli("k", p=0.5),
            np.array([0.5, 0.5]),
            np.array([0.0, 1.0]),
            id="bernoulli",
        ),
        pytest.param(
            lambda: pm.Categorical("k", p=[0.1, 0.3, 0.6]),
            np.array([0.1, 0.3, 0.6]),
            np.array([-3.0, 0.0, 3.0]),
            id="categorical",
        ),
    ],
)
def test_finite_discrete_logp(k_dist, probs, mu_vals):
    """Test that conditional gives the exact discrete posterior logp."""
    with pm.Model() as m:
        sigma = pm.HalfNormal("sigma")
        k = k_dist()
        y = pm.Normal("y", mu=pt.as_tensor(mu_vals)[k], sigma=sigma)

    marginal_m = marginalize(m, "k")
    cond_m = conditional(marginal_m)

    assert "k" in [rv.name for rv in cond_m.free_RVs]

    y_val = 2.5
    logps = compute_conditional_logprob(
        cond_m, "k", domain=range(len(mu_vals)), point={"sigma_log__": 0.0, "y": y_val}
    )

    # Manual: log P(k | y, sigma=1) ∝ log P(y | k, sigma=1) + log P(k)
    expected = scipy.special.log_softmax(
        np.log(probs) + scipy.stats.norm.logpdf(y_val, mu_vals, 1.0)
    )
    np.testing.assert_allclose(logps, expected)


def test_with_remove_value_transforms():
    """Test that remove_value_transforms + conditional gives natural-scale inputs."""
    with pm.Model() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.5)
        y = pm.Normal("y", mu=idx, sigma=sigma)

    marginal_m = marginalize(m, "idx")

    # Default: should have transformed value variable
    cond_m = conditional(marginal_m)
    assert any(
        rv.name == "sigma" and cond_m.rvs_to_transforms[rv] is not None for rv in cond_m.free_RVs
    )

    # With remove_value_transforms: natural scale
    natural_m = remove_value_transforms(marginal_m)
    cond_nat = conditional(natural_m)
    assert all(cond_nat.rvs_to_transforms[rv] is None for rv in cond_nat.free_RVs)

    # Both should give the same logp result
    logp_fn = cond_m.compile_logp(vars=[cond_m["idx"]])
    logp_fn_nat = cond_nat.compile_logp(vars=[cond_nat["idx"]])

    lp = logp_fn({"sigma_log__": 0.0, "y": 2.0, "idx": 1})
    lp_nat = logp_fn_nat({"sigma": 1.0, "y": 2.0, "idx": 1})
    np.testing.assert_allclose(lp, lp_nat)


def test_marginal_vs_full_conditional():
    """Test marginal posterior vs full conditional via unmarginalize."""
    with pm.Model() as m:
        idx = pm.Bernoulli("idx", p=0.5)
        sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor([0.3, 0.7])[idx])
        y = pm.Normal("y", mu=(idx + sub_idx) - 1, sigma=0.5)

    marginal_m = marginalize(m, ["idx", "sub_idx"])

    # Reference marginal posterior P(idx | y), sub_idx integrated out
    # (exactness of this path is covered by test_recover_nested_subset)
    lp_marginal = compute_conditional_logprob(
        conditional(marginal_m, "idx"), "idx", domain=(0, 1), point={"y": 0.5}
    )

    # Full conditional: P(idx | sub_idx, y) via unmarginalize
    partial_m = unmarginalize(marginal_m, "sub_idx")
    cond_full = conditional(partial_m, "idx")
    assert "sub_idx" in [rv.name for rv in cond_full.free_RVs]
    with pytest.warns(match="multiple dependent variables"):
        logp_full = cond_full.compile_logp(vars=[cond_full["idx"]])

    # Full conditional depends on sub_idx — different answers for different sub_idx values
    lp_given_sub0 = [logp_full({"y": 0.5, "sub_idx": 0, "idx": k}) for k in (0, 1)]
    lp_given_sub1 = [logp_full({"y": 0.5, "sub_idx": 1, "idx": k}) for k in (0, 1)]
    np.testing.assert_allclose(scipy.special.logsumexp(lp_given_sub0), 0.0, atol=1e-14)
    np.testing.assert_allclose(scipy.special.logsumexp(lp_given_sub1), 0.0, atol=1e-14)

    # Full conditionals should differ from each other and from the marginal
    assert not np.allclose(lp_given_sub0, lp_given_sub1)
    assert not np.allclose(lp_given_sub0, lp_marginal)


def test_recover_nested_subset():
    """Test recovering a nested variable with its parent integrated out.

    Uses a 3-category idx so marginal posteriors of idx and sub_idx
    have different shapes and are numerically distinguishable.
    """
    p_idx = np.array([0.1, 0.3, 0.6])
    p_sub_given_idx = np.array([0.2, 0.8, 0.5])
    mu = np.array([[-1, 1], [0, 3], [2, 5]], dtype="float64")  # [idx, sub_idx]

    with pm.Model() as m:
        idx = pm.Categorical("idx", p=p_idx)
        sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor(p_sub_given_idx)[idx])
        y = pm.Normal("y", mu=pt.as_tensor(mu)[idx, sub_idx], sigma=1.0, observed=2.5)

    marginal_m = marginalize(m, ["idx", "sub_idx"])

    # Manual joint log-probabilities: log p(idx=k, sub_idx=j, y)
    y_val = 2.5
    log_joints = np.zeros((3, 2))
    for k in range(3):
        for j in range(2):
            p_s = p_sub_given_idx[k] if j == 1 else 1 - p_sub_given_idx[k]
            log_joints[k, j] = (
                np.log(p_idx[k]) + np.log(p_s) + scipy.stats.norm.logpdf(y_val, mu[k, j], 1.0)
            )

    # Recover sub_idx only — idx integrated out
    cond_sub = conditional(marginal_m, "sub_idx")
    assert "idx" not in [rv.name for rv in cond_sub.free_RVs]
    assert "sub_idx" in [rv.name for rv in cond_sub.free_RVs]

    logp_sub_fn = cond_sub.compile_logp(vars=[cond_sub["sub_idx"]])
    actual_sub = np.array([logp_sub_fn({"sub_idx": j}) for j in range(2)])
    expected_sub = scipy.special.log_softmax(scipy.special.logsumexp(log_joints, axis=0))
    np.testing.assert_allclose(actual_sub, expected_sub)
    np.testing.assert_allclose(scipy.special.logsumexp(actual_sub), 0.0, atol=1e-14)

    # Recover idx only — sub_idx integrated out
    cond_idx = conditional(marginal_m, "idx")
    assert "sub_idx" not in [rv.name for rv in cond_idx.free_RVs]

    logp_idx_fn = cond_idx.compile_logp(vars=[cond_idx["idx"]])
    actual_idx = np.array([logp_idx_fn({"idx": k}) for k in range(3)])
    expected_idx = scipy.special.log_softmax(scipy.special.logsumexp(log_joints, axis=1))
    np.testing.assert_allclose(actual_idx, expected_idx)
    np.testing.assert_allclose(scipy.special.logsumexp(actual_idx), 0.0, atol=1e-14)


def test_recover_independent_variables():
    """Test recovering multiple independent marginalized variables."""
    with pm.Model() as m:
        idx1 = pm.Bernoulli("idx1", p=0.75)
        x = pm.Normal("x", mu=idx1)
        idx2 = pm.Bernoulli("idx2", p=0.75, shape=(5,))
        y = pm.Normal("y", mu=(idx2 * 2 - 1), shape=(5,))

    marginal_m = marginalize(m, [idx1, idx2])
    cond_m = conditional(marginal_m)

    assert set(rv.name for rv in cond_m.free_RVs) == {"idx1", "idx2", "x", "y"}

    logp_idx1 = cond_m.compile_logp(vars=[cond_m["idx1"]])

    tp = {"x": 0.5, "y": np.zeros(5), "idx1": 0, "idx2": np.zeros(5, dtype=int)}
    lp1 = [logp_idx1({**tp, "idx1": k}) for k in (0, 1)]
    np.testing.assert_allclose(scipy.special.logsumexp(lp1), 0.0, atol=1e-14)


def test_sample_posterior_predictive_single():
    """Test sample_posterior_predictive with a single recovered variable."""
    with pm.Model() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.5)
        y = pm.Normal("y", mu=idx, sigma=sigma, observed=1.0)

    marginal_m = marginalize(m, "idx")
    cond_m = conditional(marginal_m)

    # sigma=0.01 → y=1 strongly favors idx=1 (mu=1)
    idata = from_dict({"posterior": {"sigma": np.full((1, 50), 0.01)}})
    result = pm.sample_posterior_predictive(
        idata,
        model=cond_m,
        sample_vars=["idx"],
        random_seed=42,
    )
    assert result.posterior_predictive.idx.values.mean() > 0.99


@pytest.mark.parametrize(
    "build_model, marginalized_name, point",
    [
        pytest.param(
            build_normal_chain_model,
            "x",
            {"mu": 1.0, "x": 3.0, "y": 10.0},
            id="normal-normal",
        ),
        pytest.param(
            build_bernoulli_model,
            "idx",
            {"idx": 1, "y": 1.0},
            id="finite-discrete",
        ),
    ],
)
def test_roundtrip_preserves_joint_logp(build_model, marginalized_name, point):
    """marginalize() + conditional() re-factor the model but preserve the joint density.

    E.g. the normal chain factors as p(mu)*p(x|mu)*p(y|x); the conditional
    model factors it as p(mu)*p(y|mu)*p(x|y,mu) (y stays marginalized over
    x, x recovered).
    Both describe the same joint p(mu, x, y), so their total logp must match.
    """
    m = build_model()
    cond_m = conditional(marginalize(m, marginalized_name))
    np.testing.assert_allclose(
        cond_m.compile_logp()(point),
        m.compile_logp()(point),
    )


def test_nested_chain_rule_factorization():
    """Recovering all nested variables factors the posterior via the chain rule.

    P(idx, sub, subsub | y) =
        P(idx | y) * P(sub | idx, y) * P(subsub | sub, idx, y),
    with each factor checked against exact enumeration of the original
    (three-level) model.
    """
    with pm.Model() as m:
        idx = pm.Bernoulli("idx", p=0.6)
        sub = pm.Bernoulli("sub", p=pt.as_tensor([0.3, 0.7])[idx])
        subsub = pm.Bernoulli("subsub", p=pt.as_tensor([0.2, 0.9])[sub])
        pm.Normal("y", mu=subsub * 2.0, sigma=1.0)

    point = {"y": 0.7}
    ref_fn = m.compile_logp()

    def joint_lp(i, s, ss):
        return ref_fn({"idx": i, "sub": s, "subsub": ss, **point})

    ref = logsumexp([joint_lp(i, s, ss) for i, s, ss in itertools.product((0, 1), repeat=3)])

    cond_all = conditional(marginalize(m, ["idx", "sub", "subsub"]))
    test_point = {"idx": 1, "sub": 0, "subsub": 1, **point}

    # p(idx=1 | y): sub, subsub integrated out
    ref_idx = logsumexp([joint_lp(1, s, ss) for s, ss in itertools.product((0, 1), repeat=2)]) - ref
    # p(sub=0 | idx=1, y): subsub integrated out
    ref_sub = logsumexp([joint_lp(1, 0, ss) for ss in (0, 1)]) - logsumexp(
        [joint_lp(1, s, ss) for s, ss in itertools.product((0, 1), repeat=2)]
    )
    # p(subsub=1 | sub=0, idx=1, y)
    ref_subsub = joint_lp(1, 0, 1) - logsumexp([joint_lp(1, 0, ss) for ss in (0, 1)])

    for name, ref_factor in [("idx", ref_idx), ("sub", ref_sub), ("subsub", ref_subsub)]:
        factor = cond_all.compile_logp(vars=[cond_all[name]])(test_point)
        np.testing.assert_allclose(factor, ref_factor)


@pytest.mark.parametrize("explicit_model", (True, False))
def test_recover_basic(explicit_model):
    with pm.Model() as m:
        sigma = pm.HalfNormal("sigma")
        p = np.array([0.5, 0.2, 0.3])
        k = pm.Categorical("k", p=p)
        mu = np.array([-3.0, 0.0, 3.0])
        mu_ = pt.as_tensor_variable(mu)
        y = pm.Normal("y", mu=mu_[k], sigma=sigma)

    marginal_m = marginalize(m, [k])

    rng = np.random.default_rng(211)

    with marginal_m:
        prior = pm.sample_prior_predictive(
            draws=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = from_dict({"posterior": {k: np.expand_dims(v, axis=0) for k, v in prior.items()}})

    if explicit_model:
        idata = recover(idata, model=marginal_m)
    else:
        with marginal_m:
            idata = recover(idata)

    post = idata.posterior
    assert "k" in post
    assert post.k.shape == post.y.shape


def test_recover_coords():
    """Test if coords can be recovered with marginalized value had it originally"""
    with pm.Model(coords={"year": [1990, 1991, 1992]}) as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.75, dims="year")
        x = pm.Normal("x", mu=idx, sigma=sigma, dims="year")

    marginal_m = marginalize(m, [idx])
    rng = np.random.default_rng(211)

    with marginal_m:
        prior = pm.sample_prior_predictive(
            draws=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = from_dict({"posterior": {k: np.expand_dims(prior[k], axis=0) for k in prior}})

    with marginal_m:
        idata = recover(idata)
    post = idata.posterior
    assert "idx" in post
    assert post.idx.dims == ("chain", "draw", "year")


def test_recover_batched():
    """Test that marginalization works for batched random variables"""
    with pm.Model() as m:
        sigma = pm.HalfNormal("sigma")
        idx = pm.Bernoulli("idx", p=0.7, shape=(3, 2))
        y = pm.Normal("y", mu=idx.T, sigma=sigma, shape=(2, 3))

    marginal_m = marginalize(m, [idx])

    rng = np.random.default_rng(211)

    with marginal_m:
        prior = pm.sample_prior_predictive(
            draws=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = from_dict({"posterior": {k: np.expand_dims(prior[k], axis=0) for k in prior}})

        idata = recover(idata)
    post = idata.posterior
    assert post["y"].shape == (1, 20, 2, 3)
    assert post["idx"].shape == (1, 20, 3, 2)


def test_recover_nested():
    """Test that marginalization works when there are nested marginalized RVs"""

    with pm.Model() as m:
        idx = pm.Bernoulli("idx", p=0.75)
        sub_idx = pm.Bernoulli("sub_idx", p=pt.switch(pt.eq(idx, 0), 0.15, 0.95))
        sub_dep = pm.Normal("y", mu=idx + sub_idx, sigma=1.0)

    marginal_m = marginalize(m, [idx, sub_idx])

    rng = np.random.default_rng(211)

    with marginal_m:
        prior = pm.sample_prior_predictive(
            draws=20,
            random_seed=rng,
            return_inferencedata=False,
        )
        idata = from_dict({"posterior": {k: np.expand_dims(v, axis=0) for k, v in prior.items()}})

        idata = recover(idata)
    post = idata.posterior
    assert "idx" in post
    assert post.idx.shape == post.y.shape
    assert "sub_idx" in post
    assert post.sub_idx.shape == post.y.shape
