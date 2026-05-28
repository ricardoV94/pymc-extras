import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest
import scipy

from arviz import from_dict
from pymc.model.transform.conditioning import remove_value_transforms

from pymc_extras import marginalize
from pymc_extras.model.marginal.model import conditional, unmarginalize


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


class TestConditional:
    def test_finite_discrete_logp(self):
        """Test that conditional gives correct conditional logp for Bernoulli."""
        with pm.Model() as m:
            sigma = pm.HalfNormal("sigma")
            idx = pm.Bernoulli("idx", p=0.5)
            y = pm.Normal("y", mu=idx, sigma=sigma)

        marginal_m = marginalize(m, "idx")
        cond_m = conditional(marginal_m)

        assert "idx" in [rv.name for rv in cond_m.free_RVs]

        logps = compute_conditional_logprob(
            cond_m, "idx", domain=[0, 1], point={"sigma_log__": 0.0, "y": 2.0}
        )

        # Manual: log P(idx=k | y=2, sigma=1) ∝ log P(y=2|idx=k, sigma=1) + log P(idx=k)
        expected = scipy.special.log_softmax(
            [scipy.stats.norm.logpdf(2.0, k, 1) + np.log(0.5) for k in (0, 1)]
        )
        np.testing.assert_allclose(logps, expected)

    def test_with_remove_value_transforms(self):
        """Test that remove_value_transforms + conditional gives natural-scale inputs."""
        with pm.Model() as m:
            sigma = pm.HalfNormal("sigma")
            idx = pm.Bernoulli("idx", p=0.5)
            y = pm.Normal("y", mu=idx, sigma=sigma)

        marginal_m = marginalize(m, "idx")

        # Default: should have transformed value variable
        cond_m = conditional(marginal_m)
        assert any(
            rv.name == "sigma" and cond_m.rvs_to_transforms[rv] is not None
            for rv in cond_m.free_RVs
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

    def test_categorical_conditional(self):
        """Test conditional with Categorical marginalized variable."""
        p = np.array([0.1, 0.3, 0.6])
        mu = np.array([-3.0, 0.0, 3.0])

        with pm.Model() as m:
            k = pm.Categorical("k", p=p)
            y = pm.Normal("y", mu=pt.as_tensor(mu)[k], sigma=1.0)

        marginal_m = marginalize(m, "k")
        cond_m = conditional(marginal_m)

        y_val = 2.5
        logps = compute_conditional_logprob(cond_m, "k", domain=range(3), point={"y": y_val})
        expected = scipy.special.log_softmax(np.log(p) + scipy.stats.norm.logpdf(y_val, mu, 1.0))
        np.testing.assert_allclose(logps, expected)

    def test_marginal_vs_full_conditional(self):
        """Test marginal posterior vs full conditional via unmarginalize."""
        with pm.Model() as m:
            idx = pm.Bernoulli("idx", p=0.5)
            sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor([0.3, 0.7])[idx])
            y = pm.Normal("y", mu=(idx + sub_idx) - 1, sigma=0.5)

        marginal_m = marginalize(m, ["idx", "sub_idx"])

        # Marginal posterior: P(idx | y) with sub_idx integrated out
        cond_marginal = conditional(marginal_m, "idx")
        assert "sub_idx" not in [rv.name for rv in cond_marginal.free_RVs]
        logp_marginal = cond_marginal.compile_logp(vars=[cond_marginal["idx"]])
        lp_marginal = [logp_marginal({"y": 0.5, "idx": k}) for k in (0, 1)]
        np.testing.assert_allclose(scipy.special.logsumexp(lp_marginal), 0.0, atol=1e-14)

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

    def test_recover_all_nested(self):
        """Test recovering all nested variables gives chain-rule factorization."""
        with pm.Model() as m:
            idx = pm.Bernoulli("idx", p=0.5)
            sub_idx = pm.Bernoulli("sub_idx", p=pt.as_tensor([0.3, 0.7])[idx])
            y = pm.Normal("y", mu=(idx + sub_idx) - 1, sigma=0.5)

        marginal_m = marginalize(m, ["idx", "sub_idx"])
        cond_all = conditional(marginal_m)

        assert set(rv.name for rv in cond_all.free_RVs) == {"idx", "sub_idx", "y"}

        # Each logp is a valid conditional (sums to 1 over domain)
        logp_idx = cond_all.compile_logp(vars=[cond_all["idx"]])
        logp_sub = cond_all.compile_logp(vars=[cond_all["sub_idx"]])

        tp = {"y": 0.5, "idx": 1, "sub_idx": 0}
        lp_idx = [logp_idx({**tp, "idx": k}) for k in (0, 1)]
        lp_sub = [logp_sub({**tp, "sub_idx": k}) for k in (0, 1)]
        np.testing.assert_allclose(scipy.special.logsumexp(lp_idx), 0.0, atol=1e-14)
        np.testing.assert_allclose(scipy.special.logsumexp(lp_sub), 0.0, atol=1e-14)

        # Chain-rule factorization: idx has P(idx | y) (sub_idx integrated out),
        # so idx's logp does NOT depend on sub_idx
        lp_a = logp_idx({"y": 0.5, "idx": 0, "sub_idx": 0})
        lp_b = logp_idx({"y": 0.5, "idx": 0, "sub_idx": 1})
        assert np.isclose(lp_a, lp_b)

        # sub_idx has P(sub_idx | idx, y), so it DOES depend on idx
        lp_c = logp_sub({"y": 0.5, "idx": 0, "sub_idx": 0})
        lp_d = logp_sub({"y": 0.5, "idx": 1, "sub_idx": 0})
        assert not np.isclose(lp_c, lp_d)

    def test_recover_nested_subset(self):
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

    def test_recover_independent_variables(self):
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

    def test_sample_posterior_predictive_single(self):
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
