import numpy as np
import pymc as pm

from pymc.model.fgraph import fgraph_from_model

from pymc_extras.marginal import marginalize, unmarginalize
from pymc_extras.model.marginal.distributions.laplace import MarginalLaplaceRV


def test_mixed_laplace_marginalization():
    """Laplace settings survive re-marginalization, and the joint mixed call works."""

    def build_model():
        with pm.Model() as m:
            x = pm.MvNormal("x", mu=np.zeros(2), tau=np.eye(2))
            z = pm.Bernoulli("z", p=pm.math.sigmoid(x.sum()))
            pm.Normal("y", mu=z * 2.0, sigma=1.0, observed=1.5)
        return m

    minimizer_kwargs = {"method": "L-BFGS-B", "optimizer_kwargs": {"tol": 1e-6}}

    def assert_laplace_preserved(marginal_model):
        fg, _ = fgraph_from_model(marginal_model)
        [laplace_op] = [n.op for n in fg.apply_nodes if isinstance(n.op, MarginalLaplaceRV)]
        assert laplace_op.marginalized_name == "x"
        assert laplace_op.minimizer_kwargs == minimizer_kwargs

    # Sequential: laplace first, then a variable it absorbed as dependent
    laplace_m = marginalize(
        build_model(), laplace_approx={"x": np.eye(2)}, minimizer_kwargs=minimizer_kwargs
    )
    assert_laplace_preserved(marginalize(laplace_m, "z"))

    # Joint single call with mixed settings
    joint_m = marginalize(
        build_model(), "z", laplace_approx={"x": np.eye(2)}, minimizer_kwargs=minimizer_kwargs
    )
    assert_laplace_preserved(joint_m)

    # Partial unmarginalize recovers z but keeps x marginalized with its settings
    assert_laplace_preserved(unmarginalize(joint_m, "z"))
