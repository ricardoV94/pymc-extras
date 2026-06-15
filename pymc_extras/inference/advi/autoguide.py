from dataclasses import dataclass, field

import numpy as np
import pytensor.tensor as pt

from pymc.distributions import Normal
from pymc.logprob.basic import conditional_logp
from pymc.model.core import Deterministic, Model
from pymc.pytensorf import resolve_shapes
from pytensor.gradient import disconnected_grad
from pytensor.graph import ancestors
from pytensor.graph.basic import Variable, equal_computations
from pytensor.graph.replace import graph_replace


def get_value_shapes_and_dims(
    model: Model,
) -> tuple[dict[Variable, Variable], dict[Variable, tuple | None]]:
    """Return the symbolic shapes and model dims of the value variables of the model's free RVs.

    Guides parameterize the model in the space of its value variables (the unconstrained
    space), where a transformed RV may have a different shape than the RV itself
    (e.g., a Dirichlet of size n has a simplex-transformed value variable of size n - 1).

    Model dims describe each RV in its constrained space. They carry over to the value
    variable only when the transform preserves the variable's shape: there is no
    transform, the transform is elemwise (``ndim_supp == 0``), or the symbolic shapes
    of the RV and its value variable can be shown to be equal.
    """
    free_rvs = model.free_RVs
    transformed_rvs = [
        rv
        if (transform := model.rvs_to_transforms[rv]) is None
        else transform.forward(rv, *rv.owner.inputs)
        for rv in free_rvs
    ]
    shapes = resolve_shapes([var.shape for var in (*transformed_rvs, *free_rvs)])
    value_shapes, rv_shapes = shapes[: len(free_rvs)], shapes[len(free_rvs) :]
    if overlap := (set(free_rvs) & set(ancestors(value_shapes))):
        raise ValueError(f"value shapes still depend on the following rvs {overlap}")

    value_dims = {}
    for rv, value_shape, rv_shape in zip(free_rvs, value_shapes, rv_shapes):
        dims = model.named_vars_to_dims.get(rv.name, None)
        if dims is not None:
            transform = model.rvs_to_transforms[rv]
            preserves_shape = (
                transform is None
                or transform.ndim_supp == 0
                or equal_computations([value_shape], [rv_shape])
            )
            if not preserves_shape:
                dims = None
        value_dims[rv] = dims

    return dict(zip(free_rvs, value_shapes)), value_dims


@dataclass(frozen=True)
class AutoGuideModel:
    model: Model
    params_init_values: dict[Variable, np.ndarray]
    name_to_param: dict[str, Variable] = field(init=False)

    def __post_init__(self):
        object.__setattr__(
            self,
            "name_to_param",
            {x.name: x for x in self.params_init_values.keys()},
        )

    @property
    def params(self) -> tuple[Variable, ...]:
        return tuple(self.params_init_values.keys())

    def __getitem__(self, name: str) -> Variable:
        return self.name_to_param[name]

    def stochastic_logq(self, path_derivative_gradient: bool = True) -> pt.TensorVariable:
        """Returns a graph representing the logp of the guide model, evaluated under draws from its random variables.

        When ``path_derivative_gradient`` is True, the variational parameters are detached
        from the density evaluation, so that gradients flow only through the random draws.
        This yields the lower-variance path-derivative gradient estimator of _[1], also
        known as "sticking the landing".

        References
        ----------
        .. [1] Geoffrey Roeder, Yuhuai Wu, and David Duvenaud. Sticking the Landing: Simple,
               Lower-Variance Gradient Estimators for Variational Inference. NeurIPS, 2017.
        """
        logp_terms = conditional_logp(
            {rv: rv for rv in self.model.deterministics},
            warn_rvs=False,
        )
        logq = pt.sum([logp_term.sum() for logp_term in logp_terms.values()])

        if path_derivative_gradient:
            # Detach variational parameters from the density evaluation. The remaining
            # gradient flows through the random draws (the path derivative); the dropped
            # score-function term has zero expectation.
            repl = {p: disconnected_grad(p) for p in self.params}
            logq = graph_replace(logq, repl)

        return logq


def AutoDiagonalNormal(model: Model, random_seed=None) -> AutoGuideModel:
    """
    Create a guide model for ADVI with a mean-field normal distribution.

    A guide model is a variational distribution that approximates the posterior distribution of the model's free
    random variables. In this case, we use a mean-field normal distribution, which assumes that the free random
    variables are independent and normally distributed. For details, see _[1].

    The guide is parameterized in the space of the model's value variables (the unconstrained space), as in _[1].
    For each free random variable in the model, we create a corresponding random variable in the guide model with a
    normal distribution over its value variable. The mean and standard deviation of each normal distribution are
    parameterized by learnable parameters (loc and scale), which are initialized to small random values.

    Parameters
    ----------
    model : Model
        The probabilistic model for which to create the guide.
    random_seed : optional
        Seed passed to ``model.initial_point``, which is only used when a variable's
        initial value strategy is random (e.g. a prior draw).

    Returns
    -------
    guide_model : AutoGuideModel
        An AutoGuideModel containing the guide model and the initial values for its parameters.

    References
    ----------
    .. [1] Alp Kucukelbir, Dustin Tran, Rajesh Ranganath, Andrew Gelman, and David M. Blei. Automatic Differentiation
           Variational Inference. Journal of Machine Learning Research, 18(14):1–45, 2017.
    """
    coords = model.coords
    free_rvs = model.free_RVs

    if discrete_rvs := [
        rv.name for rv in free_rvs if not model.rvs_to_values[rv].type.dtype.startswith("float")
    ]:
        raise ValueError(
            f"ADVI requires continuous free RVs, but {discrete_rvs} are discrete. "
            "Marginalize them out or use another inference method."
        )

    value_shapes, value_dims = get_value_shapes_and_dims(model)
    # Initialize the guide means at the model initial point (already in the unconstrained
    # space), so the first guide draws start in a region of finite model logp
    initial_point = model.initial_point(random_seed=random_seed)
    params_init_values = {}

    # model=None detaches the guide from any model context the user may be inside,
    # which would otherwise register the guide as a nested submodel
    with Model(coords=coords, model=None) as guide_model:
        for rv in free_rvs:
            value_var = model.rvs_to_values[rv]
            loc = pt.tensor(f"{rv.name}_loc", shape=value_var.type.shape)
            scale = pt.tensor(f"{rv.name}_scale", shape=value_var.type.shape)

            loc_init = initial_point[value_var.name]
            params_init_values[loc] = loc_init
            # TODO: Make the scale init customizable
            params_init_values[scale] = np.full_like(loc_init, 0.1)

            z = Normal(
                f"{rv.name}_z",
                mu=0.0,
                sigma=1.0,
                shape=value_shapes[rv],
            )
            Deterministic(
                rv.name,
                loc + pt.softplus(scale) * z,
                dims=value_dims[rv],
            )

    return AutoGuideModel(guide_model, params_init_values)
