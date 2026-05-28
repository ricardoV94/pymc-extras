import warnings

from collections.abc import Sequence

import pytensor.tensor as pt

from pymc.distributions.transforms import Chain
from pymc.logprob.transforms import IntervalTransform
from pymc.model import Model, modelcontext
from pymc.model.fgraph import (
    ModelValuedVar,
    extract_dims,
    fgraph_from_model,
    model_from_fgraph,
)
from pymc.util import RandomState, _get_seeds_per_chain
from pytensor.compile import SharedVariable
from pytensor.graph import (
    Variable,
    graph_inputs,
)
from pytensor.graph.replace import graph_replace
from pytensor.graph.rewriting.basic import in2out
from pytensor.graph.rewriting.db import RewriteDatabaseQuery
from pytensor.graph.traversal import io_toposort
from pytensor.tensor import TensorVariable
from xarray import DataTree, merge

from pymc_extras.model.marginal.distributions.core import (
    MarginalRV,
    marginalized_conditional,
)
from pymc_extras.model.marginal.distributions.laplace import MarginalLaplaceRV
from pymc_extras.model.marginal.graph_analysis import (
    find_conditional_dependent_rvs,
    find_conditional_input_rvs,
    is_conditional_dependent,
)
from pymc_extras.model.marginal.rewrites import (
    DeferredMarginalSubgraph,
    LaplaceMarginalSubgraph,
    MarginalSubgraph,
    MarginalSubgraphBase,
    local_unmarginalize,
    marginal_rewrites_db,
)

ModelRVs = TensorVariable | Sequence[TensorVariable] | str | Sequence[str]


def _unique(seq: Sequence) -> list:
    """Copied from https://stackoverflow.com/a/480227"""
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def _get_marginalized_rv_names(model, unmarginal_model):
    model_var_names = set(rv.name for rv in model.free_RVs)
    return [rv.name for rv in unmarginal_model.free_RVs if rv.name not in model_var_names]


def replace_marginal_subgraph(
    fgraph, rv_to_marginalize, dependent_rvs, input_rvs, use_laplace=False, **marginalize_kwargs
) -> None:
    """Replace a marginalized subgraph with a flat MarginalSubgraph marker Op.

    The subgraph stays alive in the fgraph — the MS node references both
    the subgraph outputs and boundary vars as its inputs. No cloning here;
    rewrites clone at resolution time when building the OpFromGraph.

    If `use_laplace` is True, a LaplaceMarginalSubgraph marker is created
    instead, with the precision matrix Q appended as the last boundary input
    and the minimizer options stored on the marker.
    """
    raw_marg = rv_to_marginalize.owner.inputs[0]
    raw_deps = [
        dep.owner.inputs[0] if isinstance(dep.owner.op, ModelValuedVar) else dep
        for dep in dependent_rvs
    ]

    subgraph_outputs = [raw_marg, *raw_deps]
    boundary = list(input_rvs)
    boundary += [
        inp
        for inp in graph_inputs(subgraph_outputs, blockers=boundary)
        if (isinstance(inp, SharedVariable) and inp not in boundary)
    ]

    # Unwrap ModelValuedVar inside the subgraph so the interior only
    # references raw RVs. This prevents cycles when rv_to_marginalize
    # is replaced by the MS output below.
    subgraph_nodes = set(io_toposort(boundary, subgraph_outputs))
    for node in list(subgraph_nodes):
        if not isinstance(node.op, ModelValuedVar):
            continue
        model_var = node.outputs[0]
        raw_rv = node.inputs[0]
        for client_node, client_idx in list(fgraph.clients.get(model_var, [])):
            if client_node in subgraph_nodes:
                fgraph.change_node_input(client_node, client_idx, raw_rv, import_missing=True)

    marginalized_dims = extract_dims(rv_to_marginalize)
    n_dep = len(dependent_rvs)

    output_types = [out.type for out in subgraph_outputs]
    if use_laplace:
        # Q goes last so the logp implementation can pop it back
        boundary.append(marginalize_kwargs.pop("Q"))
        op = LaplaceMarginalSubgraph(
            n_dependent_rvs=n_dep,
            marginalized_dims=marginalized_dims,
            output_types=output_types,
            **marginalize_kwargs,
        )
    else:
        has_nested = any(
            rd.owner is not None and isinstance(rd.owner.op, MarginalSubgraphBase)
            for rd in raw_deps
        )
        cls = DeferredMarginalSubgraph if has_nested else MarginalSubgraph

        op = cls(
            n_dependent_rvs=n_dep,
            marginalized_dims=marginalized_dims,
            output_types=output_types,
        )

    new_outputs = op(*(subgraph_outputs + boundary))
    if not isinstance(new_outputs, list):
        new_outputs = list(new_outputs)

    for old, new in zip(subgraph_outputs, new_outputs):
        new.name = old.name

    fgraph.replace(rv_to_marginalize, new_outputs[0], import_missing=True)

    for i, dep in enumerate(dependent_rvs):
        ms_dep = new_outputs[1 + i]
        if isinstance(dep.owner.op, ModelValuedVar):
            fgraph.change_node_input(dep.owner, 0, ms_dep, import_missing=True)


def marginalize(
    model: Model,
    rvs_to_marginalize: ModelRVs,
    rewrite_query=RewriteDatabaseQuery(include=["basic"]),
    use_laplace: bool = False,
    **marginalize_kwargs,
) -> Model:
    """Marginalize a subset of variables in a PyMC model.

    This creates a new `Model`, with the specified variables marginalized.

    Notes
    -----
    Deterministics and Potentials cannot be conditionally dependent on the
    marginalized variables.

    Marginalization is resolved via logprob rewrites. The supported cases
    include finite discrete variables (Bernoulli, Categorical,
    DiscreteUniform, DiscreteMarkovChain) and closed-form conjugate pairs
    such as Normal-Normal.

    For finite discrete marginalization with batched dimensions, any
    conditionally dependent variables must use information from an individual
    batched dimension (i.e., the connecting graph must be strictly Elemwise).
    If you want to bypass this restriction you can separate each dimension
    of the marginalized variable into scalar components and stack them
    together. Note that such graphs will grow exponentially in the number of
    marginalized variables.

    Parameters
    ----------
    model : Model
        PyMC model to marginalize. Original variables will be cloned.
    rvs_to_marginalize : Sequence[TensorVariable]
        Variables to marginalize in the returned model.
    use_laplace : bool
        Whether to use Laplace approximations to marginalize out
        rvs_to_marginalize. Requires passing the precision matrix ``Q`` of the
        marginalized variable via ``marginalize_kwargs``, alongside optional
        ``minimizer_seed`` and ``minimizer_kwargs``.

    Returns
    -------
    marginal_model: Model
        Marginal model with the specified variables marginalized.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.marginal import marginalize

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

        marginal_m = marginalize(m, [x])
        idata = pm.sample(model=marginal_m)
    """
    if isinstance(rvs_to_marginalize, str | Variable):
        rvs_to_marginalize = (rvs_to_marginalize,)

    rvs_to_marginalize = [model[rv] if isinstance(rv, str) else rv for rv in rvs_to_marginalize]

    if not rvs_to_marginalize:
        return model

    for rv_to_marginalize in rvs_to_marginalize:
        if rv_to_marginalize not in model.free_RVs:
            raise ValueError(f"Marginalized RV {rv_to_marginalize} is not a free RV in the model")

    fg, memo = fgraph_from_model(model)
    rvs_to_marginalize_fg = [memo[rv] for rv in rvs_to_marginalize]

    rvs_to_marginalize = rvs_to_marginalize_fg
    toposort = fg.toposort()

    for rv_to_marginalize in sorted(
        rvs_to_marginalize,
        key=lambda rv: toposort.index(rv.owner),
        reverse=True,
    ):
        all_rvs = [node.out for node in fg.toposort() if isinstance(node.op, ModelValuedVar)]

        dependent_rvs = find_conditional_dependent_rvs(rv_to_marginalize, all_rvs)
        if not dependent_rvs:
            continue

        # Issue warning for IntervalTransform on dependent RVs
        for dependent_rv in dependent_rvs:
            transform = dependent_rv.owner.op.transform

            if isinstance(transform, IntervalTransform) or (
                isinstance(transform, Chain)
                and any(isinstance(tr, IntervalTransform) for tr in transform.transform_list)
            ):
                warnings.warn(
                    f"The transform {transform} for the variable {dependent_rv}, which depends on the "
                    f"marginalized {rv_to_marginalize} may no longer work if bounds depended on other variables.",
                    UserWarning,
                )

        # Check that no deterministics or potentials depend on the rv to marginalize
        for det in model.deterministics:
            if is_conditional_dependent(memo[det], rv_to_marginalize, all_rvs):
                raise NotImplementedError(
                    f"Cannot marginalize {rv_to_marginalize} due to dependent Deterministic {det}"
                )
        for pot in model.potentials:
            if is_conditional_dependent(memo[pot], rv_to_marginalize, all_rvs):
                raise NotImplementedError(
                    f"Cannot marginalize {rv_to_marginalize} due to dependent Potential {pot}"
                )

        marginalized_rv_input_rvs = find_conditional_input_rvs([rv_to_marginalize], all_rvs)
        other_direct_rv_ancestors = [
            rv
            for rv in find_conditional_input_rvs(dependent_rvs, all_rvs)
            if rv is not rv_to_marginalize
        ]
        input_rvs = _unique((*marginalized_rv_input_rvs, *other_direct_rv_ancestors))

        if use_laplace:
            # Q may reference variables of the original model; remap it to the fgraph clones
            Q = marginalize_kwargs["Q"]
            if not isinstance(Q, Variable):
                Q = pt.as_tensor_variable(Q)
            marginalize_kwargs["Q"] = memo.get(Q, Q).copy()

        replace_marginal_subgraph(
            fg, rv_to_marginalize, dependent_rvs, input_rvs, use_laplace, **marginalize_kwargs
        )

    rewriter = marginal_rewrites_db.query(rewrite_query)
    rewriter.rewrite(fg)

    remaining = [node for node in fg.toposort() if isinstance(node.op, MarginalSubgraphBase)]
    for node in remaining:
        marginalized_rv = node.inputs[0]
        n_dep = node.op.n_dependent_rvs
        dependent_rvs = node.inputs[1 : 1 + n_dep]
        raise NotImplementedError(
            f"Cannot marginalize {node.outputs[0]} with distribution "
            f"{marginalized_rv.owner.op} and dependent variables "
            f"{[rv.owner.op for rv in dependent_rvs]}. "
        )

    return model_from_fgraph(fg, mutate_fgraph=True)


def _validate_recover_var_names(var_names, marginalized_rv_names):
    if var_names is None:
        return list(marginalized_rv_names)
    var_names = [var if isinstance(var, str) else var.name for var in var_names]
    var_names_to_recover = [name for name in marginalized_rv_names if name in var_names]
    missing_names = [name for name in var_names if name not in marginalized_rv_names]
    if missing_names:
        raise ValueError(f"Unrecognized var_names: {missing_names}")
    return var_names_to_recover


def _find_laplace_marginalized_names(apply_nodes) -> list[str]:
    """Names of Laplace-marginalized variables, including ones absorbed into other MarginalRVs."""
    names = []
    for node in apply_nodes:
        if isinstance(node.op, MarginalLaplaceRV):
            names.append(node.op.inner_outputs[0].name)
        if isinstance(node.op, MarginalRV):
            names.extend(_find_laplace_marginalized_names(node.op.fgraph.apply_nodes))
    return names


def unmarginalize(model: Model, rvs_to_unmarginalize: str | Sequence[str] | None = None) -> Model:
    """Unmarginalize a subset of variables in a PyMC model.


    Parameters
    ----------
    model : Model
        PyMC model to unmarginalize. Original variables well be cloned.
    rvs_to_unmarginalize : str or sequence of str, optional
        Variables to unmarginalize in the returned model. If None, all variables are
        unmarginalized.

    Returns
    -------
    unmarginal_model: Model
        Model with the specified variables unmarginalized.
    """

    fg, _memo = fgraph_from_model(model)

    if rvs_to_unmarginalize is not None:
        if not isinstance(rvs_to_unmarginalize, list | tuple):
            rvs_to_unmarginalize = (rvs_to_unmarginalize,)
        rvs_to_unmarginalize = set(rvs_to_unmarginalize)

        # Laplace-marginalized RVs that are kept marginalized would be re-marginalized
        # without the Q / minimizer options they were created with, which cannot be
        # recovered from the unmarginalized graph.
        kept_laplace_rvs = [
            name
            for name in _find_laplace_marginalized_names(fg.apply_nodes)
            if name not in rvs_to_unmarginalize
        ]
        if kept_laplace_rvs:
            raise NotImplementedError(
                f"Laplace-marginalized variables {kept_laplace_rvs} cannot be kept marginalized "
                "through a partial unmarginalize, because their precision matrix Q and minimizer "
                "options are not currently preserved when re-marginalizing. Either include them "
                "in rvs_to_unmarginalize, or rebuild the model with "
                "marginalize(..., use_laplace=True) from scratch."
            )

    # Unmarginalize all the MarginalRVs
    in2out(local_unmarginalize, ignore_newtrees=False).apply(fg)
    unmarginalized_model = model_from_fgraph(fg, mutate_fgraph=True)
    if rvs_to_unmarginalize is None:
        return unmarginalized_model

    # Re-marginalize the variables we want to keep marginalized
    old_free_rv_names = set(rv.name for rv in model.free_RVs)
    new_free_rv_names = set(
        rv.name for rv in unmarginalized_model.free_RVs if rv.name not in old_free_rv_names
    )
    if rvs_to_unmarginalize - new_free_rv_names:
        raise ValueError(
            f"Unrecognized rvs_to_unmarginalize: {rvs_to_unmarginalize - new_free_rv_names}"
        )
    rvs_to_keep_marginalized = tuple(new_free_rv_names - rvs_to_unmarginalize)
    return marginalize(unmarginalized_model, rvs_to_keep_marginalized)


def conditional(
    model: Model,
    rvs_to_recover: ModelRVs | None = None,
) -> Model:
    """Replace marginalized variables with their conditional distributions.

    Returns a new model where the specified marginalized variables become
    free RVs whose distributions are their conditionals given the dependents.
    Unspecified marginalized variables stay marginalized (integrated out).

    The returned model can be used with ``pm.sample_posterior_predictive``
    to draw conditional posterior samples, or with ``model.compile_logp``
    to evaluate conditional log-probabilities.

    The input is a marginalized model. Starting from an original model
    factored as ``p(mu) * p(x|mu) * p(y|x)``, marginalizing ``x`` yields
    ``p(mu) * p(y|mu)``. ``conditional`` adds ``x`` back as its conditional
    distribution, giving ``p(mu) * p(y|mu) * p(x|y, mu)`` -- a re-factorization
    of the same full joint ``p(mu, x, y)``: the recovered variable follows the
    conditional ``p(x|y, mu)``, while each dependent stays marginalized over it.

    Selecting variables matters when evaluating logp:
    ``model.compile_logp(vars=[model["x"]])`` gives the conditional
    ``p(x|y, mu)``, while the unqualified ``model.compile_logp()`` is the full
    joint ``p(mu, x, y)``.

    Parameters
    ----------
    model : Model
        PyMC model with marginalized variables.
    rvs_to_recover : str, sequence of str, or None
        Marginalized variables to recover. Defaults to all.

    Returns
    -------
    Model
        Model with the specified variables as free RVs with conditional
        distributions.

    Examples
    --------
    **Basic usage** — recover a marginalized variable:

    .. code-block:: python

        import pymc as pm
        from pymc_extras.marginal import marginalize, conditional

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            idx = pm.Bernoulli("idx", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(idx, -10, 10), observed=[10, 10, -10])

        marginal_m = marginalize(m, [idx])
        idata = pm.sample(model=marginal_m)

        # Get model with idx's conditional posterior as its distribution
        cond_m = conditional(marginal_m)
        logp_fn = cond_m.compile_logp(vars=[cond_m["idx"]])
        pm.sample_posterior_predictive(idata, model=cond_m, sample_vars=["idx"])

    **Nested marginalization** — recover a subset (marginal posterior):

    When multiple variables are marginalized, specifying a subset recovers
    those variables with the others integrated out (marginal posterior).

    .. code-block:: python

        with pm.Model() as m:
            idx = pm.Bernoulli("idx", p=0.5)
            sub_idx = pm.Bernoulli("sub_idx", p=f(idx))
            y = pm.Normal("y", mu=idx + sub_idx, sigma=1)

        marginal_m = marginalize(m, ["idx", "sub_idx"])

        # Marginal posterior of idx (sub_idx integrated out):
        # P(idx | y, σ) = Σ_sub_idx P(idx, sub_idx | y, σ)
        cond_idx = conditional(marginal_m, "idx")

        # Marginal posterior of sub_idx (idx integrated out):
        # P(sub_idx | y, σ) = Σ_idx P(idx, sub_idx | y, σ)
        cond_sub = conditional(marginal_m, "sub_idx")

    **Recovering all nested variables** — joint posterior factorization:

    When recovering all marginalized variables at once, the joint
    posterior is factored via the chain rule in recovery order.  Each
    variable integrates out the not-yet-recovered ones and conditions
    on the already-recovered ones:

    .. code-block:: python

        # P(idx, sub_idx | y) = P(idx | y) · P(sub_idx | idx, y)
        cond_all = conditional(marginal_m)

        # idx's logp does NOT depend on sub_idx (sub_idx is integrated out):
        logp_idx = cond_all.compile_logp(vars=[cond_all["idx"]])

        # sub_idx's logp depends on idx:
        logp_sub = cond_all.compile_logp(vars=[cond_all["sub_idx"]])

    The result is a valid generative DAG — draw exact joint posterior
    samples by forward-sampling through it.

    **Full conditional via unmarginalize:**

    To get the full conditional ``P(idx | sub_idx, y)`` (conditioning
    on ``sub_idx`` rather than integrating it out), first unmarginalize
    ``sub_idx`` so it becomes a free RV with its original prior, then
    conditionalize ``idx``:

    .. code-block:: python

        from pymc_extras.marginal import unmarginalize

        partial_m = unmarginalize(marginal_m, "sub_idx")
        cond_full = conditional(partial_m, "idx")
        # User must provide sub_idx values when evaluating
    """
    unmarginal_model = unmarginalize(model)
    marginalized_rv_names = _get_marginalized_rv_names(model, unmarginal_model)

    if rvs_to_recover is None:
        var_names_to_recover = list(marginalized_rv_names)
    else:
        if isinstance(rvs_to_recover, str | Variable):
            rvs_to_recover = (rvs_to_recover,)
        var_names_to_recover = _validate_recover_var_names(rvs_to_recover, marginalized_rv_names)

    [n for n in marginalized_rv_names if n not in var_names_to_recover]

    if not var_names_to_recover:
        return model

    # Chain-rule factorization of the joint posterior.  The base is the
    # marginal model — dependents keep their marginal distribution via the
    # MarginalRV, so the conditional can reference them without cycles.
    fg, _memo = fgraph_from_model(model)

    # Check if all requested vars can be found directly in fg.
    # If any are nested, recover ALL vars via the chain-rule, then
    # re-marginalize the unwanted ones (the chain-rule model IS the joint
    # posterior, so Σ_unwanted p(all|y) = p(kept|y)).
    all_direct = all(_find_marg_rv(fg, name)[0] is not None for name in var_names_to_recover)
    vars_to_add = var_names_to_recover if all_direct else list(marginalized_rv_names)

    for var_name in vars_to_add:
        marg_node, source_fg = _find_marg_rv(fg, var_name)
        if marg_node is not None:
            _add_conditional(fg, marg_node, source_fg, var_name)
        else:
            source_fg, _ = fgraph_from_model(marginalize(unmarginal_model, [var_name]))
            marg_node, _ = _find_marg_rv(source_fg, var_name)
            if marg_node is not None:
                _add_conditional(fg, marg_node, source_fg, var_name)
            else:
                raise NotImplementedError(
                    f"Cannot build conditional for nested variable '{var_name}'. "
                    f"Use conditional(model) to recover all marginalized variables "
                    f"together, or unmarginalize the parent variables first."
                )

    result = model_from_fgraph(fg, mutate_fgraph=True)

    # Re-marginalize vars that were recovered only for the chain-rule
    # but weren't requested by the user.
    vars_to_remarginalize = [n for n in vars_to_add if n not in var_names_to_recover]
    if vars_to_remarginalize:
        result = marginalize(result, vars_to_remarginalize)

    return result


def _find_marg_rv(fg, var_name):
    """Find the MarginalRV in ``fg`` whose marginalized variable is ``var_name``."""
    for node in fg.toposort():
        if isinstance(node.op, MarginalRV) and node.op.inner_outputs[0].name == var_name:
            return node, fg
    return None, fg


def _remap_to_fg(sample_graph, source_fg, fg):
    """Remap ``sample_graph`` references from ``source_fg`` to ``fg``.

    When ``source_fg is fg`` this is a no-op.  Otherwise, maps model RVs
    and fgraph inputs by name so nothing from source_fg leaks into fg.
    """
    if source_fg is fg:
        return sample_graph

    remap = {}

    # Model RVs → fg model RVs (by name)
    for src_node in source_fg.toposort():
        if not isinstance(src_node.op, ModelValuedVar):
            continue
        name = src_node.outputs[0].name
        fg_node = next(
            (
                n
                for n in fg.toposort()
                if isinstance(n.op, ModelValuedVar) and n.outputs[0].name == name
            ),
            None,
        )
        if fg_node is not None:
            remap[src_node.outputs[0]] = fg_node.outputs[0]

    # Fgraph inputs (value variables) → fg inputs (by name)
    fg_inputs_by_name = {
        getattr(inp, "name", None): inp for inp in fg.inputs if getattr(inp, "name", None)
    }
    for src_inp in source_fg.inputs:
        src_name = getattr(src_inp, "name", None)
        if src_name and src_name in fg_inputs_by_name:
            remap[src_inp] = fg_inputs_by_name[src_name]

    if remap:
        [sample_graph] = graph_replace([sample_graph], replace=remap, strict=False)
    return sample_graph


def _add_conditional(fg, marg_node, source_fg, var_name):
    """Dispatch on ``marg_node`` (from ``source_fg``), wire result into ``fg``, add as free RV."""
    from pymc.model.fgraph import ModelObservedRV, model_free_rv

    op = marg_node.op
    n_dep = op.n_dependent_rvs

    # Dispatch → sample_graph with dep_dummies
    sample_graph, dep_dummies = marginalized_conditional(op, marg_node)
    replacements = dict(zip(op.inner_inputs, marg_node.inputs))
    [sample_graph] = graph_replace([sample_graph], replace=replacements, strict=False)

    # Map dep_dummies → fg's dependent model RVs (or observed data constants)
    dep_remap = {}
    for k, dep_output in enumerate(marg_node.outputs[1 : 1 + n_dep]):
        clients = source_fg.clients.get(dep_output, [])
        mv_client = next((c for c, _ in clients if isinstance(c.op, ModelValuedVar)), None)
        dep_name = mv_client.outputs[0].name
        is_observed = isinstance(mv_client.op, ModelObservedRV)

        fg_mv = next(
            n
            for n in fg.toposort()
            if isinstance(n.op, ModelValuedVar) and n.outputs[0].name == dep_name
        )
        dep_remap[dep_dummies[k]] = fg_mv.inputs[1] if is_observed else fg_mv.outputs[0]

    [sample_graph] = graph_replace([sample_graph], replace=dep_remap, strict=False)

    # Remap remaining source_fg references to fg
    sample_graph = _remap_to_fg(sample_graph, source_fg, fg)

    # Import new shared variables (e.g. RNGs from Categorical.dist)
    for inp in graph_inputs([sample_graph]):
        if isinstance(inp, SharedVariable) and inp not in fg.inputs:
            fg.add_input(inp)

    # Add the conditional as a new free RV
    sample_graph.name = var_name
    value = sample_graph.type()
    value.name = var_name
    fg.add_input(value)
    conditional_free_rv = model_free_rv(sample_graph, value, None, *op.marginalized_dims)
    fg.add_output(conditional_free_rv, reason="conditionalize")


def recover(
    idata: DataTree,
    *,
    model: Model | None = None,
    var_names: Sequence[str] | None = None,
    extend_inferencedata: bool = True,
    random_seed: RandomState = None,
):
    """Sample marginalized variables from their conditional posterior.

    Builds the chain-rule factorization of the joint posterior via
    :func:`conditional` and forward-samples all recovered variables
    together.  For more control, use :func:`conditional` directly.

    Parameters
    ----------
    idata : DataTree
        DataTree with posterior group.
    model : Model, optional
        PyMC model with marginalized variables.
    var_names : sequence of str, optional
        Variables to recover. Defaults to all marginalized variables.
    extend_inferencedata : bool, default True
        Whether to extend the original DataTree or return a new Dataset.
    random_seed : int, array-like of int or SeedSequence, optional
        Seed for generating samples.

    Returns
    -------
    idata : DataTree or Dataset
        DataTree with recovered samples added to posterior, or a new Dataset.

    Examples
    --------
    .. code-block:: python

        import pymc as pm
        from pymc_extras.marginal import marginalize, recover

        with pm.Model() as m:
            p = pm.Beta("p", 1, 1)
            x = pm.Bernoulli("x", p=p, shape=(3,))
            y = pm.Normal("y", pm.math.switch(x, -10, 10), observed=[10, 10, -10])

        marginal_m = marginalize(m, [x])
        idata = pm.sample(model=marginal_m)
        recover(idata, model=marginal_m)
    """
    import pymc as pm

    if isinstance(idata, Model):
        raise TypeError(
            "The order of arguments of `recover` changed. " "The first input must be an idata"
        )

    model = modelcontext(model)
    unmarginal_model = unmarginalize(model)
    marginalized_rv_names = _get_marginalized_rv_names(model, unmarginal_model)
    var_names_to_recover = _validate_recover_var_names(var_names, marginalized_rv_names)

    if random_seed is not None:
        _get_seeds_per_chain(random_seed, len(var_names_to_recover))
    else:
        [None] * len(var_names_to_recover)

    # Build a single conditional model recovering all requested variables
    # via the chain-rule factorization.  This handles nested variables
    # correctly (each conditions on the already-recovered ones and
    # integrates out the not-yet-recovered ones).  Sample all recovered
    # variables together so the chain-rule dependencies are satisfied
    # (e.g. sub_idx's conditional uses idx's sampled value).
    cond_model = conditional(model, var_names_to_recover)
    freeze = [rv.name for rv in cond_model.free_RVs if rv.name not in var_names_to_recover]

    sample_result = pm.sample_posterior_predictive(
        idata,
        model=cond_model,
        sample_vars=var_names_to_recover,
        freeze_vars=freeze,
        random_seed=random_seed,
        progressbar=False,
    )
    pp = sample_result.posterior_predictive
    pp_ds = pp.dataset if isinstance(pp, DataTree) else pp

    all_datasets = [pp_ds[var_names_to_recover]]

    if not all_datasets:
        return idata

    rv_dataset = all_datasets[0]
    for ds in all_datasets[1:]:
        rv_dataset = merge([rv_dataset, ds], compat="override")

    if extend_inferencedata:
        idata["posterior"] = idata["posterior"].assign(rv_dataset)
        return idata
    else:
        return rv_dataset


def recover_marginals(*args, return_samples: bool = True, **kwargs):
    """Deprecated alias for :func:`recover`.

    .. deprecated::
        ``recover_marginals`` has been renamed to :func:`recover` (available as
        ``pymc_extras.marginal.recover``).  Unlike the old implementation, it no
        longer returns the posterior log-probabilities of the marginalized
        variables (the ``lp_*`` arrays / ``return_samples=False`` mode); use
        :func:`conditional` together with ``Model.compile_logp`` to evaluate
        those instead.
    """
    warnings.warn(
        "`recover_marginals` has been renamed to `recover` and moved to the "
        "`pymc_extras.marginal` namespace (`pymc_extras.marginal.recover`).",
        FutureWarning,
        stacklevel=2,
    )
    if not return_samples:
        raise NotImplementedError(
            "`recover` no longer returns posterior log-probabilities of the "
            "marginalized variables. Use `conditional(...)` with "
            "`Model.compile_logp` to evaluate them instead."
        )
    return recover(*args, **kwargs)


__all__ = ["conditional", "marginalize", "recover", "recover_marginals", "unmarginalize"]
