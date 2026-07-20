"""Marginalize the coordinates of a latent that never reach the likelihood.

`marginalize` removes a whole random variable. This removes a *subset* of one:
the rows of a packed Gaussian latent that nothing downstream reads.

Packing prediction inputs into the prior is free when the latent is
marginalized, but not when it is *sampled*: the unobserved rows become
coordinates NUTS must explore, constrained only by the prior and correlated with
the observed ones through K.

Their contribution to the model density is a factor that integrates to one, so
removing them changes nothing about the posterior over what remains -- this is
marginalization in the degenerate case, with no conjugacy requirement. What is
left is recovered afterwards by `project` and `conditional_covariance`, which
are exactly ``p(f_unobserved | f_observed)``.

Scoped to slice-shaped uses (`pt.unpack`, `gp[:n]`), where the unobserved
coordinates are axis-aligned. For a general affine map the unused subspace is
the null space of ``A``, which is not a set of rows, and this declines.
"""

import pytensor.tensor as pt

from pymc import MvNormal
from pymc.model.fgraph import ModelFreeRV, fgraph_from_model, model_free_rv, model_from_fgraph
from pytensor.graph.fg import FunctionGraph, Output
from pytensor.graph.traversal import ancestors
from pytensor.tensor.basic import Split
from pytensor.tensor.subtensor import Subtensor, get_idx_list

__all__ = ["marginalize_subset"]


def _observed_prefix(fgraph, rv):
    """Length of the leading block of `rv` that reaches anything else.

    Returns None unless every use of `rv` is a slice of the form ``rv[:n]``
    (which is what both ``gp[:n]`` and ``pt.unpack`` reduce to).
    """
    lengths = set()
    for client, _ in fgraph.clients.get(rv, []):
        op = getattr(client, "op", None)
        if isinstance(op, Output):
            # every model variable is an fgraph output; not a real use
            continue
        if isinstance(op, Split):
            # only the first partition may be used
            used = [i for i, o in enumerate(client.outputs) if fgraph.clients.get(o)]
            if used != [0]:
                return None
            splits = client.inputs[1]
            try:
                lengths.add(int(splits.eval()[0]))
            except Exception:
                return None
        elif isinstance(op, Subtensor):
            [idx] = get_idx_list(client.inputs, op.idx_list)
            if (
                not isinstance(idx, slice)
                or idx.start not in (None, 0)
                or idx.step not in (None, 1)
            ):
                return None
            try:
                lengths.add(int(pt.as_tensor(idx.stop).eval()))
            except Exception:
                return None
        else:
            return None

    if len(lengths) != 1:
        return None
    return lengths.pop()


def marginalize_subset(model, name):
    """Marginalize the rows of `name` that nothing downstream reads.

    Their factor in the joint integrates to one, so the posterior over the
    remaining rows is unchanged -- this is marginalization in the degenerate
    case, needing no conjugacy. The returned model samples only the rows that
    reach the likelihood.

    Recover the dropped rows with ``project`` / ``conditional_covariance``,
    which give exactly the conditional they would have been sampled from, at
    the packed inputs or at any others.
    """
    fg, memo = fgraph_from_model(model)
    rv_out = memo[model[name]]

    [model_rv] = [
        v for v in fg.variables if v.owner and isinstance(v.owner.op, ModelFreeRV) and v is rv_out
    ]
    rv = model_rv.owner.inputs[0]
    if not isinstance(rv.owner.op, MvNormal):
        raise NotImplementedError(f"{name} is not an MvNormal")

    n_obs = _observed_prefix(fg, model_rv)
    if n_obs is None:
        raise NotImplementedError(
            f"Uses of {name} are not a single leading slice; cannot identify an unobserved block."
        )

    mu, cov = rv.owner.op.dist_params(rv.owner)
    if rv in ancestors([mu, cov]):
        raise NotImplementedError("Self-referential prior parameters")

    small = MvNormal.dist(mu=pt.atleast_1d(mu)[:n_obs], cov=cov[:n_obs, :n_obs])
    # name / dims / transform live on the Op, not among the inputs
    op = model_rv.owner.op
    [value] = model_rv.owner.inputs[1:]
    new_value = small.type()
    new_value.name = value.name
    new_model_rv = model_free_rv(small, new_value, op.transform, op.name, *op.dims)

    # every use was `rv[:n_obs]`, which the reduced variable now supplies whole
    slice_uses = [
        client.outputs[0]
        for client, _ in list(fg.clients[model_rv])
        if not isinstance(getattr(client, "op", None), Output)
    ]
    fg.replace_all([(u, new_model_rv) for u in slice_uses], import_missing=True)

    # `model_rv` is itself an fgraph output, and the replacement has a different
    # shape, so the output list is rebuilt rather than swapped in place.
    new_outputs = [new_model_rv if out is model_rv else out for out in fg.outputs]
    return model_from_fgraph(FunctionGraph(outputs=new_outputs, clone=False))
