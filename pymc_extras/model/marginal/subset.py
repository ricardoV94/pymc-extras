"""Marginalize a *named* sub-block of a Gaussian latent.

`marginalize` removes a whole random variable. This is the same call removing
part of one: the
coordinates of a packed `MvNormal` that no dependent variable reads. Their
factor integrates to one, so the posterior over the rest is unchanged and no
conjugacy is required.

Packing prediction inputs into the prior is free when the latent is
marginalized, but not when it is *sampled*: the unread rows become coordinates
NUTS must explore, constrained only by the prior and correlated with the rest
through K.

The block is identified by naming it in the model::

    gp = pgp.GP("gp", Xs, cov=k)
    f_train, f_pred = pt.unpack(gp, shapes)
    name_variable("f_pred", f_pred)
    pm.Bernoulli("y", logit_p=f_train, observed=y)

    m2 = marginalize_named_subset(m, "f_pred")  # `gp` shrinks to the train rows

A name rather than an inferred partition, for two reasons. It keeps
`marginalize`'s contract that marginalized variables are referred to by name --
they no longer exist as variables, so a name is the only handle left, and
``conditional(m2)["f_pred"]`` gives the block back. And it turns an inference
into a validation: the partition is stated, and this checks that nothing
downstream reads it, rather than scanning the graph for slice patterns and
silently declining on anything that was not a single leading slice.

`ModelNamed` is the right category for that handle. It names a variable without
making it a free RV (which the sampler would then explore) or a `Deterministic`
(which blocks marginalization of anything it reads). It is the same wrapper used
to anchor marginalized variables, so a partition goes in and comes back out
through one mechanism.

Scoped to blocks nothing reads. Marginalizing a block the likelihood *does* read
is a conjugacy problem, not this one, and declines with a pointer.
"""

import numpy as np
import pytensor.tensor as pt

from pymc import MvNormal, modelcontext
from pymc.model.fgraph import ModelFreeRV, ModelNamed, model_free_rv
from pytensor.graph.fg import Output
from pytensor.graph.traversal import ancestors
from pytensor.tensor.basic import Split
from pytensor.tensor.reshape import SplitDims
from pytensor.tensor.shape import Reshape, SpecifyShape
from pytensor.tensor.subtensor import Subtensor, get_idx_list

from pymc_extras.model.marginal.distributions.subset_gaussian import build_subset_marginal

__all__ = ["marginalize_named_subset_fgraph", "name_variable"]


def name_variable(name, var, model=None):
    """Register `var` under `name`, as neither an RV nor a Deterministic.

    The handle `marginalize` needs to identify a sub-block.
    Deliberately not a `Deterministic`: those block marginalization of anything
    they depend on, and are recomputed for every draw, neither of which suits a
    partition marker.
    """
    model = modelcontext(model)
    var = pt.as_tensor(var)
    # Name in place. `var.copy()` would insert a DeepCopyOp between the handle
    # and the block it points at, which is exactly the link that identifies the
    # partition.
    var.name = name
    model.register_data_var(var)
    return var


def _slice_positions(var, parent, length):
    """Integer positions of `parent` that `var` reads.

    Handles both forms a partition takes: a `Subtensor` (``gp[:n]``) and one
    output of a `Split` (which is what `pt.unpack` emits). None if `var` is not
    a static, axis-aligned block of `parent`'s leading axis.
    """
    node = var.owner
    if node is None:
        return None

    # `pt.unpack` reshapes each piece after splitting, so the handle sits above
    # a shape op rather than directly on the block. These preserve the leading
    # axis element-for-element, so look through them.
    if isinstance(node.op, SplitDims | Reshape | SpecifyShape):
        inner = node.inputs[0]
        if var.ndim == inner.ndim == 1:
            return _slice_positions(inner, parent, length)
        return None

    if node.inputs[0] is not parent:
        return None

    def const(v):
        return None if v is None else int(pt.as_tensor(v).eval())

    if isinstance(node.op, Subtensor):
        try:
            [idx] = get_idx_list(node.inputs, node.op.idx_list)
        except (ValueError, TypeError):
            return None
        if not isinstance(idx, slice):
            return None
        try:
            return np.arange(length)[slice(const(idx.start), const(idx.stop), const(idx.step))]
        except Exception:
            return None

    if isinstance(node.op, Split):
        # `axis` lives on the op; inputs are (x, splits).
        if getattr(node.op, "axis", 0) != 0:
            return None
        try:
            splits = np.asarray(node.inputs[1].eval(), dtype=int)
        except Exception:
            return None
        i = node.outputs.index(var)
        start = int(splits[:i].sum())
        return np.arange(start, start + int(splits[i]))

    return None


def _block_parent(var):
    """The variable `var` is a block of, walking through shape ops."""
    node = var.owner
    if node is None:
        return None
    if isinstance(node.op, SplitDims | Reshape | SpecifyShape):
        return _block_parent(node.inputs[0])
    if isinstance(node.op, Split | Subtensor):
        return node.inputs[0]
    return None


def marginalize_named_subset_fgraph(fg, name):
    """Marginalize the named sub-block `name` of a Gaussian latent, in place.

    The parent keeps its name and shrinks to the coordinates that remain.
    Recover the dropped ones with ``conditional(m2)[name]``, which carries the
    exact Gaussian conditional.
    """
    named = [
        v
        for v in fg.variables
        if v.owner and isinstance(v.owner.op, ModelNamed) and v.owner.op.name == name
    ]
    if not named:
        raise ValueError(
            f"{name!r} is not a named variable of this model. Register the block with "
            "`name_variable(name, rv[...])` before marginalizing it."
        )
    [named_var] = named

    # Structural, not an ancestor scan: hyperpriors are ancestors too, and the
    # parent is specifically the variable this block slices.
    model_rv = _block_parent(named_var.owner.inputs[0])
    if model_rv is None or not isinstance(getattr(model_rv.owner, "op", None), ModelFreeRV):
        raise NotImplementedError(f"{name!r} is not a slice of a free RV of this model.")
    rv = model_rv.owner.inputs[0]
    if not isinstance(rv.owner.op, MvNormal):
        raise NotImplementedError(f"{name!r} is a block of {model_rv.name!r}, not an MvNormal")

    [length] = rv.type.shape or (None,)
    if length is None:
        raise NotImplementedError(f"{model_rv.name!r} has no static length to partition")

    drop = _slice_positions(named_var.owner.inputs[0], model_rv, length)
    if drop is None or len(drop) == 0:
        raise NotImplementedError(
            f"{name!r} is not a non-empty static slice of {model_rv.name!r}'s leading axis."
        )

    keep_mask = np.ones(length, dtype=bool)
    keep_mask[drop] = False
    keep_idx = set(np.flatnonzero(keep_mask).tolist())

    mu, cov = rv.owner.op.dist_params(rv.owner)
    if rv in ancestors([mu, cov]):
        raise NotImplementedError("Self-referential prior parameters")

    # Every variable that reads a block of the parent, with the positions it
    # reads. Scanning is simpler than walking clients, because a block can sit
    # under a chain of shape ops.
    blocks = {}
    for v in fg.variables:
        positions = _slice_positions(v, model_rv, length)
        if positions is not None:
            blocks[v] = positions

    # Keep the outermost: a block whose own consumers are blocks too is an
    # intermediate (the Split under a reshape), not something to rewire.
    outermost = {
        v: pos
        for v, pos in blocks.items()
        if not any(
            out in blocks
            for client, _ in fg.clients.get(v, [])
            if not isinstance(getattr(client, "op", None), Output)
            for out in client.outputs
        )
    }

    # Validation, not inference: every other use of the parent must stay inside
    # the kept block, or dropping these rows would change the model.
    named_block = named_var.owner.inputs[0]

    # The block itself must be read by nothing but its own name handle.
    if [
        client
        for client, _ in fg.clients.get(named_block, [])
        if client is not named_var.owner and not isinstance(getattr(client, "op", None), Output)
    ]:
        raise NotImplementedError(
            f"{name!r} is read by the model, so its factor does not integrate to one. "
            f"Integrating out a block something depends on is a conjugacy problem, which "
            f"`marginalize` handles for a whole variable but not for a sub-block."
        )

    other_uses, kept_uses = [], []
    for v, positions in outermost.items():
        if v is named_block:
            continue
        if not fg.clients.get(v):
            continue
        if set(positions.tolist()) <= keep_idx:
            kept_uses.append(v)
        else:
            other_uses.append(v)

    direct = [
        client
        for client, _ in fg.clients.get(model_rv, [])
        if not isinstance(getattr(client, "op", None), Output)
        and not any(out in blocks for out in client.outputs)
    ]
    if other_uses or direct:
        raise NotImplementedError(
            f"Something downstream reads coordinates of {model_rv.name!r} that {name!r} "
            f"would marginalize away. Integrating those out is a conjugacy problem, which "
            f"`marginalize` handles for a whole variable but not for a sub-block."
        )

    _unobs, obs = build_subset_marginal(rv, keep_mask, marginalized_name=name)

    op = model_rv.owner.op
    [value] = model_rv.owner.inputs[1:]
    new_value = obs.type()
    new_value.name = value.name
    new_model_rv = model_free_rv(obs, new_value, op.transform, op.name, *op.dims)

    # Every remaining use read exactly the kept block, which the reduced
    # variable now supplies whole.
    if kept_uses:
        fg.replace_all([(v, new_model_rv) for v in kept_uses], import_missing=True)

    # The name handle is consumed and the parent shrinks, so its output is
    # replaced rather than swapped in place (the types differ).
    if named_var in fg.outputs:
        fg.remove_output(fg.outputs.index(named_var))
    if model_rv in fg.outputs:
        fg.remove_output(fg.outputs.index(model_rv))
    fg.add_output(new_model_rv, reason="marginalize-subset", import_missing=True)
