"""The kernel as a graph node, so it survives model transforms.

`project` must evaluate the covariance at inputs the model was not built with.
Carrying the kernel as a Python attribute on the RV works until the model is
transformed: `clone_model` clones variables and drops attributes.

Instead the kernel is an `OpFromGraph` over its two design matrices. The op is
the marker and its inputs are the design matrices, so both the function and the
data are recoverable by walking the covariance graph. Evaluating at new inputs
is calling the op again -- no graph surgery, no re-rooting, and shapes come from
real arguments each time rather than being baked in.
"""

import pytensor.tensor as pt

from pytensor.compile.builders import OpFromGraph
from pytensor.graph.basic import Constant
from pytensor.graph.traversal import ancestors
from pytensor.tensor.blockwise import Blockwise

__all__ = ["KernelOp", "build_kernel_op", "kernel_of"]


class KernelOp(OpFromGraph):
    """A covariance function, callable as ``op(X_row, X_col, *params)``."""


def build_kernel_op(cov, dtype="float64"):
    """Wrap a `Covariance` into a `KernelOp` plus the params it closes over.

    The params are the *shallowest* variables the kernel reads that do not
    depend on the design matrices. Cutting there rather than at the graph roots
    keeps hyperparameter RVs outside the op: descending past them would pull
    their RNGs in as op inputs, and an `OpFromGraph` containing a
    `RandomVariable` with no update is rejected downstream.
    """
    Xr = pt.tensor("Xr", dtype=dtype, shape=(None, None))
    Xc = pt.tensor("Xc", dtype=dtype, shape=(None, None))
    expr = cov(Xr, Xc)

    memo = {}

    def on_X(var):
        if var not in memo:
            if var is Xr or var is Xc:
                memo[var] = True
            elif var.owner is None:
                memo[var] = False
            else:
                memo[var] = False  # break cycles; graphs are acyclic anyway
                memo[var] = any(on_X(i) for i in var.owner.inputs)
        return memo[var]

    params, seen = [], set()

    def visit(var):
        if var in seen:
            return
        seen.add(var)
        if var is Xr or var is Xc:
            return
        if not on_X(var):
            if not isinstance(var, Constant):
                params.append(var)
            return
        for i in var.owner.inputs:
            visit(i)

    visit(expr)
    return KernelOp([Xr, Xc, *params], [expr], inline=False), params


def _as_kernel_op(op):
    """`KernelOp`, possibly wrapped in a `Blockwise` by `vectorize_graph`."""
    if isinstance(op, KernelOp):
        return op
    core = getattr(op, "core_op", None)
    if isinstance(op, Blockwise) and isinstance(core, KernelOp):
        return core
    return None


def kernel_of(gp):
    """Recover ``(op, X, params)`` from a GP's covariance graph."""
    cov = gp.owner.op.dist_params(gp.owner)[1]
    nodes = [
        v.owner
        for v in ancestors([cov])
        if v.owner is not None and _as_kernel_op(v.owner.op) is not None
    ]
    if not nodes:
        raise ValueError("No kernel node found; was this built by pymc_extras.gp.GP?")
    node = nodes[0]
    return _as_kernel_op(node.op), node.inputs[1], list(node.inputs[2:])
