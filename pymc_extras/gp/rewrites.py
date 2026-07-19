"""Graph rewrites that make packed GP priors cheap.

A GP prior is defined jointly over every input set of interest, then partitioned
with ``pt.pack`` / ``pt.unpack``. Only the observed partition reaches the
likelihood, so the rest should cost nothing.

That already holds when the partition is written as a plain slice: PyTensor
lifts ``Subtensor`` through the kernel's ``Elemwise``, so only the needed block
of the covariance is ever evaluated. It does *not* hold for ``pt.unpack``, which
emits a ``Split``, and no rewrite lifts ``Split`` through ``Elemwise``. The full
``(n + m) x (n + m)`` covariance is then built and immediately thrown away:

    n_pred = 1000    24.6 ms      n_pred = 4000    367.3 ms     (pt.unpack)
    n_pred = 1000     0.1 ms      n_pred = 4000     0.1 ms      (gp[:n])

Rather than teach ``Split`` to commute with every op, the rewrite below turns a
partly-unused ``Split`` back into ``Subtensor``s on the used outputs. The
existing lifting machinery takes it from there.
"""

import pytensor.tensor as pt

from pytensor.compile import optdb
from pytensor.graph.rewriting.basic import node_rewriter
from pytensor.tensor.basic import Split


@node_rewriter([Split])
def local_split_of_unused_outputs(fgraph, node):
    """Rewrite a `Split` with unused outputs into `Subtensor`s on the used ones.

    `Split` computes every partition at once, so an unused output keeps the
    whole input alive. Slicing instead lets the unused partitions become dead
    code, and lets `Subtensor` lift through whatever produced the input.

    Declines when every output is used: replacing one `Split` by k slices
    would recompute the input k times if the lift then fails.
    """
    used = [i for i, out in enumerate(node.outputs) if fgraph.clients.get(out)]
    if len(used) == len(node.outputs) or not used:
        return None

    x, splits = node.inputs
    axis = node.op.axis
    if axis < 0:
        axis += x.type.ndim

    # Offsets stay symbolic so this works for `pm.Data`-backed partitions.
    starts = pt.concatenate([pt.zeros((1,), dtype=splits.dtype), pt.cumsum(splits)])

    replacements = {}
    for i in used:
        index = (slice(None),) * axis + (slice(starts[i], starts[i + 1]),)
        replacements[node.outputs[i]] = x[index]

    return replacements


# Runs with the other subtensor lifts, so the Subtensors this produces get
# pushed into the kernel graph in the same pass.
optdb["canonicalize"].register(
    "local_split_of_unused_outputs",
    local_split_of_unused_outputs,
    "fast_run",
    "fast_compile",
)
