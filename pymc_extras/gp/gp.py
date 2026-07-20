"""A GP prior is an MvNormal over stacked inputs.

The design commitment: `GP` is a thin constructor, not a new distribution and
not an object that owns inference. Everything downstream (marginalization,
conditionals, posterior predictive) is the generic linear-Gaussian machinery in
`pymc_extras.model.marginal`, which knows nothing about GPs.

Partitioning uses stock `pt.pack` / `pt.unpack`: define the prior jointly over
every input set you care about (training points, prediction points, inducing
points), then "observed at the training points" is just `f_train`, a slice.
Slicing is affine, which is all the marginalization needs.

`packed_shapes` are symbolic graphs, not plain sizes, and when a part is a
`pm.Data` they reference that shared variable. Every pymc model transform goes
through `clone_model`, which clones shared variables, so shapes taken from the
original model keep reading the original model's data. Re-derive them against
the transformed model -- `pt.pack(X, cond_m["X_pred"], keep_axes=-1)` -- before
unpacking anything that came out of `marginalize` or `conditional`.
"""

import pymc as pm
import pytensor.tensor as pt

from pytensor.graph.traversal import ancestors

from pymc_extras.gp.data import build_kernel_op, kernel_of
from pymc_extras.model.marginal import (  # noqa: F401
    conditional,
    marginalize,
    marginalize_named_subset,
    name_variable,
)

# importing `marginal` above registers the linear-Gaussian conjugacy rewrite


def _as_2d(X):
    X = pt.as_tensor(X)
    return X[:, None] if X.ndim == 1 else X


def GP(name, X, cov, mean=0.0, jitter=1e-6, dims=None, model=None):
    """Register a GP prior over the inputs `X`.

    Parameters
    ----------
    X : array-like or tensor
        Input locations, typically the first output of `pack`.
    cov : Covariance
        Kernel object (see `pymc_extras.gp.kernels`), called as ``cov(X)``.
    mean : scalar, tensor, or callable
        Constant, vector, or a mean function called as ``mean(X)``.
    """
    model = pm.modelcontext(model)

    X = _as_2d(X)
    n = X.shape[0]
    # Separate row/column barriers so `project` can re-root only the rows.
    kernel_op, params = build_kernel_op(cov, dtype=X.type.dtype)
    K = kernel_op(X, X, *params) + jitter * pt.eye(n)
    mu = mean(X) if callable(mean) else pt.broadcast_to(pt.as_tensor(mean), (n,))

    return pm.MvNormal(name, mu=mu, cov=K, dims=dims, model=model)


def conditional_moments(model, name="gp"):
    """``(mu, cov)`` of a recovered variable's conditional, as symbolic tensors.

    `conditional` hands back a *model* in which `name` is a free RV whose
    distribution is the conditional. Its parameters are the predictive
    moments, still symbolic in the hyperparameters.
    """
    rv = model[name]
    mu, cov = rv.owner.op.dist_params(rv.owner)
    return mu, cov


def predictive_fn(model, outs):
    """Compile ``outs`` into a callable taking a posterior point.

    The wrinkle this papers over: the outputs must have RVs swapped for value
    variables before compiling. Most of `model`'s value vars are then irrelevant
    -- the recovered variable's own, and any from a prediction block added after
    fitting (`f_pred`, `y_new`) -- so only the ones the outputs actually reach
    are declared as inputs.

    Extra entries in the point are dropped, so the caller can pass a whole
    posterior point and each set of outputs takes the subset it depends on. A
    value that *is* needed and missing is left to `fn`, which raises
    ``Missing input`` -- never filled in with a default that would quietly
    give a wrong answer.
    """
    outs = model.replace_rvs_by_values(list(outs))
    needed = set(ancestors(outs))
    inputs = [v for v in model.value_vars if v in needed]
    fn = model.compile_fn(outs, inputs=inputs, point_fn=True)
    names = {v.name for v in inputs}

    def call(point):
        return fn({k: v for k, v in point.items() if k in names})

    return call


def project(gp, X_new, jitter=1e-6):
    """Noise-free projection of a GP onto new inputs: ``A @ gp``.

    ``A = K(X_new, Z) K(Z, Z)^-1`` where ``Z`` are `gp`'s own inputs. This is
    the sparse-GP building block: put a GP on a small set of inducing inputs
    and push it onto the data. The result is an *affine* function of `gp`,
    which is exactly what the linear-Gaussian marginalization handles, so
    sparse approximations need no approximation-specific machinery.
    """
    return _projection_matrix(gp, X_new, jitter) @ gp


def _projection_matrix(gp, X_new, jitter):
    """``A = K(X_new, Z) K(Z, Z)^-1``, the conditional-mean map."""
    Kzz, Kxz, _ = _kernel_blocks(gp, X_new, jitter)
    L = pt.linalg.cholesky(Kzz)
    return pt.linalg.solve_triangular(
        L.T, pt.linalg.solve_triangular(L, Kxz.T, lower=True), lower=False
    ).T


def _kernel_blocks(gp, X_new, jitter):
    """``(K_zz + jitter I, K_*z, K_**)`` from the kernel node in `gp`'s graph."""
    op, X, params = kernel_of(gp)
    X_new = _as_2d(X_new)
    Kzz = op(X, X, *params) + jitter * pt.eye(X.shape[0])
    return Kzz, op(X_new, X, *params), op(X_new, X_new, *params)


def _conditional_crossterm(gp, X_new, jitter):
    """``V`` with ``V.T @ V == K(X_new, Z) K(Z, Z)^-1 K(Z, X_new)``."""
    Kzz, Kxz, _ = _kernel_blocks(gp, X_new, jitter)
    L = pt.linalg.cholesky(Kzz)
    return pt.linalg.solve_triangular(L, Kxz.T, lower=True)


def prior_variance_correction(gp, X_new, jitter=1e-6):
    """``diag(K(X_new) - Q(X_new))``, the variance `project` throws away.

    Adding this to the observation noise turns DTC into FITC. This is the
    diagonal of `conditional_covariance`; use that when you need the full
    matrix (e.g. to draw smooth function samples rather than pointwise bands).
    """
    V = _conditional_crossterm(gp, X_new, jitter)
    _, _, Kss = _kernel_blocks(gp, X_new, jitter)
    return pt.clip(pt.diag(Kss) - pt.sum(V**2, axis=0), 0.0, pt.inf)


def conditional_covariance(gp, X_new, jitter=1e-6):
    """Full conditional covariance ``K(X_new) - K(X_new, Z) K(Z,Z)^-1 K(Z, X_new)``.

    Together with `project` (the conditional mean) this is the complete GP
    conditional, so ``MvNormal(project(gp, X_new), conditional_covariance(gp,
    X_new))`` draws *joint* -- i.e. smooth -- function values at `X_new`.
    `prior_variance_correction` is its diagonal, which is enough for pointwise
    intervals but produces independent, visibly rough draws.
    """
    V = _conditional_crossterm(gp, X_new, jitter)
    _, _, Kss = _kernel_blocks(gp, X_new, jitter)
    cov_new = Kss - V.T @ V
    cov_new = 0.5 * (cov_new + cov_new.T)  # symmetrize against round-off
    return cov_new + jitter * pt.eye(cov_new.shape[0])


def predictive_moments(gp, X_new, mu, cov, jitter=1e-6):
    """``(A mu, A cov A' + conditional_covariance)`` at `X_new`.

    Prediction from a *posterior over* the GP's own values rather than from a
    known draw of them. `project` and `conditional_covariance` condition on a
    given `f_z`; feeding them a posterior mean instead keeps the conditional
    spread but drops the posterior's own uncertainty, which understates the
    total silently. This adds it back: the same affine-Gaussian pushforward as
    everywhere else, ``A Sigma A'`` plus the part of the prior `A` throws away.

    `mu` and `cov` are the moments of `gp` -- e.g. from `conditional_moments`,
    or a variational `q(u)`.
    """
    A = _projection_matrix(gp, X_new, jitter)
    cov = pt.as_tensor(cov)
    if cov.ndim == 1:  # a diagonal q(u)
        cov = pt.diag(cov)

    pred_cov = A @ cov @ A.T + conditional_covariance(gp, X_new, jitter)
    pred_cov = 0.5 * (pred_cov + pred_cov.T)
    return A @ pt.as_tensor(mu), pred_cov


def conditional_at(name, X_new, gp, jitter=1e-6, dims=None, model=None):
    """Register the GP conditional at `X_new` as a new `MvNormal`.

    This is the whole prediction API::

        with fitted_model:
            f_pred = pgp.conditional_at("f_pred", X_new, gp)
            pm.Bernoulli("y_new", logit_p=f_pred)  # the model's own likelihood
        pm.sample_posterior_predictive(idata, sample_vars=["f_pred", "y_new"])

    It is ``MvNormal(project(gp, X_new), conditional_covariance(gp, X_new))``,
    which is the GP conditional. Building it by hand also works, and is what the
    "building blocks" section shows; the point of the helper is that the two
    pieces belong together. Reaching for `conditional_covariance` while pairing
    it with a posterior *mean* instead of draws silently understates the spread
    -- see `predictive_moments`, which is the correct closed form for that case.

    Not to be confused with `pymc_extras.gp.conditional`, which transforms a
    *model* to recover a marginalized latent. This adds a variable to the model
    you are already in, and works the same whether `gp` was integrated out,
    sampled, or fitted variationally, because all it needs is draws of `gp`.
    """
    return pm.MvNormal(
        name,
        mu=project(gp, X_new, jitter),
        cov=conditional_covariance(gp, X_new, jitter),
        dims=dims,
        model=model,
    )
