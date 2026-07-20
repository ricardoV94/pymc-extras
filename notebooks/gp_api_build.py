"""Build gp_api.ipynb from a single source of truth.

Regenerate with `python gp_api_build.py`, then `python gp_api_execute.py` to run
it and embed outputs.
"""

import glob
import json
import pathlib
import subprocess

CELLS = []


def _ruff():
    """The ruff pre-commit uses, so generated cells match what the hook wants.

    Without this, pre-commit reformats `gp_api.ipynb` on every commit and the
    committed notebook stops matching what this script emits -- which quietly
    breaks "edit the builder, never the .ipynb".
    """
    for path in glob.glob(str(pathlib.Path.home() / ".cache/pre-commit/repo*/py_env-*/bin/ruff")):
        return path
    return None


RUFF = _ruff()


def format_notebook(path):
    """Apply the same ruff passes the pre-commit hook will apply.

    Ruff understands .ipynb natively, so running it on the written file makes
    the builder's output byte-identical to what the hook produces. Otherwise
    pre-commit reformats the notebook on every commit and it stops matching
    this script -- quietly breaking "edit the builder, never the .ipynb".
    """
    if RUFF is None:
        print("  (ruff not found; notebook left unformatted)")
        return
    for args in (["check", "--fix", "--quiet"], ["format", "--quiet"]):
        subprocess.run([RUFF, *args, str(path)], capture_output=True, text=True)


def md(text):
    CELLS.append(("markdown", text.strip("\n")))


def code(text):
    CELLS.append(("code", text.strip("\n")))


# ===========================================================================
md(r"""
# A low-level Gaussian Process API

Built on one idea:

> **A GP prior is an `MvNormal`. Everything else is generic linear-Gaussian
> machinery that knows nothing about GPs.**

| Concept | What it actually is |
|---|---|
| GP prior | `pm.MvNormal(mu, K)`, `K = kernel(X)` |
| Integrating the latent out | `pymc_extras.marginalize` |
| Posterior over the latent | `pymc_extras.conditional` |
| GP conditional mean | `project` |
| GP conditional covariance | `conditional_covariance` |
| Prediction | `conditional_at`, then the model's own likelihood |
| Sparse / variational GP | `project` onto inducing inputs, stock ADVI guide |

The only new machinery is a **linear-Gaussian conjugacy rewrite**: for
`f ~ MvNormal(m, K)` and `y ~ Normal(g(f), s)` with `g` affine,

$$y \sim \mathcal{N}(Am + b,\; AKA^\top + S), \qquad
f \mid y \sim \mathcal{N}\big(m + (AK)^\top G^{-1} r,\; K - (AK)^\top G^{-1}(AK)\big)$$

with $G = AKA^\top + S$, $r = y - (Am+b)$. It registers into
`marginal_ir_rewrites_db`; `marginalize` and `conditional` then work unmodified.
It is not GP-specific — it covers regression coefficients and state-space
latents identically.

Because `g` is *any* affine function, "observed at a subset of the inputs" and
"projected from inducing points" are the same operation.

There is **one way to predict**, and it is the same two lines whether the latent
was integrated out, sampled, or fitted variationally. That repetition is the
point of this notebook, so it is left visible rather than factored away.

---

*Generated from `gp_api_build.py`. Edit that, regenerate, then run
`gp_api_execute.py`.*
""")

code("""
import numpy as np
import matplotlib.pyplot as plt
import pytensor.tensor as pt

import arviz as az
import pymc as pm
import pymc_extras.gp as pgp

RNG = np.random.default_rng(0)
plt.rcParams["figure.figsize"] = (10, 4)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

N_TRAIN = 60
X = np.linspace(0, 1, N_TRAIN)[:, None]
y = np.sin(6 * X.ravel()) + 0.2 * RNG.normal(size=N_TRAIN)
X_pred = np.linspace(-0.3, 1.3, 80)[:, None]

ETA_TRUE, LS_TRUE, SIGMA_TRUE = 1.7, 0.3, 0.2
""")

# ---------------------------------------------------------------- kernels
md("""
## Kernels

Callable objects, not matrices: `k(X)` is the covariance, `k(X, Xs)` the
cross-covariance. Keeping the *function* is what lets a model be re-evaluated at
new inputs. They compose with `+` and `*` and scale by scalars, including model
random variables, which is how hyperpriors enter.
""")

code("""
k = ETA_TRUE**2 * pgp.kernels.Matern52(ls=LS_TRUE)
print("k(X)     ->", k(X).eval().shape)
print("k(X, Xs) ->", k(X, X_pred).eval().shape, " (cross-covariance, rectangular)")

k_composite = (
    2.0 * pgp.kernels.ExpQuad(ls=np.array([0.5]))
    + pgp.kernels.Matern32(ls=0.2)
    + pgp.kernels.WhiteNoise(0.1)
)

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, kern, name in zip(axes, [k, k_composite], ["Matern52", "sum of three"]):
    im = ax.imshow(kern(X).eval(), cmap="viridis")
    ax.set_title(name)
    fig.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------- prior
md("""
## The prior

`GP` is a thin constructor, not a new distribution and not an object that owns
inference: it registers a `pm.MvNormal` whose covariance is the kernel evaluated
at `X`. Nothing downstream knows it is a GP.

The model holds the **training inputs only**. Prediction inputs are not needed
here, and putting them in is actively harmful once the latent is sampled:
unobserved rows become coordinates NUTS must explore with no data to constrain
them. The appendix covers defining the prior jointly over several input sets and
the one case that genuinely needs it.
""")

code("""
with pm.Model() as latent_model:
    ls = pm.InverseGamma("ls", alpha=3.0, beta=1.0)
    eta = pm.Exponential("eta", scale=1.0)
    sigma = pm.HalfNormal("sigma", sigma=1.0)

    gp = pgp.GP("gp", X, cov=eta**2 * pgp.kernels.Matern52(ls=ls))
    pm.Normal("y", mu=gp, sigma=sigma, observed=y)

print("gp       :", gp.type.shape, " <- training inputs only")
print("it is an :", type(gp.owner.op).__name__)
print("free_RVs :", [v.name for v in latent_model.free_RVs])
""")

# ------------------------------------------------------- building blocks
md(r"""
## The building blocks

Two functions carry all the GP-specific content, and they are the *same formula*
seen twice. For a GP over inputs $Z$ and any new inputs $X_*$:

$$A_* = K_{*z}K_{zz}^{-1}, \qquad
\mathbb{E}[f_* \mid f_z] = A_* f_z, \qquad
\operatorname{Cov}[f_* \mid f_z] = K_{**} - A_*K_{z*}$$

* `project(gp, X_new)` builds $A_* f_z$ — the **conditional mean**.
* `conditional_covariance(gp, X_new)` builds $K_{**} - A_*K_{z*}$ — the
  **conditional covariance**.
* `prior_variance_correction(gp, X_new)` is its diagonal.
* `conditional_at(name, X_new, gp)` is the two of them as one `MvNormal`, which
  is how every prediction below is written. The pieces are shown here because
  the notebook is about what the API *is*, not because you should assemble them
  by hand.

These were written for the sparse case and turned out to be the textbook GP
conditional. "Inducing points" and "prediction points" are not two ideas: both
are some other set of inputs you want the GP at, reached by the same linear map.
""")

code("""
Z_demo = np.linspace(0, 1, 10)[:, None]
X_star = np.linspace(-0.1, 1.1, 7)[:, None]

with pm.Model():
    g_demo = pgp.GP("g", Z_demo, cov=k)
    proj = pgp.project(g_demo, X_star)
    C_full = pgp.conditional_covariance(g_demo, X_star)

f_z = RNG.normal(size=10)
Kzz = k(Z_demo).eval() + 1e-6 * np.eye(10)
Ksz = k(X_star, Z_demo).eval()

print("project              == K_*z K_zz^-1 f_z :",
      np.allclose(proj.eval({g_demo: f_z}), Ksz @ np.linalg.solve(Kzz, f_z)))
print("conditional_covariance == K_** - A K_z*  :",
      np.allclose(C_full.eval() - 1e-6 * np.eye(7),
                  k(X_star).eval() - Ksz @ np.linalg.solve(Kzz, Ksz.T), atol=1e-8))
""")

md("""
The full matrix rather than the diagonal is what makes draws *joint*, i.e.
actual smooth functions.
""")

code("""
with pm.Model():
    g2 = pgp.GP("g2", X, cov=k)
    mu_j = pgp.project(g2, X_pred)
    C_j = pgp.conditional_covariance(g2, X_pred)
    d_j = pgp.prior_variance_correction(g2, X_pred)
    mu_v = mu_j.eval({g2: np.sin(6 * X.ravel())})
    C_v, d_v = C_j.eval(), d_j.eval()

fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
axes[0].plot(X_pred.ravel(), RNG.multivariate_normal(mu_v, C_v, size=4).T, lw=1)
axes[0].set_title("conditional_covariance: joint draws")
axes[1].plot(X_pred.ravel(),
             (mu_v + np.sqrt(d_v) * RNG.normal(size=(4, len(X_pred)))).T, lw=1)
axes[1].set_title("diagonal only: independent draws")
plt.tight_layout()
plt.show()
""")

# ------------------------------------------------------------ conjugate
md("""
## A conjugate likelihood: marginalize, then condition

The latent integrates out in closed form, so NUTS samples 3 hyperparameters
instead of 3 + 60. `conditional` then returns a model in which `gp` is a free RV
whose distribution *is* the posterior, still symbolic in the hyperparameters. So
the latent comes back, and prediction proceeds exactly as if it had been sampled
all along.
""")

code("""
marginal_model = pgp.marginalize(latent_model, ["gp"])
print("free_RVs before:", [v.name for v in latent_model.free_RVs])
print("free_RVs after :", [v.name for v in marginal_model.free_RVs])

with marginal_model:
    idata = pm.sample(draws=400, tune=400, chains=2, random_seed=0, progressbar=False)

print(az.summary(idata, var_names=["ls", "eta", "sigma"], round_to=3))
print("\\ntrue: ls =", LS_TRUE, " eta =", ETA_TRUE, " sigma =", SIGMA_TRUE)
print(
    "\\nsigma and ls are recovered; eta is weakly identified in a GP marginal"
    "\\nlikelihood (it trades off against ls) and the Exponential(1) prior pulls"
    "\\nit down. A property of the model, not of the machinery."
)
""")

code("""
cond_model = pgp.conditional(marginal_model)
print("free_RVs:", [v.name for v in cond_model.free_RVs], " <- gp is back")

# ---- the prediction block, verbatim in every fit path below ----
with cond_model:
    f_pred = pgp.conditional_at("f_pred", X_pred, cond_model["gp"])
    pm.Normal("y_new", mu=f_pred, sigma=cond_model["sigma"])
    pp = pm.sample_posterior_predictive(
        idata, sample_vars=["gp", "f_pred", "y_new"], random_seed=0, progressbar=False
    )
# ----------------------------------------------------------------

f_pred_draws = pp.posterior_predictive["f_pred"].to_numpy().reshape(-1, len(X_pred))
mean_pred, sd_pred = f_pred_draws.mean(0), f_pred_draws.std(0)

xg = X_pred.ravel()
fig, ax = plt.subplots()
ax.fill_between(xg, mean_pred - 2 * sd_pred, mean_pred + 2 * sd_pred, alpha=0.25,
                label="±2 sd")
ax.plot(xg, mean_pred, lw=2, label="posterior mean")
ax.plot(X.ravel(), y, "o", ms=4, label="observations")
ax.plot(xg, np.sin(6 * xg), ls="--", lw=1, label="true function")
ax.axvspan(-0.3, 0, color="k", alpha=0.05)
ax.axvspan(1.0, 1.3, color="k", alpha=0.05)
ax.set_title("Exact GP posterior predictive (shaded = extrapolation)")
ax.legend(loc="lower left", ncols=2)
plt.show()
""")

md(r"""
### The closed form, and a trap

Drawing `f_pred` is Monte Carlo over a conditional that is available in closed
form. `predictive_moments(gp, X_new, mu, cov)` gives it directly: it pushes a
*posterior over* `f_z` forward rather than conditioning on one known draw of it,

$$\mathbb{E}[f_*] = A\mu, \qquad \operatorname{Cov}[f_*] = A\Sigma A^\top + (K_{**} - A K_{z*})$$

which is the same affine-Gaussian pushforward as everything else. The two agree
to Monte Carlo error.

The trap it exists to close: `conditional_covariance` alone is
$\operatorname{Cov}[f_* \mid f_z]$, so pairing it with a posterior *mean* keeps
the conditional spread and silently drops $A\Sigma A^\top$ — the posterior's own
uncertainty. Nothing errors; the bands just come out too narrow, most severely
where the data pins `f_z` down least.
""")

code("""
mu_sym, cov_sym = pgp.conditional_moments(cond_model)
pm_sym, pcov_sym = pgp.predictive_moments(cond_model["gp"], X_pred, mu_sym, cov_sym)
closed = pgp.predictive_fn(cond_model, [pm_sym, pt.sqrt(pt.diag(pcov_sym))])
naive = pgp.predictive_fn(cond_model, [pt.sqrt(pt.diag(
    pgp.conditional_covariance(cond_model["gp"], X_pred)))])

post = idata.posterior.to_dataset().stack(sample=("chain", "draw"))
ms, ss, ns = [], [], []
for i in range(0, post.sizes["sample"], 8):
    s = post.isel(sample=i)
    point = {f"{v}_log__": float(np.log(s[v])) for v in ("ls", "eta", "sigma")}
    m_, sd_ = closed(point)
    ms.append(m_)
    ss.append(sd_)
    ns.append(naive(point)[0])

ms, ss, ns = np.array(ms), np.array(ss), np.array(ns)
cf_mean = ms.mean(0)
cf_sd = np.sqrt((ss ** 2).mean(0) + ms.var(0))     # law of total variance
naive_sd = np.sqrt((np.array(ns) ** 2).mean(0) + ms.var(0))

print("closed form vs draws: max |mean diff| =", round(float(np.abs(cf_mean - mean_pred).max()), 4))
print("                      max |sd   diff| =", round(float(np.abs(cf_sd - sd_pred).max()), 4))
print("dropping A S A^T understates variance by up to",
      round(float((cf_sd ** 2 / naive_sd ** 2).max())), "x")

fig, ax = plt.subplots()
ax.fill_between(xg, cf_mean - 2 * cf_sd, cf_mean + 2 * cf_sd, alpha=0.25, label="correct ±2 sd")
ax.plot(xg, cf_mean - 2 * naive_sd, color="C3", lw=1, ls="--", label="conditional_covariance only")
ax.plot(xg, cf_mean + 2 * naive_sd, color="C3", lw=1, ls="--")
ax.plot(X.ravel(), y, "o", ms=4, label="observations")
ax.set_title("predictive_moments vs dropping the posterior's own spread")
ax.legend(loc="lower left", ncols=2)
plt.show()
""")

# ------------------------------------------------------- sample latent
md("""
## A non-conjugate likelihood: sample the latent, project to predict

With a Bernoulli likelihood the latent no longer integrates out and
`marginalize` declines rather than guessing, so it is sampled instead.

Prediction is **the same block as above**, with `Normal` swapped for the
likelihood this model actually uses. Nothing about it knows whether `gp` arrived
by conditioning or by sampling; it only needs draws of `gp`.

Two things it depends on, both easy to get wrong:

* `f_pred` and `y_new` go on the model **after** fitting. An unobserved RV in
  the fitted model is a *free* RV, so MCMC would sample it — and packing 60
  prediction rows into this problem drops min ESS from 482 to 4.
* Pass **`sample_vars`**, not `var_names`. With `var_names`, a variable already
  in the trace is returned unchanged: no resampling, no warning, just
  `Sampling: []`.
""")

code("""
X_b = np.linspace(0, 1, 40)[:, None]
p_true = 1 / (1 + np.exp(-3 * np.sin(6 * X_b.ravel())))
y_b = RNG.binomial(1, p_true)
X_b_pred = np.linspace(-0.2, 1.2, 60)[:, None]
k_b = 3.0**2 * pgp.kernels.Matern52(ls=0.25)

with pm.Model() as bern_model:
    gp_b = pgp.GP("gp", X_b, cov=k_b)          # training inputs only
    pm.Bernoulli("y", logit_p=gp_b, observed=y_b)

try:
    pgp.marginalize(bern_model, ["gp"])
except NotImplementedError as exc:
    print("marginalize declines:", str(exc)[:70], "...")

with bern_model:
    idata_b = pm.sample(draws=500, tune=1000, chains=2, random_seed=0,
                        progressbar=False)
print("max r_hat:", float(np.nanmax(az.rhat(idata_b, var_names=["gp"])["gp"].to_numpy())))
""")

code("""
# ---- the prediction block, verbatim from the conjugate section ----
with bern_model:
    f_pred = pgp.conditional_at("f_pred", X_b_pred, gp_b)
    pm.Bernoulli("y_new", logit_p=f_pred)
    pp = pm.sample_posterior_predictive(
        idata_b, sample_vars=["f_pred", "y_new"], random_seed=0, progressbar=False
    )
# -------------------------------------------------------------------

f_draws = pp.posterior_predictive["f_pred"].to_numpy().reshape(-1, len(X_b_pred))
p_draws = 1 / (1 + np.exp(-f_draws))

xb = X_b_pred.ravel()
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(xb, f_draws[::150].T, lw=1, alpha=0.8)
axes[0].set_title("latent $f_*$ draws")
axes[1].fill_between(xb, *np.percentile(p_draws, [2.5, 97.5], axis=0), alpha=0.25,
                     label="95%")
axes[1].plot(xb, p_draws.mean(0), lw=2, label="$P(y_*=1)$")
axes[1].plot(X_b.ravel(), p_true, ls="--", lw=1, label="true probability")
axes[1].plot(X_b.ravel(), y_b, "|", ms=6, alpha=0.4, color="k")
axes[1].set_ylim(-0.05, 1.05)
axes[1].set_title("$y_*$ through the likelihood")
axes[1].legend(loc="lower left")
plt.tight_layout()
plt.show()
""")

# -------------------------------------------------------------- variational
md(r"""
## A non-conjugate likelihood: fit it variationally

Sampling the latent degrades as it grows: 120 training points already gives
`r_hat = 2.5`. Putting the prior on a few inducing inputs and `project`ing onto
the data shrinks the latent to those inducing values, which is also the
construction to reach for with many observations.

No SVGP-specific code exists. The ELBO

$$\mathrm{ELBO} = \mathbb{E}_{q(u)}\big[\log p(y \mid Au)\big] - \mathrm{KL}\big(q(u)\,\|\,p(u)\big)$$

is what `E_q[log p(y,u) - log q(u)]` already computes when `p(u)` is the model
and `q(u)` the guide, so the KL term is never implemented. Prediction is the
same three lines as above, because `Trainer.sample_posterior` returns draws that
`sample_posterior_predictive` consumes directly.
""")

code("""
from pymc_extras.inference.advi import AutoMultivariateNormal, Trainer

Z_b = np.linspace(0, 1, 15)[:, None]

with pm.Model() as svgp_model:
    u_b = pgp.GP("u", Z_b, cov=k_b)
    pm.Bernoulli("y", logit_p=pgp.project(u_b, X_b), observed=y_b)

trainer = Trainer(guide=AutoMultivariateNormal, model=svgp_model,
                  n_particles=32, random_seed=0)
trainer.fit(3000)
idata_vi = trainer.sample_posterior(draws=1000, random_seed=0)
print("latent dim:", u_b.type.shape, "| sample_posterior ->", type(idata_vi).__name__)

# ---- the prediction block again, unchanged; only `u_b` differs ----
with svgp_model:
    f_pred = pgp.conditional_at("f_pred", X_b_pred, u_b)
    pm.Bernoulli("y_new", logit_p=f_pred)
    pp_vi = pm.sample_posterior_predictive(
        idata_vi, sample_vars=["f_pred", "y_new"], random_seed=0, progressbar=False
    )
# -------------------------------------------------------------------

p_vi = 1 / (1 + np.exp(
    -pp_vi.posterior_predictive["f_pred"].to_numpy().reshape(-1, len(X_b_pred))))

fig, ax = plt.subplots()
ax.fill_between(xb, *np.percentile(p_vi, [2.5, 97.5], axis=0), alpha=0.25,
                label="variational 95%")
ax.plot(xb, p_vi.mean(0), lw=2, label="variational $P(y_*=1)$")
ax.plot(xb, p_draws.mean(0), lw=1.5, ls=":", label="sampled latent")
ax.plot(X_b.ravel(), p_true, ls="--", lw=1, label="true probability")
ax.plot(Z_b.ravel(), np.full(len(Z_b), -0.03), "^", ms=8, label="inducing points")
ax.set_ylim(-0.06, 1.05)
ax.set_title("15 inducing values")
ax.legend(loc="lower left", ncols=2)
plt.show()
""")

# ------------------------------------------------------------ wrap-up
md(r"""
## What is missing

1. **Woodbury / structured covariance.** $AKA^\top$ is densified, so the sparse
   construction is correct but scales $O(n^{2.3})$ rather than $O(nm^2)$. The
   main blocker, and the same work item as low-rank ADVI guides.
2. **Non-centered parameterization.** `GP` has no `parameterization=`, which is
   what caps the sampled-latent path at around a hundred observations.

3. **`SubsetMarginalRV`.** `marginalize_subset` (appendix) discards the dropped
   rows rather than keeping them as an unused output of a `MarginalRV`, so plain
   `conditional` cannot recover them. `project` and `predictive_moments` can, so
   this is uniformity rather than capability.

Shallow by construction: kernel coverage, no mean-function namespace, no
multi-output or Kronecker structure, no `predict_f` / `predict_y` wrappers.

### Design notes

* `A` is **never materialized**. The linear map is applied with
  `vectorize_graph`, preserving structure, so a partition stays a slice and
  PyTensor can push it into the kernel graph. `pt.unpack` emits a `Split`, which
  nothing lifts, so `pymc_extras.gp.rewrites` turns a partly-unused `Split` back
  into `Subtensor`s and lets the existing machinery do the rest.
* Affineness is checked by a conservative op whitelist; anything unrecognized
  declines cleanly rather than producing a wrong logp.
* `project` and `conditional_covariance` are the GP conditional, which is why
  all three fits share their prediction code verbatim.
""")

# ------------------------------------------------------------- appendix
md(r"""
---

## Appendix: packing, and the case that needs it

Everything above puts the prior on the training inputs and reaches new inputs
with `project`. The alternative is to define the prior **jointly** over every
input set up front, stacked with `pt.pack`, and let `pt.unpack` slice it:
"observed at the training points" is then `f_train`, a plain slice, which is
affine, which is all the marginalization needs.

This is strictly more general in *what map* relates the latent to the
observations — the conjugacy rewrite accepts any affine `g`, not just a subset —
but it is not the way to predict. It fixes the prediction inputs at build time,
and it duplicates what `project` already does.

Packing is free when the latent is marginalized (`pymc_extras.gp.rewrites` lifts
the `Split` that `pt.unpack` emits, so an unused partition never enters the
covariance: 4000 prediction rows go from 367 ms to 0.13 ms). It is *not* free
when the latent is sampled. `marginalize_subset` removes the rows nothing
downstream reads — their factor integrates to one, so no conjugacy is needed and
the posterior over what remains is unchanged.
""")

code("""
with pm.Model() as packed_model:
    ls_p = pm.InverseGamma("ls", alpha=3.0, beta=1.0)
    Xs, shapes = pt.pack(X, X_pred, keep_axes=-1)
    gp_p = pgp.GP("gp", Xs, cov=pgp.kernels.Matern52(ls=ls_p))
    f_train, f_pred_slice = pt.unpack(gp_p, shapes)
    pm.Bernoulli("y", logit_p=f_train, observed=(y > 0).astype(int))

reduced = pgp.marginalize_subset(packed_model, "gp")
print("packed gp :", packed_model["gp"].type.shape)
print("reduced   :", reduced["gp"].type.shape, " <- the 80 unread rows are gone")

# the model you would have written without packing at all
with pm.Model() as unpacked_model:
    ls_u = pm.InverseGamma("ls", alpha=3.0, beta=1.0)
    gp_u = pgp.GP("gp", X, cov=pgp.kernels.Matern52(ls=ls_u))
    pm.Bernoulli("y", logit_p=gp_u, observed=(y > 0).astype(int))

point = {"ls_log__": 0.0, "gp": np.zeros(N_TRAIN)}
print("logp matches it exactly:",
      np.isclose(reduced.compile_logp()(point), unpacked_model.compile_logp()(point)))
""")

md(r"""
The dropped rows are not discarded: they are kept as an unused output of a
`SubsetMarginalRV`, whose conditional is the Gaussian conditional of the block
given the rest. So `conditional` hands them back on its own, with no GP-specific
call and no chance of pairing a posterior mean with the wrong covariance:
""")

code("""
cond_sub = pgp.conditional(reduced)
print("free_RVs:", [v.name for v in cond_sub.free_RVs], " <- the dropped block is back")

rv_u = cond_sub["gp_unobserved"]
mu_u, cov_u = rv_u.owner.op.dist_params(rv_u.owner)
f_obs = np.sin(6 * X.ravel())
point = {"ls_log__": np.log(0.3), "gp": f_obs}
got_mu, got_sd = pgp.predictive_fn(cond_sub, [mu_u, pt.sqrt(pt.diag(cov_u))])(point)

# identical to what project / conditional_covariance compute by hand
with unpacked_model:
    ref = pgp.predictive_fn(unpacked_model, [
        pgp.project(unpacked_model["gp"], X_pred),
        pt.sqrt(pt.diag(pgp.conditional_covariance(unpacked_model["gp"], X_pred))),
    ])
ref_mu, ref_sd = ref(point)
print("vs project / conditional_covariance:  max |mu diff| =",
      float(np.abs(got_mu - ref_mu).max()))
print("                                      max |sd diff| =",
      float(np.abs(got_sd - ref_sd).max()))
print("(two equivalent factorizations of an ill-conditioned K; round-off, not disagreement)")

from pymc_extras.model.marginal.marginalize import unmarginalize
try:
    unmarginalize(reduced)
except NotImplementedError as exc:
    print("unmarginalize declines:", str(exc)[:80], "...")
""")

md(r"""
The one thing packing expresses that `project` cannot: an observation that is a
general **affine map** of the latent rather than a slice of it,
$y \sim \mathcal{N}(Wf + b, \sigma^2)$. Here the observation is not a set of
rows, so there is nothing for `project` to project. The conjugacy rewrite
handles it unchanged, and `conditional` gives closed-form joint posterior
moments over the whole latent.
""")

code("""
W = RNG.normal(size=(12, N_TRAIN))
y_w = W @ np.sin(6 * X.ravel()) + 0.1 * RNG.normal(size=12)

with pm.Model() as affine_model:
    ls_w = pm.InverseGamma("ls", alpha=3.0, beta=1.0)
    gp_w = pgp.GP("gp", X, cov=pgp.kernels.Matern52(ls=ls_w))
    pm.Normal("y", mu=W @ gp_w, sigma=0.1, observed=y_w)   # not a slice

cond_w = pgp.conditional(pgp.marginalize(affine_model, ["gp"]))
mu_w, cov_w = pgp.conditional_moments(cond_w)
mu_hat, sd_hat = pgp.predictive_fn(cond_w, [mu_w, pt.sqrt(pt.diag(cov_w))])({"ls_log__": np.log(0.3)})

fig, ax = plt.subplots()
ax.fill_between(X.ravel(), mu_hat - 2 * sd_hat, mu_hat + 2 * sd_hat, alpha=0.25, label="±2 sd")
ax.plot(X.ravel(), mu_hat, lw=2, label="posterior mean")
ax.plot(X.ravel(), np.sin(6 * X.ravel()), ls="--", lw=1, label="true function")
ax.set_title("$f$ recovered from 12 linear projections $Wf$, never observed directly")
ax.legend(loc="lower left")
plt.show()
""")

# ===========================================================================
nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": body.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, body in CELLS
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

here = pathlib.Path(__file__).parent
path = here / "gp_api.ipynb"
path.write_text(json.dumps(nb, indent=1))
format_notebook(path)
print(f"wrote gp_api.ipynb ({len(CELLS)} cells)")
