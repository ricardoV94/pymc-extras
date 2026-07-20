# GP API — handoff

Branch: `gp_linear_gaussian` (pymc-extras), based on `advi-refactor` + the three
`normal_normal_marginal` commits. Nothing pushed.

The goal is a GP API where **a GP prior is an `MvNormal` and everything else is
generic linear-Gaussian machinery that knows nothing about GPs.** That claim has
held up: sparse approximations, SVGP and prediction all fell out of existing
pieces. What remains is finishing the parts that make it usable.

---

## 1. The API we are after

One way to predict, covering every case. The model holds **training inputs
only**.

```python
with pm.Model() as m:
    ls    = pm.InverseGamma("ls", alpha=3.0, beta=1.0)
    eta   = pm.Exponential("eta", scale=1.0)
    gp    = pgp.GP("gp", X, cov=eta**2 * pgp.kernels.Matern52(ls=ls))
    pm.Normal("y", mu=gp, sigma=sigma, observed=y)     # or Bernoulli, ...
```

**Fit** — determined only by whether the likelihood is conjugate:

```python
m2 = pgp.marginalize(m, ["gp"])          # conjugate: latent integrates out
idata = pm.sample(model=m2)

idata = pm.sample(model=m)                                  # non-conjugate: sample it
idata = Trainer(guide=AutoMultivariateNormal, model=m).fit(...).sample_posterior()
```

**Predict** — identical in all three cases:

```python
with m:
    # Helper to build it?
    f_pred = pm.MvNormal("f_pred", mu=pgp.project(gp, X_new),
                          cov=pgp.conditional_covariance(gp, X_new))
    pm.Bernoulli("y_new", logit_p=f_pred)          # same likelihood, unobserved
pm.sample_posterior_predictive(idata, sample_vars=["f_pred", "y_new"])
```

`project` is the conditional mean `K_*z K_zz⁻¹ f_z`, `conditional_covariance`
the conditional covariance `K_** − K_*z K_zz⁻¹ K_z*`. Together they are the
textbook GP conditional, which is why inducing points and prediction points are
the same operation and neither needs its own code path.

Two traps that must stay documented:

* Add `f_pred` / `y_new` **after** fitting. An unobserved RV in the fitted model
  is a *free* RV, so MCMC samples it.
* Pass **`sample_vars`**, not `var_names`. With `var_names` a variable already in
  the trace is returned unchanged — no resampling, no warning (`Sampling: []`).

## 2. The general API (appendix material)

`pack`/`unpack` + `marginalize` / `marginalize_subset` / `conditional` is
strictly more general in *what map* relates latent to observations, because the
conjugacy rewrite accepts any affine `g`, not just a subset:

```python
Xs, shapes = pt.pack(X, X_pred, keep_axes=-1)
gp = pgp.GP("gp", Xs, cov=k)
f_train, f_pred = pt.unpack(gp, shapes)
pm.Normal("y", mu=f_train, sigma=sigma, observed=y)

m2 = pgp.marginalize(m, ["gp"])           # conjugate: whole RV
m2 = pgp.marginalize_subset(m, "gp")      # otherwise: just the unsampled rows
c  = pgp.conditional(m2)                  # posterior over the joint latent
```

Keep this because it is the only way to express `y ~ Normal(W @ f + b)`, where
the observation is not a slice of the latent and `project` has nothing to say.
It also gives closed-form joint posterior moments in the conjugate case.

It should **not** be the headline: it requires deciding prediction inputs at
build time, and it is a second way to do what `project` already does.

## 3. Where we stand

### Committed and working

| | |
|---|---|
| `model/marginal/distributions/linear_gaussian.py` | conjugacy rewrite: `MvNormal` latent under any affine-in-the-latent Gaussian observation. `A` never materialized (`vectorize_graph`); affineness by conservative op whitelist |
| `gp/gp.py` | `GP`, `project`, `conditional_covariance`, `prior_variance_correction`, `predictive_moments`, `conditional_moments`, `predictive_fn` |
| `gp/kernels.py` | `ExpQuad`, `Matern32/52`, `WhiteNoise`, `Constant`, `+`/`*`, ARD, `active_dims` |
| `gp/rewrites.py` | `Split` lift: partly-unused `Split` → `Subtensor`s, so `pt.unpack` costs nothing |
| `notebooks/gp_api.ipynb` | 25 cells, 7 figures, executed |
| `gp/data.py` | `KernelOp`: the kernel as an `OpFromGraph`, so it survives `clone_model` |
| `model/marginal/subset.py` | `marginalize_subset` |

Verified numerically: `project` and `conditional_covariance` against closed
forms; marginal logp and conditional moments against textbook GP formulas
(rtol 1e-6 / 1e-5); kernels against `pymc.gp.cov`; FITC with `Z = X` collapsing
to the exact GP; VI recovering the closed-form DTC posterior over `u` to 1e-3.
`tests/model/marginal/` passes (105 passed, 3 skipped) with the new rewrite
registered.

* **`model/marginal/subset.py` — `marginalize_subset(model, name)`.** Removes the
  rows of a packed Gaussian latent that nothing downstream reads. Their factor
  integrates to 1, so no conjugacy is needed. Verified: 50 → 20 coordinates,
  logp identical to an equivalent unpacked model.
* **`gp/data.py` — `KernelOp`.** The kernel as an `OpFromGraph` over its two
  design matrices, replacing the `.X` / `.cov` Python attributes that do not
  survive `clone_model`. `gp/gp.py` is rewired to it. `project`,
  `conditional_covariance`, `prior_variance_correction` match closed forms both
  on a fresh GP and off `conditional(...)`, including when the observation is a
  general affine map `W @ gp` rather than a slice.
* **`gp/gp.py` — `predictive_moments(gp, X_new, mu, cov)`.** Recovers the exact
  GP predictive to 1e-6 from the conditional's moments. The naive path it
  replaces (`conditional_covariance` on a posterior mean) understated variance
  by 6×–1780× on a 20-point problem.

  The earlier "blocked: no kernel node found" note was misdiagnosed twice over.
  The kernel is **not** absent from the conditional — it arrives via the prior
  `K` inside `post_cov`, so `linear_gaussian_conditional` needed no change. The
  actual bug was in `build_kernel_op`: it closed the op over `graph_inputs`,
  which walks *through* hyperparameter RVs and captures their RNGs, producing an
  `OpFromGraph` with an update-less `RandomVariable`. It now cuts at the
  shallowest `X`-independent variables, leaving the RVs outside the op.

## 4. Next steps, in priority order

1. **`SubsetMarginalRV`.** Rather than `marginalize_subset` discarding the
   dropped rows, keep them as an unused output of a `MarginalRV`, following
   `linear_gaussian.py`'s pattern (op + `_logprob` + `marginalized_conditional`).
   Then plain `conditional()` recovers the block with no new function, and the
   recovery information travels inside the op. Note this is a *refactor*, not new
   capability: `project` + `predictive_moments` already recover the block
   correctly. It buys API uniformity, at the cost of a new op class and an entry
   point that does not currently go through `marginalize_fgraph`.

   This is now the only item left on the original list.


## 5. The notebook

`notebooks/gp_api.ipynb`, generated from `gp_api_build.py` and executed by
`gp_api_execute.py`. **Edit the builder, never the `.ipynb`.**

**Now at the target shape**: 25 cells, 7 figures, executed clean. The idea and
concept table; kernels; the prior (training inputs only); the building blocks;
conjugate (`marginalize` → sample → predict); non-conjugate sampled; and
non-conjugate variational — all three predicting with the *same* block, marked
off by a comment banner in each so the repetition is visible:

```python
with model:
    pm.MvNormal("f_pred", mu=pgp.project(gp, X_new),
                cov=pgp.conditional_covariance(gp, X_new))
    <Likelihood>("y_new", ..., f_pred)
    pm.sample_posterior_predictive(idata, sample_vars=["f_pred", "y_new"])
```

This works in the conjugate case because `conditional` puts `gp` back as a free
RV, so `f_pred` is redrawn per posterior draw. `predictive_moments` is shown
beside it as the closed-form equivalent (they agree to Monte Carlo error) and as
the vehicle for the understated-variance trap — 360× on the notebook's data.

Packing is now **appendix only**: `pt.pack`/`unpack`, `marginalize_subset`
(140 → 60 rows, logp identical to the unpacked model), and the one case that
needs it, `y ~ Normal(W @ f, s)`, where the observation is not a slice and
`project` has nothing to say.

Keep it a demonstration of the design, not a user guide: no decision tables, no
"which path should I choose", no scaling studies in the body. State resolved
non-issues as claims rather than measuring them. It ran at 41 cells as a guide
and 19 as a design document; the shorter one was better.

## 6. Measurements worth not re-deriving

| | |
|---|---|
| Packing rows you *sample* | 40-pt Bernoulli GP, +60 prediction rows: min ESS 482 → 4 |
| Sampled latent ceiling | 40 pts `r_hat` 1.01; 120 pts `r_hat` 2.53. NUTS itself says "or reparameterize" |
| Exact GP logp | n=1000 31 ms, n=2000 140 ms, n=4000 1.1 s (×4–8 per doubling) |
| Sparse logp | ~×4–5 per doubling → O(n^2.3), i.e. Woodbury absent |
| `Split` lift | 4000 prediction rows: 367 ms → 0.13 ms, flat to 8000 |
| VI vs closed form | `q(u)` vs exact DTC posterior: mean 7e-4, cov 1e-4 |
| Conditioning | `K_zz + 1e-6` cond 3.2e8 vs `K_oo + σ²` cond 8e3 — `project` inverts an unregularized covariance, `marginalize` does not |

## 7. Gotchas already paid for

* `clone_model` clones shared variables and drops Python attributes on RVs. Any
  state on an RV is lost by every model transform. This caused three separate
  bugs; it is why (1) and (5) above matter.
* `pt.pack`'s `packed_shapes` are **symbolic graphs**, not sizes. Over a
  `pm.Data` they reference that shared variable and must be re-derived against
  the transformed model, or they silently slice by the pre-transform length.
* A statically-shaped input makes `x.shape[0]` fold to a constant at
  construction, and `graph_replace` inserts a `SpecifyShape` that fails when
  replacing with a different size. Both bite anything trying to re-root a design
  matrix.
* Pre-commit reformats files, so `git commit -F -` fails; a follow-up
  `git commit --amend --no-edit` then folds the change into the *previous*
  commit. Re-stage and commit again instead.
* This is a POC: the executed notebook is the verification. Do not run test
  suites unless asked.
