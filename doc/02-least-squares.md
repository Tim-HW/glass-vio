# Module 2 — Nonlinear least squares on a manifold

> **Prerequisite:** [Module 1](01-manifolds.md). **After this you can:** turn a set of weighted
> residuals into the normal equations $\mathbf{H}\delta\mathbf{x}=\mathbf{b}$, solve them by
> Gauss-Newton, and retract the step onto the manifold — the loop that runs *everywhere* in this
> codebase.

Every estimate in this system — the gyro bias, the metric alignment, the per-frame pose, the future
sliding window — is the answer to the same question: **which state makes all my measurements most
consistent?** This module is that question's machinery, in the abstract. Modules 3–5 supply the
actual residuals; here we build the engine that consumes them.

Code: [`gauss_newton.hpp`](../../glasslio/glass_core/include/glass_core/gauss_newton.hpp)
(`NormalEquationsN<N>`: the accumulator + solve). Full derivation: glass-lio's
[gauss-newton.md](../../glasslio/doc/gauss-newton.md).

---

## 1. The problem

A **residual** $\mathbf{r}(\mathbf{x})$ is *how wrong a measurement is* given a candidate state — it
is zero when the state explains the measurement exactly. We seek the state that minimizes the total
weighted squared residual over all measurements:

$$
\mathbf{x}^\star = \arg\min_{\mathbf{x}} \; C(\mathbf{x}), \qquad
C(\mathbf{x}) = \sum_k \lVert \mathbf{r}_k(\mathbf{x}) \rVert^2_{\Omega_k}
= \sum_k \mathbf{r}_k^\top \,\Omega_k\, \mathbf{r}_k.
$$

Each **weight** (or *information*) matrix $\Omega_k = \Sigma_k^{-1}$ is the inverse covariance of
measurement $k$: a precise measurement pulls hard, a noisy one barely. This is not a tuning knob —
it is where each sensor's own uncertainty enters, and it is why fusion needs no hand-chosen blend
coefficient ([Module 5](05-tight-coupling.md)).

The catch: $\mathbf{r}(\mathbf{x})$ is **nonlinear** in $\mathbf{x}$ (a pinhole divides by depth; a
rotation goes through `Exp`), and $\mathbf{x}$ lives partly on a **manifold**. So there is no
closed-form solution. Gauss-Newton gets there by iterating a linear approximation.

---

## 2. Gauss-Newton: linearize, accumulate, solve, retract

### 2.1 Linearize — the Jacobian is the local model

Take the current estimate $\mathbf{x}$ and a small tangent step $\delta\mathbf{x}$ (applied with `⊞`
from [Module 1](01-manifolds.md)). To first order:

$$
\mathbf{r}(\mathbf{x}\boxplus\delta\mathbf{x}) \approx \mathbf{r}(\mathbf{x}) + \mathbf{J}\,\delta\mathbf{x},
\qquad \mathbf{J} = \left.\frac{\partial\,\mathbf{r}(\mathbf{x}\boxplus\delta\mathbf{x})}{\partial\,\delta\mathbf{x}}\right|_{\delta\mathbf{x}=0}.
$$

$\mathbf{J}$ is the residual's derivative *evaluated at the current $\mathbf{x}$*, **and taken
through `⊞`** — that is the whole reason [Module 1](01-manifolds.md) mattered. It is a local model,
valid only for small $\delta\mathbf{x}$; that locality is exactly why the algorithm iterates.

### 2.2 Accumulate — the normal equations

Substitute the linear model into $C$ and set the gradient to zero. The minimizer of a sum of
squared *linear* terms is the solution of the **normal equations**, built one measurement at a time:

$$
\boxed{\;\mathbf{H}\,\delta\mathbf{x} = \mathbf{b}\;}, \qquad
\mathbf{H} = \sum_k \mathbf{J}_k^\top\,\Omega_k\,\mathbf{J}_k, \qquad
\mathbf{b} = -\sum_k \mathbf{J}_k^\top\,\Omega_k\,\mathbf{r}_k.
$$

$\mathbf{H}$ is the (Gauss-Newton approximation to the) **Hessian** — the cost's local curvature;
$\mathbf{b}$ is the negative gradient. In code this sum *is* the object `NormalEquationsN<15>`:

```cpp
NormalEquationsN<15> eq;
eq.addScalar(residual, jacobian_row, weight);     // one scalar measurement (e.g. one pixel coord)
eq.addBlock<9>(residual9, jacobian9x15, info9);    // one vector measurement (e.g. the IMU)
const NavVec dx = eq.solve();                       // H dx = b
```

`★ Insight — this sum is the fusion ────────────`
There is no separate "combine the sensors" step. Every measurement — a pixel, an IMU delta, a prior
— folds its $\mathbf{J}^\top\Omega\mathbf{J}$ into the *same* $\mathbf{H}$. Solving $\mathbf{H}$ is
what balances them, weighted automatically by each $\Omega$. [Module 5](05-tight-coupling.md) is
just this paragraph with the three specific residuals plugged in — which is why "the fusion" needs
no new machinery beyond this module.
`─────────────────────────────────────────────────`

### 2.3 Solve

$\mathbf{H} = \sum \mathbf{J}^\top\Omega\mathbf{J}$ is symmetric positive semidefinite by
construction, so `eq.solve()` is a Cholesky-type factorization — `H.ldlt().solve(b)` — not a general
matrix inverse. For a 15×15 system this is microseconds.

### 2.4 Retract — put the step back on the manifold

$\delta\mathbf{x}\in\mathbb{R}^{15}$ is a tangent vector. `boxplus` maps it onto the state:

$$
\mathbf{R}\leftarrow\mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol{\phi}),\quad
\mathbf{p}\leftarrow\mathbf{p}+\delta\mathbf{p},\quad\dots
$$

Then **re-linearize about the new estimate and repeat**, until $\delta\mathbf{x}$ is negligible.
One curved block (compose), twelve flat ones (add).

---

## 3. What the approximation throws away — GN vs Levenberg-Marquardt

Gauss-Newton drops the second-derivative term of the true Hessian (the part weighted by the
residual itself). This is excellent **when the residuals are small or nearly linear near the
optimum**, and glassvio arranges exactly that: the state is well-conditioned when enough landmarks
are in view, and each solve is *seeded by the IMU's own prediction* — already close to the answer,
where Gauss-Newton is strongest. So there is **no damping and no line search** in the per-frame
solve.

Levenberg-Marquardt adds a damping term $\lambda\mathbf{I}$ to $\mathbf{H}$, interpolating toward
gradient descent when a step would overshoot — the robust choice when the seed is poor or the
problem ill-conditioned. glassvio does not need it *because* the IMU seed and the observability
gates keep it in Gauss-Newton's sweet spot. Knowing that is knowing when the design would have to
change.

---

## 4. Robust weighting — one outlier must not win

A single mistracked corner is an *outlier*: a residual so large that, squared, it would dominate the
sum and drag the whole state to satisfy one bad pixel. The **Huber** weight caps that influence — a
residual is quadratic while small and only linear once it exceeds a threshold, so a gross outlier
contributes a bounded pull instead of an unbounded one. Camera rows go through Huber; the IMU block
does not (a preintegrated delta is not an outlier candidate the way one corner is). Details where
they bite, in [Module 5](05-tight-coupling.md).

---

## Lab — watch the loop converge, and watch it converge *wrong*

`estimator_check` drives the real estimator over EuRoC deterministically and writes a per-frame CSV;
`vio_check` is the isolated thesis check. Both are nothing but the loop above, iterated.

```bash
colcon build --packages-select glassvio
./build/glassvio/estimator_check          # writes /tmp/glassvio_run.csv
```

1. **Read `rmse` in the CSV.** It is the reprojection residual norm *after* the solve — the loop
   drove it down. It stays 2–4 px: vision fits locally, every frame.
2. **Now the point of the whole course.** `pos_err` (position vs ground truth) slowly *grows* even
   while `rmse` stays small. The solve is minimizing its cost perfectly — and the trajectory is
   drifting anyway. A converged cost is **not** a correct state; the missing information (scale,
   bias) simply is not in the residuals yet. Hold that thought for [Module 8](08-sliding-window.md).

The engine is done. The next three modules are the residuals it eats: the camera, the IMU, and the
prior.

---

← [Module 1](01-manifolds.md) · [Syllabus](README.md) · Next: **[Module 3 — The camera](03-camera.md)** →
