# Module 1 — State on a manifold

> **Prerequisite:** [Module 0](00-the-problem.md), linear algebra. **After this you can:** work
> with rotations as a Lie group — `Exp`/`Log`, `hat`/`vee`, ⊞/⊟, the right Jacobian, the adjoint —
> and say why every rotation Jacobian in this repo is tied to one perturbation convention.

Twelve of the state's fifteen numbers ($\mathbf{p},\mathbf{v},\mathbf{b}_g,\mathbf{b}_a$) live in
ordinary $\mathbb{R}^3$: you add them, subtract them, differentiate them, done. The remaining three
— the orientation $\mathbf{R}$ — do **not**, and every hard line in this codebase traces to that one
block. This module builds exactly the rotation machinery the rest of the course uses, and no more.

Code: [`sophus/so3.hpp`](../glass_core/include/sophus) (the group),
[`nav_state.hpp`](../glass_core/include/glass_core/nav_state.hpp) (`boxplus`/`boxminus`
on the full 15-DoF state). The deeper treatment is glass-lio's
[gauss-newton.md §6](../../glasslio/doc/gauss-newton.md).

---

## 1. Why a rotation is not a vector

A 3-D rotation is a $3\times3$ matrix $\mathbf{R}$ constrained by

$$
\mathbf{R}^\top\mathbf{R} = \mathbf{I}, \qquad \det\mathbf{R} = +1.
$$

Nine numbers, six constraints, **three** real degrees of freedom. The set of all such matrices is
the **special orthogonal group** $SO(3)$ — a smooth, *curved* 3-D surface sitting inside 9-D matrix
space. "Curved" has a concrete consequence: add two rotation matrices entrywise and the result
violates $\mathbf{R}^\top\mathbf{R}=\mathbf{I}$ — it is not a rotation at all. You cannot do
calculus on $SO(3)$ the naive way, because the naive step walks off the surface.

Parametrizations that *pretend* it is flat all fail somewhere: Euler angles gimbal-lock, and a
3-vector of angles has no consistent derivative. The fix is not a better parametrization; it is to
stop treating the group as flat and treat it as what it is — a **Lie group**, a manifold that is
also a group.

---

## 2. The tangent space, `hat`, and `vee`

At the identity, $SO(3)$ has a flat 3-D tangent space called $\mathfrak{so}(3)$ — the space of
**rotation vectors** $\boldsymbol{\phi} = \boldsymbol{\theta}\,\hat{\mathbf{n}}$ (axis
$\hat{\mathbf{n}}$, angle $\theta$). This *is* an ordinary vector space: you can add, scale, and
differentiate in it freely. It is where all the linear algebra will happen.

Two maps move between a 3-vector and its matrix form in $\mathfrak{so}(3)$:

$$
\widehat{\;\cdot\;} : \mathbb{R}^3 \to \mathfrak{so}(3), \qquad
\hat{\boldsymbol{\phi}} =
\begin{bmatrix} 0 & -\phi_z & \phi_y \\ \phi_z & 0 & -\phi_x \\ -\phi_y & \phi_x & 0 \end{bmatrix},
\qquad (\cdot)^\vee \text{ is its inverse.}
$$

`hat` is the skew-symmetric (cross-product) matrix: $\hat{\mathbf{a}}\,\mathbf{b} = \mathbf{a}\times\mathbf{b}$.
It is `Sophus::SO3d::hat` in the code — the same one [we just used](../src/checks/epipolar_check.cpp)
to build $[\mathbf{t}]_\times$ in the epipolar check.

---

## 3. `Exp` and `Log` — the bridge to and from the group

The **exponential map** wraps a tangent vector onto the group; **`Log`** unwraps it:

$$
\mathrm{Exp} : \mathbb{R}^3 \to SO(3), \qquad \mathrm{Log} : SO(3) \to \mathbb{R}^3, \qquad
\mathrm{Log}(\mathrm{Exp}(\boldsymbol{\phi})) = \boldsymbol{\phi}.
$$

For $SO(3)$, `Exp` has the closed form (Rodrigues' formula, $\theta = \lVert\boldsymbol{\phi}\rVert$):

$$
\mathrm{Exp}(\boldsymbol{\phi}) = \mathbf{I} + \frac{\sin\theta}{\theta}\hat{\boldsymbol{\phi}}
  + \frac{1-\cos\theta}{\theta^2}\hat{\boldsymbol{\phi}}^2.
$$

Two facts are all you need to carry forward:

- **`Exp` turns a gyro reading into a rotation.** A gyro reports a rate $\boldsymbol{\omega}$; over
  $\Delta t$ the body turns by the rotation vector $\boldsymbol{\omega}\,\Delta t$, and
  $\mathrm{Exp}(\boldsymbol{\omega}\,\Delta t)$ is the actual rotation. This is the first line of
  [preintegration](04-imu-preintegration.md).
- **Composition stays exactly on the manifold.** $\mathbf{R}_1\mathbf{R}_2 \in SO(3)$ exactly, forever —
  no drift off the surface, no periodic re-orthonormalization. This is why rotations are *composed*,
  never added: `dR_ = dR_ * dRk`, ten thousand times, still a perfect rotation.

---

## 4. ⊞ and ⊟ — calculus on the manifold

To do least squares we need to (a) take a small step from a rotation and (b) measure the difference
between two rotations, both as ordinary 3-vectors. Those are the **boxplus** and **boxminus**
operators. glassvio uses the **right** convention (perturbation applied in the body frame, on the
right):

$$
\mathbf{R} \boxplus \boldsymbol{\phi} \;=\; \mathbf{R}\,\mathrm{Exp}(\boldsymbol{\phi}),
\qquad
\mathbf{R}_2 \boxminus \mathbf{R}_1 \;=\; \mathrm{Log}(\mathbf{R}_1^\top \mathbf{R}_2).
$$

`⊞` takes a rotation and a tangent step and returns a rotation (the *retraction* — how a
Gauss-Newton update lands back on the manifold). `⊟` takes two rotations and returns the tangent
vector between them (used for the state prior in [Module 8](08-sliding-window.md)). They invert each
other: $(\mathbf{R}\boxplus\boldsymbol{\phi})\boxminus\mathbf{R} = \boldsymbol{\phi}$.

For the **full 15-DoF state**, `boxplus` is these two on the rotation block and plain `+` on the
other twelve:

$$
\mathbf{x}\boxplus\delta\mathbf{x} =
(\mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol{\phi}),\ \mathbf{p}+\delta\mathbf{p},\ \mathbf{v}+\delta\mathbf{v},\ \mathbf{b}_g+\delta\mathbf{b}_g,\ \mathbf{b}_a+\delta\mathbf{b}_a).
$$

That is [`nav_state.hpp`](../glass_core/include/glass_core/nav_state.hpp) in one line:
a 15-DoF manifold with exactly **one** non-trivial block. Everything the solver does is: linearize
in the flat tangent space, solve for $\delta\mathbf{x}$, retract with `⊞`.

---

## 5. Two objects the Jacobians need: `J_r` and the adjoint

Curvature is not free — two correction terms appear whenever a perturbation passes through `Exp` or
gets moved between frames.

**The right Jacobian $\mathbf{J}_r$.** `Exp` is nonlinear, so a perturbation does *not* pass through
it unchanged:

$$
\mathrm{Exp}(\boldsymbol{\phi} + \boldsymbol{\delta}) \approx
\mathrm{Exp}(\boldsymbol{\phi})\,\mathrm{Exp}\!\big(\mathbf{J}_r(\boldsymbol{\phi})\,\boldsymbol{\delta}\big).
$$

$\mathbf{J}_r$ is the "stretch" the manifold applies to a tangent perturbation, and
$\mathbf{J}_r \to \mathbf{I}$ as $\boldsymbol{\phi}\to\mathbf{0}$. That limit is a trap: at small
rotations, *omitting* $\mathbf{J}_r$ is nearly right — it passes unit tests and short bags — and goes
wrong exactly when you rotate hard, i.e. when the IMU mattered most. It is the most-cited bug in
preintegration code. It appears in [Module 4](04-imu-preintegration.md) as `rightJacobian(...)`.

**The adjoint $\mathrm{Ad}_\mathbf{R}$.** It converts a tangent vector from the right (body) side to
the left (world) side: $\mathbf{R}\,\mathrm{Exp}(\boldsymbol{\phi}) = \mathrm{Exp}(\mathrm{Ad}_\mathbf{R}\boldsymbol{\phi})\,\mathbf{R}$,
with $\mathrm{Ad}_\mathbf{R} = \mathbf{R}$ for $SO(3)$. You will not derive with it here, but it is
why "left vs right" is a real distinction and not pedantry.

`★ The convention is a contract ─────────────────`
Left- and right-perturbation give the **same manifold** but **incompatible Jacobians**. glassvio is
**right** everywhere ($\mathbf{R}\leftarrow\mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol{\phi})$,
because a gyro measures body-frame rate). glass-lio's `optimizeSE3`, whose increment lives in the
world frame, is **left**. Hand a right-convention Jacobian to a left-convention solve and nothing
crashes — the estimator runs, converges, and is wrong. This is why every Jacobian in the repo is
pinned against finite differences taken *through `boxplus`*: the retraction is part of the
derivative's definition.
`──────────────────────────────────────────────────`

---

## Lab — the retraction round-trips, and the Jacobians are pinned to it

`glass_core`'s nav-residual test exercises exactly this module: it perturbs a `NavState` through
`boxplus`, and checks that the analytic rotation Jacobians match finite differences taken the same
way.

```bash
colcon build --packages-select glasslio
./build/glasslio/test_nav_residual         # (ctest target; run via colcon test if not a bare binary)
colcon test --packages-select glasslio --ctest-args -R nav_residual ; colcon test-result --verbose
```

1. **Confirm it passes.** The analytic and numeric Jacobians agree — the `⊞`/`Exp`/`J_r` machinery
   above is self-consistent.
2. **Now break it.** In [`nav_residual.hpp`](../glass_core/include/glass_core/nav_residual.hpp),
   replace the right Jacobian `Jr` with the identity `I` in the rotation block, rebuild, re-run. The
   test fails — but only in the entries where the rotation is large. That failure *is* §5's trap: at
   small angles the bug is invisible. Revert.

The lesson: on a manifold, "the Jacobian" is meaningless without naming the retraction. The next
module is what consumes these Jacobians — Gauss-Newton.

---

← [Module 0](00-the-problem.md) · [Syllabus](README.md) · Next: **[Module 2 — Nonlinear least squares](02-least-squares.md)** →
