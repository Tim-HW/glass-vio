# The tightly-coupled solve — three residuals, one Gauss-Newton system

How a camera and an IMU are fused into a single pose estimate. This is the whole thesis of
glassvio and glass-lio in one place: **the fusion is not a filter or a blend — it is two sets
of residuals stacked into one normal-equations system**, solved by the same Gauss-Newton the
LiDAR path uses.

Code: [`visual_registration.cpp`](../src/vio/visual_registration.cpp) (the loop),
[`reprojection.hpp`](../include/glassvio/reprojection.hpp) (the camera factor),
[`nav_residual.hpp`](../../glasslio/glass_core/include/glass_core/nav_residual.hpp) (the IMU +
prior factors), [`gauss_newton.hpp`](../../glasslio/glass_core/include/glass_core/gauss_newton.hpp)
(the accumulator + solve).

**For the generic manifold solver** — the normal equations *derived*, what the Gauss-Newton
approximation throws away, LDLT, Huber, the retraction — read glass-lio's
[gauss-newton.md](../../glasslio/doc/gauss-newton.md) first. This doc is the VIO-specific
assembly: which residuals, the 15-DoF state, and how they combine.

---

## 1. The problem: nonlinear least squares on a 15-DoF manifold

We estimate the navigation state

$$
\mathbf{x} = (\mathbf{R},\ \mathbf{p},\ \mathbf{v},\ \mathbf{b}_g,\ \mathbf{b}_a)
\qquad 3+3+3+3+3 = 15 \text{ DoF}
$$

(orientation, position, velocity, gyro bias, accel bias — see
[nav_state.hpp](../../glasslio/glass_core/include/glass_core/nav_state.hpp)). We minimise the
total weighted squared residual over three factors:

$$
C(\mathbf{x}) \;=\; \underbrace{\sum_{\text{landmarks}} \lVert \mathbf{r}_{\text{px}} \rVert^2_{\Omega_{\text{px}}}}_{\text{camera}}
\;+\; \underbrace{\lVert \mathbf{r}_{\text{imu}} \rVert^2_{\Omega_{\text{imu}}}}_{\text{IMU}}
\;+\; \underbrace{\lVert \mathbf{r}_{\text{prior}} \rVert^2_{\Omega_{\text{prior}}}}_{\text{prior}}
$$

`R` lives on the curved group $SO(3)$; the other twelve components are ordinary $\mathbb{R}^3$.
So this is least squares on a manifold — flat everywhere except one 3-DoF rotation block. That
single curved block is the entire source of the Lie-algebra content below.

---

## 2. The three residuals

### 2.1 Reprojection (camera) — 2 per landmark

A fixed 3-D landmark $\mathbf{P}_w$, carried into the camera through the pose and the extrinsic,
must land on the pixel the tracker observed:

$$
\mathbf{P}_i = \mathbf{R}^\top(\mathbf{P}_w - \mathbf{p}), \qquad
\mathbf{P}_c = \mathbf{R}_{ci}\mathbf{P}_i + \mathbf{t}_{ci}, \qquad
\mathbf{r}_{\text{px}} = \pi(\mathbf{P}_c) - \mathbf{z}_{\text{obs}}
$$

with the pinhole projection $\pi(\mathbf{P}_c) = \big(f_x X/Z + c_x,\; f_y Y/Z + c_y\big)$.
Landmarks are **held fixed** during the solve — they play the role glass-lio's map planes play,
which is what keeps the state at 15 and not $15+3L$. See
[`reprojection.hpp`](../include/glassvio/reprojection.hpp).

### 2.2 IMU (preintegration) — 9

Between the previous state $\mathbf{x}_i$ and the current $\mathbf{x}_j$, the preintegrated delta
must agree with what the states say happened:

$$
\mathbf{r}_{\text{imu}} =
\begin{bmatrix}
\mathrm{Log}\!\big(\hat{\Delta\mathbf{R}}^\top \mathbf{R}_i^\top \mathbf{R}_j\big) \\
\mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\,\Delta t) - \hat{\Delta\mathbf{v}} \\
\mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t - \tfrac12\mathbf{g}\Delta t^2) - \hat{\Delta\mathbf{p}}
\end{bmatrix}
$$

Read each row as *"what the state says"* minus *"what the IMU says"*. How the deltas are built
is [preintegration.md](preintegration.md); the residual and its Jacobian are in
[nav_residual.hpp](../../glasslio/glass_core/include/glass_core/nav_residual.hpp).

### 2.3 Prior (bias + carried covariance) — 6

Biases are weakly observable, so left free they absorb real motion. The prior anchors them to
the previous estimate, weighted by the carried covariance plus the random-walk process noise:

$$
\mathbf{r}_{\text{prior}} = \begin{bmatrix} \mathbf{b}_{g,j} - \mathbf{b}_{g,i} \\ \mathbf{b}_{a,j} - \mathbf{b}_{a,i} \end{bmatrix}
$$

This is the half of the bias treatment that says *how wrong a bias might START*, not just how
fast it drifts — the distinction that lets $\mathbf{b}_a$ move toward its true value at all.

---

## 3. Gauss-Newton: linearise, accumulate, solve, retract

The residuals are nonlinear in $\mathbf{x}$, so we cannot solve in one shot. Gauss-Newton takes
the current estimate, **linearises every residual about it**, solves the resulting linear system
for a step, applies the step, and repeats.

### 3.1 Linearise — the Jacobian is the local model

For a small perturbation $\delta\mathbf{x}$ in the tangent space,

$$
\mathbf{r}(\mathbf{x} \boxplus \delta\mathbf{x}) \;\approx\; \mathbf{r}(\mathbf{x}) + \mathbf{J}\,\delta\mathbf{x},
\qquad \mathbf{J} = \left.\frac{\partial \mathbf{r}}{\partial \delta\mathbf{x}}\right|_{\mathbf{x}}
$$

$\mathbf{J}$ is the residual's derivative **evaluated at the current $\mathbf{x}$** — a local
linear model, valid only for small $\delta\mathbf{x}$. That locality is exactly why the whole
thing iterates: re-linearise about the new estimate each pass. §4 covers how each $\mathbf{J}$ is
computed and why it is $M \times 15$.

### 3.2 Accumulate — the normal equations, and the fusion

Substituting the linear model into $C$ and setting the gradient to zero gives the normal
equations $\mathbf{H}\,\delta\mathbf{x} = \mathbf{b}$, accumulated one factor at a time:

$$
\mathbf{H} = \sum \mathbf{J}^\top \Omega\, \mathbf{J}, \qquad
\mathbf{b} = -\sum \mathbf{J}^\top \Omega\, \mathbf{r}
$$

This sum **is** the sensor fusion. Every camera row and the IMU block fold into the *same*
15×15 $\mathbf{H}$; there is no separate blending step, no filter, no weighting coefficient
chosen by hand. In [`visual_registration.cpp`](../src/vio/visual_registration.cpp) it is literally:

```cpp
for (each landmark in view)                        // camera: 2 scalar rows each, Huber-weighted
    accumulateReprojection(eq, x, P_w, z, ...);
eq.addBlock<9>(imuResidual(...), imuJacobian(...), imu_information);   // IMU: one 9-block
eq.addBlock<6>(biasResidual(...), biasJacobian(), bias_information);  // prior: one 6-block
const NavVec dx = eq.solve();                       // H dx = b, via LDLT
x = boxplus(x, dx);                                 // retract onto the manifold
```

Each factor's weight $\Omega = \Sigma^{-1}$ is its inverse covariance — pixel noise for the
camera (whitened by $\sigma_{\text{px}}$), the propagated 9×9 for the IMU, the carried
covariance for the prior. The relative weighting is thus *self-tuning*: a longer IMU gap means a
bigger $\Sigma$, less pull. Nobody tunes it.

### 3.3 Solve and retract

$\mathbf{H} = \sum \mathbf{J}^\top\Omega\mathbf{J}$ is symmetric positive semidefinite by
construction, so `eq.solve()` is a Cholesky factorisation — `H.ldlt().solve(b)`. The step
$\delta\mathbf{x} \in \mathbb{R}^{15}$ lives in the tangent space, and `boxplus` maps it back
onto the manifold:

$$
\mathbf{R} \leftarrow \mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol{\phi}), \qquad
\mathbf{p} \leftarrow \mathbf{p} + \delta\mathbf{p}, \quad \dots
$$

One curved block (compose), twelve flat ones (add). Iterate until $\delta\mathbf{x}$ is small.

---

## 4. The Jacobians — analytic, $M\times 15$, and sparse by physics

Each factor supplies an **analytic** Jacobian (hand-derived by chain rule, not finite
differences), of shape $M \times 15$: two columns-blocks of interest per physical quantity, the
rest structurally zero.

**The reprojection Jacobian** is a three-link chain:

$$
\mathbf{J}_{\text{px}} = \frac{\partial \pi}{\partial \mathbf{P}_c}\cdot \frac{\partial \mathbf{P}_c}{\partial \mathbf{P}_i}\cdot \frac{\partial \mathbf{P}_i}{\partial \delta\mathbf{x}}
= \mathbf{J}_\pi \cdot \mathbf{R}_{ci} \cdot \frac{\partial \mathbf{P}_i}{\partial \delta\mathbf{x}}
$$

where $\mathbf{J}_\pi$ (the $2\times3$ projection derivative) carries the $1/Z$, $1/Z^2$
nonlinearity, and the manifold enters at exactly one place — the rotation block, under the
RIGHT perturbation $\mathbf{R} \leftarrow \mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol\phi)$:

$$
\frac{\partial \mathbf{P}_i}{\partial \delta\boldsymbol\phi} = \widehat{\mathbf{P}_i}, \qquad
\frac{\partial \mathbf{P}_i}{\partial \delta\mathbf{p}} = -\mathbf{R}^\top
$$

So the $2\times15$ matrix has non-zero blocks only at `kIdxPhi` and `kIdxPos`:

```cpp
J.block<2,3>(0, kIdxPhi) = J_pi * R_ci * Sophus::SO3d::hat(P_i);
J.block<2,3>(0, kIdxPos) = -J_pi * R_ci * R_iw;
// kIdxVel, kIdxBg, kIdxBa: left ZERO -- a camera cannot see them.
```

Those nine zero columns are **the physics, not an approximation**: a camera observes pose,
never velocity or bias. The IMU Jacobian (`imuJacobian`, 9×15) is dense across rotation,
velocity, position and *both* biases, because the IMU *does* see them. That asymmetry is the
whole reason the two sensors are complementary — and why loose coupling, which cannot let the
IMU write into a shared $\mathbf{H}$, could never estimate the biases at all.

### The manifold is why every Jacobian must be pinned

$\mathbf{J}$ claims to be $\partial\mathbf{r}/\partial\delta\mathbf{x}$ under the *right*
retraction. A Jacobian for the wrong retraction (or a chain-rule slip, or `Jr` stubbed to
identity) linearises the wrong local model: the solve still runs, still converges, and is
**quietly wrong**. So every Jacobian in the system is verified against finite differences taken
*through `boxplus`* — [`test_reprojection.cpp`](../test/test_reprojection.cpp) pins the camera
factor to 3e-10, [`test_nav_residual.cpp`](../../glasslio/glass_core/test/test_nav_residual.cpp)
pins the IMU and prior. The analytic form runs; the numeric form is the independent oracle.

---

## 5. Two guards the LiDAR factor never needed

- **Cheirality.** $\pi$ divides by $Z$. A landmark behind the camera projects to an
  ordinary-looking pixel (mirrored through the optical centre) with a **sign-flipped** Jacobian —
  no NaN, just a confident step the wrong way. So a landmark with $Z < Z_{\min}$ is refused
  before it enters $\mathbf{H}$ (`landmarkInCamera`). Point-to-plane is affine and has no such
  failure.
- **Robust weighting.** A KLT mismatch is an outlier; the camera rows go through `addScalar`
  with a Huber weight (which is why the 2-vector is split into two whitened scalars rather than
  an `addBlock<2>` — for isotropic pixel noise the two are identical, and the split buys Huber).
  The IMU block gets **no** Huber — a preintegrated delta is not an outlier candidate the way one
  tracked corner is.

No damping and no line search: the state is 15-DoF, well-conditioned when enough landmarks are
in view, and seeded by the IMU's own prediction (`predictState`) — already close to the optimum
where Gauss-Newton is strongest.

---

## 6. The sliding-window generalisation

Nothing above changes for the sliding window (see [next-steps.md](next-steps.md)) — it is the
*same* linearise → accumulate → solve → retract loop. Only the **shape** grows:

- $\delta\mathbf{x}$ becomes $15K$ long ($K$ keyframes) instead of 15;
- $\mathbf{H}$ becomes $15K \times 15K$;
- the IMU factor between two keyframes contributes to **two** state blocks —
  $\partial\mathbf{r}/\partial\mathbf{x}_i$ (already built: `imuJacobianI`) and
  $\partial\mathbf{r}/\partial\mathbf{x}_j$ — instead of one;
- the same accel bias appears in $K-1$ IMU factors at once, which is what finally makes it
  observable.

The Gauss-Newton machinery is identical; there is just more of it stacked into one $\mathbf{H}$
and $\mathbf{b}$.
