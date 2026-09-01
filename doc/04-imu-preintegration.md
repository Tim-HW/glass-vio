# Module 4 — The IMU: preintegration

> **Prerequisite:** [Modules 1–2](01-manifolds.md). **After this you can:** turn a gyro/accel stream
> into the deltas $\Delta\mathbf{R},\Delta\mathbf{v},\Delta\mathbf{p}$ that constrain two poses, form
> the 9-DoF IMU residual, and say why its Jacobian is dense where the camera's is zero.

The IMU's contribution to the fused solve is one factor tying the previous state to the current one:
*whatever the two states claim happened between them must match what the IMU integrated.* The
subtlety — and the reason this is a whole module — is that naively integrating the IMU couples it to
the previous state's rotation, which the solver keeps changing. **Preintegration** removes that
coupling. This module builds the delta, the residual it feeds, and its Jacobian.

Code: [`preintegration.cpp`](../glass_core/src/preintegration.cpp),
[`nav_residual.hpp`](../glass_core/include/glass_core/nav_residual.hpp) (the residual +
Jacobians). Checks: [`test_preintegration.cpp`](../glass_core/test/test_preintegration.cpp),
[`imu_dead_reckon_check.cpp`](../src/checks/imu_dead_reckon_check.cpp) (real data).

---

## 1. The IMU residual — what the deltas are *for*

Between the previous state $\mathbf{x}_i$ and the current $\mathbf{x}_j$, the preintegrated deltas
(hatted) must agree with what the two states say happened. That comparison is the residual — 9 rows,
*"what the state says"* minus *"what the IMU says"*:

$$
\mathbf{r}_{\text{imu}} =
\begin{bmatrix}
\mathrm{Log}\!\big(\widehat{\Delta\mathbf{R}}^\top \mathbf{R}_i^\top \mathbf{R}_j\big) \\[2pt]
\mathbf{R}_i^\top(\mathbf{v}_j - \mathbf{v}_i - \mathbf{g}\,\Delta t) - \widehat{\Delta\mathbf{v}} \\[2pt]
\mathbf{R}_i^\top(\mathbf{p}_j - \mathbf{p}_i - \mathbf{v}_i\Delta t - \tfrac12\mathbf{g}\Delta t^2) - \widehat{\Delta\mathbf{p}}
\end{bmatrix}
$$

Two things to notice, because they justify everything below. **Gravity $\mathbf{g}$ appears here, in
the residual, not in the deltas** — the IMU measures *specific force* (gravity included), and gravity
is subtracted out at comparison time, once, where the world frame is known. And **the deltas
$\widehat{\Delta\cdot}$ do not depend on $\mathbf{x}_i$ at all** — that independence is what lets the
solver move $\mathbf{x}_i$ freely without re-integrating. Producing deltas with that property is
preintegration; the rest of the module is *how*.

---

## 2. The object is an accumulator with a frozen bias

```cpp
ImuPreintegration pre(bias_gyro, bias_accel, gyro_noise, accel_noise);
for (each sample) pre.integrate(gyro, accel, dt);
// -> pre.dR(), pre.dv(), pre.dp(), pre.dt(), pre.covariance()
```

The bias is fixed **at construction**, and every delta is relative to *that* bias. This is the
central bargain of the scheme: the bias is a variable the solver moves, so the deltas would be
invalidated on every iteration. Instead the object carries $\partial\Delta/\partial\mathbf{b}$
alongside the deltas and shifts them along those Jacobians on demand (§7). Each instance covers **one
interval** — glassvio builds a fresh one per image pair.

---

## 3. Anatomy of one `integrate()` call

### 3.1 Setup

```cpp
if (dt <= 0.0) return;                            // a non-advancing sample says nothing
const Eigen::Vector3d w = gyro  - bias_gyro_;     // bias-corrected measurements
const Eigen::Vector3d a = accel - bias_accel_;
const Eigen::Matrix3d dR_mat = dR_.matrix();      // the state BEFORE this step
const Eigen::Matrix3d a_hat  = SO3d::hat(a);
const SO3d dRk = SO3d::exp(w * dt);               // this step's rotation increment
const Eigen::Matrix3d Jr = rightJacobian(w * dt);
```

`dR_mat` is captured **once**, and every block below reads it — a correctness condition, not an
optimization (§5). `a` is **specific force**, not acceleration: at rest it points *up* at ≈9.81, and
gravity is never subtracted here (it goes in the residual, §1), which is why $\Delta\mathbf{v}$ over
a stationary interval is not zero.

### 3.2 The state — the three lines you came for

$$
\begin{aligned}
\Delta\mathbf{p} &\mathrel{+}= \Delta\mathbf{v}\,\Delta t + \tfrac12\,\Delta\mathbf{R}\,\mathbf{a}\,\Delta t^2 \\
\Delta\mathbf{v} &\mathrel{+}= \Delta\mathbf{R}\,\mathbf{a}\,\Delta t \\
\Delta\mathbf{R} &\mathrel{=} \Delta\mathbf{R}\cdot\mathrm{Exp}(\boldsymbol{\omega}\,\Delta t)
\end{aligned}
$$

```cpp
dp_ += dv_ * dt + 0.5 * dR_mat * a * dt * dt;
dv_ += dR_mat * a * dt;
dR_  = dR_ * dRk;      // compose ON the manifold, on the RIGHT (Module 1 §3–4)
dt_ += dt;
```

Ordinary kinematics ($p += v\,dt + \tfrac12 a\,dt^2$) with two twists from [Module 1](01-manifolds.md):
$\Delta\mathbf{R}$ **composes on the right** (a gyro measures body-frame rate), never adds; and
**nothing here mentions $\mathbf{R}_i,\mathbf{p}_i,\mathbf{v}_i$** — that absence *is*
preintegration.

### 3.3 The covariance — propagated in the tangent space

The deltas are uncertain, and the solver needs to know how much to weigh the IMU against the camera
([Module 2](02-least-squares.md)'s $\Omega$). The error state $[\delta\boldsymbol{\phi};\delta\mathbf{v};\delta\mathbf{p}]$
propagates linearly, $\boldsymbol{\Sigma}\leftarrow\mathbf{A}\boldsymbol{\Sigma}\mathbf{A}^\top+\mathbf{B}\mathbf{Q}\mathbf{B}^\top$.
The revealing entry:

| block | value | what it says |
|---|---|---|
| `A(3,0)` | $-\Delta\mathbf{R}\,\hat{\mathbf{a}}\,\Delta t$ | **an orientation error tilts the accelerometer** — gravity leaks into velocity |

`A(3,0)` is why attitude error dominates inertial navigation: 1° of tilt mis-resolves 9.81 m/s² into
0.17 m/s² of phantom horizontal acceleration → **0.34 m of position error in 2 s**.

**The noise term has a trap:**

```cpp
Q.block<3,3>(0,0) = I * (gyro_noise_ * gyro_noise_ / dt);   // DIVIDED by dt
```

`gyro_noise` is a continuous-time **density** (rad/s/√Hz), not a per-sample σ. Held over a step of
length $\Delta t$ it gives variance $\sigma^2/\Delta t$ — *divided*. Sanity check: halve $\Delta t$,
take twice as many noisier steps, and the random walk grows as $\sqrt t$, not $t$. Write
$\sigma^2\cdot\Delta t$ and the covariance still looks plausible — it is simply the wrong size,
forever, and the IMU is mis-weighted against every other sensor.

### 3.4 The bias Jacobians

The object accumulates $\partial\Delta\mathbf{R}/\partial\mathbf{b}_g$,
$\partial\Delta\mathbf{v}/\partial\mathbf{b}_{g,a}$, $\partial\Delta\mathbf{p}/\partial\mathbf{b}_{g,a}$
(Forster et al. eq. 59–61). Structure worth seeing: **a gyro bias reaches position through
orientation** — $\partial\Delta\mathbf{p}/\partial\mathbf{b}_g$ chains through
$\partial\Delta\mathbf{R}/\partial\mathbf{b}_g$ and $\hat{\mathbf{a}}$. There is *no*
$\partial\Delta\mathbf{R}/\partial\mathbf{b}_a$: the accelerometer does not affect integrated
rotation, so that block is structurally absent, not merely zero.

---

## 4. The ordering rule — stated once, applied three times

> **Every block linearizes about the state *before* the step. So whatever others depend on must be
> advanced last.**

| within | reads the old… | so this goes last |
|---|---|---|
| the whole call | `dR_` (as `dR_mat`) | `dR_` (last state line) |
| the state block | `dv_` (in the `dp_` line) | `dv_` before `dR_` |
| the bias block | `dR_dbg_` | `dR_dbg_` |
| the call | current `dR_` for `A`,`B` | covariance computed **first** |

Hoist any one up a single line: **no crash, no NaN** — just position integrated with next-step's
orientation. The estimator runs, converges, and reports a confident, quietly-wrong trajectory. This
is why the accumulation is pinned against brute-force integration, not against a re-derivation.

---

## 5. `Δt` is a contract, not a hint

`integrate()` holds the sample **constant** over `dt` — a rectangle rule, honest only when `dt` is
one real sampling interval. The KITTI bag glassvio's phase-1 check uses drops **5.59 s of IMU across
28 gaps**, two ≈1.6 s. Feeding `integrate()` a `dt` of 1.67 does not fail — it invents 1.6 s of
fictitious constant acceleration, and the resulting drift measures the *gap*, not the sensor. So the
consumer, not the engine, must detect a break and end the interval:

```cpp
if (imu[k+1].t - imu[k].t > kMaxSampleGap) { /* break the window */ }
```

`glass_core` deliberately does not police this — it cannot tell a dropout from a legitimately slow
sensor. [`imu_dead_reckon_check`](../src/checks/imu_dead_reckon_check.cpp) does, skipping any window
with a gap (19 of 35 on this bag).

---

## 6. The bias correction — the payoff

```cpp
dR_corrected(dbg) → dR_ * SO3d::exp(dR_dbg_ * dbg);        // manifold: compose, RIGHT
dv_corrected(...) → dv_ + dv_dbg_ * dbg + dv_dba_ * dba;   // ℝ³: plain addition
dp_corrected(...) → dp_ + dp_dbg_ * dbg + dp_dba_ * dba;
```

`dbg`/`dba` are the **offset from the bias the deltas were integrated with**, not the new bias
(`imuResidual` computes `xj.bg - pre.bias_gyro()`). This first-order Taylor expansion is the trade:
a little accuracy for a bias that barely moved, against re-integrating hundreds of samples inside
every solver iteration. `test_preintegration` verifies it is genuinely second-order accurate — halve
the offset, the error falls 4×. If the bias ever drifts too far, the answer is to *re-integrate*, not
to add higher-order terms.

---

## 7. The IMU residual's Jacobian — dense where the camera's was zero

The residual (§1) is differentiated w.r.t. the state through `⊞`, giving `imuJacobian` — a **9×15**
block that is *dense* across rotation, velocity, position, and **both biases**, because the IMU sees
all of them (the mirror image of the camera's nine zero columns, [Module 3 §4](03-camera.md)). The
rotation rows carry a `rightJacobianInverse` — [Module 1](01-manifolds.md)'s $\mathbf{J}_r$ trap,
inverted.

There is a second Jacobian the LiDAR path never needed: `imuJacobianI` = $\partial\mathbf{r}/\partial\mathbf{x}_i$,
the derivative w.r.t. the *previous* state. glass-lio held $\mathbf{x}_i$ fixed and never formed it;
glassvio needs it in two places — to give the previous state a finite certainty in the per-frame
solve ([Module 5 §3](05-tight-coupling.md)), and to connect two keyframes in the sliding window
([Module 8](08-sliding-window.md)). Both are **built and finite-diff-pinned** in `test_nav_residual`.

`★ Insight — asymmetry is the whole point ──────`
Camera Jacobian: dense in pose, zero in velocity and bias. IMU Jacobian: dense in velocity and bias,
present in pose. Stack them ([Module 5](05-tight-coupling.md)) and each fills the other's zeros —
that is *why* two sensors see what neither can alone, and why the fusion must be tight (both writing
into one $\mathbf{H}$) to ever observe the biases.
`─────────────────────────────────────────────────`

---

## Lab — dead-reckon on real data, cross-checked by an independent oracle

`imu_dead_reckon_check` seeds a `NavState` from ground truth, integrates ~200 samples,
`predictState`s forward, and compares to truth — exercising §3.2's accumulation on real KITTI data.

```bash
colcon build --packages-select glassvio
./build/glassvio/imu_dead_reckon_check
```

| window | travelled | drift | |
|---|---|---|---|
| best (cruising) | 15.35 m | **0.021 m** | 0.14 % |
| median of 16 | — | **0.242 m** | — |
| worst | 6.94 m | 1.011 m | 14.6 % |

1. **Read the median (~0.24 m over ~15 m).** That is the IMU alone, over ~2 s — good short-term,
   already drifting. Exactly the "drifts as $t^2$" of [Module 0](00-the-problem.md); vision's job is
   to stop it.
2. **The oracle that matters.** An independent NumPy reimplementation of the same physics lands at
   0.298 m median — agreement to 0.06 m between two implementations *sharing no code*. A sign error
   or bad convention in either would show up as disagreement, not as a plausible trajectory. This is
   the testing philosophy of the whole course in one number.
3. **What this does NOT exercise:** with $b_g=b_a=0$ and the raw deltas, the covariance (§3.3), bias
   Jacobians (§3.4), and bias correction (§6) are computed and never read. They stay pinned by
   `test_preintegration` until the solver moves the biases for real — which is the next module.

---

## Parameters

| Name | Meaning | Note |
|---|---|---|
| `bias_gyro`, `bias_accel` | the bias the deltas are relative to | frozen at construction; move off it via §6 |
| `gyro_noise` | rad/s/√Hz | continuous **density** — enters `Q` as `σ²/Δt` |
| `accel_noise` | m/s²/√Hz | same; these set how much the IMU is *trusted* |

---

← [Module 3](03-camera.md) · [Syllabus](README.md) · Next: **[Module 5 — Tight coupling](05-tight-coupling.md)** →
