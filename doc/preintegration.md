# Preintegration — how the deltas are actually accumulated

The IMU factor's engine room: how a stream of gyro/accel samples becomes the three
quantities `ΔR, Δv, Δp` that constrain two poses, and the accumulated Jacobians and
covariance that travel with them.

Code: [`preintegration.cpp`](../../glasslio/glass_core/src/preintegration.cpp),
[`preintegration.hpp`](../../glasslio/glass_core/include/glass_core/preintegration.hpp).
Self-check: [`test_preintegration.cpp`](../../glasslio/glass_core/test/test_preintegration.cpp).
Measured on real data: [`imu_dead_reckon_check.cpp`](../src/imu_dead_reckon_check.cpp).

**This doc is the *how*.** For *why* preintegration exists at all — the `R_i` dependence
it removes, why gravity is excluded, how the deltas enter the residual — read glass-lio's
[§7.4 and §7.5](../../glasslio/doc/7-tight-coupling.md) first. This one picks up where that
leaves off: what one `integrate()` call physically does, and the ordering discipline that
is the difference between a correct estimator and a plausible-looking wrong one.

---

## 1. The object is an accumulator with a frozen bias

```cpp
ImuPreintegration pre(bias_gyro, bias_accel, gyro_noise, accel_noise);
for (each sample) pre.integrate(gyro, accel, dt);
// -> pre.dR(), pre.dv(), pre.dp(), pre.dt(), pre.covariance()
```

The bias is fixed **at construction** and every delta is relative to *that* bias. This is
the central bargain of the whole scheme and the reason the class exists rather than a free
function: the bias is a variable the solver moves, so the deltas would be invalidated on
every iteration. Instead the object carries `∂Δ/∂b` alongside the deltas and shifts them
along those Jacobians on demand (§6).

Each instance covers **one interval**. It is not reset or reused — glasslio builds a fresh
one per scan; glassvio will build one per image pair.

---

## 2. Anatomy of one `integrate()` call

### 2.1 The setup

```cpp
if (dt <= 0.0) return;                            // a non-advancing sample says nothing

const Eigen::Vector3d w = gyro  - bias_gyro_;     // bias-corrected measurements
const Eigen::Vector3d a = accel - bias_accel_;

const Eigen::Matrix3d dR_mat = dR_.matrix();      // the state BEFORE this step
const Eigen::Matrix3d a_hat  = SO3d::hat(a);      // â x = a × x
const SO3d dRk = SO3d::exp(w * dt);               // this step's rotation increment
const Eigen::Matrix3d Jr = rightJacobian(w * dt);
```

`dR_mat` is captured **once, here**, and every block below reads it. That is not an
optimisation — it is the correctness condition. See §4.

`a` is the **specific force**, not acceleration: at rest it points *up* with magnitude
≈9.81. Gravity is not subtracted here and never will be (that happens in the residual),
which is why `Δv` over a stationary interval is *not* zero.

### 2.2 The state — the three lines you came for

$$
\begin{aligned}
\Delta\mathbf{p} &\mathrel{+}= \Delta\mathbf{v}\,\Delta t + \tfrac{1}{2}\,\Delta\mathbf{R}\,\mathbf{a}\,\Delta t^2 \\
\Delta\mathbf{v} &\mathrel{+}= \Delta\mathbf{R}\,\mathbf{a}\,\Delta t \\
\Delta\mathbf{R} &\mathrel{=} \Delta\mathbf{R} \cdot \mathrm{Exp}(\boldsymbol{\omega}\,\Delta t)
\end{aligned}
$$

```cpp
dp_ += dv_ * dt + 0.5 * dR_mat * a * dt * dt;
dv_ += dR_mat * a * dt;
dR_  = dR_ * dRk;      // compose ON the manifold, on the RIGHT
dt_ += dt;
```

Ordinary kinematics — `p += v·dt + ½a·dt²` — with two differences that matter.

**`ΔR` composes, it does not add**, and it composes on the *right*. That one line is the
only hard thing in the file, and §3 is about why.

**Nothing here mentions `R_i`, `p_i` or `v_i`.** That absence *is* preintegration.

### 2.3 The covariance — propagated in the tangent space

The deltas are not certain, and the solver needs to know how uncertain in order to weigh
the IMU against the camera. The error state `[δφ; δv; δp]` propagates linearly:

$$
\boldsymbol{\epsilon}_{k+1} = \mathbf{A}\,\boldsymbol{\epsilon}_k + \mathbf{B}\,\mathbf{n}
\qquad\Longrightarrow\qquad
\boldsymbol{\Sigma} \leftarrow \mathbf{A}\,\boldsymbol{\Sigma}\,\mathbf{A}^\top + \mathbf{B}\,\mathbf{Q}\,\mathbf{B}^\top
$$

The interesting entries of `A` are readable physics, not bookkeeping:

| block | value | what it says |
|---|---|---|
| `A(0,0)` | $\Delta\mathbf{R}_k^\top$ | orientation error rotates with the inverse increment |
| `A(3,0)` | $-\Delta\mathbf{R}\,\hat{\mathbf{a}}\,\Delta t$ | **an orientation error tilts the accelerometer** — gravity leaks into velocity |
| `A(6,0)` | $-\tfrac{1}{2}\Delta\mathbf{R}\,\hat{\mathbf{a}}\,\Delta t^2$ | the same leak, integrated once more into position |
| `A(6,3)` | $\mathbf{I}\,\Delta t$ | velocity error integrates straight into position error |

`A(3,0)` is the one worth staring at. It is why attitude error is the dominant killer in
inertial navigation: a small tilt mis-resolves a 9.81 m/s² vector, and 1° of tilt injects
0.17 m/s² of phantom horizontal acceleration — which double-integrates into **0.34 m of
position error in 2 s**.

**The noise term has a trap in it:**

```cpp
Q.block<3,3>(0,0) = I * (gyro_noise_ * gyro_noise_ / dt);   // note: DIVIDED by dt
```

`gyro_noise` is a continuous-time **density** (rad/s/√Hz), not a per-sample standard
deviation. Holding it over a step of length `Δt` gives variance `σ²/Δt`, *divided*, not
multiplied. The sanity check is the physics: halve `Δt` and you take twice as many steps,
each individually noisier — so the accumulated random walk grows as **√t, not t**. Write
`σ²·Δt` instead and the covariance still looks entirely plausible; it is simply the wrong
size, forever, and the IMU is mis-weighted against every other sensor.

### 2.4 The bias Jacobians

$$
\begin{aligned}
\frac{\partial \Delta\mathbf{p}}{\partial \mathbf{b}_a} &\mathrel{+}= \frac{\partial \Delta\mathbf{v}}{\partial \mathbf{b}_a}\Delta t - \tfrac{1}{2}\Delta\mathbf{R}\,\Delta t^2
&\qquad
\frac{\partial \Delta\mathbf{v}}{\partial \mathbf{b}_a} &\mathrel{+}= -\Delta\mathbf{R}\,\Delta t \\
\frac{\partial \Delta\mathbf{p}}{\partial \mathbf{b}_g} &\mathrel{+}= \frac{\partial \Delta\mathbf{v}}{\partial \mathbf{b}_g}\Delta t - \tfrac{1}{2}\Delta\mathbf{R}\,\hat{\mathbf{a}}\,\frac{\partial \Delta\mathbf{R}}{\partial \mathbf{b}_g}\Delta t^2
&\qquad
\frac{\partial \Delta\mathbf{v}}{\partial \mathbf{b}_g} &\mathrel{+}= -\Delta\mathbf{R}\,\hat{\mathbf{a}}\,\frac{\partial \Delta\mathbf{R}}{\partial \mathbf{b}_g}\Delta t \\
\frac{\partial \Delta\mathbf{R}}{\partial \mathbf{b}_g} &\mathrel{=} \Delta\mathbf{R}_k^\top \frac{\partial \Delta\mathbf{R}}{\partial \mathbf{b}_g} - \mathbf{J}_r\,\Delta t
\end{aligned}
$$

(Forster et al., eq. 59–61.) Note the structure: **a gyro bias error reaches position
through orientation** — `∂Δp/∂b_g` contains `∂ΔR/∂b_g`, chained through `â`. There is no
`∂ΔR/∂b_a` because the accelerometer does not affect the integrated rotation at all; that
block is structurally absent, not merely zero.

---

## 3. Why `ΔR` is the only hard line — the Lie-algebra content

Two of the three deltas are ordinary vectors. One is not, and every awkward thing in this
file traces back to that single asymmetry.

For the general treatment — the retraction, why not Euler angles or quaternions, the
solver's *left* convention — see glass-lio's
[gauss-newton.md §6](../../glasslio/doc/gauss-newton.md). What follows is only what
preintegration itself forces.

### Why the rotation cannot simply be added

`Δv` and `Δp` live in ℝ³, so `+=` is legal and means exactly what it looks like. `ΔR` lives
in **SO(3)**: 3×3 matrices constrained by $\mathbf{R}^\top\mathbf{R} = \mathbf{I}$ and
$\det\mathbf{R} = +1$. Nine numbers, six constraints, **three** real degrees of freedom — a
*curved* 3-D manifold. Add two rotation matrices and orthonormality is destroyed; the sum is
not a rotation at all.

`Exp` is the bridge:

$$
\mathrm{Exp} : \mathbb{R}^3 \longrightarrow SO(3)
$$

`SO3d::exp(w * dt)` takes the gyro's **rotation vector** (axis × angle — an element of the
tangent space `so(3)`, which *is* a flat vector space) and returns an actual rotation. The
gyro reports a rate; `w·dt` is a tangent vector; `Exp` is what makes it a group element that
can be composed.

### Composition is what keeps it exact

```cpp
dR_ = dR_ * dRk;
```

Composition is a **group operation**, so the result is *exactly* an element of SO(3) — after
one step or ten thousand. No drift off the manifold, no periodic re-orthonormalisation, no
creeping violation of $\mathbf{R}^\top\mathbf{R} = \mathbf{I}$. Contrast the naive
alternative: update additively, then project back onto the manifold. That projection is a
correction that is no part of the integration, applied to an error that should never have
existed.

### …and on the RIGHT, because the gyro is strapped down

`dR_ * dRk`, not `dRk * dR_`. A gyro measures angular rate in the **body** frame — the frame
the sensor occupies *now*, not the one the interval started in — so its increment composes
on the right. This is not taste; it is dictated by the frame the measurement lives in:

| | increment measured in | composes on |
|---|---|---|
| `optimizeSE3` | the **world** frame | the **left** |
| this file, `boxplus`, `NavState` | the **body** frame (a gyro) | the **right** |

Same manifold, same `Exp`, opposite side, **incompatible Jacobians**. Swap them and the
estimator still runs, still converges, and is wrong.

**That makes it a contract, not a detail.** Every Jacobian this file accumulates is valid
*only* under the right perturbation $\mathbf{R} \leftarrow \mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol{\phi})$
that `nav_state.hpp`'s `boxplus` applies. Hand them to a solver that perturbs on the left
and nothing complains.

### The price: `J_r`

Curvature is not free. `Exp` is nonlinear, so a perturbation does **not** pass through it
unchanged:

$$
\mathrm{Exp}(\boldsymbol{\phi} + \boldsymbol{\delta}) \approx \mathrm{Exp}(\boldsymbol{\phi})\,\mathrm{Exp}\!\left(\mathbf{J}_r(\boldsymbol{\phi})\,\boldsymbol{\delta}\right)
$$

$\mathbf{J}_r$ is the stretch the manifold applies to your perturbation. It is why
`rightJacobian(w * dt)` is computed once in the setup and used **exactly twice**:

| where | line | why |
|---|---|---|
| covariance | `B.block<3,3>(0,0) = Jr * dt` | gyro noise reaches the rotation error through the manifold, stretched by $\mathbf{J}_r$ |
| bias Jacobian | `dR_dbg_ = dRk.transpose() * dR_dbg_ - Jr * dt` | a gyro-bias offset is a tangent perturbation — same stretch |

$\mathbf{J}_r \to \mathbf{I}$ as $\boldsymbol{\phi} \to \mathbf{0}$. **That is precisely why
omitting it is the most-cited bug in preintegration code**: at small rotations the wrong
answer is nearly right, so it passes unit tests, demos, and short bags — and degrades exactly
when you rotate hard, i.e. when the IMU mattered most. The same identity shows up inverted in
`imuJacobian` as `rightJacobianInverse`, whose comment says so outright.

### The payoff

Three deltas; one curved block, two flat:

| delta | space | accumulation (§2.2) | bias correction (§6) |
|---|---|---|---|
| `ΔR` | SO(3), **curved** | `dR_ * dRk` — compose | `dR_ * Exp(dR_dbg·dbg)` — compose |
| `Δv` | ℝ³, flat | `+=` | `+` |
| `Δp` | ℝ³, flat | `+=` | `+` |

That table is the whole argument for doing it this way. Lie algebra is not complicating
these quantities — it is **quarantining the complication to one of them**, so the other two
stay ordinary arithmetic. `nav_state.hpp` makes the identical point one level up, about the
full 15-DoF state: *"a 15-DoF manifold with exactly ONE non-trivial block."*

---

## 4. The ordering rule — stated once, applied three times

> **Every block linearises about the state *before* the step. So whatever others depend
> on must be advanced last.**

This single rule explains every ordering in the file:

| within | reads the old… | so this goes last |
|---|---|---|
| the whole call | `dR_` (captured as `dR_mat`) | `dR_` (§2.2, last state line) |
| the state block | `dv_` (in the `dp_` line) | `dv_` before `dR_` |
| the bias block | `dR_dbg_` (in the `dp_`/`dv_` lines) | `dR_dbg_` |
| the call | the current `dR_` for `A`, `B` | covariance computed **first** |

Hoist any one of those up a single line and you get **no crash, no NaN, no exception** —
just position integrated with next-step's orientation. The estimator runs, converges, and
reports a confident trajectory that is quietly wrong. This is the defining hazard the
[testing doc](../../glasslio/doc/testing.md) is about, and the reason the accumulation is
pinned against brute-force integration rather than against a re-derivation.

---

## 5. `Δt` is a contract, not a hint

`integrate()` holds the sample **constant** over `dt`. That is a rectangle rule, and it is
only honest when `dt` is one real sampling interval.

This is not academic. The KITTI bag used by glassvio's phase-1 check drops **5.59 s of IMU
across 28 gaps**, two of them ≈1.6 s:

| gap | at |
|---|---|
| 1.67 s | t = 14.6 s |
| 1.61 s | t = 52.4 s |
| 50–270 ms × 26 | clustered from t = 38 s |

Feeding `integrate()` a `dt` of 1.67 does not fail. It cheerfully invents **1.6 seconds of
fictitious constant acceleration**, and the resulting drift measures the gap rather than
the sensor. So the consumer's job — not the engine's — is to detect a break in the stream
and end the interval:

```cpp
if (imu[k+1].t - imu[k].t > kMaxSampleGap) { /* break the window */ }
```

`glass_core` deliberately does not police this: it cannot know whether a large `dt` means a
dropout or a legitimately slow sensor. The check that *does* police it is
[`imu_dead_reckon_check.cpp`](../src/imu_dead_reckon_check.cpp), which skips any window
containing a gap and reports how many it dropped (19 of 35, on this bag).

---

## 6. The bias correction — the payoff

```cpp
dR_corrected(dbg) → dR_ * SO3d::exp(dR_dbg_ * dbg);          // manifold: compose, RIGHT
dv_corrected(...) → dv_ + dv_dbg_ * dbg + dv_dba_ * dba;     // ℝ³: plain addition
dp_corrected(...) → dp_ + dp_dbg_ * dbg + dp_dba_ * dba;
```

`dbg`/`dba` are the **offset from the bias the deltas were integrated with** — not the new
bias itself. `imuResidual` computes them as `xj.bg - pre.bias_gyro()`.

This is a first-order Taylor expansion, and that is the whole trade: a little accuracy for
a bias that has barely moved, against re-integrating hundreds of samples inside every
solver iteration. `test_preintegration` verifies it is genuinely second-order accurate —
halve the bias offset and the error falls **4×**. If the bias ever drifts far enough that
this stops holding, the answer is to **re-integrate**, not to add higher-order terms.

Note the asymmetry: the rotation correction *composes on the right* because it lives in the
tangent space at `dR_`, while `dv`/`dp` live in ℝ³ and simply add. One curved block, two
flat ones — the same split as `boxplus`.

---

## 7. What this has actually been measured to do

Phase 1 of glassvio ran the accumulator against KITTI ground truth
([`imu_dead_reckon_check.cpp`](../src/imu_dead_reckon_check.cpp)): seed a `NavState` from
`/kitti/pose`, integrate ~200 samples at `dt ≈ 0.011 s` (91.5 Hz), `predictState` forward,
compare to truth.

| window | travelled | drift | |
|---|---|---|---|
| best (cruising) | 15.35 m | **0.021 m** | 0.14 % |
| median of 16 | — | **0.242 m** | — |
| worst | 6.94 m | 1.011 m | 14.6 % |

An independent NumPy reimplementation of the same physics landed on a median of 0.298 m —
**agreement to 0.06 m between two implementations sharing no code.** That cross-check is
worth more than either number alone: it is an [oracle](../../glasslio/doc/testing.md), and
a sign error or a bad convention in either would have shown up as disagreement, not as a
plausible trajectory.

The worst windows correlate with *low distance travelled* — i.e. braking and turning — and
are likely dominated by the check's own ground-truth velocity seeding (a ±0.25 s finite
difference is biased exactly when acceleration is changing; 0.4 m/s of seed error alone
produces 0.8 m over 2 s). The sensor is probably better than the table's tail suggests.

---

## 8. What that check does *not* exercise

Worth being precise, because the table above is reassuring and only covers a third of the
file.

`predictState` consumes `pre.dR()`, `pre.dv()`, `pre.dp()` — the **raw** deltas — and phase
1 seeded `b_g = b_a = 0`. So:

- ✅ **§2.2 the state accumulation**, the SO(3) composition, the gravity convention.
- ❌ **§2.3 the covariance** — computed on every one of those ~200 calls, then never read.
- ❌ **§2.4 the bias Jacobians** — likewise accumulated and discarded.
- ❌ **§6 the bias correction** — `dbg = 0` makes `dR_corrected` an identity.

Those three are pinned by `test_preintegration` against finite differences and brute-force
integration, but they remain unexercised **on real data** until the reprojection factor
lands and the solver starts moving `b_g`/`b_a` for real. That is phase 3.

---

## Parameters

| Name | Meaning | Note |
|---|---|---|
| `bias_gyro`, `bias_accel` | the bias the deltas are relative to | frozen at construction; move off it via §6, don't re-integrate |
| `gyro_noise` | rad/s/√Hz | continuous **density** — enters `Q` as `σ²/Δt` |
| `accel_noise` | m/s²/√Hz | same; these two set how much the IMU is *trusted* |
