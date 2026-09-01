# Module 5 — Tight coupling: stacking IS the fusion

> **Prerequisite:** [Modules 2–4](02-least-squares.md). **After this you can:** assemble the camera
> residual, the IMU residual, and a bias prior into one 15-DoF Gauss-Newton system, and explain why
> that single stack — no filter, no blend — is the entire fusion.

This is the thesis of glassvio (and glass-lio) in one place: **the fusion is not a filter or a
weighted average — it is two sets of residuals stacked into one normal-equations system**, solved by
the same Gauss-Newton the LiDAR path uses. You already have the pieces: the engine
([Module 2](02-least-squares.md)), the camera Jacobian ([Module 3](03-camera.md)), the IMU
Jacobian ([Module 4](04-imu-preintegration.md)). This module bolts them together and runs the
per-frame loop.

Code: [`visual_registration.cpp`](../src/vio/visual_registration.cpp) (`solveFrame`, the loop),
[`vio_estimator.cpp`](../src/vio/vio_estimator.cpp) (the frame driver),
[`sync.hpp`](../include/glassvio/sync.hpp) (pairing). Check:
[`vio_check.cpp`](../src/checks/vio_check.cpp).

---

## 1. The third residual — the bias prior

Two residuals are done. But biases are only *weakly* observable per frame ([Module 4](04-imu-preintegration.md)):
left completely free, they absorb real motion into themselves. A **prior** anchors them to the
previous estimate, weighted by the carried covariance plus random-walk process noise:

$$
\mathbf{r}_{\text{prior}} = \begin{bmatrix} \mathbf{b}_{g,j}-\mathbf{b}_{g,i} \\ \mathbf{b}_{a,j}-\mathbf{b}_{a,i} \end{bmatrix} \quad (6\text{ rows}).
$$

This encodes *how wrong a bias might start* and *how fast it drifts* — the piece that lets
$\mathbf{b}_a$ move toward its true value at all instead of being pinned at zero or wandering free.

So the full cost is three factors over the 15-DoF state:

$$
C(\mathbf{x}) = \underbrace{\sum_{\text{landmarks}}\lVert\mathbf{r}_{\text{px}}\rVert^2_{\Omega_{\text{px}}}}_{\text{camera, Module 3}}
+ \underbrace{\lVert\mathbf{r}_{\text{imu}}\rVert^2_{\Omega_{\text{imu}}}}_{\text{IMU, Module 4}}
+ \underbrace{\lVert\mathbf{r}_{\text{prior}}\rVert^2_{\Omega_{\text{prior}}}}_{\text{prior}}.
$$

---

## 2. The assembly — one `H`, and why that is the fusion

[Module 2](02-least-squares.md)'s accumulation, with the three specific factors plugged in. In
[`visual_registration.cpp`](../src/vio/visual_registration.cpp) it is literally:

```cpp
for (each landmark in view)                        // camera: 2 scalar rows each, Huber-weighted
    accumulateReprojection(eq, x, P_w, z, ...);
eq.addBlock<9>(imuResidual(...), imuJacobian(...), imu_information);   // IMU: one 9-block
eq.addBlock<6>(biasResidual(...), biasJacobian(), bias_information);  // prior: one 6-block
const NavVec dx = eq.solve();                       // H dx = b, via LDLT
x = boxplus(x, dx);                                 // retract onto the manifold
```

`★ Insight — there is no fusion step ───────────`
Every camera row and the IMU block fold into the *same* 15×15 $\mathbf{H} = \sum\mathbf{J}^\top\Omega\mathbf{J}$.
No separate blending, no filter, no hand-chosen weighting coefficient. Solving $\mathbf{H}$ balances
them, and each factor's weight $\Omega=\Sigma^{-1}$ is its *own* inverse covariance — pixel noise for
the camera, the propagated 9×9 for the IMU, the carried covariance for the prior. The balance is
**self-tuning**: a longer IMU gap means a bigger $\Sigma$ and less pull. Nobody tunes it. And because
the camera fills the IMU's zero columns and vice versa ([Modules 3–4](03-camera.md)), the stack
observes what neither sensor could alone — the definition of *tight* coupling.
`─────────────────────────────────────────────────`

**Robust weighting.** The camera rows go through `addScalar` with a Huber weight (which is why the
2-vector is split into two whitened scalars rather than `addBlock<2>` — identical for isotropic pixel
noise, but the split buys Huber). The IMU block gets **no** Huber: a preintegrated delta is not an
outlier candidate the way one tracked corner is.

---

## 3. Weighting the IMU by its OWN uncertainty — inflated

A subtle bug lives here, and its fix is why [Module 4](04-imu-preintegration.md)'s `imuJacobianI`
exists. The IMU factor's information is **not** $\Sigma_{\text{pre}}^{-1}$ alone — that would assert
the previous state $\mathbf{x}_i$ was *perfect* (it enters `imuResidual` as a constant). Over a short
interval the IMU's information is then effectively infinite and it overrules every pixel. Measured
before the fix: reprojection error grew to **643 px** with 135 landmarks in view, unheeded.

The fix propagates $\mathbf{x}_i$'s uncertainty through the *other* half of the IMU Jacobian:

$$
\Sigma_{\text{eff}} = \Sigma_{\text{pre}} + \mathbf{J}_i\,\mathbf{P}_i\,\mathbf{J}_i^\top,
\qquad \mathbf{J}_i = \frac{\partial\mathbf{r}}{\partial\mathbf{x}_i} = \texttt{imuJacobianI}.
$$

This gives the previous state a *finite* certainty and lets vision and the IMU actually negotiate.
The posterior $\mathbf{H}^{-1}$ is carried forward as the next $\mathbf{P}$ — what `NormalEquationsN::H()`
has always been for.

---

## 4. Sync — where the two-pose factor gets its measurements

The IMU factor ties pose $i$ to pose $j$, so it needs *exactly* the IMU spanning the gap between the
two images. A camera frame is an **instant** (unlike a LiDAR sweep), so the requirement is IMU
covering `[t_prev_image, t_image]` exactly:

```
IMU:    *----*----*----*----*----*----*----*
             ^                         ^
         at/before t_prev          at/after t_cur
images:      |                         |
           t_prev                    t_cur
```

Two rules make this exact ([`sync.hpp`](../include/glassvio/sync.hpp)): a frame is **held** until the
IMU stream has a sample at or past its stamp (release early and preintegration silently covers only
part of the interval — `dt` short, delta truncated, wrong in a plausible way); and consumed IMU is
**not eagerly dropped**, because the next interval needs the sample this one used as its right edge.
It syncs *tracks*, not images (23 KB vs 455 KB) — forced by the threading
([pipeline.md](pipeline.md)) — and a dropped group splices its IMU onto the next, so the
preintegration chain survives a queue overflow.

---

## 5. The per-frame loop — predict, solve, then map

Order matters: **solve first, then map.**

```cpp
guess = predictState(x, pre, gravity);        // the IMU's own prediction seeds GN (Module 2 §3)
if (pnpFromMap(obs, pnp)) guess.R/p = pnp;    // vision re-anchor, immune to any time gap
res = solveFrame(map.landmarks(), obs, x, P, pre, gravity, guess, bias_prior, calib);
x = res.state;
map.insert(features, T_world_cam);            // insert with the pose we JUST solved
```

Inside `solveFrame`, one Gauss-Newton loop stacks §2's factors. **Landmarks are held fixed** during
the solve — that is what keeps the state at 15 DoF (not $15+3L$) and lets the shared
`NormalEquationsN<15>` be reused untouched, exactly as glass-lio holds its map planes fixed while ICP
runs.

**Inserting after solving is the trick.** A new landmark is anchored to a pose the camera itself just
agreed with, not the IMU's guess — insert-before-solve would triangulate against a prediction and
then fit to it, a feedback loop. The sliding-window map ([`landmark_map.hpp`](../include/glassvio/landmark_map.hpp))
is glass-lio's `LocalMap` with points instead of planes: triangulated as tracks mature (gated on
parallax *angle* and cheirality), held fixed during a solve, pruned when they leave view or stop
fitting.

---

## 6. Warmup — surviving a transient without dying

The bootstrap ([Modules 6–7](06-epipolar-and-sfm.md)) is expensive and runs on the worker; while it
runs the queue drops frames and KLT ids churn, so the first tracking frame can share almost no ids
with the map. For `warmup_frames` after bootstrap, an under-constrained solve *or* a broken IMU chain
**coasts** on the prediction and keeps inserting, instead of declaring LOST — the state dead-reckons
(good to 0.16 m / 2 s, [Module 4](04-imu-preintegration.md)) while the map repopulates with live ids.
The budget replenishes on every good solve, so it also rides out a mid-run dip (a blank wall, a hard
turn). Only after the budget is spent is a starved solve a real loss.

---

## Lab — the thesis, isolated

`vio_check` is the tight-coupling claim on a plate: a run engineered so **vision alone would starve**,
where only the IMU factor sharing $\mathbf{H}$ keeps the trajectory alive.

```bash
colcon build --packages-select glassvio
./build/glassvio/vio_check          # reports final aligned error, ~0.036 m
```

1. **Read the 0.036 m.** Camera + IMU in one `NormalEquationsN<15>` tracks through the starvation.
2. **Break the coupling.** Zero the IMU information (`imu_information → 0`) so only reprojection rows
   enter $\mathbf{H}$, and re-run: the starved stretch now diverges — the pixels alone under-constrain
   the pose. That gap between 0.036 m and divergence *is* the fusion. Revert.

You now have a working tracker — *given a metric starting state*. The next two modules are how that
state is conjured from nothing, which is the hardest part of monocular VIO.

---

← [Module 4](04-imu-preintegration.md) · [Syllabus](README.md) · Next: **[Module 6 — Epipolar geometry & SfM](06-epipolar-and-sfm.md)** →
