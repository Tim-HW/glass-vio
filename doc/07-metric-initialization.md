# Module 7 — Metric initialization: recovering the metre

> **Prerequisite:** [Modules 2, 4, 6](02-least-squares.md). **After this you can:** estimate the gyro
> bias from vision, solve the linear visual-inertial alignment for scale/gravity/velocity, and gate
> on whether the scale is actually observable.

[Module 6](06-epipolar-and-sfm.md) produced shape in an invented ruler. This module hands that
reconstruction the accelerometer's metre — the one thing with real units — plus gravity, velocity,
and the gyro bias, yielding a **metric** `NavState` the tracker ([Module 5](05-tight-coupling.md))
can start from. It is a **gate**: nothing tracks until it completes, and completing it *badly* is
worse than waiting, so half the module is about knowing when to refuse.

Code: [`vio_initializer.cpp`](../src/vio/vio_initializer.cpp),
[`sfm_window.hpp`](../include/glassvio/sfm_window.hpp). Checks:
[`gyro_bias_check.cpp`](../src/checks/gyro_bias_check.cpp),
[`vi_align_check.cpp`](../src/checks/vi_align_check.cpp).

---

## 1. Why a bootstrap exists, and its ordering

No single sensor supplies a metric start:

> the camera sees landmarks but has no units; the accelerometer has units but has never seen a
> landmark.

So the sub-stages are ordered by what each needs from the last:

| sub-stage | needs | produces | check |
|---|---|---|---|
| **reconstruct** ([Module 6](06-epipolar-and-sfm.md)) | nothing | shape, invented ruler ($\lVert\mathbf{t}\rVert=1$) | `sfm_check` |
| **gyro bias** | rotations only (scale-free) | $\mathbf{b}_g$ | `gyro_bias_check` |
| **align** | reconstruction + IMU | the metre: scale, gravity, velocity | `vi_align_check` |

There is **no static-window `ImuInit`** here. glass-lio's LiDAR bootstrap assumes the sensor started
at rest — an assumption a flying MAV never honours (on KITTI it produced a "bias" that was really the
car's yaw rate). glassvio reads the gyro bias off vision's rotations and gravity out of the linear
solve; no static window needed.

---

## 2. Gyro bias — from rotations, before the metre exists

The essential matrix's relative rotation ([Module 6 §2](06-epipolar-and-sfm.md)) is **scale-free**, so
the gyro bias is observable *before* any scale is. Between consecutive frames, vision says the camera
rotated by $\mathbf{R}^{\text{cam}}_{k,k+1}$; preintegration says $\Delta\mathbf{R}_{k,k+1}(\mathbf{b}_g)$.
The bias is the value that reconciles them, over the whole stream:

$$
\mathbf{b}_g = \arg\min_{\mathbf{b}_g}\sum_k \big\lVert \mathrm{Log}\!\big(\Delta\mathbf{R}_k(\mathbf{b}_g)^\top\,\mathbf{R}^{\text{cam}}_k\big)\big\rVert^2,
$$

linearized via the bias Jacobian $\partial\Delta\mathbf{R}/\partial\mathbf{b}_g$ from
[Module 4](04-imu-preintegration.md) and solved in `glass_core`'s own `NormalEquationsN<3>` — the same
accumulator, three columns wide. It gets the **whole stream**, not just the SfM window: the bias is a
constant of the sensor, so every frame pair is evidence and averaging pulls the estimate below any one
pair's noise floor. On EuRoC it recovers the real 0.076 rad/s bias to ~2%.

---

## 3. The visual-inertial alignment — the metre in one linear solve

Here is the crux. SfM says "the camera moved 1.0 baselines"; preintegration says "it moved 0.47
metres." The accelerometer is the only thing with units, so it — and nothing else — sets what a
baseline is worth. Write the IMU's kinematic relations between consecutive SfM poses, taking the
**per-frame velocity, gravity, and scale as unknowns**, and every equation is *linear* in those
unknowns:

$$
\begin{aligned}
\text{from } \Delta\mathbf{p}:\quad & s\,(\mathbf{p}^c_{k+1}-\mathbf{p}^c_k) - \mathbf{v}_k\,\Delta t - \tfrac12\mathbf{g}\,\Delta t^2 = \mathbf{R}_{b_k}\,\widehat{\Delta\mathbf{p}} + (\mathbf{R}^c_{k+1}-\mathbf{R}^c_k)\,\mathbf{t}_{ci} \\
\text{from } \Delta\mathbf{v}:\quad & -\mathbf{v}_k + \mathbf{v}_{k+1} - \mathbf{g}\,\Delta t = \mathbf{R}_{b_k}\,\widehat{\Delta\mathbf{v}}
\end{aligned}
$$

Stack these over all consecutive pairs into $\mathbf{A}\mathbf{x} = \mathbf{b}$ with unknown vector

$$
\mathbf{x} = [\,\mathbf{v}_0,\ \mathbf{v}_1,\ \dots,\ \mathbf{v}_n,\ \mathbf{g},\ s\,]^\top
\qquad (3(n{+}1)+3+1 \text{ unknowns}),
$$

and solve **one least-squares system**. The scale $s$ that comes out is the metres-per-baseline
converting the whole reconstruction to metric.

**The free oracle:** $\lVert\mathbf{g}\rVert$ is *not* constrained in the solve — that its magnitude
lands on 9.81 is a fact nothing told the system, and the only self-check available without ground
truth. `vi_align_check` watches exactly that.

---

## 4. The scale-observability gate — the subtle one

Solving $\mathbf{A}\mathbf{x}=\mathbf{b}$ *always* returns a number for $s$. Whether that number
**means** anything is a separate question, and getting it wrong is the difference between a 1.6 m room
and a 3 cm one.

Scale reaches the estimator only through the accelerometer's **non-gravity** part (§3's $\Delta\mathbf{v}$
term). A window without excitation — a near-constant-velocity glide — makes $\mathbf{A}$ nearly
rank-deficient *in the scale direction*: $s$ comes out positive but metrically meaningless. The gate
is the **marginal relative uncertainty** of the scale:

$$
\boxed{\;\frac{\sigma_s}{|s|} = \frac{\sqrt{\sigma^2\,(\mathbf{A}^\top\mathbf{A})^{-1}_{ss}}}{|s|}\;} \;<\; \texttt{max\_scale\_uncertainty}.
$$

`★ Insight — why *marginal*, and why not the condition number ─`
$(\mathbf{A}^\top\mathbf{A})^{-1}_{ss}$ is the **marginal** variance of $s$ — it already accounts for
$s$ trading off against $\mathbf{v}$ and $\mathbf{g}$, which is the exact ridge a badly-excited window
hides in (a raw diagonal would miss it). Dividing by $|s|$ makes it dimensionless, so the threshold
needs no per-dataset units. And the **raw condition number of $\mathbf{A}$ does not work**: its columns
span $\Delta t$, $\Delta t^2$, and ruler units, so its conditioning measures *column scaling*, not
observability — measured, a good window and a bad one both gave ~60.
`──────────────────────────────────────────────`

On EuRoC V1_01, $\sigma_s/|s|$ of 0.02 gave a 2%-accurate scale, 0.10 gave 32% off, 0.2+ degenerate.
The gate at **0.06** makes the estimator wait for a genuinely well-excited window (1.62 m landmark
depth, velocity to ~80% of truth); 0.15 let a 3×-biased window through. It checks **conditioning, not
accuracy** — a well-conditioned window can still be biased (that residual 20% is
[Module 8](08-sliding-window.md)'s problem) — so re-derive the threshold from the
uncertainty-vs-accuracy curve on a new sequence.

---

## 5. Defining the world, and handing off

On success the **world frame is defined** (it is not given — a monocular VIO cannot know where the
origin is or which way it faces): origin at the first body pose, +Z along **measured gravity**, yaw
arbitrary (gravity constrains two of three rotational DoF; the third is unobservable, left at zero via
`FromTwoVectors`, exactly as glass-lio's IMU init). Landmarks become metric ($s\cdot\text{ruler}$, in
`world`); the state is seeded with the aligned pose, velocity, and gyro bias. The **accel bias stays
0** — nothing estimates it here, and its initial covariance is left loose to admit that
([Module 8](08-sliding-window.md)).

**Two spans, not one** — conflating them is a bug made three times in this project. The gyro bias
wants MANY frames (a constant — more is better); the SfM window wants FEW (its landmarks die as the
camera moves). So `bootstrap_frames` (collect) and `sfm_window_frames` (reconstruct) are separate
parameters. The offline harness never saw the coupling because it loads whole bags; online, "the end
of the stream" is *now*.

---

## Lab — the two sub-stages, each against its oracle

```bash
colcon build --packages-select glassvio
./build/glassvio/gyro_bias_check     # vision-rotation bias vs the dataset's stated b_g
./build/glassvio/vi_align_check      # the alignment; watch |g| land on 9.81
```

1. **`gyro_bias_check`** recovers ~0.076 rad/s to ~2% from vision alone — no IMU integration of
   position, just rotations (§2). Confirm the scale-free claim holds.
2. **`vi_align_check`** — read the recovered $\lVert\mathbf{g}\rVert$. It lands near 9.81 though nothing
   constrained it: the alignment is self-consistent (§3's free oracle). Note the reported
   $\sigma_s/|s|$ — this is the gate deciding whether to trust the scale.
3. **Break the gate.** Loosen `max_scale_uncertainty` from 0.06 toward 0.15 and re-run the full
   `estimator_check`: it will bootstrap on a poorly-excited window and reconstruct the room at the
   wrong size — *conditioning was the difference*, not any code error. Revert.

You now bootstrap at metric scale. The final module is where — and why — the estimate still drifts.

---

← [Module 6](06-epipolar-and-sfm.md) · [Syllabus](README.md) · Next: **[Module 8 — Sliding window & drift](08-sliding-window.md)** →
