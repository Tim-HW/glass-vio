# Module 8 — Sliding window & drift

> **Prerequisite:** [Modules 5, 7](05-tight-coupling.md). **After this you can:** localize an
> estimator's drift to one loop by substituting ground truth for one quantity at a time, explain why
> that loop needs landmarks *in the state* rather than a wider window of poses, and stage the bundle
> adjustment that is the real backend.

The system so far works: it bootstraps at metric scale and tracks. It also *drifts*, in a specific,
measurable way. This capstone is the honest state of the estimator, the story of how that drift was
traced — including the plausible explanation that turned out to be wrong — and the design of what
completes it. It doubles as the **cold-start handoff**: read it first if you are picking glassvio
back up.

Code target: [`vio_estimator.cpp`](../src/vio/vio_estimator.cpp),
[`landmark_map.cpp`](../src/vio/landmark_map.cpp). Measurement tool:
[`estimator_check.cpp`](../src/checks/estimator_check.cpp).

---

## 1. Where we are — measured

A working monocular VIO on the ROS node: bootstraps at true metric scale, tracks ~20–30 s on EuRoC
V1_01, maintains a sliding-window landmark map live, publishes odom + TF. The unit suites are green;
the offline thesis check `vio_check` is 0.036 m. On the deterministic harness:

| metric | per-frame tracker alone | **today's default** (§6) |
|---|---|---|
| bootstrap landmark depth | 1.62 m (true room scale) | 2.01 m (a later bootstrap window: §6) |
| velocity accuracy $\lVert\mathbf{v}\rVert_{\text{est}}/\lVert\mathbf{v}\rVert_{\text{gt}}$ | ~0.78 | **~1.00** |
| tracked | ~29 s | **129 s** — from the 15.4 s bootstrap to the end of the ground truth |
| median position drift (aligned to GT at bootstrap) | 0.65 m | **0.36 m** (over 129 s) |
| error first passed 1 m | — | **never** — at every tracker gate, with the true bias, with gravity re-estimated (§6) |
| V1_02_medium / V1_03_difficult, first >1 m | — | 74.2 s / 97.0 s (§6) |

§2–§5 are the story of the left column — how its drift was traced. §6 is the right column.

The thesis — camera reprojection + IMU in ONE `NormalEquationsN<15>` — is demonstrated end to end.

---

## 2. The drift — and the story we told about it

Two residuals remain:

1. **A ~20% scale shrink.** $\lVert\mathbf{v}\rVert_{\text{est}}/\lVert\mathbf{v}\rVert_{\text{gt}}\approx0.78$,
   and the trajectory itself agrees: over each half-second, the estimate moves ~0.8× as far as the
   truth. A *metric* error, not a tracking one — `rmse` stays 2–4 px throughout.
2. **Divergence in the fast final section** ($\lVert\mathbf{v}\rVert$ runs away at ~t = 42 s).

The first explanation — and for a while the documented one — was the **accelerometer bias**. The
bootstrap ([Module 7](07-metric-initialization.md)) solves the scale with $\mathbf{b}_a = 0$, and
EuRoC's is 0.55 m/s², large against the motion. Worse, $\mathbf{b}_a$ is only *weakly* observable per
frame ($\partial\Delta\mathbf{v}/\partial\mathbf{b}_a\sim\Delta t\sim0.05$): measured across four prior
settings (`sigma_accel_bias` 0.1 / 0.2 / 0.4 / 1.0), $b_{a,y}$ **never** converges to its true 0.548 — a
loose prior lets it wander, a tight one pins it at zero. A window of $K$ keyframes would constrain the
same $\mathbf{b}_a$ through $K-1$ IMU factors jointly. So: build a fixed-lag smoother.

Every sentence of that is true. It was not what was wrong.

---

## 3. Before you build it — glass-lio made this exact argument

It is also, nearly word for word, the argument glass-lio made about *its* tight-coupling divergence.
The LiDAR side reasoned that $\mathbf{x}_i$ is held infinitely certain, so a single solve can never
correct it, so the fix is a real sliding window. It *built the primitives* — gravity as a state, the
closed-form $\mathbf{x}_i$ marginalization, the Schur kernel, the state-transition Jacobian — pinned
each against finite differences, and re-ran. **It still diverged: 1.5 million metres**
([`testing.md` §12](https://github.com/Tim-HW/glass-lio/blob/main/doc/testing.md#12-case-study--making-tight-coupling-work-on-the-real-bag)).
The real cause, found by logging the *state* instead of the pose, was a gravity prior anchored to its
own estimate — a random walk with no restoring force. One line. Fixed, and with `lidar_sigma`
calibrated to the sensor, tight coupling reached parity with the trusted loose path: two numbers, zero
new architecture.

`★ Insight — the sophisticated wrong story ───────`
The glass-lio diagnosis was careful, and it named a **real** deficiency. It simply was not what
dominated the error. **A defect being real does not make it the one you are measuring.** So before
building a window, glassvio ran the one-day experiment that would have saved glass-lio the rewrite.
`──────────────────────────────────────────────────`

---

## 4. The falsification — ground truth, one quantity at a time

The deterministic harness can replace any single estimated quantity with ground truth. If the drift
vanishes, that quantity carries it; if it stays, that quantity is a bystander. Each row is one flag
and about a minute of CPU:

| `estimator_check` flag | replaced with ground truth | tracked | median pos err | position scale (15–35 s) |
|---|---|---|---|---|
| *(none)* | nothing | 29 s | 0.652 m | ≈ 0.79 |
| `--oracle-ba` | $\mathbf{b}_a$ at bootstrap | 74 s | 0.777 m † | ≈ 0.76 |
| `--parallax=3` | nothing — triangulation gate 1° → 3° | 38 s | 0.803 m | ≈ 0.75 |
| `--oracle-map` | the poses new landmarks are triangulated from | **133 s — whole sequence** | **0.024 m** | **≈ 0.97** |
| `--oracle-ba --oracle-map` | both | **130 s — whole sequence** | **0.023 m** | **≈ 0.98** |

† Over a run 2.5× longer; on the span both tracked (15–42 s) it is 0.62 m against the baseline's 0.66 m.
*Position scale* is the estimate's displacement over the truth's, summed over half-second steps.

What each row says:

- **$\mathbf{b}_a = 0$ at bootstrap is a real error, but a secondary one.** The alignment absorbs the
  bias as a tilted gravity — 4.04° at bootstrap, against 0.41° with the true bias
  ($\arctan(0.548/9.81) = 3.2°$) — and that tilt costs the fast section (29 s → 74 s). But it is **not**
  the scale error: from the oracle's metric bootstrap (speed ratio 1.06), position scale is back to 0.8
  within five seconds. And seeded at the true 0.548, tracking drags $b_{a,y}$ to −0.25. The bias is
  *absorbing* an error, not causing one.
- **Estimating $\mathbf{b}_a$ in the alignment cannot fix it either.** `estimate_accel_bias`
  ([Module 7](07-metric-initialization.md)) adds $\mathbf{b}_a$ as three more linear unknowns, gated on
  their marginal std. Over the 1 s SfM window that std is **4.6–6.8 m/s²** — without rotation,
  $\mathbf{b}_a$ is indistinguishable from a tilt of $\mathbf{g}$ — so the gate refuses, correctly.
- **Not a triangulation selection bias.** Each track is triangulated at the first frame its parallax
  clears the gate, which favours noise that makes points look *closer*. A 3° gate, far above the noise,
  changes nothing.
- **It is the map.** Triangulate new landmarks from ground-truth poses and the *same* estimator — same
  solve, same IMU factor, same biases, even the 4°-tilted $\mathbf{b}_a = 0$ bootstrap — holds metric
  scale for the entire sequence at 2 cm.

`★ Insight — oracle substitution ────────────────`
This is the cheapest discriminator an estimator has, and the reason `estimator_check` exists. The
fourth row cost a minute and is worth more than weeks of architecture: it says the solve, the IMU
factor, and the bootstrap are all fine *given a correct map* — and so it says where **not** to build.
`─────────────────────────────────────────────────`

---

## 5. The loop, and why a window of poses would not break it

The drift lives in a loop that runs every frame:

```
solve the pose against FIXED landmarks  →  triangulate new landmarks FROM that pose  →  repeat
```

Any error in a solved pose is written into the landmarks it triangulates, and those landmarks then
set the next pose as if they were ground truth. This is exactly the loop that makes monocular visual
odometry drift in scale. The IMU is supposed to be the anchor — the one sensor with a metre — but in
this design it only ever touches the *pose*: a pixel-precise map outvotes one 0.05 s IMU interval
every frame, and the landmarks themselves never feel the accelerometer at all.

So the first-planned Stage A — a window of keyframe *states* with the landmarks still held fixed —
would leave the loop intact. The landmarks would still come from solved poses and still be frozen
while the window solves. It would make $\mathbf{b}_a$ more observable, and $\mathbf{b}_a$ would
converge to whatever value best explains a shrinking map.

Breaking the loop means **landmarks join the state**: estimate $K$ keyframes and the landmarks they
observe jointly, so the IMU factors between keyframes pull the *whole structure* — poses and points —
to metric scale. That is bundle adjustment, and it is the same Gauss-Newton as
[Module 2](02-least-squares.md), just wider:

- $\delta\mathbf{x}$ becomes $15K + 3L$ ($K$ keyframes, $L$ landmarks);
- each reprojection row gains a $2\times3$ block for its landmark,
  $\partial\mathbf{r}/\partial\mathbf{P}_w = \mathbf{J}_\pi\mathbf{R}_{ci}\mathbf{R}^\top$ — the exact
  negative of its position block ([Module 3 §4](03-camera.md)): a landmark moving is the camera moving
  the other way;
- the landmark part of $\mathbf{H}$ is block-diagonal ($3\times3$ per landmark), so the **Schur
  complement** eliminates it cheaply, leaving a $15K$ system over the keyframes;
- each IMU factor touches **two** keyframe blocks — `imuJacobianI` and `imuJacobian`
  ([Module 4](04-imu-preintegration.md));
- the same $\mathbf{b}_a$ appears in $K-1$ IMU factors — §2's observability argument, still true, now
  with a map that is not shrinking underneath it.

---

## 6. The plan — smallest discriminating step first

### Step 0 — can weighting make the IMU hold the metre? (done: no)

glass-lio's lesson applies once more: before architecture, check the numbers. If the camera and IMU
were merely mis-weighted, a knob would close the gap — so every knob was swept with
`estimator_check`, against the `--oracle-map` row as target:

| knob (solved-pose map) | speed ratio | outcome |
|---|---|---|
| IMU weight ≈ 0 — pure vision | 0.53–0.61 | the map loop **contracts by itself**: scale 0.86 → 0.55 in 20 s |
| IMU weight 1 (default) | 0.78 | an equilibrium near 0.8 |
| IMU weight 2 | 0.87 | lost after 19 s |
| IMU weight 4–16 | — | diverges within 5–10 s — **even on a ground-truth map** |
| pixel sigma 2 / 4 / 8 / 16 px | 0.84 / 0.81 / 0.74 / 0.68 | best at 2 px, far short of 1.0 |
| IMU noise densities ×10 / ×40 | 0.61 / 0.53 | a weaker IMU — worse |

Three conclusions:

- **The contraction belongs to the vision loop, and the IMU is the only thing resisting it.** More
  IMU weight restores scale monotonically.
- **No weight reaches the target.** Past ×2 the estimator destabilizes, even on a ground-truth map
  and even with ×10 noise. That is the knob, not the factor: the extra weight compounds through the
  carried covariance ($\mathbf{P} = \mathbf{H}^{-1}$ feeds the next frame's $\Sigma_{\text{eff}}$), an
  over-confidence spiral.
- **So the fix is structural.** The IMU has to be able to move the landmarks, not only the pose —
  Stage A.

**A second mis-set constant, found on the way.** `gravity_world_` is frozen at the bootstrap's
estimate. From the $\mathbf{b}_a = 0$ bootstrap that estimate is tilted 4.04° — a permanent world-frame
error of $9.81\sin 4.04° = 0.69$ m/s² — and the tracker's $\mathbf{b}_a$, a *body*-frame state, chases
it as the vehicle turns. Even on a ground-truth map it never converges: $\lvert\mathbf{b}_a - \text{truth}\rvert$
oscillates at 0.6–0.8 m/s² for 130 s. From the true-$\mathbf{b}_a$ bootstrap (0.41° tilt) it holds
within 0.03–0.22. This is exactly §3's warning — a window cannot fix a wrong constant — so gravity's
direction (2 DoF) must become estimable after the bootstrap: a state in the Stage A window, or a
re-run of the inertial alignment over a longer, rotating trajectory (ORB-SLAM3 refines its IMU
initialization this way).

**The target stays measured, not guessed:** with solved poses, the position scale and
$\lVert\mathbf{v}\rVert$ ratio should approach 1.0, and `pos_err` should head toward the
`--oracle-map` row's 0.02 m.

### Stage A — windowed bundle adjustment: built, and it fixes the scale

Code: [`keyframe_window.cpp`](../src/vio/keyframe_window.cpp), hooked into the tracker in
[`vio_estimator.cpp`](../src/vio/vio_estimator.cpp). Before writing it, the two reference systems
were read for their actual choices:

| | VINS-Fusion | ORB-SLAM3 (`LocalInertialBA`) | glassvio |
|---|---|---|---|
| window | ~10 frames; keyframe on 10 px parallax | 10 keyframes, temporal chain | 10 keyframes, one every 4 frames (0.2 s) |
| landmarks | inverse depth, anchored in a frame | XYZ, Schur-eliminated | XYZ, Schur-eliminated (`schurSolve`) |
| oldest state | marginalized into a prior | the keyframe before it, fixed | the anchor, fixed (default) or gauge-only (`--anchor-gauge`) |
| solver | Ceres dogleg, 8 iterations | g2o LM, 10 iterations | LM, 8 iterations |

ORB-SLAM3 is the closer fit because its split *is* ours: per-frame tracking against fixed map points
(our `solveFrame`), plus a local BA on keyframes that refines the points — which is exactly what
breaks §5's loop. Every factor is glass_core's and was already pinned: the IMU residual with *both*
halves of its Jacobian (`imuJacobianI` finally earns its keep), the bias random walk, the anchor
prior, the reprojection factor. The two new pieces are pinned too — the landmark block of the
reprojection Jacobian (`test_reprojection`, 1e-9) and the Schur solve, against a dense solve (1e-14)
and glass_core's `schurMarginalize` (6e-16) in `test_keyframe_window`.

Measured on V1_01:

| `estimator_check` | tracked | median pos err | speed ratio |
|---|---|---|---|
| `--no-window` — byte-identical to the pre-Stage-A tracker | 29 s | 0.652 m | 0.775 |
| **window, anchor fixed (default)** | **80 s** | **0.456 m** | **1.019** |
| window, `--anchor-gauge` | 76 s | 0.677 m | 1.020 |
| window + `--oracle-ba` | 91 s | 0.397 m | 1.002 |
| window, IMU noise ×10 / ×50 | 51 / 31 s | 0.42 / 0.37 m | 0.95 / 0.92 |
| `--oracle-map` — the target | 133 s | 0.024 m | 0.995 |

**The scale is fixed**: position scale holds 0.97–1.07 through the run, because the IMU now moves
the map. What the measurements leave open:

- **The anchor.** Gauge-only is the one that can re-level against gravity, yet the fixed anchor wins
  on both counts — so it is the default. Why the re-levelling does not pay is not yet understood.
- **$\mathbf{b}_a$ still does not converge** — $\lvert\mathbf{b}_a-\text{truth}\rvert$ stays 0.4–0.6 m/s²
  even from the true bias. Something the window cannot absorb is still mis-set.
- **Inflating the IMU noise** toward VINS-Fusion's values makes everything worse: the datasheet
  densities stay.
- **The deaths are the tracker's, not the window's.** Before each loss the window moves the state
  by only a few centimetres. In the fast sections (84–92 s, 103 s) the map thins below 20, and a
  frame with a handful of features lets a degenerate PnP guess through the warmup coast; every
  solved-pose run also meets one tracker frame near 89.6 s with over 1 600 px of reprojection
  error. Pending tracks pile up there too (150–440), which first looked like the cause. It is not —
  see below.
- **Cost**: ~100 ms per keyframe solve, most of it accumulating the reduced system over landmark
  pairs. Fine offline; the live node needs it trimmed (symmetry, fewer iterations) or threaded.

**Triangulating from the window — built, measured, and off.** The pile-up suggested that pending
tracks never mature because they are triangulated from the *tracker's* stored poses, which the
window later corrects. So `KeyframeWindow::triangulate` does what VINS-Fusion and ORB-SLAM3 do: every
track the newest keyframe sees, triangulated linearly over all its keyframe observations at the
*optimized* poses, and kept only through the parallax, cheirality and 3 px reprojection gates
(pinned in `test_keyframe_window`). Then the map was instrumented to say *why* each pending track
fails to triangulate (the `tri_*` columns), and run against ground-truth poses:

| 80–90 s, summed over frames | tried | matured | too little parallax | behind a camera / too far |
|---|---|---|---|---|
| Stage A (solved poses) | 11 546 | 2 329 | 5 701 | 3 516 |
| `--oracle-map` (ground-truth poses) | 12 921 | 2 480 | 5 913 | 4 528 |

The hypothesis was wrong. With *perfect* poses the map matures almost as many tracks and fails as
many on parallax and depth: the pile-up is what fast, rotating motion looks like, not a symptom.
**Supply was never short — the landmarks' quality is what differs.** And the change hurt by the
honest score: with it on, the error first passed 1 m at 40 s instead of 89 s. So it is off by default
(`--window-tri` to compare), kept and pinned for when the poses it trusts beat the map's.

`★ Insight — survival time lies ─────────────────`
That run was first reported as an *improvement* — "tracked 92.5 s against 80 s" — because the
estimator only declares a loss when it runs out of features. It had diverged at ~92 s and flew on at
ten times the true speed, 13 m off, until 105 s. The harness now prints the first time the error
passed 1 m. A plausible number is still a number the estimator chose to report.
`─────────────────────────────────────────────────`

**Gravity's direction as a window state — built, measured, and off.** Step 0's second finding was
a mis-set constant: `gravity_world_` frozen at the bootstrap's 4° tilt, a permanent 0.69 m/s² error.
So the window gained two unknowns — gravity's tilt, $\mathbf{g}(\boldsymbol\theta) =
\mathrm{Exp}([\theta_x,\theta_y,0])\,\mathbf{g}_\text{ref}$, its magnitude fixed — observable against
the fixed anchor, and entering every IMU factor through glass_core's `imuGravityJacobian` times the
$3\times2$ tilt map (pinned by finite differences in `test_keyframe_window`); the tracker adopts the
refined gravity after each solve.

| `estimator_check` | first >1 m | median err | gravity error while tracking | $\lvert\mathbf{b}_a-\text{truth}\rvert$ |
|---|---|---|---|---|
| gravity frozen (default) | **89.0 s** | 0.46 m | 4.04° | 0.58 → 0.43 m/s² |
| `--gravity` (prior 1°, none, or 0.3° alike) | 87.9 s | 0.46 m | **1.8–2.7°** | 0.55 → 0.47 m/s² |

It works as a gravity estimate — the tilt halves while tracking — and buys nothing where it
counts: the same position error, the same first >1 m, and the accel bias still unconverged. (Its
last-frame gravity error first read 15° — that was the divergence at 90 s, not the estimator;
end-of-run numbers lie the way survival time does.) So it is off by default.

ORB-SLAM3 does this differently in kind, not degree: at 2, 5 and 15 s it runs a **global** inertial
optimization over *all* keyframes — gravity `Rwg`, scale and both biases jointly, where the
trajectory's rotation finally separates $\mathbf{b}_a$ from tilt — and then `ApplyScaledRotation`
re-levels the *whole map*, followed by scale-and-gravity refinements every 10 s until 75 s. A window
tilting gravity inside itself is not that.

**Refusing the tracker's blow-ups — built, measured, on.** The 1 600 px frame at 89.6 s turned out
to be downstream. Frame by frame, the break starts at 87.60 s with one *accepted* solve that is
almost pure garbage — 1 inlier of 94 — after which the map, pruned against that pose, falls from
143 landmarks to 60 in the same frame. So the tracker now refuses a solve when fewer than half its
observations agree with it (`VisualParams::min_inlier_fraction`: ORB-SLAM3's inlier check after pose
optimization, as a fraction; healthy frames sit at a median of 0.99 and a 5th percentile of 0.85).

That alone was not enough, and the reason is the most instructive bug in this module. A refused
frame *coasts*, which was meant to dead-reckon on the IMU. But the seed it adopted had its R and p
overwritten by PnP, and on a frame whose solve was just refused, PnP rotates the pose ~1.7° a frame
(normal p99: 1.0°). Coast after coast, the attitude walked from 6° to 14° off in 0.3 s; the keyframe
window then met a 13.5° gyro disagreement (normal: 0.03°) and resolved it by moving the pose 0.4 m
and writing a 6× velocity. The camera could not object — a pose that fits every pixel can carry any
velocity. Now a coast adopts the IMU's own prediction, and PnP stays the Gauss-Newton seed, which is
its job (`EstimatorParams::coast_on_pnp`).

Measured before the fix that follows (`--no-forget --window-outlier-px=0` reproduces it):

| `estimator_check` | first >1 m | median err |
|---|---|---|
| no gate (`--inlier-fraction=0`, as Stage A left it) | 89.0 s | 0.456 m |
| gate 0.5, coasting on PnP (`--coast-on-pnp`) | never — but lost at 90 s | 0.449 m |
| gate 0.5, coasting on the IMU | 130.5 s — to the end of the ground truth | 0.540 m over 132 s |
| gate 0.3 / 0.7, coasting on the IMU | 88.5 s / 88.5 s | 0.451 / 0.474 m |

Read the last row. Both mechanisms were fixed for good — the map no longer collapses, the attitude
no longer walks — but at 0.3 and 0.7 the run still broke near 88.5 s, so 130 s was one good outcome
of a fragile section. What broke there was on the vision side: at the first window solve after the
coast, vision's share of the starting cost was 20–150× normal, and 591 of 634 views sat more than
20 px off. (Not inserting refused frames at all was tried; it starves the map and fails at 40 s.)

**Forgetting what the map dropped — built, measured, on.** Split by keyframe, that spike was *not* in
the newest keyframe (2.7k, normal) but spread across the older ones (75k–147k each). The chain:

1. a refused frame coasts and is inserted into the map at the IMU's guess;
2. at that guess, good landmarks reproject more than 8 px off, so the map drops them as outliers;
3. their KLT ids live on, and a few frames later re-triangulate from the coast poses at *new*
   positions — 70 such tracks in the one frame at 87.75 s (median: 1);
4. the newest keyframe agrees with the new positions — it was taken at those poses — while every
   older keyframe still holds its view of the *old* landmark, under the same id.

The window was fitting ten keyframes to points half of them never saw. ORB-SLAM3 closes this with
`EraseMapPointMatch`: an outlier's observations leave the keyframes. Now so do ours
(`KeyframeWindow::forget`, `WindowParams::forget_outliers`). The window also drops, before each solve,
every view more than 8 px off at its starting state (`WindowParams::outlier_px` — ORB-SLAM3's chi²
outlier edges, VINS-Fusion's 3 px cut); a normal solve loses a median of one view. Never re-triangulating
a dropped track instead (`--no-recycle`) starves the map: lost at 78 s at every gate.

| first >1 m · median err | gate 0 | gate 0.3 | gate 0.5 | gate 0.7 |
|---|---|---|---|---|
| neither (`--no-forget --window-outlier-px=0`) | 88.4 s · 0.455 | 88.5 s · 0.451 | 130.5 s · 0.540 | 88.5 s · 0.474 |
| 8 px drop only (`--no-forget`) | never · 0.379 | never · 0.329 | never · 0.327 | never · 0.347 |
| forget only (`--window-outlier-px=0`) | never · 0.380 | never · 0.375 | never · 0.414 | never · 0.386 |
| **both (default)** | never · 0.403 | never · 0.338 | **never · 0.330** | never · 0.359 |

Either one alone holds, and they do it differently. The drop alone survives by going blind: at
87.85 s it strips 526 of 552 views and the IMU carries the window. Forget removes the *cause* — the
stale views — and the window keeps ~100 honest ones through the coast. Both are on: forget because
it is the fix, the drop because it is the standard guard against the next source of bad views. The
drop's threshold is not delicate either — 5, 10, 20 and 40 px all hold at every gate.

**The tracker was freezing its own points — found, fixed.** Two runs that should have been *better*
broke the section again: the true accelerometer bias at bootstrap (`--oracle-ba`, 0.41° of tilt
instead of 4.04°) passed 1 m at 90.9 s, and `--gravity` at 90.2 s. So gravity was not what the
section needed, and the global refinement above was not built. Both crossed 87.6–91.3 s mostly
blind — frame after frame with no tracker solve — and lived or died on how well the IMU coasted.
The images there show a wall of plain mattresses and curtains: FAST at threshold 20 finds 23–42
corners where it finds 70–90 elsewhere (at 7: 160–200). Low texture, not blur.

And low texture tripped a bug. `FeatureTracker::track` re-seeded whenever the previous frame had
fewer than `min_features` (150) tracks — but `detectInto` *appends*, so the survivors went back out
at the previous frame's pixels, unflowed, under their old ids. Replaying the tracker over V1_01:
72 frames did it, 22 of them in 87.55–88.96 s alone, each handing the estimator ~140 points a median
6.5 px (p90 13.4 px) from where KLT says they went. The tracker solve refused those frames —
correctly — so the blind stretch was largely self-inflicted. Now only a cold start re-seeds; below
the floor the survivors are flowed and topped up like on any other frame (`test_feature_tracker`
pins it: the old code reports 0 px of motion for a 3 px shift).

| `estimator_check` | before the fix: first >1 m | after: first >1 m · median err |
|---|---|---|
| V1_01, gates 0 / 0.3 / 0.5 / 0.7 | never | never · 0.36–0.37 m |
| V1_01 `--oracle-ba` | 90.9 s | never · 0.34 m |
| V1_01 `--gravity` | 90.2 s | never · 0.38 m |
| V1_02_medium | bootstrap 62.2 s, then 70.5 s | bootstrap 32.9 s, then **74.2 s** · 0.47 m |
| V1_03_difficult | bootstrap 33.6 s, then 37.0 s (lost at 39 s) | bootstrap 44.4 s, then **97.0 s** · 0.54 m |

The tracker feeds the bootstrap too, so V1_01 now bootstraps at 15.4 s instead of 12.9 s (2.0 m
median depth, 3.1° tilt) and its medians run over 129 s instead of 132. V1_02 and V1_03 are the
first numbers from a sequence the estimator was not developed on — and on V1_02 the harness's 60 s
minimum is out of reach when the bootstrap is at 32.9 s of an 85 s flight.

**Why the bootstrap waits — measured, one bug fixed, the main cause open.** V1_02 bootstraps at
32.9 s and V1_03 at 44.4 s, while the vehicle flies at 0.3–1.7 m/s from the first seconds. With
`estimator_check --init-log`, every attempt prints the stage that refused it and that stage's
numbers, and — against ground truth, scale-free — how wrong the reconstruction was:

| attempts refused by (before the PnP fix below) | V1_01 | V1_02 | V1_03 |
|---|---|---|---|
| [2] SfM: no base pair with ≥ 100 landmarks | 7 | 47 | 119 |
| [3] gyro bias: too few pairs | 9 | 3 | 15 |
| [4] \|g\| implausible | 2 | 18 | 8 |
| [4] scale ≤ 0, or σ_s/s above 0.06 | 19 | 39 | 11 |

- **SfM is starved, not blind.** Most refusals tried a base pair and triangulated 40–99 landmarks
  against `min_landmarks` 100; the rest shared fewer than 60 tracks with the base frame. More
  corners (`--fast=12`) let SfM through far more often — and V1_02 then bootstraps at *81 s*,
  because the windows it lets through die in the alignment. The SfM gate was hiding the next one.
- **A flipped PnP pose — fixed.** The \|g\| refusals carried a pose ~177° off: 70 of 70 came from
  the PnP that propagates the ruler, none from the base pair. `solvePnPRansac` never had its inlier
  count checked, and a pose rotated half a turn with the landmarks *behind* the camera reprojects
  them just as well. It now needs `min_pnp_points` inliers, all in front. Flips: 70 → 1; the
  refusals moved to the scale gate, and no bootstrap time changed.
- **The scale collapses — open.** Windows refused on scale have decent geometry (rotation 0.6–0.9°,
  translation direction 4–8° off) yet solve s ≈ 0.03× the truth; accepted ones land at 0.87–0.96×.
  A longer window (`--sfm-window=30/40`) is not the fix: it moves V1_03's bootstrap to 58 s with a
  1.8× speed error that the 0.06 gate *accepted*.

**How the state of the art does it.** VINS-Fusion, ORB-SLAM3 and OpenVINS all reach 2–7 cm RMS
ATE on these rooms (ORB-SLAM3's Table II, monocular-inertial: V1_01/02/03 0.049/0.015/0.037 m;
VINS-Mono 0.047/0.066/0.180). Scored the same way — SE(3)-aligned RMS ATE — glassvio is 0.095 m on
V1_01 from its 15 s bootstrap, 0.32 m on V1_02 and 0.62 m on V1_03. The design difference that
matters here: **none of them demands a right scale from one 1 s linear solve.** VINS-Fusion keeps
10 parallax-spaced keyframes, asks the SfM for 20 shared tracks and then *bundle-adjusts* the
window, checks only s > 0, and refines gravity with |g| fixed. ORB-SLAM3 builds 2 s of visual map,
runs a nonlinear inertial optimization and re-estimates scale at 5 s, 15 s and every 10 s to 75 s.
OpenVINS constrains |g| in its linear solve and refines by MLE.

**Gravity with |g| fixed — built, measured, off.** VINS-Fusion's RefineGravity: once the free |g| passes
the oracle check, fix it at 9.81 and re-solve with gravity as two tangent coordinates
(`InitializerParams::refine_gravity`, `--refine-gravity`). The bootstrap it accepts can be
metrically better — V1_02's starting speed ratio 0.81 → 1.01 — but it does *not* cure the collapse
(refused windows still solve s ≈ 0.02–0.06× the truth), and downstream it hurt two sequences of
three: V1_01's median error 0.36 → 0.53 m, V1_03 failing at 47.5 s. Off until the reconstruction
below is fixed; then it is worth measuring again.

| FAST 20 | V1_01 | V1_02 | V1_03 |
|---|---|---|---|
| **free \|g\| (default): bootstrap · first >1 m · ATE** | 15.4 s · never · 0.095 m | 32.9 s · 74.2 s · 0.316 m | 44.4 s · 97.0 s · 0.618 m |
| \|g\| fixed (`--refine-gravity`) | 12.4 s · never · 0.119 m | 32.9 s · never · 0.221 m | 44.4 s · 47.5 s · 0.837 m |

**The collapse is the reconstruction's positions — measured by substitution.** `--oracle-sfm`
replaces the reconstruction's rotations and/or positions with ground truth, in its own ruler, just
before stage [4]:

| `--oracle-sfm` | V1_02 bootstrap · s / s_true | V1_03 bootstrap · s / s_true |
|---|---|---|
| none | 32.9 s · median 0.02 over 58 windows | 44.4 s · median 0.02 over 20 windows |
| `rot` | 32.9 s · median 0.02 | 44.4 s · median 0.03 |
| `pos` | **7.4 s · 1.10**, the first window through | **13.4 s · 0.91**, the first window through |

True rotations change nothing; true positions bootstrap on the first window that reaches the
alignment. The alignment is sound — it is being fed translations (4–8° off in direction, propagated
frame to frame by PnP) that no metric trajectory fits.

**So what is next** is the step VINS-Fusion runs and glassvio skips: bundle-adjust the reconstruction
window — poses and landmarks, reprojection only — before stage [4].

### Stage B — marginalization (only if Stage A's dropped-oldest loss matters)

Schur-complement the oldest keyframe and its landmarks into a **prior** on the remaining window,
instead of dropping it. This is where VINS-Mono spends most of its backend complexity, and consistency
matters: a wrong marginalization injects spurious information. The *kernel* is already built and
exact — `schurMarginalize` in [`marginalization.hpp`](../glass_core/include/glass_core/marginalization.hpp),
pinned to 2.1e-17 against both the full solve (must match) and the naive hold-fixed solve (must
differ). Stage B is the bookkeeping around it: what to marginalize, keeping the prior consistent as
the window slides, not double-counting. Do **not** start here.

### Stage C — feature supply for fast motion (demoted)

The tracker's top-up is gated below `min_features` to protect the bootstrap SfM
([Module 3](03-camera.md)), and it was a suspect for the fast-section divergence. It is now a weak
one: the `--oracle-map` runs cross that section with the *same* tracker and the *same* supply, and do
not diverge. Phase-aware top-up is still a sensible cleanup, but it is not the fix.

---

## 7. Landmines (each cost real time on this project)

- **Online ≠ offline, over and over.** Every node-only bug was "the end of the buffer is not NOW":
  bias pairs (2912 frames vs 50), dropped IMU chains, stale SfM windows, churned KLT ids.
  `estimator_check` removes all of it. When the node fails but the harness does not, it is plumbing,
  not logic.
- **A real defect is not necessarily the dominant one.** §2's argument was correct and beside the
  point. Substitute ground truth before you build.
- **Survival time lies.** An estimator declares a loss when it runs out of features, not when it is
  wrong — a run can diverge and "track" on for seconds. Score by the first time the error passed
  1 m; `estimator_check` prints it.
- **PnP is a seed, not a state.** Adopted unfused on a frame whose solve was just refused, it walked
  the attitude 8° in 0.3 s. And a pose that fits every pixel can still carry a garbage velocity: a
  camera does not observe velocity, so a reprojection check can never vouch for one.
- **A track id is not a landmark.** KLT ids outlive the map's verdict on them: an id dropped as an
  outlier and re-triangulated is a new point under an old name, and whatever still holds the old
  views — here the keyframe window — is fitting a point that no longer exists. Drop the point, drop
  its views.
- **Suspect your own front end first.** A "blind stretch" blamed on the scene was 22 frames of the
  tracker handing back the previous frame's pixels. Replay the front end offline on the real images
  before blaming blur, texture or the IMU — and a run that gets *worse* with a truer input (here,
  the true bias) is pointing at something else entirely.
- **`imu_prior_weight` > 1 is not a test of the IMU.** It compounds through the carried covariance
  and spirals into over-confidence; it diverged even on a ground-truth map. Calibrate the noise
  densities, never the weight.
- **The oracle hooks are test-only.** `EstimatorParams::oracle_insert_pose` and
  `InitializerParams::accel_bias` exist for `estimator_check`. If either is ever set from a launch
  file, the estimator is cheating.
- **A count is not a duration.** Every "how many frames" parameter means different seconds at a
  different rate. Derive from the measured rate, never hardcode.
- **`-UNDEBUG` in CMake is load-bearing.** Release defines `NDEBUG`, which deletes every `assert()` —
  the tests would pass while checking nothing. Already set; do not remove.
- **Eigen `auto` + `.inverse().translation()` dangles.** Name the type (`-> Eigen::Vector3d`).
- **Ruler units vs metres.** Only the SfM translation scales by $s$; the extrinsic is already metric.
- **The scale gate checks CONDITIONING, not accuracy.** 0.06 is tuned for EuRoC V1_01 — re-derive
  from the uncertainty-vs-accuracy curve on a new sequence.

---

## Lab — the measurement discipline itself

`estimator_check` drives the **real** `VioEstimator` deterministically (no ROS, no threads, no dropped
frames) and writes a 63-column per-frame CSV. Past the pose, velocity and bias columns it records
the latest window solve (landmarks, cost before and after and its split into prior / IMU / vision,
vision's starting cost per keyframe, views it dropped as outliers and views more than 20 px off,
iterations, how far it moved the newest keyframe, landmarks it triangulated, and its worst IMU
factor's interval and rotation / velocity / position residuals), dropped tracks re-triangulated,
why the map's pending tracks did not
mature that frame, the tracker's fit (median pixel error and inliers), its attitude and gravity error
against the truth, and how far PnP moved the seed. Error is aligned to ground truth at the bootstrap
instant — a raw $\lVert\mathbf{p}_{\text{est}}-\mathbf{p}_{\text{gt}}\rVert$ is meaningless because
the estimator defines its own gravity-aligned world.

```bash
colcon build --packages-select glassvio
./build/glassvio/estimator_check                    # the baseline
./build/glassvio/estimator_check --oracle-ba        # true b_a at bootstrap
./build/glassvio/estimator_check --oracle-map       # landmarks from ground-truth poses
./build/glassvio/estimator_check --parallax=3       # a stricter triangulation gate
```

Each prints the gravity tilt at bootstrap, the median position error, and the median
$\lVert\mathbf{v}\rVert/\lVert\mathbf{v}_{\text{gt}}\rVert$ while tracking.

1. **Reproduce §4's table.** Before reading further, predict each row. Which ones surprised you?
2. **Plot `bay` against `gbay` from the `--oracle-ba` run.** Seeded at the truth, dragged away from it
   within seconds. That is what a bias *absorbing* an error looks like — and why a bias that never
   converges is weak evidence that the bias is the problem.
3. **Take on Step 0.** Change the camera/IMU weighting and try to move the solved-pose run toward the
   `--oracle-map` row. You have a measured target, so every attempt is a yes or a no.
4. **The habit to keep:** measure here *before* every change and re-measure *after*. Every
   self-deception this project caught — a bias-prior loosening that made drift worse, a
   condition-number gate that did not discriminate, a feature-supply hypothesis, and finally the
   accel-bias story itself — was caught by this CSV, not by reasoning. That habit is the last thing
   the course has to teach.

---

## Quick reference

```bash
./build/glassvio/estimator_check [--oracle-ba|--no-est-ba] [--oracle-map] [--parallax=DEG] \
                                 [--px-sigma=PX] [--imu-weight=W] [--imu-noise-x=K] \
                                 [--no-window] [--anchor-gauge] [--kf-every=N] [--window=K] \
                                 [--window-tri] [--no-map-tri] [--gravity] [--gravity-sigma=DEG] \
                                 [--inlier-fraction=F] [--no-refused-insert] [--coast-on-pnp] \
                                 [--no-forget] [--window-outlier-px=PX] [--no-recycle] \
                                 [--init-log] [--fast=N] [--sfm-window=N] \
                                 [--refine-gravity] [--oracle-sfm=rot|pos|both]
./build/glassvio/vio_check           # the offline tight-coupling thesis check
./run_euroc.sh                       # the node, live, with RViz
colcon test --packages-select glassvio   # the six suites: glass_core's four + reprojection + tracker
```

---

← [Module 7](07-metric-initialization.md) · [Syllabus](README.md) · [Pipeline reference](pipeline.md)
