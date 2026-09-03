# Module 8 — Sliding window & drift

> **Prerequisite:** [Modules 5, 7](05-tight-coupling.md). **After this you can:** diagnose why a
> single-frame solve leaves a residual scale bias, explain why a window of keyframes fixes it, and
> stage the sliding-window bundle adjustment that is the real backend.

The system so far works: it bootstraps at metric scale and tracks. It also *drifts*, in a specific,
measurable way — and the fix is the one piece not yet built. This capstone is both the honest state
of the estimator and the design of what completes it. It doubles as the **cold-start handoff**: read
it first if you are picking glassvio back up.

Code target: a windowed solver over [`vio_estimator.cpp`](../src/vio/vio_estimator.cpp).
Measurement tool: [`estimator_check.cpp`](../src/checks/estimator_check.cpp).

---

## 1. Where we are — measured

A working monocular VIO on the ROS node: bootstraps at true metric scale, tracks ~20–30 s on EuRoC
V1_01, maintains a sliding-window landmark map live, publishes odom + TF. 71 unit tests green; the
offline thesis check `vio_check` is 0.036 m. On the deterministic harness:

| metric | value |
|---|---|
| bootstrap landmark depth | 1.62 m (true room scale) |
| velocity accuracy $\lVert\mathbf{v}\rVert_{\text{est}}/\lVert\mathbf{v}\rVert_{\text{gt}}$ | ~0.80 |
| tracked before losing the scene | ~29 s |
| median position drift (aligned to GT at bootstrap) | 0.65 m |

The thesis — camera reprojection + IMU in ONE `NormalEquationsN<15>` — is demonstrated end to end.

---

## 2. The two remaining residuals, and their measured cause

Both trace to the same root, and both are the sliding window's job.

1. **~20% residual scale under-estimate** ($\lVert\mathbf{v}\rVert_{\text{est}}\approx0.80$). The
   bootstrap scale is computed by [Module 7](07-metric-initialization.md) from *accelerometer motion /
   vision motion* with $\mathbf{b}_a=0$. EuRoC's accel bias is 0.55 m/s² — huge relative to the motion
   — so it biases the scale. Tightening the observability gate got us 3×-off → 20%-off by picking a
   better-excited window; the residual *is* the un-estimated bias.
2. **Divergence in the fast final section** ($\lVert\mathbf{v}\rVert$ runs to ~6 m/s at ~t=42 s). Two
   things stack: the map thins faster than it re-triangulates under fast motion, and the
   weakly-observable states (velocity, accel bias) drift once vision stops constraining them.

---

## 3. Why the fix is a window — the observability argument

The accel bias is only **weakly** observable per frame: $\partial\Delta\mathbf{v}/\partial\mathbf{b}_a\sim\Delta t\sim0.05$.
Measured across four prior settings (`sigma_accel_bias` 0.1 / 0.2 / 0.4 / 1.0), $b_{a,y}$ **never**
converges to its true 0.548 — a loose prior only lets it *wander*, a tight one *pins it at zero*.

`★ Insight — one frame cannot pin a weak state ─`
A single-frame solve constrains $\mathbf{b}_a$ through *one* short interval — barely. A window of $K$
keyframes constrains the *same* $\mathbf{b}_a$ through all $K-1$ IMU factors **jointly**, and now the
tiny per-frame sensitivities add up to something the solve can actually resolve. That is the entire
argument, and it is why every drift thread in this project lands here. This is also exactly
[Module 2](02-least-squares.md)'s loop — just with a bigger $\mathbf{x}$.
`─────────────────────────────────────────────────`

(Loose coupling was considered and rejected: it trades these fixable bugs for worse ones — scale drift
per segment, rotation degeneracy — and abandons the tight-coupling thesis the repo exists to show.)

---

## 4. It is the same Gauss-Newton, just wider

Nothing about [Module 2](02-least-squares.md) changes — same linearize → accumulate → solve → retract.
Only the **shape** grows:

- $\delta\mathbf{x}$ becomes $15K$ long ($K$ keyframes) instead of 15;
- $\mathbf{H}$ becomes $15K\times15K$;
- the IMU factor between two keyframes contributes to **two** state blocks —
  $\partial\mathbf{r}/\partial\mathbf{x}_i$ (`imuJacobianI`, [Module 4](04-imu-preintegration.md)) *and*
  $\partial\mathbf{r}/\partial\mathbf{x}_j$ — instead of one;
- the same accel bias appears in $K-1$ IMU factors at once, which is what finally makes it observable.

---

## 5. Before you build it — glass-lio made this exact argument, and was wrong

§3 is a good argument. It is also, nearly word for word, the argument glass-lio made about *its*
tight-coupling divergence — and glass-lio has since documented, at length, that the argument was not
what was actually wrong.

The LiDAR side reasoned: $\mathbf{x}_i$ is held infinitely certain, a single solve can never correct
it, so the fix is a real sliding window. It then *built the primitives* — gravity promoted to a
state, the closed-form $\mathbf{x}_i$ marginalization, the Schur kernel, the state-transition
Jacobian — pinned each against finite differences, and re-ran. **It still diverged: 1.5 million
metres** ([`7-tight-coupling.md` §7.8b](https://github.com/Tim-HW/glass-lio/blob/main/doc/7-tight-coupling.md)).

The real cause, found in §7.8c by logging the *state* instead of the pose: the gravity prior anchored
$\mathbf{g}$ to its own carried estimate every scan — a random walk with no restoring force. Gravity
is near-unobservable over one 0.1 s scan, so the solver explained every small error by tilting
$\mathbf{g}$, the anchor chased it, and it compounded. One line. With that fixed and `lidar_sigma`
calibrated to the sensor's actual ~2 cm, tight coupling went from *broken* to **parity with the
trusted loose path** — two numbers, zero new architecture.

`★ Insight — the sophisticated wrong story ───────`
The §7.8b diagnosis was not lazy. It was careful, primitive-by-primitive, and it named a **real**
deficiency: $\mathbf{x}_i$ *is* held certain, and that *is* wrong. It simply was not what dominated
the error. **A defect being real does not make it the one you are measuring.** The pose said only
"everything is huge"; the state said "gravity, specifically."
`──────────────────────────────────────────────────`

**What that means here.** glassvio's gravity is a fixed constant — `gravity_world_` in
[`vio_estimator.hpp`](../include/glassvio/vio_estimator.hpp), set at bootstrap and never
re-estimated — so it cannot run away the way glass-lio's did. But the failure *mode* generalizes:
§2 attributes the scale bias to an un-estimated $\mathbf{b}_a$, and that attribution has not been
measured against the alternative — that some *fixed* quantity (the bootstrap gravity, the extrinsic,
the scale gate's threshold) is simply mis-set. A window cannot fix a wrong constant.

So before Stage A, spend the day [the Lab](#lab--the-measurement-discipline-itself) asks for:

1. Add gravity-error and **per-axis** $\mathbf{b}_a$ columns to `estimator_check`'s CSV — it logs
   `bay`/`gbay` only.
2. Check whether the ~20% scale error and the t≈42 s divergence track a *fixed* miscalibration, or a
   genuinely unconverged state.

If $\mathbf{b}_a$ never converges under any single-frame weighting — which §3's four-setting sweep
already suggests — Stage A is right and you have lost a day. If it does converge once something else
is corrected, you have saved the weeks Stage A costs. glass-lio paid the second price; this module
exists so this project does not pay it twice.

---

## 6. The plan — stage it, smallest useful build first

### Stage A — 3-keyframe fixed-lag smoother (do this first)

A window of $K=3$–5 keyframe `NavState`s optimized jointly, oldest simply **dropped** when the window
slides (no marginalization yet — accept the small information loss). The minimum that makes the accel
bias observable.

| piece | status |
|---|---|
| IMU factor between two keyframes: `∂r/∂x_j` AND `∂r/∂x_i` | **built + pinned** (`imuJacobian`, `imuJacobianI`, `test_nav_residual`) |
| State prior / `boxminus` for the window's oldest state | **built + pinned** (`priorResidual`, `priorJacobian`, `boxminus`) |
| Reprojection factor (2×15 per landmark) | **built + pinned to 3e-10** (`reprojection.hpp`) |
| Schur-complement marginalization (Stage B's primitive) | **built + pinned to 2.1e-17** (`marginalization.hpp`, `test_marginalization`) |
| State-transition Jacobian $\mathbf{F}$, for $\mathbf{P}_j=\mathbf{F}\mathbf{P}_i\mathbf{F}^\top+\mathbf{G}\mathbf{Q}\mathbf{G}^\top$ | **built + pinned** (`imuStateTransition`; the noise half is `ImuPreintegration::covariance()`) |
| Dynamic-size normal equations ($15K$ wide) | **NEW** — `NormalEquationsN<N>` is fixed-size; needs a windowed solver |
| Keyframe selection + management | **NEW** — parallax/time-based insertion |

Only the last two rows are actually missing. The rest arrived with `glass_core` — glass-lio built
them chasing §5's divergence, and they are shared here verbatim.

**Success test:** re-run `estimator_check`, watch `bay` (accel-bias-y estimate) climb toward 0.548 and
`pos_err` flatten. If it does, the window works — and the residual scale bias should shrink too, since
bias and scale are coupled through [Module 7 §3](07-metric-initialization.md)'s equation.

### Stage B — marginalization (only if Stage A's dropped-oldest loss matters)

Schur-complement the oldest keyframe + its landmarks into a **prior** on the remaining window, instead
of dropping it. This is the hard, essential part of a real VIO backend (where VINS-Mono spends most of
its backend complexity), and consistency matters: a wrong marginalization injects spurious
information. Landmarks are marginalized out each step, leaving a dense system over just the $K$
keyframe states ($15K$, small).

The *kernel* is already built and exact — `schurMarginalize` in
[`marginalization.hpp`](../glass_core/include/glass_core/marginalization.hpp), pinned to 2.1e-17
against both the full solve (must match) and the naive hold-fixed solve (must differ). What Stage B
adds is the bookkeeping around it: deciding what to marginalize, keeping the resulting prior
consistent as the window slides, and not double-counting information. Do **not** start here.

### Stage C — feature supply for fast motion (independent, smaller)

The fast-motion divergence is partly *supply*: the tracker top-up is gated below `min_features` to
protect the bootstrap SfM ([Module 3](03-camera.md)), which starves young-track supply during
tracking. The conflict is **temporal** — SfM runs once at bootstrap; aggressive top-up during tracking
is harmless. Fix: phase-aware top-up (gentle while bootstrapping, full while tracking). The wrinkle:
the tracker runs in the node's callback, upstream of the estimator's phase, so making it phase-aware
and testable in `estimator_check` needs a little plumbing. Separable from the window; can go anytime.

---

## 7. Landmines (each cost real time on this project)

- **Online ≠ offline, over and over.** Every node-only bug was "the end of the buffer is not NOW":
  bias pairs (2912 frames vs 50), dropped IMU chains, stale SfM windows, churned KLT ids.
  `estimator_check` removes all of it. When the node fails but the harness does not, it is plumbing,
  not logic.
- **A count is not a duration.** Every "how many frames" parameter means different seconds at a
  different rate. Derive from the measured rate, never hardcode.
- **`-UNDEBUG` in CMake is load-bearing.** Release defines `NDEBUG`, which deletes every `assert()` —
  the tests would pass while checking nothing. Already set; do not remove.
- **Eigen `auto` + `.inverse().translation()` dangles.** Name the type (`-> Eigen::Vector3d`).
- **Ruler units vs metres.** Only the SfM translation scales by $s$; the extrinsic is already metric.
  Composing the two Isometries directly mixes them silently.
- **The scale gate checks CONDITIONING, not accuracy.** 0.06 is tuned for EuRoC V1_01 — re-derive from
  the uncertainty-vs-accuracy curve on a new sequence.

---

## Lab — the measurement discipline itself

`estimator_check` is the highest-leverage tool in the project: it drives the **real** `VioEstimator`
deterministically (no ROS, no threads, no dropped frames) and writes a 35-column per-frame CSV.

```bash
colcon build --packages-select glassvio
./build/glassvio/estimator_check          # writes /tmp/glassvio_run.csv
```

Columns that matter: `pos_err`, `vel_err`, `feats`, `map`, `pending`, `bay`/`gbay` (accel bias est vs
truth). Error is aligned to ground truth at the bootstrap instant — a raw
$\lVert\mathbf{p}_{\text{est}}-\mathbf{p}_{\text{gt}}\rVert$ is meaningless because the estimator
defines its own gravity-aligned world.

1. **Plot `bay` against `gbay`.** Watch the estimated accel bias fail to reach the true value — §3's
   argument, on your screen. This is the number Stage A must move.
2. **Plot `vel_err`.** The ~20% under-estimate of §2.1, steady — a *metric* error, not a tracking one
   (`rmse` stays 2–4 px throughout).
3. **Add the columns §5 asks for, and try to falsify §3.** Log the gravity error and each axis of
   $\mathbf{b}_a$ separately, then ask whether the drift tracks a *fixed* miscalibration rather than an
   unconverged state. This is the step that would have saved glass-lio a rewrite, and it costs a day
   against Stage A's weeks. Do it before you build the window, not after.
4. **The habit to keep:** measure here *before* every change and re-measure *after*. Every self-deception
   this project caught — a bias-prior loosening that made drift worse, a condition-number gate that did
   not discriminate, a feature-supply hypothesis that was wrong — was caught by this CSV, not by
   reasoning. That habit is the last thing the course has to teach.

---

## Quick reference

```bash
./build/glassvio/estimator_check     # deterministic drive + CSV — measure here first
./build/glassvio/vio_check           # the offline tight-coupling thesis check
./run_euroc.sh                       # the node, live, with RViz
colcon test --packages-select glassvio   # the six suites: glass_core's four + reprojection + tracker
```

---

← [Module 7](07-metric-initialization.md) · [Syllabus](README.md) · [Pipeline reference](pipeline.md)
