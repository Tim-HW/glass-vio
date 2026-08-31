# Next steps — sliding-window BA, and where the estimator stands

A cold-start handoff. Read this first if you are picking glassvio back up.

## Where we are

**A working monocular VIO on the ROS node.** It bootstraps at true metric scale, tracks
~20–30 s on EuRoC V1_01, maintains a sliding-window landmark map live, and publishes odom +
TF (visualised in RViz via `./run_euroc.sh`). 71 unit tests green; the offline thesis check
`vio_check` is 0.036 m.

Measured on the deterministic harness (`estimator_check`, see below):

| metric | value |
|---|---|
| bootstrap landmark depth | 1.62 m (true room scale) |
| velocity accuracy `\|v\|est / \|v\|gt` | ~0.80 |
| tracked before losing the scene | ~29 s |
| median position drift (aligned to GT at bootstrap) | 0.65 m |

The thesis — camera reprojection + IMU in ONE `NormalEquationsN<15>`, the same Gauss-Newton
the LiDAR path uses — is demonstrated end to end.

## Housekeeping: `glass_core` is now a proper git submodule

Two things were wrong and are now fixed:

1. `CMakeLists.txt` pointed `add_subdirectory` at `../glasslio/glass_core`, a sibling-checkout
   path that never matched the actual directory name (`glass-lio`) — any real build should have
   hit the `FATAL_ERROR` at configure time.
2. This repo also carried its own committed copy of `glass_core/` that no CMake target ever
   referenced — dead weight, stale relative to glass-lio's.

Both are resolved by extracting `glass_core` into its own repo,
[glass-core](https://github.com/Tim-HW/glass-core), and vendoring it here as a **git
submodule** at `glass_core/`. glassvio no longer needs glass-lio checked out beside it at all —
clone, then `git submodule update --init --recursive`. `CMakeLists.txt` now points
`GLASS_CORE_DIR` at the local `glass_core/` submodule instead of reaching into a sibling repo.

This matters for the plan below: glass-lio's `glass_core` (the same content, now the submodule's
source of truth) had moved ahead of what this repo was built against, so pieces the table below
used to list as speculative are
now sitting right there once you build against the fixed path.

## THE key tool: `estimator_check`

`src/checks/estimator_check.cpp` drives the REAL `VioEstimator` deterministically (no ROS, no
threads, no dropped frames), and writes a 35-column per-frame CSV:

```
./install/glassvio/lib/glassvio/estimator_check [bag] [config] [out.csv]   # default /tmp/glassvio_run.csv
```

This is the highest-leverage thing built in the whole project. Every online bug this session
was timing-dependent and confounded ~5 variables through the node; this tool made each a
deterministic yes/no in seconds. **Use it to measure before every fix, and re-measure after.**
The CSV columns that matter: `pos_err`, `vel_err`, `feats`, `map`, `pending`, `bay`/`gbay`
(accel bias est vs truth). Error is aligned to ground truth at the bootstrap instant — a raw
`\|p_est - p_gt\|` is meaningless because the estimator defines its own gravity-aligned world.

## The two remaining residuals, with measured causes

Both trace to the same root and both are the sliding window's job.

1. **~20% residual scale under-estimate** (`\|v\|est` ~0.80, not 1.0). The bootstrap scale is
   computed by stage [4] from `accelerometer motion / vision motion` with `b_a = 0`. EuRoC's
   accel bias is 0.55 m/s^2 — huge relative to the motion — so it biases the scale. Tightening
   the scale-observability gate (`max_scale_uncertainty` 0.15 -> 0.06) got us from 3x-off to
   20%-off by picking a better-excited window, but the residual is the un-estimated bias.

2. **Divergence in the fast final section** (`\|v\|` runs away to ~6 m/s at ~t=42 s). Two
   things stack: the map thins faster than it re-triangulates under fast motion, and the
   weakly-observable states (velocity, accel bias) drift once vision stops constraining them.

## Why the fix is a sliding window (and not what it looks like)

The accel bias is only WEAKLY observable per frame (`dv/db_a ~ dt ~ 0.05`). Measured across
four prior settings (`sigma_accel_bias` 0.1 / 0.2 / 0.4 / 1.0): `b_a,y` NEVER converges to its
true 0.548 — a loose prior only lets it WANDER, a tight one pins it at zero. A single-frame
solve cannot pin a weakly-observable bias. A window of many frames CAN, because the bias is
constrained by all of them jointly. That is the entire argument, and it is why every drift
thread in this project lands here.

(Loose coupling was considered and rejected — see the discussion in the git history: it trades
these fixable bugs for worse ones, scale drift per segment and rotation degeneracy, and
abandons the tight-coupling thesis the repo exists to show.)

## Read this before Stage A: glass-lio built the exact same argument, and it was wrong

glass-lio's tight-coupling work ([`doc/7-tight-coupling.md`](../glass-lio/doc/7-tight-coupling.md)
§7.8b→§7.8c, in the sibling repo) made *precisely* this argument on the LiDAR side — "`x_i` is
held infinitely certain, the fix is a real sliding window" — and built the primitives for it:
gravity promoted to an estimated state, the `x_i`-uncertainty inflation, and a general
Schur-complement `marginalization.hpp`. **It still diverged catastrophically (pose to 1.5M m).**

The actual bug was two boring calibration numbers: a gravity prior anchored to its own running
estimate (a random walk with no restoring force — small per-scan errors got shovelled into
gravity and compounded), and a mistuned `lidar_sigma`. Fixing those two — zero new architecture
— brought it to parity with the trusted loose path. The lesson they drew, twice (they got it
wrong once even after being warned): **instrument the state, not the pose/error metric**, before
trusting a "this needs a bigger architecture" diagnosis.

Why this is worth pausing on here: glass-vio's own gravity is a **fixed constant**
(`gravity_world_` in `vio_estimator.hpp`), never re-estimated after bootstrap, and the diagnosed
scale bug ("EuRoC's accel bias is huge relative to the motion, biases the scale") has the same
shape as glass-lio's gravity/accel-bias conflation bug. Before committing to Stage A's multi-day
build, it's worth cheaply checking: add gravity-error and per-axis `b_a` columns to
`estimator_check`'s CSV (not just `bay`/`gbay` magnitude) and see whether the ~20% scale error
and the fast-motion divergence correlate with a specific miscalibrated scalar rather than
genuinely needing multi-frame joint observability. If they do, that's a day of calibration, not
a windowed solver. If they don't — if the bias truly never converges under any single-frame
weighting, as the four-setting sweep already suggests — Stage A is the right call and now has
more of its primitives built than the table below used to say.

## The plan: stage it, smallest useful build first

### Stage A — 3-keyframe fixed-lag smoother (do this first)

A window of K=3–5 keyframe NavStates optimised jointly, oldest simply DROPPED when the window
slides (no marginalisation yet — accept the small information loss). This is the minimum that
makes the accel bias observable.

What it needs, and what already exists:

| piece | status |
|---|---|
| IMU factor between two keyframes: `∂r/∂x_j` AND `∂r/∂x_i` | **built + finite-diff-pinned** (`imuJacobian`, `imuJacobianI` in glass_core, `test_nav_residual`) |
| State prior / `boxminus` for the window's oldest state | **built + pinned** (`priorResidual`, `priorJacobian`, `boxminus`) |
| Reprojection factor (2x15 per landmark) | **built + pinned to 3e-10** (`reprojection.hpp`, `test_reprojection`) |
| Schur-complement marginalization (Stage B's primitive, exact) | **built + pinned to 2.1e-17** (`marginalization.hpp`, `test_marginalization`) — pulled forward from Stage B since it now exists in glass_core anyway |
| State-transition Jacobian `F` (for `P_j = F P_i F^T + G Q G^T`) | **built + pinned** (`imuStateTransition`) — the noise half was always `ImuPreintegration::covariance()` |
| Dynamic-size normal equations (15*K wide) | **NEW** — `NormalEquationsN<N>` is fixed-size; needs a windowed/dynamic solver |
| Keyframe selection + management | **NEW** — parallax/time-based keyframe insertion |

The first four rows are only true as of the `glass_core` link fix above — they live in
glass-lio's copy, which this repo wasn't actually building against until now.

Success test: re-run `estimator_check`, watch `bay` climb to ~0.548 and `pos_err` flatten. If
it does, the window works and the residual scale bias should shrink too (bias and scale are
coupled through stage [4]'s equation).

### Stage B — marginalisation (only if Stage A's dropped-oldest loss matters)

Schur-complement the oldest keyframe + its landmarks into a prior on the remaining window,
instead of dropping it. This is the hard, essential part of a real VIO backend (it is where
VINS-Mono spends most of its backend complexity) — and consistency matters: a wrong
marginalisation injects spurious information. Landmarks get marginalised out each step, leaving
a dense system over just the K keyframe states (15K, small). Do NOT start here.

### Stage C — feature supply for fast motion (independent, smaller)

The node's fast-motion divergence is partly supply: the tracker top-up is gated (below
`min_features`) to protect the bootstrap SfM, which starves the map's young-track supply during
tracking. The conflict is TEMPORAL — SfM only runs once at bootstrap; aggressive top-up during
tracking is harmless. Fix: phase-aware top-up (gentle while bootstrapping, full while tracking).
The wrinkle: the tracker runs in the node's callback, upstream of the estimator's phase, and
`EurocDataset` pre-tracks all frames offline — so making it phase-aware and testable in
`estimator_check` needs a little plumbing. This is separable from the window and can go anytime.

## Landmines to remember (all cost real time this project)

- **Online != offline, over and over.** Every node-only bug was "the end of the buffer is not
  NOW": bias pairs (2912 frames vs 50), dropped IMU chains, stale SfM windows, stale anchors,
  churned KLT ids. `estimator_check` removes all of it. When the node fails but the harness
  does not, it is plumbing, not logic.
- **A count is not a duration.** `num_samples`, `bootstrap_frames` vs `window_frames` — every
  "how many frames" parameter means different seconds at a different rate. Derive from the
  measured rate, never hardcode.
- **`-UNDEBUG` in CMake is load-bearing.** Release defines `NDEBUG` which deletes every
  `assert()`; the tests would pass while checking nothing. Already set — do not remove.
- **Eigen `auto` + `.inverse().translation()` dangles.** Name the type
  (`-> Eigen::Vector3d`); a deduced `auto` returns a Block into a dead temporary.
- **Ruler units vs metres.** SfM positions are in the invented ruler; the extrinsic is in
  metres. Only the SfM translation scales by `s`. Composing the two Isometries directly mixes
  them silently.
- **`cv::triangulatePoints` returns `CV_32F` for `Point2f` input.** Reading it as `double`
  yields fake points-at-infinity. Convert.
- **The scale gate checks CONDITIONING, not accuracy.** `σ_s/\|s\|` ensures scale is
  well-constrained; a well-conditioned window can still be biased. 0.06 is tuned for EuRoC
  V1_01 — re-derive it from the uncertainty-vs-accuracy curve on a new sequence.

## Quick reference

```bash
# deterministic drive + CSV (measure here first)
./install/glassvio/lib/glassvio/estimator_check
# the offline thesis check (tight coupling, one solver two sensors)
./install/glassvio/lib/glassvio/vio_check
# the node, live, with RViz
./run_euroc.sh
# unit tests (glass_core Jacobians + reprojection + tracker)
colcon test --packages-select glassvio glasslio
```
