# The VIO pipeline — execution-order reference

> This is the **reference map** of the running system, in the order the code executes at runtime.
> To *learn* the system, start with the **[course syllabus](README.md)** instead — it builds the
> same parts bottom-up. This page is for orienting once you know them.

Monocular visual-inertial odometry, sharing glass-lio's estimation engine (`glass_core`) — the same
Gauss-Newton on SE(3), the same IMU preintegration and 15-DoF nav state, with a **camera reprojection
residual** where the LiDAR point-to-plane was. One solver, two sensors, zero copied Jacobians.

```
  image ─► track (FAST+KLT, undistort) ─┐
                                          ├─► sync ─► MeasureGroup ─► collect ─► BOOTSTRAP ─► track
  imu ────────────────────────────────── ┘          (image+IMU gap)   (gate)              (tight solve)
```

## The stages, in execution order → the modules that explain them

| # | Stage | Course module | Code |
|---|---|---|---|
| **1** | **Feature tracking** — FAST + KLT, persistent ids, points undistorted at the boundary | [Module 3](03-camera.md) | [`feature_tracker.hpp`](../include/glassvio/feature_tracker.hpp) |
| **2** | **Sync** — pair a frame's tracks with the IMU spanning the gap since the last frame | [Module 5 §4](05-tight-coupling.md) | [`sync.hpp`](../include/glassvio/sync.hpp) |
| **3** | **Bootstrap** — recover metric scale, gravity, velocity, gyro bias. *A gate: tracking cannot start until it completes.* | [Modules 6–7](06-epipolar-and-sfm.md) | [`vio_initializer.cpp`](../src/vio/vio_initializer.cpp) |
| **4** | **Track** — reprojection + IMU in one 15-DoF Gauss-Newton solve; sliding-window landmark map | [Module 5](05-tight-coupling.md) | [`visual_registration.cpp`](../src/vio/visual_registration.cpp), [`landmark_map.cpp`](../src/vio/landmark_map.cpp) |

**The foundations under all of it** (not runtime stages): [Module 1 — manifolds](01-manifolds.md),
[Module 2 — Gauss-Newton](02-least-squares.md), [Module 4 — preintegration](04-imu-preintegration.md),
and [Module 8 — the sliding-window backend](08-sliding-window.md) that the drift needs. glass-lio's
[gauss-newton.md](https://github.com/Tim-HW/glass-lio/blob/main/doc/gauss-newton.md) and [testing.md](https://github.com/Tim-HW/glass-lio/blob/main/doc/testing.md)
apply unchanged — `glass_core` is shared.

## The one-paragraph version

Each image is FAST+KLT-**tracked**, its points **undistorted**, and **synced** with the IMU samples
spanning the gap since the last frame. A monocular camera has no scale, so the system first
**bootstraps**: an up-to-scale reconstruction (essential matrix + PnP), the gyro bias from vision's
rotations, and a linear visual-inertial alignment that hands the accelerometer's metre to the
reconstruction — recovering scale, gravity, and velocity. Then it **tracks**: every frame folds a
camera reprojection residual and an IMU preintegration factor into the same `NormalEquationsN<15>`
glass-lio's LiDAR path uses, with a sliding-window landmark map inserting new metric landmarks as old
ones leave view. Output is `~/odom` plus a TF.

## Frames

- **world** (`odom`) — DEFINED at bootstrap, not given. Origin at the first body pose, +Z along
  **measured gravity**, yaw arbitrary (gravity constrains two of three rotational DoF; the third is
  unobservable, left at zero — `FromTwoVectors`, exactly as glass-lio's IMU init). See
  [Module 7 §5](07-metric-initialization.md).
- **imu / body** — the state's frame. `NavState = (R, p, v, b_g, b_a)` lives here.
- **camera** — related to the body by the fixed extrinsic `T_cam_imu` from the Kalibr calib. The ruler
  SfM invents dies at bootstrap; everything after is metric in `world`.

## Threading

Feature tracking runs in the **image callback**, not the worker. This is the one place glassvio must
NOT copy glass-lio. glass-lio drops the oldest scan when its worker falls behind — safe, because ICP
registers against a stateless map. **KLT is not stateless**: its entire state is the previous frame,
so a dropped image sends it N → N+2, the flow doubles, tracks die, ids churn — exactly what the
tracker exists to prevent. So the tracker sits where nothing is dropped, and the queue carries its
OUTPUT (23 KB of tracks, not a 455 KB frame).

```
  image cb (own group)          queue              worker
  ────────────────────      ─────────────         ────────
  track → undistort → sync ──► MeasureGroup ──► collect → bootstrap → track
  imu cb (own group)   ─────►  sync                (VioEstimator)
```

The fix for doing work in a callback is not to move it but to stop it sharing a thread with the IMU:
each subscription has its own **mutually-exclusive callback group** on a `MultiThreadedExecutor`, so
tracking and IMU intake run concurrently. The hand-off is a **bounded queue** (`max_queue_size`) — but
when it overflows it drops the oldest frame's *observations* while **splicing its IMU onto the next
group**, so the preintegration chain is never broken.

### Ownership is the invariant

- `tracker_` — touched only by the image callback.
- `estimator_` (`VioEstimator`, and all its state) — touched only by the worker.
- `sync_` and the sensor buffers — only under `buf_mutex_`.
- The queue is the single hand-off point.

## Output

`~/odom` (`nav_msgs/Odometry`, `world → body`, velocity in the body frame) and the matching TF,
published every tracked frame. The feature overlay is on `~/features` for RViz.

## Current status

**A working monocular VIO.** Measured on the deterministic harness (`estimator_check`) over EuRoC
V1_01:

| Stage | State |
|---|---|
| [1] Feature tracking | ✅ FAST + KLT, persistent ids, radtan undistortion at the boundary. |
| [2] Sync | ✅ Working; IMU spliced across dropped frames. |
| [3] Bootstrap | ✅ Metric scale via the observability gate — 1.62 m landmark depth, gyro bias to ~2%. |
| [4] Track | ✅ Tight solve + sliding-window map, plus Stage A's keyframe-window bundle adjustment and an inlier gate on every tracker solve. Tracks V1_01 from its 15 s bootstrap to the end of the ground truth at metric scale (speed ratio 1.00), median drift 0.36 m, never above 1 m; the faster V1_02 and V1_03 first pass 1 m at 74 s and 97 s. |

The offline thesis check `vio_check` is **0.036 m**; the unit suites are green. **Two residuals
remain** ([Module 8](08-sliding-window.md)): a ~20% scale shrink and a fast-motion divergence. Traced
by substituting ground truth one quantity at a time, both come from the loop in which solved poses
triangulate the map that sets the next poses — with landmarks triangulated from ground-truth poses
instead, the same estimator tracks the whole 130 s sequence at 0.023 m.

## Not implemented (yet)

- **Marginalization (Stage B)** — the window drops its oldest keyframe outright; a Schur prior would
  keep what it knew. The kernel (`schurMarginalize`) is built and pinned. Stage A itself — the
  keyframe-window bundle adjustment — is built: [Module 8 §6](08-sliding-window.md).
- **Gravity re-levelling** — `gravity_world_` stays frozen at the bootstrap's estimate (a 4° tilt).
  Re-estimating it inside the window was built and halves the tilt, but gains no position accuracy,
  so it is off; ORB-SLAM3's answer, a global inertial optimization that re-levels the whole map, is
  not built. [Module 8 §6](08-sliding-window.md). (Triangulating from the window: built, measured
  worse, off.)
- **Loop closure / relocalisation** — none. This is odometry, not SLAM.
- **Stereo / multi-camera** — the bag has cam1, but only cam0 is used.

## Parameters

Parsed by the node from ROS params; defaults in [`vio_estimator.hpp`](../include/glassvio/vio_estimator.hpp).

| Param | Why it matters |
|---|---|
| `calib_dir` | K, the extrinsic (Kalibr's `T_imu_cam` is **inverted** to get `T_cam_imu`), radtan coeffs, datasheet noise densities. No sane default — the node fails loudly without it. [Module 3](03-camera.md) |
| `imu_topic` | **`/imu0`** on EuRoC — the ADIS16448 the calib describes. `/fcu/imu` is a *different* sensor the calib does not fit. |
| `bootstrap_frames` / `sfm_window_frames` | Two DIFFERENT spans; tying them together is a bug made three times. [Module 7 §5](07-metric-initialization.md) |
| `init.max_scale_uncertainty` | The scale-observability gate (`σ_s/\|s\|`). 0.06 admits only well-excited windows; 0.15 let a 3×-biased window through. [Module 7 §4](07-metric-initialization.md) |
| `warmup_frames` | Frames the tracker may COAST on the IMU when the visual solve is momentarily under-constrained. [Module 5 §6](05-tight-coupling.md) |
| `sigma_accel_bias` | Initial accel-bias uncertainty. A per-frame solve cannot pin it either way — that is the sliding window's job. [Module 8](08-sliding-window.md) |
| `max_queue_size` | Worker backlog. A dropped frame keeps its IMU but loses its observations. |

## Running

```bash
# from a colcon workspace with glassvio under src/ (glass_core arrives as its submodule)
git submodule update --init --recursive
./scripts/download_bag.sh      # once: EuRoC V1_01_easy, converted to ROS 2
colcon build --packages-select glassvio
./run_euroc.sh                 # V1_01_easy at 1x, with RViz
./run_euroc.sh rviz:=false     # headless, for clean numbers

# deterministic offline drive + per-frame CSV -- measure here, it has ground truth
./install/glassvio/lib/glassvio/estimator_check
```
