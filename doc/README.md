# A course in visual-inertial SLAM

This is not a manual for a black box. It is a **course** — build a monocular visual-inertial
odometry (VIO) from the estimation problem up, one runnable module at a time, on a real codebase
that fuses a camera and an IMU into one metric trajectory.

The system is [glassvio](../README.md): a monocular VIO that shares glass-lio's on-manifold
estimation engine (`glass_core`) and swaps the LiDAR point-to-plane residual for a camera
reprojection residual. **One solver, two sensors.** Everything here is real code that runs, with
an oracle-scored check behind every claim.

## How this course is different

Most SLAM courses derive equations you never run, or ship code you never derive. This does both,
and adds a third thing that is the whole point:

`★ The defining idea ─────────────────────────────`
**Every serious bug in an estimator produces plausible output, not a crash.** A sign-flipped
Jacobian still converges. A 3×-biased scale still tracks. A wrong retraction still descends the
cost. So this course is built on *oracle-scored checks*: each module ends with a **Lab** — a real
binary you run, whose number an independent ground truth already knows. You will make a system
that "works," measure that it is quietly wrong, and fix it. That loop *is* estimation engineering.
`──────────────────────────────────────────────────`

## The arc — bottom-up

The modules build in **learning order**, not the order the code executes at runtime (for that, see
the [pipeline reference](pipeline.md)). Foundations first, then each sensor, then the fusion, then
the hard problem monocular VIO exists to solve — recovering the missing metric scale — and finally
where it still drifts.

| # | Module | You will be able to… | Lab |
|---|---|---|---|
| **0** | [The problem](00-the-problem.md) | State what VIO estimates and *why a monocular camera has no ruler* | — |
| **1** | [State on a manifold](01-manifolds.md) | Work with SO(3): `Exp`/`Log`, ⊞/⊟, the right Jacobian, the adjoint | `test_nav_residual` |
| **2** | [Nonlinear least squares](02-least-squares.md) | Turn residuals into `H`,`b`, solve by Gauss-Newton, retract onto the manifold | `estimator_check` |
| **3** | [The camera](03-camera.md) | Project a landmark, undistort, and derive the reprojection Jacobian | `test_reprojection` |
| **4** | [The IMU](04-imu-preintegration.md) | Preintegrate a gyro/accel stream and form the 9-DoF IMU residual | `imu_dead_reckon_check` |
| **5** | [Tight coupling](05-tight-coupling.md) | Stack three residuals into one 15-DoF system — *stacking IS the fusion* | `vio_check` |
| **6** | [Epipolar geometry & SfM](06-epipolar-and-sfm.md) | Recover up-to-scale structure: essential matrix, triangulation, the invented ruler | `epipolar_check`, `sfm_check` |
| **7** | [Metric initialization](07-metric-initialization.md) | Hand the accelerometer's metre to the reconstruction; gate on scale observability | `gyro_bias_check`, `vi_align_check` |
| **8** | [Sliding window & drift](08-sliding-window.md) | See where a single-frame solve fails, and why a window fixes it | `estimator_check` (CSV) |

## How to use it

- **Read in order.** Each module assumes the one before it. Modules 1–2 are the math spine;
  skimming them makes 3–7 opaque.
- **Do every Lab.** They are not exercises bolted on — they are the argument. Build from a colcon
  workspace with glassvio and glasslio side by side under `src/`, then run the named binary:
  ```bash
  colcon build --packages-select glassvio
  ./build/glassvio/<lab_binary>        # e.g. ./build/glassvio/epipolar_check
  ```
- **Break things on purpose.** Most Labs end with a *"now break it"* step — flip a sign, drop a
  Jacobian term, loosen a gate — and watch the estimator stay confident while going wrong. That is
  the lesson no derivation teaches.

## Prerequisites

Linear algebra (least squares, eigen/SVD), multivariable calculus (Jacobians, the chain rule), and
enough C++ to read the code. No prior SLAM. The Lie-group machinery is built from scratch in
[Module 1](01-manifolds.md).

## Two reference companions (not lessons)

- **[pipeline.md](pipeline.md)** — the same system in *execution* order: the diagram, the frames,
  the threading, the live status. The map of the running program; read it once you know the parts.
- glass-lio's **[gauss-newton.md](../../glasslio/doc/gauss-newton.md)** and
  **[testing.md](../../glasslio/doc/testing.md)** — the shared engine's own derivation of the
  manifold solver and its testing philosophy. `glass_core` is shared verbatim, so both apply here.

## The one-paragraph version of what you will build

Each image is FAST+KLT-tracked, its points undistorted, and synced with the IMU spanning the gap
since the last frame. A monocular camera has no scale, so the system first **bootstraps** — SfM for
shape, vision for the gyro bias, a linear alignment that hands the accelerometer's metre to the
reconstruction (scale, gravity, velocity). Then it **tracks**: a camera reprojection residual and
an IMU factor in the *same* `NormalEquationsN<15>` glass-lio's LiDAR path uses, over a
sliding-window landmark map. It bootstraps at metric scale and tracks ~29 s on EuRoC V1_01 at
0.65 m median drift — with a residual scale bias that [Module 8](08-sliding-window.md) explains and
a sliding-window BA will fix.
