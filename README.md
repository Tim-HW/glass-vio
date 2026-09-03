# glass-VIO

**A transparent visual-inertial odometry — sharing [glass-lio](https://github.com/Tim-HW/glass-lio)'s estimation engine.**

GlassVIO is the visual sibling of GlassLIO. It reuses the *exact same* on-manifold
estimation core — `glass_core`: Gauss-Newton on SE(3), IMU preintegration, the 15-DoF nav
state, the IMU factor — and swaps the LiDAR point-to-plane residual for a **camera
reprojection residual**. One solver, two sensors, zero copied Jacobians.

> The whole reason `glass_core` was extracted from GlassLIO is *this repo*. If the same
> subtly-wrong Jacobian can sink a LiDAR odometry and a visual one, it should live in one
> place, tested once — not copy-pasted and left to drift.

## Docs — a course in visual-inertial SLAM

This repo is written to be **read**, and its docs are structured as a **bottom-up course**: build a
monocular VIO from the estimation problem up, one runnable module at a time, with an oracle-scored
lab behind every claim.

- **[doc/README.md](doc/README.md)** — **start here.** The syllabus: nine modules from the manifold
  ([1](doc/01-manifolds.md)) through Gauss-Newton ([2](doc/02-least-squares.md)), the camera
  ([3](doc/03-camera.md)) and IMU ([4](doc/04-imu-preintegration.md)), the tight-coupling thesis
  ([5](doc/05-tight-coupling.md)), monocular bootstrap ([6](doc/06-epipolar-and-sfm.md)–[7](doc/07-metric-initialization.md)),
  and where it still drifts ([8](doc/08-sliding-window.md)) — each with a **Lab** you run and then
  break on purpose.
- **[doc/pipeline.md](doc/pipeline.md)** — the same system in *execution* order: the diagram, the
  frames, the threading, the live status. The reference map, for once you know the parts.
- glass-lio's **[gauss-newton.md](https://github.com/Tim-HW/glass-lio/blob/main/doc/gauss-newton.md)**
  and **[testing.md](https://github.com/Tim-HW/glass-lio/blob/main/doc/testing.md)** — the shared
  engine's own derivation and testing philosophy; `glass_core` is shared verbatim.

## Status

**Early skeleton.** What runs today:

- **IMU initialization** on the shared engine — gyro bias, gravity, gravity-aligned world
  frame — the same bootstrap GlassLIO uses (`glass_core::ImuInit`). Verified against the
  GlassLIO test bag: `|g|` = 9.781, mount tilt 7.33°, matching GlassLIO exactly.

Next phases (the actual VIO):

1. **Feature front-end** — FAST/Shi-Tomasi detection + KLT optical-flow tracking.
2. **Reprojection factor** — pixel error of a projected 3D landmark, analytic Jacobian
   pinned against finite differences, folded into the same `glass_core` Gauss-Newton solver.
3. **Landmarks + sliding window** — triangulation and a small joint solve over visual +
   IMU factors.

## How it shares the engine

`glass_core` is **not** a colcon package — it is a pure-CMake sub-project with its own repo,
[glass-core](https://github.com/Tim-HW/glass-core), vendored here as a **git submodule** at
`glass_core/` and pulled in with `add_subdirectory`. glass-lio's copy is the same submodule —
one engine, one history, two front-ends, no copy-paste of Jacobians.

```bash
# after cloning glassvio, fetch the submodule too:
git submodule update --init --recursive
```

## Build & run

```bash
# from a colcon workspace with glassvio under src/
git submodule update --init --recursive   # once: pulls glass_core
./scripts/download_bag.sh                 # once: EuRoC V1_01_easy, converted to ROS 2
colcon build --packages-select glassvio
source install/setup.bash

./run_euroc.sh                            # the node, live, with RViz
./build/glassvio/estimator_check          # the deterministic harness, straight to a CSV
colcon test --packages-select glassvio    # the Jacobian suites
```

## License

**[MIT](LICENSE)** — code, config, docs. The shared `glass_core` engine and its vendored
Sophus headers keep their own (also MIT) terms; see the [glass-core](https://github.com/Tim-HW/glass-core) repo.
