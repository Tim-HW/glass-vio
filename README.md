# glass-VIO

**A transparent visual-inertial odometry — sharing [glass-lio](https://github.com/Tim-HW/glass-lio)'s estimation engine.**

GlassVIO is the visual sibling of GlassLIO. It reuses the *exact same* on-manifold
estimation core — `glass_core`: Gauss-Newton on SE(3), IMU preintegration, the 15-DoF nav
state, the IMU factor — and swaps the LiDAR point-to-plane residual for a **camera
reprojection residual**. One solver, two sensors, zero copied Jacobians.

> The whole reason `glass_core` was extracted from GlassLIO is *this repo*. If the same
> subtly-wrong Jacobian can sink a LiDAR odometry and a visual one, it should live in one
> place, tested once — not copy-pasted and left to drift.

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

`glass_core` is **not** a colcon package — it is a pure-CMake sub-project that physically
lives in the glass-lio repo at `glasslio/glass_core`. glassvio is a **sibling checkout** and
pulls it in with `add_subdirectory`:

```
src/
├── glasslio/            (glass-lio repo)
│   └── glass_core/      the shared engine
└── glassvio/            (this repo)  → add_subdirectory(../glasslio/glass_core)
```

So **glass-lio must be checked out beside glassvio** — CMake errors out clearly if the
`../glasslio/glass_core` path is missing.

## Build & run

```bash
# from a colcon workspace with both repos under src/
colcon build --packages-select glassvio
source install/setup.bash
ros2 run glassvio glassvio_node --ros-args -p imu_topic:=/livox/imu
# then play a bag with an IMU stream; the node logs when init completes
```

## License

**[MIT](LICENSE)** — code, config, docs. The shared `glass_core` engine and its vendored
Sophus headers keep their own (also MIT) terms; see the glass-lio repo.
