# glass-VIO

**A transparent visual-inertial odometry — sharing [glass-lio](https://github.com/Tim-HW/glass-lio)'s estimation engine.**

GlassVIO is the visual sibling of GlassLIO. It reuses the *exact same* on-manifold
estimation core — `glass_core`: Gauss-Newton on SE(3), IMU preintegration, the 15-DoF nav
state, the IMU factor — and swaps the LiDAR point-to-plane residual for a **camera
reprojection residual**. One solver, two sensors, zero copied Jacobians.

> The whole reason `glass_core` was extracted from GlassLIO is *this repo*. If the same
> subtly-wrong Jacobian can sink a LiDAR odometry and a visual one, it should live in one
> place, tested once — not copy-pasted and left to drift.

## Docs

Like glass-lio, this repo is written to be **read**.

- **[doc/solver.md](doc/solver.md)** — the tightly-coupled solve: the **three residuals**
  (reprojection, IMU, prior) stacked into one 15-DoF Gauss-Newton system, how the Jacobians are
  computed and why they are sparse by physics, and why *stacking into one `H` IS the fusion* —
  no filter, no blend. The whole "one solver, two sensors" thesis, worked through.
- **[doc/preintegration.md](doc/preintegration.md)** — how the IMU factor's engine room
  actually works: what one `integrate()` call does, the **ordering discipline** that
  separates a correct estimator from a plausible-looking wrong one, why the noise density
  is *divided* by `Δt`, and what the KITTI dead-reckon check does and does not prove.
  Complements glass-lio's [§7.4](../glass-lio/doc/7-tight-coupling.md), which covers *why*
  preintegration exists; this covers *how*.
- **[doc/next-steps.md](doc/next-steps.md)** — the cold-start handoff: current state, the
  `estimator_check` measurement harness, the two remaining residuals with their measured causes,
  and the staged plan for the sliding-window BA.

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
# from a colcon workspace, glassvio under src/ (glass_core comes along as its submodule)
colcon build --packages-select glassvio
source install/setup.bash
ros2 run glassvio glassvio_node --ros-args -p imu_topic:=/livox/imu
# then play a bag with an IMU stream; the node logs when init completes
```

## License

**[MIT](LICENSE)** — code, config, docs. The shared `glass_core` engine and its vendored
Sophus headers keep their own (also MIT) terms; see the [glass-core](https://github.com/Tim-HW/glass-core) repo.
