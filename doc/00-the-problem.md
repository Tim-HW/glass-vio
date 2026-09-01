# Module 0 — The problem

> **Prerequisite:** none. **After this you can:** state what VIO estimates, and explain precisely
> why one camera cannot measure scale but a camera plus an accelerometer can.

Odometry is the problem of answering *"where am I now, relative to where I started?"* from a stream
of sensor data, with no map and no external fix (no GPS, no loop closure). Visual-inertial odometry
answers it from two cheap sensors that fail in *opposite* ways — and the entire design of this
system is organized around making each one cover the other's blind spot.

---

## 1. What we are estimating

At each instant the system holds a **navigation state**:

$$
\mathbf{x} = (\mathbf{R},\ \mathbf{p},\ \mathbf{v},\ \mathbf{b}_g,\ \mathbf{b}_a)
\qquad 3+3+3+3+3 = 15 \text{ degrees of freedom}
$$

- $\mathbf{R}$ — orientation of the body in the world (a rotation; [Module 1](01-manifolds.md))
- $\mathbf{p}$ — position of the body in the world
- $\mathbf{v}$ — velocity of the body
- $\mathbf{b}_g,\ \mathbf{b}_a$ — the slowly-drifting gyroscope and accelerometer **biases**

The last two are not what we want to *know* — nobody cares about the bias — but they must be in the
state, because if they are not estimated they corrupt everything that is. Half of this course is
really about the biases.

---

## 2. The two sensors, and their opposite failures

| | **Camera** | **IMU (gyro + accelerometer)** |
|---|---|---|
| measures | where world points *appear* (pixels) | angular rate, and specific force |
| rate | ~20 Hz | ~200 Hz |
| strong at | drift-free structure, orientation over the long run | short-term motion through blur, darkness, texture-less walls |
| **blind to** | **absolute scale** (§3), and it dies in blur/dark/blank | absolute pose — it only sees *changes*, and they drift as $t^2$ |

Neither is usable alone for metric odometry. The camera is a drift-free ruler with no marked
units; the IMU has units but forgets where it is within a second. Fused, the accelerometer lends
the camera its metre and the camera stops the IMU's drift. That trade is the whole system.

---

## 3. Why one camera has no ruler — the scale ambiguity

This is the single fact that makes *monocular* VIO hard, so it is worth seeing exactly.

A pinhole camera maps a 3-D point $\mathbf{P}$ to a pixel by projecting through the optical centre:
the pixel depends only on the **direction** to $\mathbf{P}$, not its distance. Formally, for any
scalar $s>0$,

$$
\pi(\mathbf{P}) = \pi(s\,\mathbf{P}).
$$

So take a whole scene, the camera trajectory through it, and multiply *every* 3-D coordinate and
every camera position by the same $s$. Every point still projects to exactly the same pixel in
every frame. **The images cannot tell the two scenes apart.** A dollhouse filmed up close and a
real house filmed from far away produce identical video.

`★ Insight ─────────────────────────────────────`
- Monocular structure-from-motion recovers the scene **up to an unknown scale** — the shape is
  right, the size is a free multiplier. This is not a weakness of a particular algorithm; it is
  information that is *physically absent from the pixels*.
- The accelerometer breaks the tie because it measures force in real m/s². Integrated twice, it
  says "the camera moved 0.47 **metres**" while vision says "the camera moved 1.0 **baselines**."
  One division fixes the ruler. [Module 7](07-metric-initialization.md) is that division, done
  carefully.
- The catch: that division only works when the accelerometer's signal rises above its noise and
  bias — i.e. when the platform actually accelerates. A constant-velocity fly-through is scale-blind
  even *with* an IMU. This is the **scale-observability** problem, and it recurs through the course.
`─────────────────────────────────────────────────`

---

## 4. The shape of the whole system

Everything downstream is one of three jobs:

1. **Front end** — turn images into correspondences (which pixel here is which pixel there) and IMU
   samples into motion constraints. Modules [3](03-camera.md) and [4](04-imu-preintegration.md).
2. **Bootstrap** — from nothing, produce a *metric* starting state: shape (up to scale), then the
   metre, then gravity and velocity. A one-time gate. Modules [6](06-epipolar-and-sfm.md) and
   [7](07-metric-initialization.md).
3. **Tracking** — per frame, fold the camera and IMU constraints into one optimization and update
   the state. Modules [2](02-least-squares.md) and [5](05-tight-coupling.md).

The optimization in (3) is where the fusion actually happens, and it is the same Gauss-Newton on a
manifold that runs everywhere in the codebase — so that is what we build first, starting with the
manifold itself.

---

→ Next: **[Module 1 — State on a manifold](01-manifolds.md)** · [Syllabus](README.md)
