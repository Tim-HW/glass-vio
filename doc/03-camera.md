# Module 3 — The camera

> **Prerequisite:** [Modules 1–2](01-manifolds.md). **After this you can:** turn a video stream
> into persistent correspondences, project a 3-D landmark to a pixel, and derive the reprojection
> residual's Jacobian — the camera's contribution to the fused solve.

The camera's job in the fusion is one residual: *a known 3-D landmark, carried through the current
pose, should land on the pixel the tracker saw it at.* Getting there takes a front end (which pixel
is which landmark) and a projection model (where a landmark *should* appear). This module is both,
ending at the Jacobian [Module 5](05-tight-coupling.md) will stack.

Code: [`feature_tracker.hpp`](../include/glassvio/feature_tracker.hpp) (front end),
[`camera_calib.hpp`](../include/glassvio/camera_calib.hpp) (projection, undistortion),
[`reprojection.hpp`](../include/glassvio/reprojection.hpp) (the residual + Jacobian).
Checks: [`test_feature_tracker.cpp`](../test/test_feature_tracker.cpp),
[`test_reprojection.cpp`](../test/test_reprojection.cpp).

---

## 1. The front end — persistent correspondences

The back end can only triangulate or reproject a landmark if it knows *the pixel here in frame N is
the same world point as the pixel there in frame N+1*. That identity is the front end's whole
output: per frame, a list of pixel locations, each with a stable `long` **id**.

The tracker is two branches (`track()`):

- **Flow.** `calcOpticalFlowPyrLK` follows each previous corner into the new frame (3-level pyramid,
  21×21 window). Survivors keep their id; a track failing the error or border check is dropped. This
  is a *local* search: it assumes the corner moved a little and its brightness did not change — which
  is why the tracker must run where no frames are dropped (a skipped frame doubles the flow and
  breaks the assumption; see [pipeline.md#threading](pipeline.md)).
- **Top up.** When survivors fall below `min_features`, `detectInto` runs FAST, sorts by response,
  and appends the strongest corners ≥ `kMinSpacing` from an existing feature (each gets a fresh id).
  The spacing rule stops FAST stacking corners on one edge and leaving regions unwatched.

The [self-check](../test/test_feature_tracker.cpp) shifts a fixed noise field a few pixels and
asserts most ids survive the flow (843/883 in the current build) and are unique per frame — the one
property the back end depends on.

---

## 2. The pinhole model and undistortion

A pinhole camera projects a point $\mathbf{P}_c = (X,Y,Z)$ in the *camera* frame to a pixel by the
intrinsics $\mathbf{K}$:

$$
\pi(\mathbf{P}_c) = \Big(f_x \tfrac{X}{Z} + c_x,\; f_y \tfrac{Y}{Z} + c_y\Big).
$$

The division by $Z$ is the projection — and the source of both the scale ambiguity
([Module 0](00-the-problem.md)) and the nonlinearity the Jacobian must handle (§4).

Real lenses are not pinholes. EuRoC ships **raw** frames with a radial-tangential distortion (a
strong barrel, $k_1=-0.283$), so the geometry only works after undistortion. It is undone at the
tracker→estimator boundary, **on the points, not the image**:

```cpp
r.points = calib_.undistort(r.points);    // cv::undistortPoints with P = K
```

`★ Insight — undistort points, not pixels ──────`
KLT is a local search and distortion is a smooth warp, so tracking on the *raw* frame is correct and
cheap; only the downstream geometry needs true rays. Undistorting ~200 points beats remapping a
752×480 image every frame. With `P = K` the output is pixels in the same `K`, so every stage
downstream treats the camera as a clean pinhole and never learns distortion existed — and on a
rectified stream `undistort()` is a no-op, so callers never ask which dataset they have. (A corner
at (10,10) moves 155 px under this correction; the principal point moves 0.)
`─────────────────────────────────────────────────`

**The top-up trap (a real, measured bug).** `detectInto` *appends*, so it is tempting to top up to
the max every frame. That was tried and it **broke structure-from-motion**: `sfm_check`'s rigidity
residual went 2.64% → 76% and the bootstrap reconstructed the room at ~3 cm instead of metres. So
top-up stays gated below `min_features`. The cost — the map can go short of *young* tracks in fast
motion — is a separate problem with a separate fix ([Module 8](08-sliding-window.md), Stage C). The
mechanism is still open; the result is deterministic. A reminder that in this domain, a plausible
change can silently corrupt a stage two hops away.

---

## 3. The reprojection residual

Now the fusion term. A landmark $\mathbf{P}_w$ (world frame, held **fixed** during the solve —
[Module 5](05-tight-coupling.md) explains why) is carried into the camera through the current pose
$(\mathbf{R},\mathbf{p})$ and the fixed extrinsic $(\mathbf{R}_{ci},\mathbf{t}_{ci})$, then projected:

$$
\mathbf{P}_i = \mathbf{R}^\top(\mathbf{P}_w - \mathbf{p}), \qquad
\mathbf{P}_c = \mathbf{R}_{ci}\mathbf{P}_i + \mathbf{t}_{ci}, \qquad
\boxed{\;\mathbf{r}_{\text{px}} = \pi(\mathbf{P}_c) - \mathbf{z}_{\text{obs}}\;}
$$

Two scalars per landmark: predicted pixel minus observed pixel. Zero when the pose puts the landmark
exactly where the tracker saw it.

---

## 4. The reprojection Jacobian — a three-link chain

This is the derivative $\mathbf{J}_{\text{px}} = \partial\mathbf{r}_{\text{px}}/\partial\delta\mathbf{x}$
([Module 2](02-least-squares.md)), through the `⊞` retraction ([Module 1](01-manifolds.md)). It
factors by the chain rule along $\delta\mathbf{x}\to\mathbf{P}_i\to\mathbf{P}_c\to\pi$:

$$
\mathbf{J}_{\text{px}}
= \underbrace{\frac{\partial\pi}{\partial\mathbf{P}_c}}_{\mathbf{J}_\pi,\ 2\times3}
\cdot \underbrace{\frac{\partial\mathbf{P}_c}{\partial\mathbf{P}_i}}_{\mathbf{R}_{ci}}
\cdot \frac{\partial\mathbf{P}_i}{\partial\delta\mathbf{x}}.
$$

### Link 1 — the projection derivative $\mathbf{J}_\pi$ ($2\times3$)

Write $\pi$ one component at a time, $u = f_x X/Z + c_x$ and $v = f_y Y/Z + c_y$, and differentiate
each with respect to $X$, $Y$, $Z$ (quotient rule on the $Z$ column):

$$
\frac{\partial u}{\partial X} = \frac{f_x}{Z},\quad
\frac{\partial u}{\partial Y} = 0,\quad
\frac{\partial u}{\partial Z} = -\frac{f_x X}{Z^2},
\qquad
\frac{\partial v}{\partial X} = 0,\quad
\frac{\partial v}{\partial Y} = \frac{f_y}{Z},\quad
\frac{\partial v}{\partial Z} = -\frac{f_y Y}{Z^2}.
$$

Stacked, one row per pixel coordinate:

$$
\mathbf{J}_\pi = \frac{\partial\pi}{\partial\mathbf{P}_c}
= \begin{bmatrix} f_x/Z & 0 & -f_x X/Z^2 \\ 0 & f_y/Z & -f_y Y/Z^2 \end{bmatrix}
\quad [\text{px/m}].
$$

The principal point $c_x, c_y$ is a constant offset and drops out. Every entry carries $1/Z$: the
same sideways shift moves a point twice as far away by half as many pixels. The $Z$ column is the
nonlinearity — the $1/Z^2$ terms are why Gauss-Newton has to iterate, and why $Z\to0$ is refused
(see *cheirality* below).

One direction produces **no** pixel change at all:
$\mathbf{J}_\pi\mathbf{P}_c = \big(f_xX/Z - f_xXZ/Z^2,\ f_yY/Z - f_yYZ/Z^2\big) = \mathbf{0}$.
Sliding a point along its own viewing ray leaves its pixel where it was — depth is invisible to a
single observation.

### Link 2 — the extrinsic $\partial\mathbf{P}_c/\partial\mathbf{P}_i$ ($3\times3$)

$\mathbf{P}_c = \mathbf{R}_{ci}\mathbf{P}_i + \mathbf{t}_{ci}$ is affine in $\mathbf{P}_i$, and the
extrinsic is calibrated and fixed, so the derivative is just the rotation:

$$
\frac{\partial\mathbf{P}_c}{\partial\mathbf{P}_i} = \mathbf{R}_{ci}.
$$

$\mathbf{t}_{ci}$ vanishes from the *derivative* but not from the *geometry*: it is already inside
$\mathbf{P}_c$ (so inside $\mathbf{J}_\pi$'s $X,Y,Z$), and the IMU→camera lever arm reaches the
rotation block through $\mathbf{P}_i$ in Link 3 — which is measured from the IMU origin, not the
camera's. Rotating the body swings the camera around on that arm, and $\widehat{\mathbf{P}_i}$
(not $\widehat{\mathbf{P}_c}$) is what accounts for it.

### Link 3 — the pose $\partial\mathbf{P}_i/\partial\delta\mathbf{x}$ ($3\times15$): the manifold enters here, at exactly one place

Perturb the pose with `⊞` — right convention, $\mathbf{R}\leftarrow\mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol{\phi})$,
$\mathbf{p}\leftarrow\mathbf{p}+\delta\mathbf{p}$ ([Module 1](01-manifolds.md) §4) — and
differentiate $\mathbf{P}_i = \mathbf{R}^\top(\mathbf{P}_w-\mathbf{p})$ one block at a time.

**Rotation.** $\mathbf{P}_i$ contains $\mathbf{R}^\top$, not $\mathbf{R}$, so first see what the
step does to the transpose. With $\mathrm{Exp}(\boldsymbol\phi)^\top = \mathrm{Exp}(-\boldsymbol\phi)$
and the first-order $\mathrm{Exp}(\boldsymbol\phi)\approx\mathbf{I}+\widehat{\boldsymbol\phi}$:

$$
\mathbf{R}^\top \;\leftarrow\; \big(\mathbf{R}\,\mathrm{Exp}(\delta\boldsymbol\phi)\big)^\top
= \mathrm{Exp}(-\delta\boldsymbol\phi)\,\mathbf{R}^\top
\approx (\mathbf{I}-\widehat{\delta\boldsymbol\phi})\,\mathbf{R}^\top .
$$

The step sits on the right of $\mathbf{R}$, so it lands on the *left* of $\mathbf{R}^\top$ — directly
against $\mathbf{P}_i$:

$$
\mathbf{P}_i' = (\mathbf{I}-\widehat{\delta\boldsymbol\phi})\,\mathbf{R}^\top(\mathbf{P}_w-\mathbf{p})
= \mathbf{P}_i - \delta\boldsymbol\phi\times\mathbf{P}_i
= \mathbf{P}_i + \mathbf{P}_i\times\delta\boldsymbol\phi
= \mathbf{P}_i + \widehat{\mathbf{P}_i}\,\delta\boldsymbol\phi
\;\;\Rightarrow\;\;
\boxed{\frac{\partial\mathbf{P}_i}{\partial\delta\boldsymbol{\phi}} = \widehat{\mathbf{P}_i}}
$$

The swap $\mathbf{a}\times\mathbf{b} = -\mathbf{b}\times\mathbf{a}$ is the whole trick: it moves
$\delta\boldsymbol\phi$ to the right of a matrix so the derivative can be read off. Sanity check:
$\widehat{\mathbf{P}_i}\,\mathbf{P}_i = \mathbf{0}$ — rotating about the axis from the IMU origin
through the landmark leaves the landmark where it was.

**Position.** `⊞` adds $\delta\mathbf{p}$ in the *world* frame, no `Exp`:

$$
\mathbf{P}_i' = \mathbf{R}^\top(\mathbf{P}_w - \mathbf{p} - \delta\mathbf{p})
= \mathbf{P}_i - \mathbf{R}^\top\delta\mathbf{p}
\;\;\Rightarrow\;\;
\boxed{\frac{\partial\mathbf{P}_i}{\partial\delta\mathbf{p}} = -\mathbf{R}^\top}
$$

$\mathbf{R}^\top$ re-expresses the world-frame step in the body frame; the minus sign is relative
motion — the body steps forward, the landmark slides backward in its view.

**Velocity and biases.** $\mathbf{P}_i$ does not contain $\mathbf{v}$, $\mathbf{b}_g$ or
$\mathbf{b}_a$, so those nine columns are exactly zero.

### Assembled

Multiply the three links. The $2\times15$ Jacobian has non-zero blocks **only** at the rotation and
position columns:

$$
\frac{\partial\mathbf{r}_{\text{px}}}{\partial\delta\boldsymbol\phi}
= \mathbf{J}_\pi\,\mathbf{R}_{ci}\,\widehat{\mathbf{P}_i}\quad[\text{px/rad}],
\qquad
\frac{\partial\mathbf{r}_{\text{px}}}{\partial\delta\mathbf{p}}
= -\mathbf{J}_\pi\,\mathbf{R}_{ci}\,\mathbf{R}^\top\quad[\text{px/m}].
$$

```cpp
J.block<2,3>(0, kIdxPhi) =  J_pi * R_ci * Sophus::SO3d::hat(P_i);
J.block<2,3>(0, kIdxPos) = -J_pi * R_ci * R_iw;
// kIdxVel, kIdxBg, kIdxBa: left ZERO — a camera cannot see them.
```

`★ Insight — those nine zeros are physics ──────`
The velocity and bias columns are structurally zero because a single image observes *pose*, never
velocity or bias. The IMU Jacobian ([Module 4](04-imu-preintegration.md)) is the mirror image —
dense across velocity and both biases, blind to nothing it moves. That asymmetry is *why the two
sensors are complementary*, and why loose coupling (which cannot let the IMU write into a shared
$\mathbf{H}$) could never estimate the biases at all.
`─────────────────────────────────────────────────`

`★ Insight — far points fix rotation, near points fix translation ──`
Read the $1/Z$ scaling off the two blocks. The position block is $\mathbf{J}_\pi$ times rotations,
so it shrinks like $1/Z$: with EuRoC's $f_x = 458.654$, a 1 cm sideways step moves a landmark at
5 m by ≈0.9 px and one at 50 m by ≈0.09 px — below the tracker's noise. The rotation block has
$\widehat{\mathbf{P}_i}$, which *grows* like $Z$ and cancels the $1/Z$: 1 mrad moves a pixel by
≈0.46 px at any depth. A frame that sees only distant structure knows its attitude well and its
position poorly.
`─────────────────────────────────────────────────`

### The landmark block — for when landmarks are free ([Module 8](08-sliding-window.md))

The tracker holds $\mathbf{P}_w$ fixed; the sliding window solves for it too, so it also needs
$\partial\mathbf{r}_{\text{px}}/\partial\mathbf{P}_w$. No new derivation: $\mathbf{P}_w$ is a
plain vector ($\mathbf{P}_w\leftarrow\mathbf{P}_w+\delta\mathbf{P}_w$) and enters $\mathbf{P}_i$
only through the difference $\mathbf{P}_w-\mathbf{p}$, so

$$
\frac{\partial\mathbf{P}_i}{\partial\mathbf{P}_w} = +\mathbf{R}^\top = -\frac{\partial\mathbf{P}_i}{\partial\delta\mathbf{p}}
\;\;\Rightarrow\;\;
\mathbf{J}_l = -\frac{\partial\mathbf{r}_{\text{px}}}{\partial\delta\mathbf{p}} .
$$

Moving the landmark is moving the camera the other way. The code is one negation
(`reprojectionLandmarkJacobian` returns `-J_pose.block<2,3>(0, kIdxPos)`), pinned separately in
the same test — a sign slip in a copied block is the cheapest bug there is. And Link 1's null
direction reappears here: one observation constrains only two of $\mathbf{P}_w$'s three
coordinates, which is why a free landmark needs a second view with real baseline
([Module 6](06-epipolar-and-sfm.md)).

**One guard the LiDAR factor never needed — cheirality.** $\pi$ divides by $Z$. A landmark *behind*
the camera ($Z<0$) projects to an ordinary-looking pixel with a **sign-flipped** Jacobian — no NaN,
just a confident step the wrong way. So a landmark with $Z<Z_{\min}$ is refused before it enters
$\mathbf{H}$.

---

## Lab — pin the Jacobian to finite differences, then break it

The analytic Jacobian above claims to be $\partial\mathbf{r}/\partial\delta\mathbf{x}$ under the
right retraction. `test_reprojection` verifies it against finite differences taken *through*
`boxplus` — to **3e-10**.

```bash
colcon build --packages-select glassvio
colcon test --packages-select glassvio --ctest-args -R reprojection ; colcon test-result --verbose
```

1. **Confirm it passes** — analytic and numeric agree to 3e-10.
2. **Break it.** In [`reprojection.hpp`](../include/glassvio/reprojection.hpp) flip the sign of the
   `kIdxPos` block (`+J_pi*...` instead of `-`). Re-run: the pin fails. But if instead you fed this
   flipped Jacobian to the *full estimator* with no pin, it would still converge to a plausible,
   wrong trajectory — which is the entire reason this test exists. Revert.

---

← [Module 2](02-least-squares.md) · [Syllabus](README.md) · Next: **[Module 4 — The IMU](04-imu-preintegration.md)** →
