# Module 6 — Epipolar geometry & structure from motion

> **Prerequisite:** [Modules 1, 3](01-manifolds.md). **After this you can:** relate two views by the
> essential matrix, recover their relative pose *up to scale*, triangulate landmarks, and explain why
> the reconstruction has an invented ruler.

[Module 0](00-the-problem.md) said a monocular camera has no scale. Now we confront it head-on: to
*start* the estimator we need landmarks and a pose from vision alone, before the IMU has said
anything. That is structure from motion (SfM), and it recovers everything but the ruler — which
[Module 7](07-metric-initialization.md) then supplies. This module is the geometry SfM stands on.

Code: [`sfm_window.hpp`](../include/glassvio/sfm_window.hpp),
[`vio_initializer.cpp`](../src/vio/vio_initializer.cpp). Checks:
[`epipolar_check.cpp`](../src/checks/epipolar_check.cpp),
[`sfm_check.cpp`](../src/checks/sfm_check.cpp).

---

## 1. The epipolar constraint — two views of one point

A world point seen in two camera frames obeys a rigid rule. Let $\mathbf{x}_1,\mathbf{x}_2$ be the
same point's pixels in views 1 and 2, and let the second camera be $(\mathbf{R},\mathbf{t})$ relative
to the first. The point, its two projection rays, and the baseline all lie in one plane — the
**epipolar plane** — and that coplanarity is a single scalar equation.

In **normalized** coordinates $\bar{\mathbf{x}} = \mathbf{K}^{-1}\mathbf{x}$ (rays, intrinsics
removed), it is the **essential matrix** $\mathbf{E}$:

$$
\bar{\mathbf{x}}_2^\top\,\mathbf{E}\,\bar{\mathbf{x}}_1 = 0, \qquad
\mathbf{E} = [\mathbf{t}]_\times\,\mathbf{R},
$$

where $[\mathbf{t}]_\times = \widehat{\mathbf{t}}$ is [Module 1](01-manifolds.md)'s `hat`. In raw
**pixels** it is the **fundamental matrix** $\mathbf{F}$, which folds the intrinsics back in:

$$
\mathbf{x}_2^\top\,\mathbf{F}\,\mathbf{x}_1 = 0, \qquad
\boxed{\;\mathbf{F} = \mathbf{K}^{-\top}\,[\mathbf{t}]_\times\mathbf{R}\,\mathbf{K}^{-1}\;}
$$

Geometrically, $\mathbf{F}\mathbf{x}_1$ is the **epipolar line** in image 2 on which $\mathbf{x}_2$
must lie. This is the exact formula the lab builds, and it carries the crucial dependence: **$\mathbf{F}$
scales with $\lVert\mathbf{t}\rVert$**, so as the baseline shrinks to zero, $\mathbf{F}\to0$ and the
constraint says nothing — the fact that makes the [Lab](#lab) below non-trivial to get right.

---

## 2. From correspondences to relative pose (up to scale)

Given ≥5–8 correspondences you can *solve* for $\mathbf{E}$ (the five-point or normalized eight-point
algorithm — a linear system in the entries of $\mathbf{E}$, refined). Then decompose it:

$$
\mathbf{E} = [\mathbf{t}]_\times\mathbf{R} \;\xrightarrow{\text{SVD}}\; (\mathbf{R},\mathbf{t}).
$$

The SVD of $\mathbf{E}$ yields **four** candidate $(\mathbf{R},\mathbf{t})$ pairs (a rotation
ambiguity × a sign of $\mathbf{t}$); the correct one is picked by **cheirality** — only one puts the
triangulated points *in front of both cameras*. OpenCV's `recoverPose` does exactly this. In code,
this is `cv::findEssentialMat` → `cv::recoverPose`.

`★ Insight — the ruler is invented right here ──`
$\mathbf{E} = [\mathbf{t}]_\times\mathbf{R}$ is **homogeneous** in $\mathbf{t}$: scaling $\mathbf{t}$
by any $s>0$ scales $\mathbf{E}$ by $s$ but satisfies the *same* constraint. So the decomposition can
only ever return $\mathbf{t}$ as a **unit vector** — a direction, never a length. `recoverPose`
returns $\lVert\mathbf{t}\rVert=1$ *by fiat*: "the camera moved one baseline." That is the scale
ambiguity of [Module 0](00-the-problem.md), made concrete. Everything triangulated from this pose is
in units of *that invented baseline*, until [Module 7](07-metric-initialization.md) tells us what a
baseline is worth in metres.
`─────────────────────────────────────────────────`

---

## 3. Triangulation — landmarks in the invented ruler

With the (up-to-scale) relative pose known, each correspondence's 3-D point is recovered by
**triangulation**: each view gives two linear equations in the point's homogeneous coordinates
$\mathbf{X}=(X,Y,Z,W)$ (the projection $\mathbf{x}\times(\mathbf{P}\mathbf{X})=0$ for camera matrix
$\mathbf{P}=\mathbf{K}[\mathbf{R}\,|\,\mathbf{t}]$), stacked into a $4\times4$ system solved by SVD —
the **Direct Linear Transform**. The point is the null-space vector, dehomogenized by $/W$.

Two failure modes to respect (both are guarded in the code):

- **Cheirality:** a solution with $Z<0$ is behind the camera — rejected.
- **Small parallax:** if the two rays are nearly parallel (baseline tiny, or the point far away), $W\to0$
  and the depth is numerically hopeless. Triangulation is gated on parallax **angle** (≥1°), *not*
  pixel flow — a rotating camera produces plenty of flow with zero baseline, and only the angle catches
  it. (Watch for `cv::triangulatePoints` returning `CV_32F` for `Point2f` input; reading it as `double`
  yields fake points-at-infinity.)

---

## 4. Propagating scale across the window — PnP, not chained essentials

SfM runs over a small window of frames, not just a pair. You cannot *chain* essential matrices —
each pair invents its **own** arbitrary $\lVert\mathbf{t}\rVert$, so the units would reset every step.
Instead:

1. Pick a **base pair** with real baseline, run §2–3 → landmarks in the invented ruler.
2. For every other frame, run **PnP** (`solvePnP`) against those landmarks → its pose *in the same
   ruler*. **PnP is what carries the scale**: it consumes points that already have the ruler and
   returns a pose in the same units.

The base pair is chosen the way VINS-Mono does — parallax as a *filter*, geometry as the *decision*:
a candidate clearing the pixel-flow threshold is still tried end-to-end (essential → recoverPose →
triangulate) and accepted only if enough landmarks survive cheirality and the 1° gate. `min_landmarks`
guards the reconstruction's density.

---

## Lab — make the calibration gate actually gate

`epipolar_check` is the oracle for §1: ground truth gives the true relative camera pose, §1's formula
turns it into $\mathbf{F}$, and a *correct* calibration puts every real KLT track on its epipolar line
(low Sampson distance). It tests the whole chain — $\mathbf{K}$, both extrinsics, the composition
order — with no estimator in the loop.

```bash
colcon build --packages-select glassvio
./build/glassvio/epipolar_check     # from the glassvio dir (needs config/ and data/)
```

1. **Read the result** — 58 frame pairs, median Sampson ~1.9 px < 3.0 px gate → ok. The tracks obey
   the epipolar geometry the calibration predicts.
2. **The subtlety this lab was built to teach.** $\mathbf{F}$ scales with $\lVert\mathbf{t}\rVert$
   (§1), so if you test *adjacent* frames (~1 cm baseline) the Sampson numerator *and* denominator
   both collapse toward 0 and the metric reads "perfect" no matter whether the calibration is right —
   the test passes by measuring nothing. The check pairs frames `kGap=10` apart (~0.25 m) precisely to
   give $\mathbf{F}$ leverage. **Set `kGap = 1`, rebuild, re-run:** the median drops toward a vacuous
   ~0, "passing" a test that now proves nothing. Restore `kGap = 10`. A gate is only as honest as the
   leverage behind it.
3. **`sfm_check`** scores the reconstruction's rigidity directly — run it to see the up-to-scale
   structure hold together (and recall the [Module 3](03-camera.md) top-up trap sent it 2.64% → 76%).

You now have shape, in an invented ruler. Next: the metre.

---

← [Module 5](05-tight-coupling.md) · [Syllabus](README.md) · Next: **[Module 7 — Metric initialization](07-metric-initialization.md)** →
