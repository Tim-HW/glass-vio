#ifndef GLASSVIO_SFM_BUNDLE_HPP
#define GLASSVIO_SFM_BUNDLE_HPP

#include <vector>

#include "glassvio/camera_calib.hpp"
#include "glassvio/sfm_window.hpp"

namespace glassvio
{

/// What bundleAdjust did -- median reprojection error before and after, over the observations
/// it used.
struct SfmBundleStats
{
  bool ran = false;
  int observations = 0;
  int iterations = 0;
  double median_px_before = 0.0;
  double median_px_after = 0.0;
};

/// BUNDLE-ADJUST THE BOOTSTRAP RECONSTRUCTION -- VINS-Fusion's GlobalSFM Ceres step, before the
/// alignment ever sees it.
///
/// WHY. buildSfmWindow invents the ruler at one base pair and PROPAGATES it frame by frame with
/// PnP against that pair's landmarks alone. Each PnP pose inherits the landmarks' error and adds
/// its own, nothing ever reconciles them, and stage [4] then asks one metric trajectory to fit
/// every translation at once. Measured on V1_02/V1_03 (doc/08 §6): with ground-truth POSITIONS
/// in the same ruler the alignment bootstraps on the first window it sees; with these it solves
/// s ~0.02x the truth for 30-40 s.
///
/// WHAT. Every posed frame and every landmark, jointly, on reprojection error alone (Huber).
/// The GAUGE: a reconstruction is fixed only up to a similarity (7 DoF), so the base frame and
/// its base-pair partner stay FIXED -- which also keeps |t| = 1 at the pair, i.e. the ruler the
/// alignment's s is measured in. VINS-Fusion fixes the same two frames. With fix_pair_frame false only
/// the base is fixed; the pair frame moves, and the solution is rescaled afterwards so its distance from
/// the base is unchanged -- same ruler, but the essential matrix's baseline DIRECTION can be corrected.
///
/// HOW. Levenberg-Marquardt with each landmark eliminated by its own 3x3 Schur block, the same
/// structure as KeyframeWindow's. Cameras reuse the reprojection factor through a NavState whose
/// "IMU" is the camera itself (identity extrinsic) -- R = R_c0_ck, p = camera centre -- so the
/// residual and its right-perturbation Jacobians are the ones test_reprojection pins.
SfmBundleStats bundleAdjust(
  const std::vector<SfmFrame> & frames, SfmWindow & w, const CameraCalib & calib,
  int max_iterations = 10, double huber_delta_px = 2.0, bool fix_pair_frame = true);

}  // namespace glassvio

#endif  // GLASSVIO_SFM_BUNDLE_HPP
