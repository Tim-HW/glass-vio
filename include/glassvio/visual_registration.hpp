#ifndef GLASSVIO_VISUAL_REGISTRATION_HPP
#define GLASSVIO_VISUAL_REGISTRATION_HPP

#include <unordered_map>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "glass_core/nav_residual.hpp"
#include "glass_core/nav_state.hpp"
#include "glass_core/preintegration.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/reprojection.hpp"

namespace glassvio
{

using namespace glass_core;  // NOLINT(build/namespaces)

struct VisualParams
{
  int max_iterations = 30;
  double eps_translation = 1e-3;   ///< m
  double eps_rotation = 1e-4;      ///< rad
  ReprojectionParams reproj;
  /// Below this many landmarks in view, the 15-DoF problem is under-constrained: refuse
  /// rather than invent a state. This is also the STARVATION signal -- with landmarks held
  /// fixed, tracks die as the camera flies past them and eventually this fires.
  int min_features = 12;
  /// Scales the IMU's own information. 0 would make this pure vision -- useful for measuring
  /// what the IMU actually buys, which is the only honest way to justify tight coupling.
  double imu_prior_weight = 1.0;
  /// Pixel error at or below which an observation AGREES with the solved pose -- the map's own
  /// outlier bound (LandmarkMapParams::max_reprojection_px). Counted into VisualResult::inliers.
  double inlier_px = 8.0;
  /// A solve is REFUSED when fewer than this fraction of its observations agree with it (within
  /// inlier_px) -- ORB-SLAM3's inlier count after pose optimization, as a fraction. Measured on
  /// EuRoC V1_01: healthy frames sit at a median of 0.99 and a 5th percentile of 0.85; the solve
  /// that started the 87.6 s divergence had 1 inlier of 94. 0 disables.
  double min_inlier_fraction = 0.5;
};

struct VisualResult
{
  /// Enough features and a finite solve: the state is usable. This -- NOT `converged` -- is
  /// the trust signal, exactly as in glasslio.
  bool valid = false;
  /// The increment fell below eps before max_iterations. Not a trust signal: the solver
  /// routinely plateaus above eps while sitting on a good fit.
  bool converged = false;
  int iterations = 0;
  int features = 0;         ///< landmarks that contributed (in view AND in front)
  int rejected_cheirality = 0;
  double rmse_px = 0.0;     ///< VISION-only, in pixels, so it is directly interpretable
  /// The fit at the FINAL state, robustly. rmse_px is a mean taken before the last step, so a
  /// few wild tracks read as a blow-up under a good pose; these say how much of the view agrees.
  double median_px = 0.0;
  int inliers = 0;          ///< observations within VisualParams::inlier_px
  /// The inlier gate refused it: it converged, but too little of the view agreed. Distinct from
  /// a STARVED solve (too few features), which is invalid for a different reason.
  bool refused = false;
  NavState state;
  Eigen::Matrix<double, kNavDim, kNavDim> H = Eigen::Matrix<double, kNavDim, kNavDim>::Zero();
};

/// THE TIGHTLY-COUPLED VISUAL SOLVE -- glassvio's alignTightlyCoupled.
///
/// One frame: fixed landmarks, a preintegrated IMU delta, and a 15-DoF state to move. Every
/// factor folds into ONE NormalEquationsN<15>, and that sum IS the sensor fusion -- there is
/// no filter, no blending coefficient, no second fusion step.
///
/// WHAT MAKES THIS SIMPLER THAN THE LIDAR PATH, not harder. glasslio re-runs closestPlane
/// INSIDE every iteration: that is ICP's outer loop, re-guessing correspondence from the
/// current pose because a laser return is anonymous. A tracked corner is not. The KLT id IS
/// the correspondence, so the association loop DELETES -- one less loop, no nearest-neighbour
/// search, and the cost function stops changing between iterations.
///
/// LANDMARKS ARE FIXED, in the world frame and in METRES. They play the role the LiDAR map's
/// planes play, which is exactly what keeps the state at 15 and lets NormalEquationsN<15> be
/// reused untouched. Putting them in the state is bundle adjustment: H grows to 15 + 3L and
/// the Schur complement stops being optional.
/// `P_i` is x_i's covariance -- and passing it is the whole difference between an estimator
/// that tracks and one that dead-reckons. See the note on Sigma_eff in the .cpp: with P_i = 0
/// this asserts the previous state was perfect, the IMU factor's information becomes
/// effectively infinite, and vision is outvoted. Measured on EuRoC: 643 px of reprojection
/// error with 135 landmarks in view.
VisualResult solveFrame(
  const std::unordered_map<long, Eigen::Vector3d> & landmarks,
  const std::unordered_map<long, cv::Point2f> & observations,
  const NavState & xi,
  const Eigen::Matrix<double, kNavDim, kNavDim> & P_i,
  const ImuPreintegration & pre,
  const Eigen::Vector3d & gravity,
  const NavState & guess,
  const Eigen::Matrix<double, 6, 6> & bias_information,
  const CameraCalib & calib,
  const VisualParams & params);

}  // namespace glassvio

#endif  // GLASSVIO_VISUAL_REGISTRATION_HPP
