#ifndef GLASSVIO_SFM_WINDOW_HPP
#define GLASSVIO_SFM_WINDOW_HPP

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include "sophus/so3.hpp"

#include "glassvio/camera_calib.hpp"

namespace glassvio
{

/// One tracked image: the feature positions, keyed by the persistent track id that makes
/// them correspondences.
struct SfmFrame
{
  double t = 0.0;
  std::unordered_map<long, cv::Point2f> by_id;
};

struct SfmParams
{
  /// A FILTER, not a decision: pixel flow does not imply baseline (a rotating camera produces
  /// plenty with none). Candidates clearing it are still tried end-to-end and may be
  /// rejected. VINS-Mono's equivalent is `average_parallax * 460 > 30`.
  double min_parallax_px = 18.0;
  /// Ray angle below which a landmark's depth is noise. 1.0 is what ORB-SLAM3 hardcodes at
  /// its Reconstruct() call site; it is not a knob to loosen when baselines get small.
  double min_parallax_deg = 1.0;
  double min_depth = 0.1;           ///< cheirality guard, in ruler units
  int min_pnp_points = 12;
  int min_shared = 60;
  /// recoverPose inliers a base pair must yield. VINS-Mono's solveRelativeRT requires >12.
  int min_inliers = 15;
  /// Landmarks a base pair must produce to be accepted. Below this, keep scanning -- and if
  /// nothing in the window clears it, FAIL rather than return a reconstruction that is
  /// geometrically hopeless.
  ///
  /// 40 was KITTI-tuned and is dangerously permissive anywhere else. KITTI passed with 58
  /// landmarks and a 1.06% residual -- because its baseline was 2.4 m. EuRoC produces 61
  /// landmarks over a 5 cm baseline and a 62% residual. The COUNT was never the quality
  /// signal; the baseline-to-depth ratio was, and KITTI's metre-scale motion made the two
  /// look like the same thing. Measured on V1_01_easy: >=139 landmarks -> residual <7%;
  /// <=61 -> residual >49%. There is no middle.
  int min_landmarks = 100;
  /// After PnP, triangulate every other track the window's posed frames share -- from the first
  /// and last frame that see it -- as VINS-Fusion's GlobalSFM does before its bundle adjustment.
  /// Without it the reconstruction is the base pair's landmarks alone. Kept only through the same
  /// parallax and cheirality gates, and if it reprojects within max_window_reproj_px in EVERY
  /// posed view. Off: measured, it raised the collapsed scales a little but pushed V1_02's
  /// bootstrap from 16.9 to 32.7 s and its ATE from 0.16 to 0.38 m (doc/08 §6). (--sfm-tri)
  bool triangulate_window = false;
  double max_window_reproj_px = 3.0;
};

/// An up-to-scale reconstruction: poses and structure in a world stretched by one unknown
/// factor. See sfm_check.cpp's header for why that factor cannot be recovered here.
struct SfmWindow
{
  bool valid = false;
  int base = -1;                    ///< index of the reference frame (its pose is identity)
  int second = -1;                  ///< the other half of the base pair, chosen on parallax
  double base_parallax_px = 0.0;
  int inliers = 0;
  int rejected_parallax = 0;
  int rejected_cheirality = 0;
  /// Base pairs that cleared the pixel-flow filter and were tried end-to-end. More than one
  /// means candidates were rejected by the GEOMETRY -- i.e. they were rotation-dominated,
  /// which is the failure a flow threshold alone cannot see.
  int candidates_tried = 0;
  /// The most tracks any frame shared with the base (against min_shared), and the most
  /// landmarks any tried candidate triangulated (against min_landmarks) -- which gate a failed
  /// window fell short of, and by how much.
  int max_shared = 0;
  int max_trial_landmarks = 0;
  int window_triangulated = 0;   ///< landmarks added beyond the base pair (triangulate_window)

  /// X_ck = pose[k] * X_c0. Translations are in RULER units, not metres.
  std::unordered_map<int, Eigen::Isometry3d> pose;
  /// Landmarks in the base camera's frame, likewise in ruler units.
  std::unordered_map<long, Eigen::Vector3d> landmark;
};

/// Triangulate one candidate base pair into `out`, gated on cheirality and parallax ANGLE.
///
/// Separate from buildSfmWindow because it is now run PER CANDIDATE: a pair that clears the
/// pixel-flow filter can still be rotation-dominated and produce nothing, and the only way to
/// find out is to try. This is the expensive half of VINS-Mono's
/// `parallax > thresh && solveRelativeRT(...)`.
void triangulateBasePair(
  const std::vector<cv::Point2f> & p1, const std::vector<cv::Point2f> & p2,
  const std::vector<long> & ids, const Eigen::Isometry3d & T2, const cv::Mat & mask,
  const CameraCalib & calib, const SfmParams & p, SfmWindow & out);

/// The relative BODY rotation between two frames, from vision alone.
///
/// The essential matrix's rotation is the one thing two views give away for free: it needs no
/// scale, no triangulation, and no baseline magnitude. Measured against ground truth on the
/// KITTI bag it is good to 0.05-0.15 deg, which is why gyro-bias estimation can run before
/// anything else in the bootstrap.
///
/// Returns false if there are too few correspondences or the essential matrix degenerates.
bool relativeBodyRotation(
  const SfmFrame & a, const SfmFrame & b, const CameraCalib & calib,
  Sophus::SO3d & out, int min_correspondences = 60);

/// Reconstruct frames [begin, end) up to scale.
///
/// The ruler is invented at the base pair (|t| = 1 from the essential matrix) and then
/// propagated by PnP, which consumes landmarks already carrying it. Nothing here knows a
/// metre; that is phase 2b-3's job.
SfmWindow buildSfmWindow(
  const std::vector<SfmFrame> & frames, int begin, int end,
  const CameraCalib & calib, const SfmParams & p = {});

}  // namespace glassvio

#endif  // GLASSVIO_SFM_WINDOW_HPP
