#ifndef GLASSVIO_SFM_WINDOW_HPP
#define GLASSVIO_SFM_WINDOW_HPP

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

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

  /// X_ck = pose[k] * X_c0. Translations are in RULER units, not metres.
  std::unordered_map<int, Eigen::Isometry3d> pose;
  /// Landmarks in the base camera's frame, likewise in ruler units.
  std::unordered_map<long, Eigen::Vector3d> landmark;
};

namespace detail
{

inline Eigen::Matrix3d matFromCv(const cv::Mat & m)
{
  Eigen::Matrix3d out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = m.at<double>(i, j);
    }
  }
  return out;
}

inline std::vector<long> sharedIds(const SfmFrame & a, const SfmFrame & b)
{
  std::vector<long> ids;
  for (const auto & entry : a.by_id) {
    if (b.by_id.count(entry.first)) {
      ids.push_back(entry.first);
    }
  }
  return ids;
}

}  // namespace detail

/// Triangulate one candidate base pair into `out`, gated on cheirality and parallax ANGLE.
///
/// Separate from buildSfmWindow because it is now run PER CANDIDATE: a pair that clears the
/// pixel-flow filter can still be rotation-dominated and produce nothing, and the only way to
/// find out is to try. This is the expensive half of VINS-Mono's
/// `parallax > thresh && solveRelativeRT(...)`.
inline void triangulateBasePair(
  const std::vector<cv::Point2f> & p1, const std::vector<cv::Point2f> & p2,
  const std::vector<long> & ids, const Eigen::Isometry3d & T2, const cv::Mat & mask,
  const CameraCalib & calib, const SfmParams & p, SfmWindow & out)
{
  const cv::Mat K_cv = calib.cvK();
  const Eigen::Matrix3d Kinv = calib.K.inverse();

  cv::Mat P1 = cv::Mat::eye(3, 4, CV_64F);
  cv::Mat P2(3, 4, CV_64F);
  cv::Mat R_cv(3, 3, CV_64F), t_cv(3, 1, CV_64F);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R_cv.at<double>(i, j) = T2.linear()(i, j);
    }
    t_cv.at<double>(i) = T2.translation()(i);
  }
  R_cv.copyTo(P2(cv::Rect(0, 0, 3, 3)));
  t_cv.copyTo(P2(cv::Rect(3, 0, 1, 3)));
  P1 = K_cv * P1;
  P2 = K_cv * P2;

  cv::Mat X4f;
  cv::triangulatePoints(P1, P2, p1, p2, X4f);
  // triangulatePoints returns CV_32F for Point2f input. Reading that with at<double>() does
  // not throw -- it reinterprets two floats as one double and produces values that look
  // exactly like points at infinity. Convert; never assume.
  cv::Mat X4;
  X4f.convertTo(X4, CV_64F);

  for (int i = 0; i < X4.cols; ++i) {
    if (!mask.empty() && !mask.at<unsigned char>(i)) {
      continue;
    }
    const double hw = X4.at<double>(3, i);
    if (std::abs(hw) < 1e-12) {
      continue;
    }
    const Eigen::Vector3d X(
      X4.at<double>(0, i) / hw, X4.at<double>(1, i) / hw, X4.at<double>(2, i) / hw);

    // Cheirality: in front of BOTH cameras. A point behind still projects to a perfectly
    // ordinary-looking pixel -- the wrong one, with a sign-flipped Jacobian.
    if (X.z() < p.min_depth || (T2 * X).z() < p.min_depth) {
      ++out.rejected_cheirality;
      continue;
    }

    // Parallax ANGLE between the two viewing rays. THIS is what a pixel-flow threshold
    // cannot measure: rotation-induced flow clears the filter and dies right here, because
    // the rays stay parallel no matter how far the pixels moved.
    const Eigen::Vector3d r1 = (Kinv * Eigen::Vector3d(p1[i].x, p1[i].y, 1.0)).normalized();
    const Eigen::Vector3d r2 =
      (T2.linear().transpose() *
      (Kinv * Eigen::Vector3d(p2[i].x, p2[i].y, 1.0)).normalized()).normalized();
    const double ang = std::acos(std::clamp(r1.dot(r2), -1.0, 1.0)) * 180.0 / M_PI;
    if (ang < p.min_parallax_deg) {
      ++out.rejected_parallax;
      continue;
    }
    out.landmark.emplace(ids[i], X);
  }
}

/// The relative BODY rotation between two frames, from vision alone.
///
/// The essential matrix's rotation is the one thing two views give away for free: it needs no
/// scale, no triangulation, and no baseline magnitude. Measured against ground truth on the
/// KITTI bag it is good to 0.05-0.15 deg, which is why gyro-bias estimation can run before
/// anything else in the bootstrap.
///
/// Returns false if there are too few correspondences or the essential matrix degenerates.
inline bool relativeBodyRotation(
  const SfmFrame & a, const SfmFrame & b, const CameraCalib & calib,
  Sophus::SO3d & out, int min_correspondences = 60)
{
  std::vector<cv::Point2f> p1, p2;
  for (const auto & entry : a.by_id) {
    const auto it = b.by_id.find(entry.first);
    if (it != b.by_id.end()) {
      p1.push_back(entry.second);
      p2.push_back(it->second);
    }
  }
  if (static_cast<int>(p1.size()) < min_correspondences) {
    return false;
  }

  const cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
    calib.K(0, 0), 0, calib.K(0, 2), 0, calib.K(1, 1), calib.K(1, 2), 0, 0, 1);
  cv::Mat mask;
  const cv::Mat E = cv::findEssentialMat(p1, p2, K_cv, cv::RANSAC, 0.999, 1.0, mask);
  if (E.rows != 3 || E.cols != 3) {
    return false;
  }
  cv::Mat R_cv, t_cv;
  if (cv::recoverPose(E, p1, p2, K_cv, R_cv, t_cv, mask) < min_correspondences) {
    return false;
  }

  // Camera rotation -> body rotation. From the pose chain,
  //   R_cj_ci = R_cam_imu . dR_ij^T . R_cam_imu^T
  // so inverting gives dR_ij = R_cam_imu^T . R_cj_ci^T . R_cam_imu.
  const Eigen::Matrix3d R_cam_imu = calib.T_cam_imu.linear();
  const Eigen::Matrix3d R_cj_ci = detail::matFromCv(R_cv);
  out = Sophus::SO3d(
    Sophus::SO3d::fitToSO3(R_cam_imu.transpose() * R_cj_ci.transpose() * R_cam_imu));
  return true;
}

/// Reconstruct frames [begin, end) up to scale.
///
/// The ruler is invented at the base pair (|t| = 1 from the essential matrix) and then
/// propagated by PnP, which consumes landmarks already carrying it. Nothing here knows a
/// metre; that is phase 2b-3's job.
inline SfmWindow buildSfmWindow(
  const std::vector<SfmFrame> & frames, int begin, int end,
  const CameraCalib & calib, const SfmParams & p = {})
{
  SfmWindow w;
  w.base = begin;

  const cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
    calib.K(0, 0), 0, calib.K(0, 2), 0, calib.K(1, 1), calib.K(1, 2), 0, 0, 1);
  const Eigen::Matrix3d Kinv = calib.K.inverse();
  const SfmFrame & base = frames[begin];

  // --- 1+2+3. THE BASE PAIR: parallax is only a FILTER; the geometry decides.
  //
  // Pixel flow does NOT mean baseline. On a forward-driving car the two are nearly the same
  // thing, which is why taking the first frame past a flow threshold worked on KITTI. A MAV
  // ROTATES, and rotation produces flow with a literally zero baseline -- 20 px of it, and
  // nothing to triangulate. Committing to the first candidate that clears the threshold then
  // fails the whole window, which is exactly what happened on EuRoC (windows 100/400/800:
  // "no parallax, too few landmarks").
  //
  // VINS-Mono's relativePose() is the fix, and it is one keyword:
  //
  //     if (average_parallax * 460 > 30 && m_estimator.solveRelativeRT(corres, R, T))
  //     { l = i; return true; }                                      // else keep scanning
  //
  // The `&&` means a candidate that clears parallax but fails the geometry is silently
  // skipped and the loop moves on. So we try each candidate END TO END -- essential matrix,
  // recoverPose, triangulate -- and accept the first that actually yields structure.
  //
  // NOT DEROTATION. VINS-Mono's compensatedParallax2() contains a rotation-compensation line
  // that is COMMENTED OUT (`p_i_comp = p_i;` runs instead), and ORB-SLAM3 does not derotate
  // either: it runs a homography and a fundamental matrix in parallel and picks on
  // RH = SH/(SH+SF), which DETECTS the rotation/planar degeneracy rather than stumbling into
  // it. That is the better answer and the next rung; this is the cheap one.
  //
  // min_parallax_deg stays at 1.0 -- ORB-SLAM3 hardcodes exactly that at its call site. The
  // small baselines here are the selector picking bad pairs, not a threshold to loosen.
  std::vector<cv::Point2f> p1, p2;
  std::vector<long> pair_ids;
  cv::Mat mask, R_cv, t_cv;
  Eigen::Isometry3d T2 = Eigen::Isometry3d::Identity();

  for (int k = begin + 2; k < end; ++k) {
    const auto ids = detail::sharedIds(base, frames[k]);
    if (static_cast<int>(ids.size()) < p.min_shared) {
      continue;
    }
    std::vector<double> flow;
    for (long id : ids) {
      const auto & a = base.by_id.at(id);
      const auto & b = frames[k].by_id.at(id);
      flow.push_back(std::hypot(a.x - b.x, a.y - b.y));
    }
    std::nth_element(flow.begin(), flow.begin() + flow.size() / 2, flow.end());
    const double parallax_px = flow[flow.size() / 2];
    if (parallax_px < p.min_parallax_px) {
      continue;   // the cheap filter
    }
    ++w.candidates_tried;

    // --- The essential matrix. |t| = 1 here IS the ruler.
    p1.clear();
    p2.clear();
    pair_ids.clear();
    for (long id : ids) {
      p1.push_back(base.by_id.at(id));
      p2.push_back(frames[k].by_id.at(id));
      pair_ids.push_back(id);
    }
    const cv::Mat E = cv::findEssentialMat(p1, p2, K_cv, cv::RANSAC, 0.999, 1.0, mask);
    if (E.rows != 3 || E.cols != 3) {
      continue;
    }
    const int inliers = cv::recoverPose(E, p1, p2, K_cv, R_cv, t_cv, mask);
    if (inliers < p.min_inliers) {
      continue;   // rotation-dominated or degenerate: try the next frame
    }

    T2 = Eigen::Isometry3d::Identity();
    T2.linear() = detail::matFromCv(R_cv);
    T2.translation() = Eigen::Vector3d(
      t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    // --- Triangulate, gated on parallax ANGLE rather than depth. This is the test that a
    //     pixel-flow threshold cannot make: rotation survives the filter above and dies here.
    SfmWindow trial;
    triangulateBasePair(p1, p2, pair_ids, T2, mask, calib, p, trial);
    if (static_cast<int>(trial.landmark.size()) < p.min_landmarks) {
      w.rejected_parallax += trial.rejected_parallax;
      w.rejected_cheirality += trial.rejected_cheirality;
      continue;   // cleared the filter, produced no structure -- VINS-Mono's `&&`
    }

    w.second = k;
    w.base_parallax_px = parallax_px;
    w.inliers = inliers;
    w.landmark = std::move(trial.landmark);
    w.rejected_parallax += trial.rejected_parallax;
    w.rejected_cheirality += trial.rejected_cheirality;
    w.pose[begin] = Eigen::Isometry3d::Identity();
    w.pose[k] = T2;
    break;
  }
  if (w.second < 0) {
    return w;
  }

  // --- 4. PnP for the rest. THIS is what propagates the ruler: it consumes landmarks that
  //        already carry it and returns a pose in the same units.
  for (int k = begin; k < end; ++k) {
    if (w.pose.count(k)) {
      continue;
    }
    std::vector<cv::Point3f> obj;
    std::vector<cv::Point2f> img;
    for (const auto & entry : w.landmark) {
      const auto it = frames[k].by_id.find(entry.first);
      if (it == frames[k].by_id.end()) {
        continue;
      }
      obj.emplace_back(entry.second.x(), entry.second.y(), entry.second.z());
      img.push_back(it->second);
    }
    if (static_cast<int>(obj.size()) < p.min_pnp_points) {
      continue;
    }
    cv::Mat rvec, tvec;
    if (!cv::solvePnPRansac(obj, img, K_cv, cv::Mat(), rvec, tvec, false, 100, 2.0, 0.99)) {
      continue;
    }
    cv::Mat Rk;
    cv::Rodrigues(rvec, Rk);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = detail::matFromCv(Rk);
    T.translation() = Eigen::Vector3d(
      tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    w.pose[k] = T;
  }

  w.valid = w.pose.size() >= 5;
  return w;
}

}  // namespace glassvio

#endif  // GLASSVIO_SFM_WINDOW_HPP
