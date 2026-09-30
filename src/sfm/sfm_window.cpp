#include "glassvio/sfm_window.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

#include <opencv2/calib3d.hpp>

namespace glassvio
{

namespace
{


Eigen::Matrix3d matFromCv(const cv::Mat & m)
{
  Eigen::Matrix3d out;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      out(i, j) = m.at<double>(i, j);
    }
  }
  return out;
}

std::vector<long> sharedIds(const SfmFrame & a, const SfmFrame & b)
{
  std::vector<long> ids;
  for (const auto & entry : a.by_id) {
    if (b.by_id.count(entry.first)) {
      ids.push_back(entry.first);
    }
  }
  return ids;
}

}  // namespace

void triangulateBasePair(
  const std::vector<cv::Point2f> & p1, const std::vector<cv::Point2f> & p2,
  const std::vector<long> & ids, const Eigen::Isometry3d & T2, const cv::Mat & mask,
  const CameraCalib & calib, const SfmParams & p, SfmWindow & out)
{
  const cv::Mat K_cv = calib.cvK();
  const Eigen::Matrix3d Kinv = calib.K.inverse();

  // STEP 1 -- the two projection matrices, P = K [R | t]. Camera 1 is the reference, so it sits
  // at the origin ([I | 0]); camera 2 is where T2 puts it (X_cam2 = T2 * X_cam1).
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

  // STEP 2 -- triangulate every pair of pixels at once: the 3D point whose two projections land
  // closest to p1[i] and p2[i]. The answer is homogeneous (x, y, z, w), one column per point,
  // expressed in camera 1's frame.
  cv::Mat X4f;
  cv::triangulatePoints(P1, P2, p1, p2, X4f);
  // triangulatePoints returns CV_32F for Point2f input. Reading that with at<double>() does
  // not throw -- it reinterprets two floats as one double and produces values that look
  // exactly like points at infinity. Convert; never assume.
  cv::Mat X4;
  X4f.convertTo(X4, CV_64F);

  // STEP 3 -- keep a point only if it survives every gate below.
  for (int i = 0; i < X4.cols; ++i) {
    // Gate a: RANSAC already called this pair an outlier of the essential matrix.
    if (!mask.empty() && !mask.at<unsigned char>(i)) {
      continue;
    }
    // Gate b: w = 0 is a point at infinity (parallel rays). Otherwise divide by w to get metric-
    // looking coordinates -- in RULER units, not metres.
    const double hw = X4.at<double>(3, i);
    if (std::abs(hw) < 1e-12) {
      continue;
    }
    const Eigen::Vector3d X(
      X4.at<double>(0, i) / hw, X4.at<double>(1, i) / hw, X4.at<double>(2, i) / hw);

    // Gate c -- cheirality: in front of BOTH cameras. A point behind still projects to a perfectly
    // ordinary-looking pixel -- the wrong one, with a sign-flipped Jacobian.
    if (X.z() < p.min_depth || (T2 * X).z() < p.min_depth) {
      ++out.rejected_cheirality;
      continue;
    }

    // Gate d -- the parallax ANGLE between the two viewing rays (r1, r2: the unit directions
    // from each camera centre through its pixel, both written in camera 1's frame). THIS is what a pixel-flow threshold
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
    out.landmark.emplace(ids[i], X);   // a landmark: track id -> 3D point in camera 1's frame
  }
}

bool relativeBodyRotation(
  const SfmFrame & a, const SfmFrame & b, const CameraCalib & calib,
  Sophus::SO3d & out, int min_correspondences)
{
  // STEP 1 -- correspondences: every track id present in both frames gives one pixel pair.
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
  // STEP 2 -- the essential matrix E relates the two views (x2^T E x1 = 0 for every true pair),
  // fitted by RANSAC so mistracked points do not bend it; then recoverPose decomposes E (an SVD)
  // into a rotation and a translation DIRECTION, picking the one of four solutions that puts the
  // points in front of both cameras. Only the rotation is used here; t_cv is discarded.
  cv::Mat mask;
  const cv::Mat E = cv::findEssentialMat(p1, p2, K_cv, cv::RANSAC, 0.999, 1.0, mask);
  if (E.rows != 3 || E.cols != 3) {
    return false;
  }
  cv::Mat R_cv, t_cv;
  if (cv::recoverPose(E, p1, p2, K_cv, R_cv, t_cv, mask) < min_correspondences) {
    return false;
  }

  // STEP 3 -- camera rotation -> body (IMU) rotation. From the pose chain,
  //   R_cj_ci = R_cam_imu . dR_ij^T . R_cam_imu^T
  // so inverting gives dR_ij = R_cam_imu^T . R_cj_ci^T . R_cam_imu.
  const Eigen::Matrix3d R_cam_imu = calib.T_cam_imu.linear();
  const Eigen::Matrix3d R_cj_ci = matFromCv(R_cv);
  out = Sophus::SO3d(
    Sophus::SO3d::fitToSO3(R_cam_imu.transpose() * R_cj_ci.transpose() * R_cam_imu));
  return true;
}

namespace
{

SfmWindow selectBasePair(
  const std::vector<SfmFrame> & frames, int begin, int end,
  const CameraCalib & calib, const SfmParams & p)
{
  // Pixel flow selects candidates; only recovered geometry makes a usable base pair.
  SfmWindow w;
  w.base = begin;

  const cv::Mat K_cv = (cv::Mat_<double>(3, 3) <<
    calib.K(0, 0), 0, calib.K(0, 2), 0, calib.K(1, 1), calib.K(1, 2), 0, 0, 1);
  const SfmFrame & base = frames[begin];

  // --- A + B. THE BASE PAIR: parallax is only a FILTER; the geometry decides.
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

  // Try each later frame k as the base frame's partner, nearest first, and take the FIRST that
  // passes every test below. `continue` means "not this one, try the next frame".
  for (int k = begin + 2; k < end; ++k) {
    // Test 1 -- enough tracks seen in BOTH frames to fit an essential matrix robustly.
    const auto ids = sharedIds(base, frames[k]);
    w.max_shared = std::max(w.max_shared, static_cast<int>(ids.size()));
    if (static_cast<int>(ids.size()) < p.min_shared) {
      continue;
    }
    // Test 2 -- enough pixel motion: the median distance each shared track moved between the
    // two frames. Cheap, and only a filter: rotation alone moves pixels too (see above).
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

    // Test 3 -- the geometry. Fit the essential matrix E to the shared pixel pairs (RANSAC, 1 px),
    // then decompose it: recoverPose returns the rotation R_cv and the translation DIRECTION t_cv
    // of frame k relative to the base, and how many pairs end up in front of both cameras.
    // |t_cv| = 1 by construction, and that 1 IS the ruler.
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

    // The partner's pose as one transform: X_ck = T2 * X_c0.
    T2 = Eigen::Isometry3d::Identity();
    T2.linear() = matFromCv(R_cv);
    T2.translation() = Eigen::Vector3d(
      t_cv.at<double>(0), t_cv.at<double>(1), t_cv.at<double>(2));

    // Test 4 -- does this pair actually yield structure? Triangulate, gated on parallax ANGLE
    // rather than depth. This is the test a pixel-flow threshold cannot make: rotation survives
    // the filter above and dies here. Into `trial`, so a rejected candidate leaves `w` untouched.
    SfmWindow trial;
    triangulateBasePair(p1, p2, pair_ids, T2, mask, calib, p, trial);
    w.max_trial_landmarks =
      std::max(w.max_trial_landmarks, static_cast<int>(trial.landmark.size()));
    if (static_cast<int>(trial.landmark.size()) < p.min_landmarks) {
      w.rejected_parallax += trial.rejected_parallax;
      w.rejected_cheirality += trial.rejected_cheirality;
      continue;   // cleared the filter, produced no structure -- VINS-Mono's `&&`
    }

    // Accepted: frame k is the base pair's partner. Record both poses and the landmarks.
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
  return w;
}

void poseRemainingFrames(
  const std::vector<SfmFrame> & frames, int begin, int end,
  const CameraCalib & calib, const SfmParams & p, SfmWindow & w)
{
  const cv::Mat K_cv = calib.cvK();
  // PnP carries the base pair's arbitrary ruler into every other camera pose.
  for (int k = begin; k < end; ++k) {
    if (w.pose.count(k)) {
      continue;   // the two base-pair frames already have their poses
    }
    // The 3D-2D pairs for this frame: each landmark it still tracks, with the pixel it sees it at.
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
    // Solve for the camera pose that projects those landmarks onto those pixels (RANSAC, 2 px).
    // Landmarks are held FIXED here -- which is why their errors flow into every pose, and why
    // bundleAdjust() is run afterwards.
    cv::Mat rvec, tvec, inliers;
    if (!cv::solvePnPRansac(
        obj, img, K_cv, cv::Mat(), rvec, tvec, false, 100, 2.0, 0.99, inliers) ||
      inliers.rows < p.min_pnp_points)
    {
      continue;
    }
    // OpenCV returns the rotation as an axis-angle vector; Rodrigues turns it into a matrix.
    cv::Mat Rk;
    cv::Rodrigues(rvec, Rk);
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
    T.linear() = matFromCv(Rk);
    T.translation() = Eigen::Vector3d(
      tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    // CHEIRALITY. A pose rotated ~180 deg with the landmarks BEHIND the camera reprojects them
    // just as well, and RANSAC happily returns it from a few points: measured on V1_02, every
    // flipped pose in a refused bootstrap (70 of 70) came from here, not from the base pair.
    int in_front = 0;
    for (int i = 0; i < inliers.rows; ++i) {
      const cv::Point3f & X = obj[inliers.at<int>(i)];
      in_front += (T * Eigen::Vector3d(X.x, X.y, X.z)).z() > 0.0;
    }
    if (in_front < inliers.rows) {
      continue;
    }
    w.pose[k] = T;
  }
}

void addOptionalWindowLandmarks(
  const std::vector<SfmFrame> & frames, int begin, int end,
  const CameraCalib & calib, const SfmParams & p, SfmWindow & w)
{
  // Group untriangulated tracks by their first and last posed views.
  if (p.triangulate_window) {
    // For every track that is not yet a landmark, find the FIRST and LAST posed frame that see
    // it (the widest baseline available), and group the tracks by that pair of frames.
    std::map<std::pair<int, int>, std::vector<long>> by_span;
    {
      std::unordered_map<long, std::pair<int, int>> span;
      for (int k = begin; k < end; ++k) {
        if (!w.pose.count(k)) {
          continue;
        }
        for (const auto & e : frames[k].by_id) {
          if (w.landmark.count(e.first)) {
            continue;
          }
          const auto it = span.find(e.first);
          if (it == span.end()) {
            span.emplace(e.first, std::make_pair(k, k));
          } else {
            it->second.second = k;
          }
        }
      }
      for (const auto & kv : span) {
        if (kv.second.first != kv.second.second) {
          by_span[kv.second].push_back(kv.first);
        }
      }
    }
    for (const auto & group : by_span) {
      const int a = group.first.first, b = group.first.second;
      std::vector<cv::Point2f> pa, pb;
      for (long id : group.second) {
        pa.push_back(frames[a].by_id.at(id));
        pb.push_back(frames[b].by_id.at(id));
      }
      SfmWindow trial;
      triangulateBasePair(
        pa, pb, group.second, w.pose.at(b) * w.pose.at(a).inverse(), cv::Mat(), calib, p, trial);
      // The helper returns points in frame a's camera; move them into the base camera's frame,
      // and keep one only if it reprojects well in EVERY posed frame that tracked it.
      const Eigen::Isometry3d T_c0_ca = w.pose.at(a).inverse();
      for (const auto & lm : trial.landmark) {
        const Eigen::Vector3d X = T_c0_ca * lm.second;
        bool consistent = true;
        for (int k = a; k <= b && consistent; ++k) {
          const auto pose = w.pose.find(k);
          const auto obs = frames[k].by_id.find(lm.first);
          if (pose == w.pose.end() || obs == frames[k].by_id.end()) {
            continue;
          }
          const Eigen::Vector3d P = pose->second * X;
          consistent = P.z() > p.min_depth &&
            std::hypot(
            calib.fx() * P.x() / P.z() + calib.cx() - obs->second.x,
            calib.fy() * P.y() / P.z() + calib.cy() - obs->second.y) < p.max_window_reproj_px;
        }
        if (consistent) {
          w.landmark.emplace(lm.first, X);
          ++w.window_triangulated;
        }
      }
    }
  }
}

}  // namespace

SfmWindow buildSfmWindow(
  const std::vector<SfmFrame> & frames, int begin, int end,
  const CameraCalib & calib, const SfmParams & p)
{
  SfmWindow w = selectBasePair(frames, begin, end, calib, p);
  if (w.second < 0) {return w;}
  poseRemainingFrames(frames, begin, end, calib, p, w);
  addOptionalWindowLandmarks(frames, begin, end, calib, p, w);
  // Alignment needs poses from at least five frames.
  w.valid = w.pose.size() >= 5;
  return w;
}

}  // namespace glassvio
