// PHASE-2 GATE: is the calibration right, and do the tracks obey epipolar geometry?
//
// Everything downstream -- the essential matrix, the visual-inertial alignment, the
// reprojection factor -- is built on K and T_cam_imu. A wrong extrinsic does not announce
// itself: it produces a slightly worse trajectory, indistinguishable from a slightly worse
// estimator. So it gets proven here, before anything sits on it.
//
// THE ORACLE. Ground truth gives the relative camera pose between two frames; the calibration
// turns that into a fundamental matrix; a correct F puts every true correspondence on its
// epipolar line. So the Sampson distance of real KLT tracks measures the whole chain -- K,
// R_rect_00, both extrinsics, and the composition order -- with no estimator in the loop to
// hide an error.
//
// The tracks come from FeatureTracker rather than a direct OpenCV call: its persistent IDs
// ARE the correspondence, so this exercises the same front-end the VIO will use.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glassvio/dataset.hpp"
#include "glassvio/camera_calib.hpp"

/// EuRoC V1_01_easy: the ROS2 bag for sensors, the ASL CSV for ground truth. The CSV is not
/// in the bag -- it is the dataset's own batch solution over VICON and IMU, and it states
/// velocity and both biases outright, which the /vicon topic does not.
static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

constexpr double kMaxMedianSampson = 3.0;   ///< px; the gate. A correct calibration on real KLT
                                            ///< tracks + VICON ground truth over a ~0.25 m
                                            ///< baseline lands ~1.7 px (tracking noise + gt
                                            ///< interpolation), so a WRONG extrinsic (tens of px,
                                            ///< see the small-baseline rows) is cleanly caught.
constexpr int kStride = 40;                 ///< sample every Nth frame (which pairs to test)
/// Minimum baseline for a pair to be SCORED. Below this the camera has barely translated, the
/// epipole sits in-frame, and Sampson goes numerically wild (both tiny and huge) regardless of
/// calibration -- no leverage, so don't let those rows pollute the median.
constexpr double kMinBaseline = 0.10;       ///< metres
/// Frames BETWEEN the two views of each tested pair. NOT 1: adjacent EuRoC frames are 0.05 s
/// apart (~1 cm baseline), and F = K^-T [t]x R K^-1 -> 0 as |t| -> 0, so BOTH the Sampson
/// numerator and denominator collapse and the metric goes to ~0 no matter whether the
/// calibration is right -- the test passes by measuring nothing. 10 frames gives a
/// ~0.2-0.3 m baseline, where a wrong extrinsic actually blows the Sampson error up.
constexpr int kGap = 10;

/// Sampson distance: the first-order approximation to the geometric distance from a
/// correspondence to its epipolar line, in pixels. Preferred over the raw algebraic error
/// x2' F x1, which is not a distance and scales with the coordinates.
///
/// Returns a NEGATIVE sentinel when the point is degenerate (at/near the epipole, where the
/// epipolar line is undefined and `den` -> 0). Such a point carries no epipolar constraint, so
/// the caller SKIPS it rather than scoring it 0 -- a fake-perfect 0 is exactly what let the
/// old zero-baseline test pass.
double sampson(
  const Eigen::Matrix3d & F, const Eigen::Vector2d & x1, const Eigen::Vector2d & x2)
{
  const Eigen::Vector3d p1(x1.x(), x1.y(), 1.0);
  const Eigen::Vector3d p2(x2.x(), x2.y(), 1.0);
  const Eigen::Vector3d Fp1 = F * p1;
  const Eigen::Vector3d Ftp2 = F.transpose() * p2;
  const double den = Fp1.head<2>().squaredNorm() + Ftp2.head<2>().squaredNorm();
  return den > 1e-12 ? std::abs(p2.dot(Fp1)) / std::sqrt(den) : -1.0;
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::string bag_path = (argc > 1) ? argv[1] : kDefaultBag;
  const std::string calib_dir = (argc > 2) ? argv[2] : "config";

  glassvio::CameraCalib calib;
  glassvio::EurocDataset bag;
  try {
    calib = glassvio::loadEurocCalib(calib_dir);
    bag = glassvio::EurocDataset::load(bag_path, kDefaultGt, calib, {true, -1});
  } catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  std::printf(
    "K: fx=%.3f fy=%.3f cx=%.3f cy=%.3f\n", calib.fx(), calib.fy(), calib.cx(), calib.cy());
  std::printf(
    "T_cam_imu translation: [%.3f %.3f %.3f] m  (IMU seen from the camera)\n",
    calib.T_cam_imu.translation().x(), calib.T_cam_imu.translation().y(),
    calib.T_cam_imu.translation().z());
  if (bag.frames.size() < 50 || bag.gt.empty()) {
    std::fprintf(stderr, "need image and pose streams\n");
    return 2;
  }

  const Eigen::Matrix3d Kinv = calib.K.inverse();
  std::vector<double> medians;
  std::printf("\n frame   tracks   baseline   median Sampson\n");

  for (std::size_t i = kStride; i < bag.frames.size(); i += kStride) {
    const auto & prev = bag.frames[i - kGap];
    const auto & cur = bag.frames[i];

    // Relative camera pose from ground truth: X_j = T_cj_ci * X_i.
    const Eigen::Isometry3d T_cj_ci =
      calib.T_cam_imu * bag.gt.at(cur.t).inverse() * bag.gt.at(prev.t) *
      calib.T_cam_imu.inverse();

    if (T_cj_ci.translation().norm() < kMinBaseline) {
      continue;   // no leverage: the epipole is in-frame and Sampson is meaningless
    }

    // E = [t]x R, then F = K^-T E K^-1 pulls it back into pixels.
    const Eigen::Matrix3d F =
      Kinv.transpose() * (Sophus::SO3d::hat(T_cj_ci.translation()) * T_cj_ci.linear()) * Kinv;

    std::vector<double> d;
    for (const auto & [id, p2] : cur.by_id) {
      const auto it = prev.by_id.find(id);
      if (it == prev.by_id.end()) {
        continue;   // track was born this frame: no correspondence to test
      }
      const double s =
        sampson(F, Eigen::Vector2d(it->second.x, it->second.y), Eigen::Vector2d(p2.x, p2.y));
      if (s >= 0.0) {
        d.push_back(s);   // negative = degenerate (at the epipole); no constraint, skip it
      }
    }
    if (d.size() < 50) {
      continue;
    }
    std::nth_element(d.begin(), d.begin() + d.size() / 2, d.end());
    medians.push_back(d[d.size() / 2]);
    std::printf(
      "  %4zu   %5zu   %6.3f m   %8.3f px\n",
      i, d.size(), T_cj_ci.translation().norm(), d[d.size() / 2]);
  }

  if (medians.empty()) {
    std::fprintf(stderr, "no frame pairs with enough shared tracks\n");
    return 2;
  }
  std::sort(medians.begin(), medians.end());
  const double overall = medians[medians.size() / 2];
  std::printf(
    "\n%zu frame pairs. overall median Sampson: %.3f px   gate: < %.2f px\n",
    medians.size(), overall, kMaxMedianSampson);

  if (overall > kMaxMedianSampson) {
    std::printf("FAIL: calibration or tracking does not satisfy epipolar geometry\n");
    return 1;
  }
  std::printf("ok: K and T_cam_imu are consistent with the tracks and ground truth\n");
  return 0;
}
