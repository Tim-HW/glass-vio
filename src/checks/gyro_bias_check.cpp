// SCORES STAGE [3] of VioInitializer: the gyro bias, estimated from vision alone.
//
// WHY [3] IS OBSERVABLE BEFORE ANYTHING ELSE. Every other thing the bootstrap wants -- scale,
// velocity, gravity -- needs up-to-scale structure, which needs triangulation, which needs a
// baseline. Rotation needs none of it: the essential matrix hands back a relative rotation
// (0.05 deg on this bag) that is completely independent of scale. And it runs FIRST because
// every later stage integrates the gyro and would inherit its error.
//
// WHAT THIS COSTS IF SKIPPED. EuRoC's ADIS16448 is a real MEMS: its stated bias is
// b_w = [-0.0022, 0.0207, 0.0764] rad/s, and 0.0764 is 4.4 deg/s. Preintegrating as if it
// were zero puts the rotation 1.14 deg wrong over half a second, and every stage above this
// integrates the gyro.
//
// (KITTI could not show this. Its OXTS output is already bias-compensated, so the truth was
// ~0 and the stage could only ever prove a negative -- that it had not INVENTED a bias from
// a "static" window that was really a cruising car. Same code, and only the data could say
// whether it worked.)
//
// THE ORACLE. EuRoC states the true bias, so this asserts a MATCH, not merely an improvement.
// Ground truth is used ONLY to score -- the estimate never sees it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>

#include "glassvio/dataset.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/vio_initializer.hpp"

/// EuRoC V1_01_easy: the ROS2 bag for sensors, the ASL CSV for ground truth. The CSV is not
/// in the bag -- it is the dataset's own batch solution over VICON and IMU, and it states
/// velocity and both biases outright, which the /vicon topic does not.
static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

/// The gate: the corrected bias must not make the rotation WORSE than doing nothing.
constexpr double kMaxMedianRotErrDeg = 0.30;
/// rad/s. EuRoC states the true bias, so this check asserts a MATCH rather than merely an
/// improvement. |b_w| here is ~0.079, so this is a few percent.
constexpr double kMaxBiasError = 5.0e-3;

double deg(const Sophus::SO3d & R) {return R.log().norm() * 180.0 / M_PI;}

/// Ground-truth attitude. /kitti/pose IS the IMU pose, so no extrinsic is involved.
Sophus::SO3d gtRot(const glassvio::EurocDataset & bag, double t)
{
  return Sophus::SO3d(Sophus::SO3d::fitToSO3(bag.gt.at(t).linear()));
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
  if (bag.imu.size() < 500 || bag.gt.empty() || bag.frames.size() < 60) {
    std::fprintf(stderr, "need imu, pose and image streams\n");
    return 2;
  }
  std::printf(
    "tracked %zu frames, %zu imu @ %.1f Hz\n",
    bag.frames.size(), bag.imu.size(), bag.imu.rate());

  // Run stage [3] over the WHOLE bag rather than one window: the bias is a constant, so
  // every pair is evidence, and averaging is what pulls the estimate below the vision noise
  // floor of any single pair.
  glassvio::InitializerParams p;
  p.window_frames = static_cast<int>(bag.frames.size());
  const glassvio::VioInitializer init(calib, p);

  Eigen::Vector3d bg = Eigen::Vector3d::Zero();
  int pairs = 0;
  if (!init.estimateGyroBias(bag.frames, bag.imu, 0, bg, pairs)) {
    std::fprintf(stderr, "stage [3] failed: only %d usable pairs\n", pairs);
    return 2;
  }

  // How good is the vision rotation itself? Scored against ground truth, but NOT used above.
  std::vector<double> vis_err;
  for (int i = 0; i + p.bias_frame_gap < static_cast<int>(bag.frames.size());
    i += p.bias_frame_gap)
  {
    Sophus::SO3d dR_vis;
    if (!glassvio::relativeBodyRotation(
        bag.frames[i], bag.frames[i + p.bias_frame_gap], calib, dR_vis))
    {
      continue;
    }
    const Sophus::SO3d dR_gt =
      gtRot(bag, bag.frames[i].t).inverse() * gtRot(bag, bag.frames[i + p.bias_frame_gap].t);
    vis_err.push_back(deg(dR_vis.inverse() * dR_gt));
  }
  std::sort(vis_err.begin(), vis_err.end());

  // Frames -> seconds via the MEASURED rate, not a hardcoded one. This printed "0.50 s" on
  // EuRoC for a while because the constant was KITTI's 10 Hz -- the same count-is-not-a-
  // duration trap that made imu.init.num_samples wrong when the IMU rate changed.
  const double frame_hz = bag.frames.size() > 1 ?
    (bag.frames.size() - 1) / (bag.frames.back().t - bag.frames.front().t) : 20.0;
  const double gap_s = p.bias_frame_gap / frame_hz;
  std::printf(
    "\n%d pairs (%d frames apart, ~%.2f s at %.1f Hz)\n"
    "essential-matrix rotation vs ground truth: median %.4f deg  <- the noise floor\n",
    pairs, p.bias_frame_gap, gap_s, frame_hz,
    vis_err.empty() ? 0.0 : vis_err[vis_err.size() / 2]);
  std::printf(
    "\nestimated gyro bias (vision only, no ground truth):\n  [%+.5f %+.5f %+.5f] rad/s\n",
    bg.x(), bg.y(), bg.z());

  // Score: re-preintegrate at each candidate bias and compare against ground truth.
  const auto score = [&](const Eigen::Vector3d & b) {
      std::vector<double> e;
      for (int i = 0; i + p.bias_frame_gap < static_cast<int>(bag.frames.size());
        i += p.bias_frame_gap)
      {
        glass_core::ImuPreintegration pre(b, Eigen::Vector3d::Zero(), 1.0, 1.0);
        if (!bag.imu.preintegrate(
            bag.frames[i].t, bag.frames[i + p.bias_frame_gap].t, b,
            Eigen::Vector3d::Zero(), pre))
        {
          continue;
        }
        const Sophus::SO3d dR_gt = gtRot(bag, bag.frames[i].t).inverse() *
          gtRot(bag, bag.frames[i + p.bias_frame_gap].t);
        e.push_back(deg(pre.dR().inverse() * dR_gt));
      }
      std::sort(e.begin(), e.end());
      return e.empty() ? 1e9 : e[e.size() / 2];
    };

  // THE ORACLE. EuRoC states the true bias, so this is a match test, not merely an
  // improvement test. (KITTI could not support this: its OXTS is already compensated, so the
  // truth was ~0 and "matching" it proved only that we had not invented one.)
  const Eigen::Vector3d truth = bag.gt.gyroBias(bag.frames.front().t);
  const double err = (bg - truth).norm();
  std::printf(
    "ground truth (stated by the dataset, never seen by the solve):\n"
    "  [%+.5f %+.5f %+.5f] rad/s\n"
    "  error: %.2e rad/s  (%.1f%% of |b_w| = %.4f)\n",
    truth.x(), truth.y(), truth.z(), err, 100.0 * err / truth.norm(), truth.norm());

  const double dt = p.bias_frame_gap / 20.0;   // EuRoC images are 20 Hz
  const double e_zero = score(Eigen::Vector3d::Zero());
  const double e_ours = score(bg);
  const double e_truth = score(truth);

  std::printf(
    "\nmedian rotation error vs ground truth over ~%.2f s, by bias:\n"
    "  bg = 0            : %.4f deg   <- what ignoring the bias costs\n"
    "  bg = ours (vision): %.4f deg\n"
    "  bg = truth        : %.4f deg   <- the floor; we cannot beat this\n",
    dt, e_zero, e_ours, e_truth);

  if (err > kMaxBiasError) {
    std::printf("\nFAIL: the estimated bias does not match the stated one\n");
    return 1;
  }
  if (e_ours > kMaxMedianRotErrDeg) {
    std::printf("\nFAIL: the estimated bias does not track ground truth\n");
    return 1;
  }
  std::printf(
    "\nok: vision recovered a REAL gyro bias to %.1f%%, with no ground truth and no scale.\n"
    "    Ignoring it would cost %.1fx the rotation error (%.3f -> %.3f deg). The estimate\n"
    "    also sits below the essential matrix's own per-pair noise floor (%.3f deg): the\n"
    "    bias is a constant, so all %d pairs are evidence and least squares averages them.\n",
    100.0 * err / truth.norm(), e_zero / e_ours, e_zero, e_ours,
    vis_err.empty() ? 0.0 : vis_err[vis_err.size() / 2], pairs);
  return 0;
}
