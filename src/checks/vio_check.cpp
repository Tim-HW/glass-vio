// THE GATE FOR THE WHOLE THESIS: bootstrap, then solve frames tightly-coupled, and see how
// far the trajectory tracks ground truth.
//
// Everything below this has been proven in isolation -- the IMU dead-reckons (0.16 m/2 s), the
// calibration satisfies epipolar geometry (0.19 px), the bias comes out of vision (1.8%), the
// reconstruction is rigid (2.6%), and the accelerometer supplies the metre (s to 1.8%). This
// asks the only question left: does a camera residual folded into the SAME normal equations as
// the LiDAR path actually produce a trajectory?
//
// WHAT THIS CUT DELIBERATELY DOES NOT DO. Landmarks are triangulated ONCE by the bootstrap and
// held fixed forever, exactly as glasslio holds its map's planes fixed within a solve. As the
// MAV flies past them they leave view, tracks die, and the solve starves -- min_features fires
// and we stop. That is not a bug to hide; it is the measurement this check exists to take,
// because the number it reports is precisely what a sliding window has to fix. Re-triangulation
// and marginalisation are phase 4.
//
// THE HONEST COMPARISON. Running with imu_prior_weight = 0 makes it pure vision. If tight
// coupling buys nothing, that comparison says so, and no amount of Lie algebra changes it.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glassvio/dataset.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/vio_initializer.hpp"
#include "glassvio/visual_registration.hpp"

/// EuRoC V1_01_easy. Window 100 is the one that clears stage [2] (2.64% residual).
static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

constexpr double kGravityMag = 9.80665;
/// The gate: median position error while the solve is alive.
constexpr double kMaxMedianErr = 0.30;   ///< m
constexpr int kMinFramesTracked = 10;

}  // namespace

int main(int argc, char ** argv)
{
  const std::string bag_path = (argc > 1) ? argv[1] : kDefaultBag;
  const std::string calib_dir = (argc > 2) ? argv[2] : "config";
  const int window_start = (argc > 3) ? std::atoi(argv[3]) : 100;
  const double imu_weight = (argc > 4) ? std::atof(argv[4]) : 1.0;

  glassvio::CameraCalib calib;
  glassvio::EurocDataset bag;
  try {
    calib = glassvio::loadEurocCalib(calib_dir);
    bag = glassvio::EurocDataset::load(bag_path, kDefaultGt, calib, {true, -1});
  } catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (bag.imu.empty() || bag.gt.empty() || bag.frames.size() < 200) {
    std::fprintf(stderr, "need imu, pose and image streams\n");
    return 2;
  }

  // --- Bootstrap: [2] reconstruct, [3] gyro bias, [4] align.
  glassvio::InitializerParams ip;
  ip.window_frames = 30;
  const glassvio::VioInitializer init(calib, ip);
  const glassvio::InitResult r = init.run(bag.frames, bag.imu, window_start);
  if (!r.ok || !r.scale_observable) {
    std::fprintf(stderr, "bootstrap failed at window %d\n", window_start);
    return 2;
  }
  std::printf(
    "bootstrap @ %d: %zu landmarks, s = %.4f, |g| = %.3f, bg = [%+.4f %+.4f %+.4f]\n",
    window_start, r.sfm.landmark.size(), r.scale, r.gravity_sfm.norm(),
    r.gyro_bias.x(), r.gyro_bias.y(), r.gyro_bias.z());

  // --- Landmarks -> WORLD, in METRES.
  //
  // The bootstrap's landmarks are in ruler units in the base camera's frame. Two things turn
  // them into something the factor can use: multiply by s (the metre), and compose through the
  // base camera's world pose. The world frame is taken from ground truth AT THE BASE FRAME
  // ONLY -- a monocular VIO cannot know where the world's origin is, and anchoring the first
  // pose is the standard gauge choice, not a cheat. Everything after is estimated.
  const int base = r.sfm.base;
  const Eigen::Isometry3d T_world_imu0 = bag.gt.at(bag.frames[base].t);
  const Eigen::Isometry3d T_world_cam0 = T_world_imu0 * calib.T_cam_imu.inverse();

  std::unordered_map<long, Eigen::Vector3d> landmarks_world;
  for (const auto & lm : r.sfm.landmark) {
    landmarks_world.emplace(lm.first, T_world_cam0 * (r.scale * lm.second));
  }

  // --- Seed the state. Everything here came from the bootstrap except the world anchor.
  glass_core::NavState x;
  x.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_world_imu0.linear()));
  x.p = T_world_imu0.translation();
  x.v = T_world_cam0.linear() * r.velocity_sfm.front();   // stage [4]'s velocity, into world
  x.bg = r.gyro_bias;                                     // stage [3]
  const Eigen::Vector3d gravity(0.0, 0.0, -kGravityMag);

  glassvio::VisualParams vp;
  vp.imu_prior_weight = imu_weight;
  vp.reproj.sigma_px = 1.0;

  // Bias prior: how far the bias may drift between frames, from the DATASHEET rather than a
  // guess -- the random walks the Kalibr file states.
  Eigen::Matrix<double, 6, 6> bias_information = Eigen::Matrix<double, 6, 6>::Identity();
  bias_information.topLeftCorner<3, 3>() /=
    (calib.gyro_random_walk * calib.gyro_random_walk);
  bias_information.bottomRightCorner<3, 3>() /=
    (calib.accel_random_walk * calib.accel_random_walk);

  // --- The carried covariance. THIS is what stops the IMU factor from asserting that the
  // previous state was perfect.
  //
  // P_0 is the BOOTSTRAP's uncertainty, and it is emphatically not zero: stage [4] left scale
  // 1.8% off, v_0 0.04 m/s off, and |g| 0.4% off. Seeding it too tight would reinvent the very
  // bug this exists to fix -- an over-confident prior is indistinguishable from a fixed one.
  Eigen::Matrix<double, glass_core::kNavDim, glass_core::kNavDim> P =
    Eigen::Matrix<double, glass_core::kNavDim, glass_core::kNavDim>::Zero();
  P.block<3, 3>(glass_core::kIdxPhi, glass_core::kIdxPhi).diagonal().setConstant(
    std::pow(2.0 * M_PI / 180.0, 2));                                  // ~2 deg attitude
  P.block<3, 3>(glass_core::kIdxPos, glass_core::kIdxPos).diagonal().setConstant(
    std::pow(0.05, 2));                                                // 5 cm: the gauge anchor
  P.block<3, 3>(glass_core::kIdxVel, glass_core::kIdxVel).diagonal().setConstant(
    std::pow(0.10, 2));                                                // stage [4] gave 0.04
  P.block<3, 3>(glass_core::kIdxBg, glass_core::kIdxBg).diagonal().setConstant(
    std::pow(2.0e-3, 2));                                              // stage [3] gave 1.4e-3
  P.block<3, 3>(glass_core::kIdxBa, glass_core::kIdxBa).diagonal().setConstant(
    std::pow(0.10, 2));                                                // never estimated at all

  std::printf(
    "\n frame   feats  behind   rmse_px   err_m   |  imu_weight = %.1f\n", imu_weight);

  std::vector<double> errors;
  int tracked = 0;
  int stopped_at = -1;

  for (int k = base + 1; k < static_cast<int>(bag.frames.size()); ++k) {
    glass_core::ImuPreintegration pre(
      x.bg, x.ba, calib.gyro_noise, calib.accel_noise);
    if (!bag.imu.preintegrate(
        bag.frames[k - 1].t, bag.frames[k].t, x.bg, x.ba, pre,
        calib.gyro_noise, calib.accel_noise))
    {
      stopped_at = k;
      break;   // IMU dropout: the factor has no delta
    }

    // The IMU's own prediction seeds Gauss-Newton -- strictly better than constant velocity,
    // because it uses the accelerometer that actually measured the change.
    const glass_core::NavState guess = glass_core::predictState(x, pre, gravity);

    std::unordered_map<long, cv::Point2f> obs;
    for (const auto & entry : bag.frames[k].by_id) {
      if (landmarks_world.count(entry.first)) {
        obs.emplace(entry.first, entry.second);
      }
    }

    const glassvio::VisualResult res = glassvio::solveFrame(
      landmarks_world, obs, x, P, pre, gravity, guess, bias_information, calib, vp);
    if (!res.valid) {
      stopped_at = k;
      break;   // starved: the camera has flown past everything it was triangulated against
    }
    x = res.state;

    // --- Carry the posterior forward. res.H is the accumulated information from every factor
    // this frame -- vision, the IMU, and the bias prior -- so its inverse IS the new
    // covariance. That is precisely what NormalEquationsN::H()'s comment describes:
    // "P_posterior = (P_prior^-1 + H)^-1". The prior is already inside H via sigma_eff, so
    // this is a plain inverse rather than a second addition.
    //
    // LDLT, not inverse(): H is a sum of J^T Omega J and is symmetric positive SEMI-definite
    // by construction -- never indefinite, but singular when the geometry degenerates. If the
    // solve leaves it rank-deficient, keep the old P rather than propagate garbage.
    const Eigen::Matrix<double, glass_core::kNavDim, glass_core::kNavDim> P_new =
      res.H.ldlt().solve(
      Eigen::Matrix<double, glass_core::kNavDim, glass_core::kNavDim>::Identity());
    if (P_new.allFinite() && P_new.diagonal().minCoeff() > 0.0) {
      P = P_new;
    }

    const double err = (x.p - bag.gt.at(bag.frames[k].t).translation()).norm();
    errors.push_back(err);
    ++tracked;
    if (tracked <= 8 || tracked % 10 == 0) {
      std::printf(
        "  %4d   %5d   %5d   %7.3f   %6.3f\n",
        k, res.features, res.rejected_cheirality, res.rmse_px, err);
    }
  }

  if (tracked < kMinFramesTracked) {
    std::fprintf(stderr, "\nonly %d frames tracked -- the solve never got going\n", tracked);
    return 1;
  }

  std::vector<double> sorted = errors;
  std::sort(sorted.begin(), sorted.end());
  const double median = sorted[sorted.size() / 2];

  std::printf(
    "\ntracked %d frames (%.2f s) before %s at frame %d\n"
    "median position error: %.3f m   final: %.3f m   gate: < %.2f m\n",
    tracked, tracked / 20.0,
    stopped_at < 0 ? "the sequence ended" : "starving", stopped_at,
    median, errors.back(), kMaxMedianErr);

  if (median > kMaxMedianErr) {
    std::printf("\nFAIL: the tightly-coupled solve does not track ground truth\n");
    return 1;
  }
  std::printf(
    "\nok: a camera residual and an IMU factor in ONE NormalEquationsN<15>, solved by the\n"
    "    same Gauss-Newton the LiDAR path uses. Landmarks are held fixed, so starving after\n"
    "    %.2f s is expected -- that is exactly what a sliding window exists to fix.\n",
    tracked / 20.0);
  return 0;
}
