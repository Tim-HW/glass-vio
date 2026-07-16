// PHASE-1 GATE: can the IMU on this bag dead-reckon well enough to carry a VIO?
//
// Nothing above this matters if the answer is no. A reprojection factor folded into the same
// solver is only worth building if the IMU actually constrains the poses between frames -- so
// this asks the question directly, against ground truth, before any camera code exists.
//
// THE EXPERIMENT. Seed a NavState from /kitti/pose (which IS the OXTS/IMU pose in world),
// run glass_core's preintegration forward over a window, and compare predictState's answer to
// where the vehicle actually was.
//
// WHY VELOCITY IS SEEDED FROM GROUND TRUTH, and why that is not cheating. Dead reckoning
// needs v_i, and "can the IMU integrate" and "can we initialise velocity" are two different
// questions. This bag starts with the car already doing ~5 m/s, so ImuInit's static-window
// bootstrap -- which reports success here, because constant-velocity cruising passes a static
// check -- leaves v at zero and is wrong by 5 m/s immediately. That is a real problem, but it
// belongs to the initialiser, not the integrator. Seeding v from truth isolates the
// integrator, which is what this gate is about. (And v_i turns out to be 100% of the
// triangulation baseline -- see vi_align_check for where that bill comes due.)
//
// Dropout handling lives in ImuBuffer: this bag drops 5.6 s of IMU across 28 gaps, two of
// them ~1.6 s, and integrate() would turn a 1.67 s dt into 1.67 s of invented acceleration.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glass_core/nav_residual.hpp"   // predictState
#include "glass_core/nav_state.hpp"
#include "glass_core/preintegration.hpp"
#include "glassvio/dataset.hpp"

using glass_core::ImuPreintegration;
using glass_core::NavState;
using glass_core::predictState;

/// EuRoC V1_01_easy: the ROS2 bag for sensors, the ASL CSV for ground truth. The CSV is not
/// in the bag -- it is the dataset's own batch solution over VICON and IMU, and it states
/// velocity and both biases outright, which the /vicon topic does not.
static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

constexpr double kWindowSeconds = 2.0;    ///< how far to dead-reckon before comparing
constexpr double kStrideSeconds = 2.0;
constexpr double kMaxMedianDrift = 1.0;   ///< m; the gate

/// Gravity in the pose's `world` frame. Confirmed against this bag, not assumed: with R from
/// /kitti/pose, (R * accel + g) has zero mean and correlates 0.84/0.98/0.92 per axis with the
/// trajectory's own acceleration, once that trajectory is differentiated over a baseline wide
/// enough to escape GPS quantisation noise.
const Eigen::Vector3d kGravityWorld(0.0, 0.0, -9.80665);

}  // namespace

int main(int argc, char ** argv)
{
  const std::string bag_path = (argc > 1) ? argv[1] : kDefaultBag;

  glassvio::EurocDataset bag;
  try {
    // No calibration: this gate is IMU-only and never asks for an image, so the calib is
    // only ever consulted to undistort tracked points. A default one is rectified, which
    // makes undistort() a no-op -- and nothing calls it anyway with track_images off.
    bag = glassvio::EurocDataset::load(bag_path, kDefaultGt, glassvio::CameraCalib{});
  } catch (const std::exception & e) {
    std::fprintf(stderr, "cannot read bag '%s': %s\n", bag_path.c_str(), e.what());
    return 2;
  }
  std::printf(
    "bag '%s': %zu imu @ %.1f Hz, %zu pose\n",
    bag_path.c_str(), bag.imu.size(), bag.imu.rate(), bag.gt.size());
  if (bag.imu.size() < 500 || bag.gt.empty()) {
    std::fprintf(stderr, "need paired imu/pose streams\n");
    return 2;
  }

  const auto & samples = bag.imu.samples();
  const double t0 = samples.front().t;

  std::vector<double> drifts;
  int skipped = 0;
  std::printf("\n  t (s)   window   travelled    drift     drift/dist\n");

  for (double t = t0 + 0.5; t + kWindowSeconds < samples.back().t - 0.5; t += kStrideSeconds) {
    const double t_end = t + kWindowSeconds;

    NavState xi;
    // BIASES FROM TRUTH, for the same reason velocity is: "can the IMU integrate" and "can we
    // estimate its biases" are different questions, and this gate asks only the first.
    //
    // On KITTI zero was right -- the OXTS output is already compensated (vision measures the
    // residual at ~2e-4 rad/s, see gyro_bias_check). EuRoC's ADIS16448 is a real MEMS and
    // states b_a ~ [0, 0.55, 0.07] m/s^2. Integrating that as if it were zero produces 1.85 m
    // of drift over 2 s WHILE THE MAV IS STATIONARY: 0.5 * 0.55 * 2^2 = 1.1 m of it is the
    // bias alone, and none of it says anything about the integrator.
    xi.bg = bag.gt.gyroBias(t);
    xi.ba = bag.gt.accelBias(t);
    ImuPreintegration pre(xi.bg, xi.ba, 1e-3, 1e-2);
    if (!bag.imu.preintegrate(t, t_end, xi.bg, xi.ba, pre)) {
      ++skipped;   // IMU dropout inside the window
      continue;
    }

    const Eigen::Isometry3d T_i = bag.gt.at(t);
    const Eigen::Isometry3d T_j = bag.gt.at(t_end);
    xi.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_i.linear()));
    xi.p = T_i.translation();
    xi.v = bag.gt.velocity(t);

    const NavState xj = predictState(xi, pre, kGravityWorld);

    const double drift = (xj.p - T_j.translation()).norm();
    const double travelled = (T_j.translation() - T_i.translation()).norm();
    drifts.push_back(drift);
    std::printf(
      "  %5.1f   %5.2fs   %7.2f m   %6.3f m   %6.2f%%\n",
      t - t0, pre.dt(), travelled, drift,
      travelled > 0.1 ? 100.0 * drift / travelled : 0.0);
  }

  if (drifts.empty()) {
    std::fprintf(stderr, "no gap-free windows -- cannot judge the IMU\n");
    return 2;
  }

  std::sort(drifts.begin(), drifts.end());
  const double median = drifts[drifts.size() / 2];

  std::printf(
    "\n%zu windows (%d skipped for IMU dropouts)\n"
    "median %.1fs drift: %.3f m   worst: %.3f m   gate: < %.2f m\n",
    drifts.size(), skipped, kWindowSeconds, median, drifts.back(), kMaxMedianDrift);

  if (median > kMaxMedianDrift) {
    std::printf("FAIL: the IMU cannot carry %.1fs of dead reckoning on this bag\n",
      kWindowSeconds);
    return 1;
  }
  std::printf("ok: preintegration tracks ground truth; the visual factor has a base to sit on\n");
  return 0;
}
