// DETERMINISTIC OFFLINE DRIVE of the real VioEstimator -- the tool that splits "the node
// fails" into a single yes/no.
//
// The node's bootstrap produces a reconstruction ~100x too small (landmark depths ~3 cm
// against a ~metre scene), so every landmark is rejected and tracking never starts. But the
// node is nondeterministic: a slow bootstrap on the worker drops frames, and which frames
// drop depends on timing. That confounds every diagnosis.
//
// This removes ALL of it. It feeds VioEstimator::process() the exact frames the offline
// dataset tracked, in order, with zero drops -- the same estimator object the node runs, with
// none of the plumbing. Two outcomes, and they point in opposite directions:
//
//   * metric bootstrap here  -> the LOGIC is sound; the node's dropped frames corrupt the
//                               SfM window. Fix the online frame handling.
//   * tiny bootstrap here    -> bootstrap()'s own scale/window math is wrong, independent of
//                               ROS. Fix the estimator.
//
// It reuses EurocDataset (which already tracked + undistorted the images) and only synthesises
// the sensor_msgs the MeasureGroup wants -- so the feature and IMU data are byte-identical to
// what the offline gates run on.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/imu.hpp>

#include "glassvio/camera_calib.hpp"
#include "glassvio/dataset.hpp"
#include "glassvio/types.hpp"
#include "glassvio/vio_estimator.hpp"

static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

/// StampedImu -> the sensor_msgs the MeasureGroup carries. The estimator converts straight
/// back at its boundary; this just satisfies the type.
sensor_msgs::msg::Imu::ConstSharedPtr toMsg(const glassvio::StampedImu & s)
{
  auto m = std::make_shared<sensor_msgs::msg::Imu>();
  m->header.stamp.sec = static_cast<int32_t>(std::floor(s.t));
  m->header.stamp.nanosec =
    static_cast<uint32_t>((s.t - std::floor(s.t)) * 1e9);
  m->angular_velocity.x = s.gyro.x();
  m->angular_velocity.y = s.gyro.y();
  m->angular_velocity.z = s.gyro.z();
  m->linear_acceleration.x = s.accel.x();
  m->linear_acceleration.y = s.accel.y();
  m->linear_acceleration.z = s.accel.z();
  return m;
}

}  // namespace

int main(int argc, char ** argv)
{
  const std::string bag_path = (argc > 1) ? argv[1] : kDefaultBag;
  const std::string calib_dir = (argc > 2) ? argv[2] : "config";
  const std::string csv_path = (argc > 3) ? argv[3] : "/tmp/glassvio_run.csv";

  glassvio::CameraCalib calib;
  glassvio::EurocDataset bag;
  try {
    calib = glassvio::loadEurocCalib(calib_dir);
    bag = glassvio::EurocDataset::load(bag_path, kDefaultGt, calib, {true, -1});
  } catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (bag.frames.size() < 100 || bag.imu.empty() || bag.gt.empty()) {
    std::fprintf(stderr, "need image, imu and ground-truth streams\n");
    return 2;
  }
  std::printf(
    "deterministic drive: %zu frames, %zu imu, %zu gt\n",
    bag.frames.size(), bag.imu.size(), bag.gt.size());

  glassvio::EstimatorParams ep;   // exactly the node's defaults
  glassvio::VioEstimator est(calib, ep);

  const auto & imu = bag.imu.samples();
  std::size_t imu_cursor = 0;
  double t_prev = imu.empty() ? 0.0 : imu.front().t - 1.0;

  // Per-frame CSV, so a run can be plotted rather than squinted at. One row per frame.
  std::ofstream csv(csv_path);
  csv <<
    "t,stage,feats,map,pending,rmse_px,"
    "pos_err,vel_err,"
    "px,py,pz,gx,gy,gz,vx,vy,vz,gvx,gvy,gvz,"
    "bgx,bgy,bgz,gbgx,gbgy,gbgz,bax,bay,baz,gbax,gbay,gbaz\n";
  csv.setf(std::ios::fixed);
  csv.precision(6);

  int bootstrapped_at = -1;
  int tracked = 0;
  int lost_at = -1;
  double boot_depth_median = 0.0;
  std::size_t boot_landmarks = 0;
  std::vector<double> errors;

  // ALIGNMENT AT BOOTSTRAP, so the error is honest. The estimator DEFINES its own world:
  // origin at the first body pose, +Z along gravity, yaw arbitrary. That frame is NOT the
  // ground-truth frame, so a raw |p_est - p_gt| conflates real drift with a fixed origin/yaw
  // offset -- which is exactly what inflated the earlier "3.4 m". T_align is the one rigid
  // transform between the two worlds, fixed at the bootstrap instant; everything after is the
  // drift THROUGH it.
  Eigen::Isometry3d T_align = Eigen::Isometry3d::Identity();
  const double t0 = bag.frames.front().t;

  for (std::size_t k = 0; k < bag.frames.size(); ++k) {
    const auto & f = bag.frames[k];

    glassvio::MeasureGroup g;
    g.header.stamp.sec = static_cast<int32_t>(std::floor(f.t));
    g.header.stamp.nanosec = static_cast<uint32_t>((f.t - std::floor(f.t)) * 1e9);
    for (const auto & entry : f.by_id) {
      g.features.ids.push_back(entry.first);
      g.features.points.push_back(entry.second);
    }
    // Every IMU sample in (t_prev, f.t], in order, exactly once -- an unbroken chain, which is
    // precisely what the node's dropped frames destroy.
    for (; imu_cursor < imu.size() && imu[imu_cursor].t <= f.t; ++imu_cursor) {
      if (imu[imu_cursor].t > t_prev) {
        g.imu.push_back(toMsg(imu[imu_cursor]));
      }
    }
    t_prev = f.t;

    const glassvio::FrameResult r = est.process(g);

    // The estimator body pose in ITS world, this frame.
    Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
    T_wb.linear() = est.state().R.matrix();
    T_wb.translation() = est.state().p;

    if (r.stage == glassvio::FrameResult::Stage::Bootstrapped) {
      bootstrapped_at = static_cast<int>(k);
      boot_landmarks = est.landmarks().size();
      // The one rigid transform between est-world and gt-world, fixed here forever.
      T_align = bag.gt.at(f.t) * T_wb.inverse();

      const Eigen::Isometry3d T_cw = (T_wb * calib.T_cam_imu.inverse()).inverse();
      std::vector<double> depths;
      for (const auto & lm : est.landmarks()) {
        depths.push_back((T_cw * lm.second).z());
      }
      std::sort(depths.begin(), depths.end());
      boot_depth_median = depths.empty() ? 0.0 : depths[depths.size() / 2];

      std::printf(
        "\nBOOTSTRAP at frame %d (%.1f s): %zu landmarks, median depth %.3f m, "
        "|v|=%.2f\n  bg = [%+.4f %+.4f %+.4f]  (truth [%+.4f %+.4f %+.4f])\n",
        bootstrapped_at, f.t - t0, boot_landmarks, boot_depth_median, est.state().v.norm(),
        est.state().bg.x(), est.state().bg.y(), est.state().bg.z(),
        bag.gt.gyroBias(f.t).x(), bag.gt.gyroBias(f.t).y(), bag.gt.gyroBias(f.t).z());
    } else if (r.stage == glassvio::FrameResult::Stage::Lost) {
      if (lost_at < 0 && bootstrapped_at >= 0) {
        lost_at = static_cast<int>(k);
      }
    }

    // --- CSV row, and the tracked-error accumulation, both off the ALIGNED pose.
    if (bootstrapped_at >= 0 &&
      (r.stage == glassvio::FrameResult::Stage::Tracking ||
      r.stage == glassvio::FrameResult::Stage::Bootstrapped))
    {
      const Eigen::Isometry3d est_in_gt = T_align * T_wb;
      const Eigen::Vector3d p = est_in_gt.translation();
      const Eigen::Vector3d gp = bag.gt.at(f.t).translation();
      const Eigen::Vector3d v = T_align.linear() * est.state().v;
      const Eigen::Vector3d gv = bag.gt.velocity(f.t);
      const double pos_err = (p - gp).norm();
      const double vel_err = (v - gv).norm();
      if (r.stage == glassvio::FrameResult::Stage::Tracking) {
        ++tracked;
        errors.push_back(pos_err);
      }

      const auto & s = est.state();
      const Eigen::Vector3d gbg = bag.gt.gyroBias(f.t);
      const Eigen::Vector3d gba = bag.gt.accelBias(f.t);
      csv << (f.t - t0) << ","
          << (r.stage == glassvio::FrameResult::Stage::Bootstrapped ? "boot" : "track") << ","
          << r.features << "," << est.map().size() << "," << est.map().pending() << ","
          << r.rmse_px << "," << pos_err << "," << vel_err << ","
          << p.x() << "," << p.y() << "," << p.z() << ","
          << gp.x() << "," << gp.y() << "," << gp.z() << ","
          << v.x() << "," << v.y() << "," << v.z() << ","
          << gv.x() << "," << gv.y() << "," << gv.z() << ","
          << s.bg.x() << "," << s.bg.y() << "," << s.bg.z() << ","
          << gbg.x() << "," << gbg.y() << "," << gbg.z() << ","
          << s.ba.x() << "," << s.ba.y() << "," << s.ba.z() << ","
          << gba.x() << "," << gba.y() << "," << gba.z() << "\n";
    }
  }

  csv.close();
  std::printf("\nper-frame CSV -> %s\n\n=== VERDICT ===\n", csv_path.c_str());
  if (bootstrapped_at < 0) {
    std::printf("never bootstrapped -- stage [2]/[3]/[4] never cleared offline either\n");
    return 1;
  }

  const double fps = (bag.frames.size() - 1) /
    (bag.frames.back().t - bag.frames.front().t);
  std::printf(
    "bootstrapped, then tracked %d frames (%.2f s) before %s\n",
    tracked, tracked / fps,
    lost_at < 0 ? "the sequence ended" : "losing the scene");

  if (boot_depth_median < 0.3) {
    std::printf(
      "\nLANDMARKS ARE %.0fx TOO CLOSE (median depth %.3f m in a metre-scale room).\n"
      "The bug reproduces DETERMINISTICALLY here, with no dropped frames. So it is NOT the\n"
      "online plumbing -- bootstrap()'s own scale/window selection is wrong. Fix the\n"
      "estimator, not the node.\n",
      3.0 / std::max(boot_depth_median, 1e-3), boot_depth_median);
    return 1;
  }

  std::printf(
    "\nlandmarks at a sane %.2f m median depth, and it tracked %.2f s.\n"
    "The bootstrap LOGIC is sound: the node's failure is its ONLINE frame handling\n"
    "(dropped frames corrupting the SfM window), not the estimator.\n",
    boot_depth_median, tracked / fps);
  if (!errors.empty()) {
    std::sort(errors.begin(), errors.end());
    std::printf("median position error vs ground truth: %.3f m\n", errors[errors.size() / 2]);
  }
  return 0;
}
