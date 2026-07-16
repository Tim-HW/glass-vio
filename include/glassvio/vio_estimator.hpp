#ifndef GLASSVIO_VIO_ESTIMATOR_HPP
#define GLASSVIO_VIO_ESTIMATOR_HPP

#include <memory>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glassvio/camera_calib.hpp"
#include "glassvio/dataset.hpp"
#include "glassvio/types.hpp"
#include "glassvio/vio_initializer.hpp"
#include "glassvio/visual_registration.hpp"

namespace glassvio
{

using namespace glass_core;  // NOLINT(build/namespaces)

struct EstimatorParams
{
  /// Frames to collect before attempting the bootstrap. Must be enough for stage [2] to find
  /// a base pair with real parallax; 30 is ~1.5 s at EuRoC's 20 Hz.
  int bootstrap_frames = 30;
  InitializerParams init;
  VisualParams visual;

  /// The bootstrap's own uncertainty, as standard deviations. NOT zero, and the temptation to
  /// make it small is the bug this whole design exists to avoid: an over-confident prior is
  /// indistinguishable from a fixed state, and a fixed state is what makes the IMU factor
  /// overrule every pixel. Measured on EuRoC V1_01, stage [4] leaves v_0 ~0.04 m/s out and
  /// stage [3] leaves b_g ~1.4e-3 out, so these are deliberately looser than that.
  double sigma_attitude_rad = 2.0 * M_PI / 180.0;
  double sigma_position_m = 0.05;
  double sigma_velocity_mps = 0.10;
  double sigma_gyro_bias = 2.0e-3;
  /// The loosest, because NOTHING estimates it. Stage [3] recovers only the gyro bias, and
  /// EuRoC's accel bias is ~0.55 m/s^2 on one axis. The covariance admits that ignorance
  /// rather than hiding it.
  double sigma_accel_bias = 0.10;
};

struct FrameResult
{
  enum class Stage { Collecting, Bootstrapped, Tracking, Lost };
  Stage stage = Stage::Collecting;

  /// The pipeline produced a state we trust. NOT the same as `stage == Tracking`.
  bool pose_trusted = false;
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
  int features = 0;
  double rmse_px = 0.0;
};

/// THE ESTIMATOR -- collect, bootstrap, then track. glasslio's LioEstimator, for a camera.
///
/// No subscriptions, no publishers, no parameters, no threads: the caller feeds it
/// MeasureGroups and asks what came out. That separation is what makes it drivable from a
/// test, which is the only reason the bootstrap's failure modes were ever measurable.
///
/// NO ImuInit HERE, and that is deliberate. glass_core's static-window bootstrap estimates a
/// gyro bias and gravity by assuming the sensor was at REST -- an assumption a MAV never
/// honours, and which on KITTI produced a "bias" that was really the car's yaw rate (3.9x
/// worse than using zero). VioInitializer needs no such assumption: stage [3] reads the bias
/// off vision's rotations, and stage [4] reads gravity out of a linear solve. The static
/// window is not needed, so it is not used.
///
/// THE GAUGE. A monocular VIO cannot know where the world's origin is or which way it faces.
/// The world frame is therefore DEFINED here: origin at the first body pose, +Z along
/// measured gravity, yaw arbitrary (gravity constrains two of three DoF -- the third is
/// unobservable, exactly as in ImuInit's FromTwoVectors). Everything after is estimated.
class VioEstimator
{
public:
  VioEstimator(const CameraCalib & calib, const EstimatorParams & params);

  /// Feed one synced frame. Safe to call before initialisation -- it collects.
  FrameResult process(const MeasureGroup & group);

  void reset();

  bool initialized() const {return initialized_;}
  const NavState & state() const {return x_;}
  /// Landmarks, metric, in the world frame. Fixed once the bootstrap sets them -- which is
  /// why tracking starves after ~2.5 s and a sliding window is the next thing.
  const std::unordered_map<long, Eigen::Vector3d> & landmarks() const {return landmarks_;}

private:
  bool bootstrap();

  CameraCalib calib_;
  EstimatorParams p_;
  std::unique_ptr<VioInitializer> init_;

  // Collected until the bootstrap fires. SfmFrame is the initialiser's own vocabulary, so the
  // MeasureGroup's tracks are converted at the boundary.
  std::vector<SfmFrame> frames_;
  ImuBuffer imu_;

  bool initialized_ = false;
  bool lost_ = false;
  NavState x_;
  Eigen::Matrix<double, kNavDim, kNavDim> P_ =
    Eigen::Matrix<double, kNavDim, kNavDim>::Zero();
  Eigen::Matrix<double, 6, 6> bias_information_ = Eigen::Matrix<double, 6, 6>::Identity();
  Eigen::Vector3d gravity_world_{0.0, 0.0, -kGravity};
  std::unordered_map<long, Eigen::Vector3d> landmarks_;
  double t_prev_ = 0.0;
};

}  // namespace glassvio

#endif  // GLASSVIO_VIO_ESTIMATOR_HPP
