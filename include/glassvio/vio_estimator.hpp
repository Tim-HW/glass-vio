#ifndef GLASSVIO_VIO_ESTIMATOR_HPP
#define GLASSVIO_VIO_ESTIMATOR_HPP

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glassvio/camera_calib.hpp"
#include "glassvio/dataset.hpp"
#include "glassvio/landmark_map.hpp"
#include "glassvio/types.hpp"
#include "glassvio/vio_initializer.hpp"
#include "glassvio/visual_registration.hpp"

namespace glassvio
{

using namespace glass_core;  // NOLINT(build/namespaces)

struct EstimatorParams
{
  /// Frames COLLECTED before attempting the bootstrap. Deliberately NOT the same as
  /// init.window_frames, and tying them together is a bug that has now been made three times
  /// in this project.
  ///
  /// Stage [2] wants a SHORT window -- landmarks die as the camera moves, so a long one
  /// reconstructs nothing. Stage [3] wants a LONG one -- the gyro bias is a constant of the
  /// sensor, so every pair is evidence, and averaging is what pulls the estimate below the
  /// vision noise floor of any single pair (~0.28 deg on EuRoC).
  ///
  /// At bias_frame_gap = 5, N collected frames yield (N - 5) / 5 pairs. 30 frames gives 5 --
  /// below bias_min_pairs = 8, so stage [3] failed, run() bailed, and the node collected
  /// forever. 50 gives 9. The offline checks never saw this because they load the whole bag,
  /// so "scan to the end of the stream" silently meant 2912 frames; online, the end is now.
  int bootstrap_frames = 120;
  /// Frames between bootstrap ATTEMPTS. An attempt is a full SfM plus ~10 essential matrices;
  /// running one per frame starved the worker and made the queue drop groups. A slid window
  /// shares all but one frame with the last attempt, so retrying immediately re-asks a
  /// question whose answer cannot have changed.
  int bootstrap_retry_every = 5;
  /// Frames the tracker may COAST on the IMU prediction when the visual solve is
  /// under-constrained, before declaring LOST. Its job is the post-bootstrap hole: the
  /// expensive bootstrap drops frames, KLT ids churn, and the map needs a few frames of
  /// baseline to re-triangulate against live ids. Dead reckoning is good to 0.16 m / 2 s, so a
  /// short coast is cheap; replenished on every successful solve, so it also rides out a
  /// transient dip (a blank wall, a hard turn).
  int warmup_frames = 20;
  /// THE SLIDING WINDOW. Landmarks are maintained rather than frozen: triangulated as tracks
  /// mature against the (metric) state, dropped when they leave view or stop fitting. Without
  /// it, tracking starved after ~4 s with 0 landmarks in view -- measured, not predicted.
  LandmarkMapParams map;
  /// Stage [2]'s window is a SUBSET of what is collected. Leave it short.
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
  /// MODERATE, and the story here is a real measured limit. EuRoC's b_a is 0.55 m/s^2 and we
  /// seed it at 0, so the instinct is a loose prior to let the solve reach it (OpenVINS's
  /// init_dyn_inflation_ba: 100). Measured across 0.1 / 0.4 / 1.0, b_a NEVER converges to
  /// 0.55 -- it is only weakly observable per frame (dv/db_a ~ dt ~ 0.05), so a loose prior
  /// only lets it WANDER and absorb the bootstrap's velocity/scale error, which made drift
  /// WORSE (0.27 m at 0.1 -> 1.98 m at 1.0). A single-frame solve cannot pin a weakly-observable
  /// bias; that is the sliding window's job. So this stays moderate: loose enough to adapt if
  /// excitation is genuinely strong, tight enough not to soak up error to avoid declaring LOST.
  double sigma_accel_bias = 0.2;
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
  /// Why the last bootstrap attempt failed, by STAGE. Empty when it succeeded. A bare bool
  /// here left the node logging "collecting..." forever with the reason invisible.
  const std::string & lastFailure() const {return last_failure_;}
  int lastLandmarks() const {return last_landmarks_;}
  int lastBiasPairs() const {return last_bias_pairs_;}
  const NavState & state() const {return x_;}
  /// Landmarks, metric, in the world frame. Maintained by LandmarkMap: fixed WITHIN a solve
  /// (that is what keeps the state at 15 DoF), but grown and pruned between them.
  const std::unordered_map<long, Eigen::Vector3d> & landmarks() const
  {
    return map_->landmarks();
  }
  const LandmarkMap & map() const {return *map_;}

private:
  bool bootstrap();

  /// Pose from vision alone: PnP the current frame's observed landmarks against the metric
  /// map. This is what closes the bootstrap latency gap. The bootstrap runs on the worker
  /// while the queue drops ~1 s of frames, so the first tracking frame is ~1 s after the
  /// anchor -- and predictState dead-reckoning across that whole hole lands badly enough to
  /// push near-field landmarks behind the camera (measured: 64 matched, 0 pass cheirality).
  /// PnP needs no previous pose and no time base, so it is immune to the gap: it reads the
  /// pose straight off the 3D-2D correspondences. Fills only R and p; v and the biases stay
  /// with predictState, which PnP cannot see. False if too few inliers.
  bool pnpFromMap(
    const std::unordered_map<long, cv::Point2f> & obs, NavState & out) const;

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
  Eigen::Vector3d gravity_world_{0.0, 0.0, -kGravity};
  std::unique_ptr<LandmarkMap> map_;
  double t_prev_ = 0.0;
  std::string last_failure_;
  int last_landmarks_ = 0;
  int last_bias_pairs_ = 0;
  int since_attempt_ = 0;
  int warmup_ = 0;
};

}  // namespace glassvio

#endif  // GLASSVIO_VIO_ESTIMATOR_HPP
