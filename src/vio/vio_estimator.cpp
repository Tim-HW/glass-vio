#include "glassvio/vio_estimator.hpp"

#include <algorithm>
#include <cmath>

namespace glassvio
{

VioEstimator::VioEstimator(const CameraCalib & calib, const EstimatorParams & params)
: calib_(calib), p_(params)
{
  init_ = std::make_unique<VioInitializer>(calib_, p_.init);

  // The bias random-walk prior: how fast a bias may DRIFT between frames, from the datasheet
  // rather than a guess. Note it says nothing about how WRONG a bias might be to begin with --
  // that is P_'s job, and conflating the two is what glasslio's tight path warns about.
  bias_information_.setIdentity();
  bias_information_.topLeftCorner<3, 3>() /=
    (calib_.gyro_random_walk * calib_.gyro_random_walk);
  bias_information_.bottomRightCorner<3, 3>() /=
    (calib_.accel_random_walk * calib_.accel_random_walk);
}

void VioEstimator::reset()
{
  frames_.clear();
  imu_ = ImuBuffer();
  landmarks_.clear();
  initialized_ = false;
  lost_ = false;
  x_ = NavState();
  P_.setZero();
  t_prev_ = 0.0;
}

// =================================================================================
// THE GAUGE. A monocular VIO cannot know where the world is; it DEFINES it.
// =================================================================================

bool VioEstimator::bootstrap()
{
  const InitResult r = init_->run(frames_, imu_, 0);
  if (!r.ok || !r.scale_observable) {
    return false;
  }

  // Gravity fixes two of three rotational DoF. Yaw is unobservable and is left at zero --
  // FromTwoVectors gives the MINIMAL rotation, which is exactly right for the same reason
  // ImuInit uses it: there is no information to determine the third axis, so do not invent it.
  const Eigen::Quaterniond q = Eigen::Quaterniond::FromTwoVectors(
    r.gravity_sfm.normalized(), Eigen::Vector3d(0.0, 0.0, -1.0));
  Eigen::Isometry3d T_world_c0 = Eigen::Isometry3d::Identity();
  T_world_c0.linear() = q.normalized().toRotationMatrix();

  // Origin at the first BODY pose, not the first camera. In the SfM frame the base camera is
  // the origin, so the body sits at the extrinsic's translation -- already metric, since the
  // extrinsic never carried the ruler.
  const Eigen::Isometry3d T_c0_b0 = calib_.T_cam_imu;
  T_world_c0.translation() = -T_world_c0.linear() * T_c0_b0.translation();

  // Landmarks -> world, in METRES. This is where the ruler finally dies: s converts the
  // invented baseline into metres, and T_world_c0 carries them into the frame just defined.
  landmarks_.clear();
  for (const auto & lm : r.sfm.landmark) {
    landmarks_.emplace(lm.first, T_world_c0 * (r.scale * lm.second));
  }

  const Eigen::Isometry3d T_world_b0 = T_world_c0 * T_c0_b0;
  x_ = NavState();
  x_.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_world_b0.linear()));
  x_.p = T_world_b0.translation();          // zero by construction: the origin is here
  x_.v = T_world_c0.linear() * r.velocity_sfm.front();   // stage [4]
  x_.bg = r.gyro_bias;                                   // stage [3]
  // x_.ba stays zero: NOTHING estimates it. P_ below admits that rather than hiding it.

  // The bootstrap's own uncertainty. Emphatically not zero -- an over-confident prior is
  // indistinguishable from a fixed state, which is precisely the failure this design exists
  // to avoid.
  P_.setZero();
  P_.block<3, 3>(kIdxPhi, kIdxPhi).diagonal().setConstant(
    p_.sigma_attitude_rad * p_.sigma_attitude_rad);
  P_.block<3, 3>(kIdxPos, kIdxPos).diagonal().setConstant(
    p_.sigma_position_m * p_.sigma_position_m);
  P_.block<3, 3>(kIdxVel, kIdxVel).diagonal().setConstant(
    p_.sigma_velocity_mps * p_.sigma_velocity_mps);
  P_.block<3, 3>(kIdxBg, kIdxBg).diagonal().setConstant(
    p_.sigma_gyro_bias * p_.sigma_gyro_bias);
  P_.block<3, 3>(kIdxBa, kIdxBa).diagonal().setConstant(
    p_.sigma_accel_bias * p_.sigma_accel_bias);

  t_prev_ = frames_[r.sfm.base].t;
  initialized_ = true;
  return true;
}

// =================================================================================
// COLLECT -> BOOTSTRAP -> TRACK
// =================================================================================

FrameResult VioEstimator::process(const MeasureGroup & group)
{
  FrameResult out;
  const double t = static_cast<double>(group.header.stamp.sec) +
    group.header.stamp.nanosec * 1e-9;

  // Every sample, always: the buffer must span the bootstrap window AND every interval after
  // it, and the sync guarantees these arrive covering [t_prev, t].
  for (const auto & m : group.imu) {
    imu_.add(
      {static_cast<double>(m->header.stamp.sec) + m->header.stamp.nanosec * 1e-9,
        {m->angular_velocity.x, m->angular_velocity.y, m->angular_velocity.z},
        {m->linear_acceleration.x, m->linear_acceleration.y, m->linear_acceleration.z}});
  }

  if (!initialized_) {
    SfmFrame f;
    f.t = t;
    for (std::size_t i = 0; i < group.features.ids.size(); ++i) {
      f.by_id.emplace(group.features.ids[i], group.features.points[i]);
    }
    frames_.push_back(std::move(f));

    if (static_cast<int>(frames_.size()) < p_.bootstrap_frames) {
      out.stage = FrameResult::Stage::Collecting;
      return out;
    }
    if (!bootstrap()) {
      // Not a failure yet: slide the window on. A MAV that is rotating in place gives stage
      // [2] no baseline, and the honest response is to wait for translation rather than
      // reconstruct from nothing.
      frames_.erase(frames_.begin());
      out.stage = FrameResult::Stage::Collecting;
      return out;
    }
    out.stage = FrameResult::Stage::Bootstrapped;
    out.pose_trusted = true;
    out.pose = Eigen::Isometry3d::Identity();
    out.pose.linear() = x_.R.matrix();
    out.pose.translation() = x_.p;
    out.velocity = x_.v;
    return out;
  }

  if (lost_) {
    out.stage = FrameResult::Stage::Lost;
    return out;
  }

  // --- Track.
  ImuPreintegration pre(x_.bg, x_.ba, calib_.gyro_noise, calib_.accel_noise);
  if (!imu_.preintegrate(
      t_prev_, t, x_.bg, x_.ba, pre, calib_.gyro_noise, calib_.accel_noise))
  {
    lost_ = true;   // a hole in the stream: the factor has no delta to offer
    out.stage = FrameResult::Stage::Lost;
    return out;
  }

  // The IMU's own prediction seeds Gauss-Newton -- strictly better than constant velocity,
  // because it uses the accelerometer that actually measured the change.
  const NavState guess = predictState(x_, pre, gravity_world_);

  std::unordered_map<long, cv::Point2f> obs;
  for (std::size_t i = 0; i < group.features.ids.size(); ++i) {
    if (landmarks_.count(group.features.ids[i])) {
      obs.emplace(group.features.ids[i], group.features.points[i]);
    }
  }

  const VisualResult res = solveFrame(
    landmarks_, obs, x_, P_, pre, gravity_world_, guess, bias_information_, calib_, p_.visual);
  if (!res.valid) {
    // STARVATION, and it is expected: landmarks are triangulated once and held fixed, so the
    // camera eventually flies past all of them. A sliding window is what fixes this, not a
    // looser min_features.
    lost_ = true;
    out.stage = FrameResult::Stage::Lost;
    out.features = res.features;
    return out;
  }

  x_ = res.state;
  t_prev_ = t;

  // Carry the posterior. res.H is every factor's accumulated information, so its inverse IS
  // the new covariance -- what NormalEquationsN::H() has always been for. LDLT because H is
  // positive SEMI-definite by construction: singular when the geometry degenerates, never
  // indefinite. If it comes back unusable, keep the old P rather than propagate garbage.
  const Eigen::Matrix<double, kNavDim, kNavDim> P_new =
    res.H.ldlt().solve(Eigen::Matrix<double, kNavDim, kNavDim>::Identity());
  if (P_new.allFinite() && P_new.diagonal().minCoeff() > 0.0) {
    P_ = P_new;
  }

  out.stage = FrameResult::Stage::Tracking;
  out.pose_trusted = true;
  out.pose.linear() = x_.R.matrix();
  out.pose.translation() = x_.p;
  out.velocity = x_.v;
  out.features = res.features;
  out.rmse_px = res.rmse_px;
  return out;
}

}  // namespace glassvio
