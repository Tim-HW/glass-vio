#include "glassvio/vio_estimator.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include <opencv2/calib3d.hpp>

namespace glassvio
{

VioEstimator::VioEstimator(const CameraCalib & calib, const EstimatorParams & params)
: calib_(calib), p_(params)
{
  init_ = std::make_unique<VioInitializer>(calib_, p_.init);
  map_ = std::make_unique<LandmarkMap>(calib_, p_.map);

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
  map_->clear();
  initialized_ = false;
  lost_ = false;
  warmup_ = 0;
  x_ = NavState();
  P_.setZero();
  t_prev_ = 0.0;
}

// =================================================================================
// THE GAUGE. A monocular VIO cannot know where the world is; it DEFINES it.
// =================================================================================

bool VioEstimator::bootstrap()
{
  // THE SFM WINDOW IS THE NEWEST FRAMES, NOT THE OLDEST, and getting this backwards is why
  // the node collected forever. Two reasons, and both are fatal:
  //
  //   * Its landmarks must be ALIVE NOW. Triangulating from frames 6 s in the past yields
  //     points the camera has already flown past -- the solve starves on its first step.
  //   * The MAV starts on the ground. frames_[0..30) is the stationary part for the whole
  //     first buffer, so stage [2] would report "no parallax" no matter how much the vehicle
  //     later moved.
  //
  // Stage [3] still gets EVERYTHING (bias_window_frames = 0 scans to the end): the bias is a
  // constant of the sensor, so every frame is evidence, while the reconstruction wants only
  // the recent ones. Two spans, again.
  const int begin = std::max(
    0, static_cast<int>(frames_.size()) - p_.init.window_frames);
  const InitResult r = init_->run(frames_, imu_, begin);

  // WHICH STAGE FAILED, not merely that one did. A bare bool here cost an hour: the node sat
  // on "collecting..." forever and the reason (stage [3] starved of pairs) was invisible.
  last_failure_ = r.sfm.pose.empty() ?
    "[2] SfM: no base pair with parallax AND landmarks (the MAV is rotating, not translating)" :
    r.bias_pairs < p_.init.bias_min_pairs ?
    "[3] gyro bias: too few pairs" :
    r.align_intervals < p_.init.align_min_intervals ?
    "[4] align: too few intervals" :
    !r.ok ?
    "[4] align: |g| implausible -- formulation, frames or extrinsic" :
    !r.scale_observable ?
    "[4] scale not observable (s <= 0: the s/v_0 ridge won -- no accelerometer excitation)" :
    "";
  last_landmarks_ = static_cast<int>(r.sfm.landmark.size());
  last_bias_pairs_ = r.bias_pairs;

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
  std::unordered_map<long, Eigen::Vector3d> seed;
  for (const auto & lm : r.sfm.landmark) {
    seed.emplace(lm.first, T_world_c0 * (r.scale * lm.second));
  }
  map_->clear();
  map_->seed(seed);

  // ANCHOR AT THE LAST POSE SFM SOLVED, NOT THE BASE. The base is `window_frames` in the past;
  // anchoring there would hand the tracker a state ~1.5 s stale and a t_prev to match, so its
  // very first preintegration would span the whole window and its landmarks would already be
  // behind it. The last solved frame IS the present.
  const int last = r.frames.back();
  const Eigen::Isometry3d T_c0_ck = r.sfm.pose.at(last).inverse();

  // THE RULER DIES HERE, and only the translation carries it. sfm.pose's rotation is
  // scale-free, but its position is in ruler units and must be multiplied by s; the extrinsic
  // is already metric and must NOT be. Composing the two Isometries directly would mix the
  // units silently -- the same split the alignment's dp equation is built on.
  Eigen::Isometry3d T_c0_bk = Eigen::Isometry3d::Identity();
  T_c0_bk.linear() = T_c0_ck.linear() * calib_.T_cam_imu.linear();
  T_c0_bk.translation() =
    r.scale * T_c0_ck.translation() + T_c0_ck.linear() * calib_.T_cam_imu.translation();
  const Eigen::Isometry3d T_world_bk = T_world_c0 * T_c0_bk;

  x_ = NavState();
  x_.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_world_bk.linear()));
  x_.p = T_world_bk.translation();
  x_.v = T_world_c0.linear() * r.velocity_sfm.back();    // stage [4], at that same frame
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

  // The tracker resumes from the frame we anchored at -- the LAST one SfM solved, not the
  // base. Anchoring at the base would make the first preintegration span the whole window.
  t_prev_ = frames_[last].t;
  warmup_ = p_.warmup_frames;   // the post-bootstrap id-churn hole is exactly what this rides out
  initialized_ = true;
  return true;
}

bool VioEstimator::pnpFromMap(
  const std::unordered_map<long, cv::Point2f> & obs, NavState & out) const
{
  std::vector<cv::Point3f> obj;
  std::vector<cv::Point2f> img;
  obj.reserve(obs.size());
  img.reserve(obs.size());
  for (const auto & entry : obs) {
    const auto lm = map_->landmarks().find(entry.first);
    if (lm == map_->landmarks().end()) {
      continue;
    }
    obj.emplace_back(lm->second.x(), lm->second.y(), lm->second.z());
    img.push_back(entry.second);
  }
  if (static_cast<int>(obj.size()) < p_.visual.min_features) {
    return false;
  }

  cv::Mat rvec, tvec;
  // RANSAC: a single stale/drifted track would otherwise drag the whole pose. 3 px reprojection
  // threshold matches the map's own outlier gate.
  if (!cv::solvePnPRansac(
      obj, img, calib_.cvK(), cv::Mat(), rvec, tvec, false, 100, 3.0, 0.99))
  {
    return false;
  }

  // solvePnP returns X_cam = R X_world + t, i.e. T_cam_world. The camera pose in the world is
  // its inverse; the BODY pose composes the extrinsic (X_cam = T_cam_imu X_imu).
  cv::Mat Rm;
  cv::Rodrigues(rvec, Rm);
  Eigen::Matrix3d R;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R(i, j) = Rm.at<double>(i, j);
    }
  }
  Eigen::Isometry3d T_cam_world = Eigen::Isometry3d::Identity();
  T_cam_world.linear() = R;
  T_cam_world.translation() =
    Eigen::Vector3d(tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
  const Eigen::Isometry3d T_world_imu = T_cam_world.inverse() * calib_.T_cam_imu;

  out.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_world_imu.linear()));
  out.p = T_world_imu.translation();
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

    // RETRY EVERY Nth FRAME, NOT EVERY FRAME. A bootstrap attempt is a full SfM plus ~10
    // essential matrices -- tens of milliseconds against a 50 ms budget. Running it on every
    // frame made the worker fall behind, which made the queue drop groups, which (before the
    // splice above) punched holes in the IMU chain and failed the very stage we were retrying.
    // A slid window shares 49 of its 50 frames with the last attempt anyway, so retrying
    // immediately re-asks a question whose answer cannot have changed much.
    if (++since_attempt_ < p_.bootstrap_retry_every) {
      frames_.erase(frames_.begin());
      out.stage = FrameResult::Stage::Collecting;
      return out;
    }
    since_attempt_ = 0;

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

  std::unordered_map<long, cv::Point2f> obs;
  for (std::size_t i = 0; i < group.features.ids.size(); ++i) {
    if (map_->landmarks().count(group.features.ids[i])) {
      obs.emplace(group.features.ids[i], group.features.points[i]);
    }
  }

  // The seed for Gauss-Newton. predictState is the right seed in steady state -- it uses the
  // accelerometer that actually measured the motion. But across the bootstrap latency gap the
  // first interval is ~1 s, and dead-reckoning that far lands outside the solve's basin. PnP
  // reads the pose straight off the map with no time base, so it is immune to the gap. Use it
  // for R and p when it succeeds; velocity and biases stay with predictState, which PnP cannot
  // see. This makes the handoff robust without special-casing "the first frame".
  NavState guess = predictState(x_, pre, gravity_world_);
  NavState pnp;
  if (pnpFromMap(obs, pnp)) {
    guess.R = pnp.R;
    guess.p = pnp.p;
  }

  // SOLVE FIRST, MAP SECOND. Landmarks are FIXED during the solve -- that is what keeps the
  // state at 15 DoF and lets NormalEquationsN<15> be reused, exactly as glasslio holds its
  // map's planes fixed while ICP runs.
  const VisualResult res = solveFrame(
    map_->landmarks(), obs, x_, P_, pre, gravity_world_, guess, bias_information_, calib_,
    p_.visual);
  if (!res.valid) {
    // WARMUP GRACE, and it is not papering over a failure -- it fixes the last online/offline
    // gap. The bootstrap is expensive and runs on the worker; while it runs, the queue drops
    // frames and the KLT ids churn, so the frame we resume on shares almost no ids with the
    // SfM landmarks. Offline never sees this (bootstrap is instant, the next frame is
    // adjacent); online there is a ~1 s hole.
    //
    // The map CAN recover -- but a fresh track needs a few frames of widening baseline before
    // it has 1 deg of parallax and can be triangulated, and solveFrame would declare LOST
    // first. So for a short window after the bootstrap we COAST on the IMU prediction and keep
    // inserting: the state dead-reckons (good to 0.16 m / 2 s, phase 1) while the map
    // repopulates with live ids, and vision takes back over once it can.
    if (warmup_ > 0) {
      --warmup_;
      x_ = guess;
      t_prev_ = t;
      Eigen::Isometry3d T_wc = Eigen::Isometry3d::Identity();
      T_wc.linear() = x_.R.matrix();
      T_wc.translation() = x_.p;
      map_->insert(group.features, T_wc * calib_.T_cam_imu.inverse());
      out.stage = FrameResult::Stage::Tracking;
      out.pose_trusted = true;   // IMU dead reckoning over a few frames is trustworthy
      out.pose = T_wc;
      out.velocity = x_.v;
      out.features = res.features;
      return out;
    }
    // Grace exhausted: a real loss. The tracker lost the scene (blank wall, motion blur, a
    // hard turn), not merely outlived a frozen set.
    lost_ = true;
    out.stage = FrameResult::Stage::Lost;
    out.features = res.features;
    return out;
  }

  x_ = res.state;
  t_prev_ = t;
  warmup_ = p_.warmup_frames;   // vision is healthy again -- restore the coast budget

  // --- THE SLIDING WINDOW: fold this frame into the map, using the pose we JUST solved.
  //
  // The ordering is the whole trick. Insert before solving and the map would be built from the
  // IMU's prediction rather than a vision-corrected pose -- triangulating against a guess, then
  // fitting to the result, which is a feedback loop that would happily converge on nonsense.
  // Insert after, and every new landmark is anchored to a pose the camera itself just agreed
  // with.
  //
  // And this is where the ruler stays dead: x_ is metric, so triangulation yields metres with
  // no scale, no alignment and no gauge. Stage [2]'s invented baseline was a one-time cost.
  Eigen::Isometry3d T_world_cam = Eigen::Isometry3d::Identity();
  T_world_cam.linear() = x_.R.matrix();
  T_world_cam.translation() = x_.p;
  T_world_cam = T_world_cam * calib_.T_cam_imu.inverse();
  map_->insert(group.features, T_world_cam);

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
