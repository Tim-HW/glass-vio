#include "glassvio/vio_initializer.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "glass_core/gauss_newton.hpp"

namespace glassvio
{

VioInitializer::VioInitializer(const CameraCalib & calib, const InitializerParams & params)
: calib_(calib), p_(params)
{
}

// =================================================================================
// [2] RECONSTRUCT -- shape, in an invented ruler.
// =================================================================================

SfmWindow VioInitializer::reconstruct(const std::vector<SfmFrame> & frames, int begin) const
{
  const int end = std::min<int>(begin + p_.window_frames, static_cast<int>(frames.size()));
  return buildSfmWindow(frames, begin, end, calib_, p_.sfm);
}

// =================================================================================
// [3] GYRO BIAS -- from rotations, which are free of scale.
// =================================================================================

bool VioInitializer::estimateGyroBias(
  const std::vector<SfmFrame> & frames, const ImuBuffer & imu, int begin,
  Eigen::Vector3d & bias, int & pairs) const
{
  // Three unknowns, so glass_core's own normal equations at N = 3. The same accumulator the
  // LiDAR path uses at N = 15, and the same one the reprojection factor will use -- one
  // engine, and it does not care how wide the state is.
  NormalEquationsN<3> eq;
  pairs = 0;

  // NOT window_frames: see InitializerParams::bias_window_frames. The bias is a constant, so
  // it gets the whole stream, while the SfM window stays short.
  const int end = p_.bias_window_frames > 0 ?
    std::min<int>(begin + p_.bias_window_frames, static_cast<int>(frames.size())) :
    static_cast<int>(frames.size());
  for (int i = begin; i + p_.bias_frame_gap < end; i += p_.bias_frame_gap) {
    const SfmFrame & fi = frames[i];
    const SfmFrame & fj = frames[i + p_.bias_frame_gap];

    Sophus::SO3d dR_vis;
    if (!relativeBodyRotation(fi, fj, calib_, dR_vis, p_.sfm.min_shared)) {
      continue;
    }

    ImuPreintegration pre(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), 1.0, 1.0);
    if (!imu.preintegrate(
        fi.t, fj.t, Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(), pre,
        calib_.gyro_noise, calib_.accel_noise))
    {
      continue;   // IMU dropout inside the interval
    }

    // With dR_hat the preintegrated rotation and dR_vis the essential matrix's,
    //
    //   r(dbg) = Log( (dR_hat . Exp(J_bg . dbg))^T . dR_vis ) ~= r_0 - J_bg . dbg
    //
    // which is linear in dbg, so one solve suffices. d r / d dbg = -J_bg to first order.
    const Eigen::Vector3d r0 = (pre.dR().inverse() * dR_vis).log();
    eq.addBlock<3>(r0, -pre.dR_dbg(), Eigen::Matrix3d::Identity());
    ++pairs;
  }

  if (pairs < p_.bias_min_pairs) {
    return false;
  }
  bias = eq.solve();
  return bias.allFinite();
}

// =================================================================================
// [4] ALIGN -- the metre.
// =================================================================================

bool VioInitializer::align(
  const std::vector<SfmFrame> & frames, const ImuBuffer & imu,
  const SfmWindow & sfm, const Eigen::Vector3d & gyro_bias, InitResult & out) const
{
  out.frames.clear();
  for (const auto & entry : sfm.pose) {
    out.frames.push_back(entry.first);
  }
  std::sort(out.frames.begin(), out.frames.end());
  const int n = static_cast<int>(out.frames.size());
  if (n < 3) {
    return false;
  }

  // Body pose in the SfM frame. The rotation is scale-free; the translation splits into a
  // part that SCALES (the SfM position, in ruler units) and a part that does NOT (the
  // extrinsic, which is already in metres) -- which is exactly why s multiplies one and not
  // the other in the equations below.
  //
  // EXPLICIT RETURN TYPES ARE LOAD-BEARING. `inverse()` yields a temporary and
  // `.translation()` / `.linear()` return Block expressions referencing it. With a deduced
  // `auto` the lambda hands back the Block, the temporary dies at the semicolon, and the
  // Block dangles -- no crash, no warning, just silent garbage that read as zeros and made
  // the whole reconstruction look motionless. Eigen expression templates and `auto` do not
  // mix.
  const Eigen::Vector3d t_ci = calib_.T_cam_imu.translation();
  const auto R_cam = [&](int k) -> Eigen::Matrix3d {
      return sfm.pose.at(k).inverse().linear();
    };
  const auto p_cam = [&](int k) -> Eigen::Vector3d {
      return sfm.pose.at(k).inverse().translation();
    };
  const auto R_body = [&](int k) -> Eigen::Matrix3d {
      return R_cam(k) * calib_.T_cam_imu.linear();
    };

  // Unknowns: [v_0 .. v_n (3 each), g (3), s (1)]. Every equation below is LINEAR in all of
  // them, which is why this is one least-squares solve and not an optimisation.
  const int dim = 3 * n + 3 + 1;
  const int gi = 3 * n;
  const int si = 3 * n + 3;

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6 * (n - 1), dim);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * (n - 1));
  int row = 0;
  int intervals = 0;

  for (int a = 0; a + 1 < n; ++a) {
    const int k0 = out.frames[a], k1 = out.frames[a + 1];
    ImuPreintegration pre(gyro_bias, Eigen::Vector3d::Zero(), calib_.gyro_noise,
      calib_.accel_noise);
    if (!imu.preintegrate(
        frames[k0].t, frames[k1].t, gyro_bias, Eigen::Vector3d::Zero(), pre,
        calib_.gyro_noise, calib_.accel_noise))
    {
      continue;
    }
    const double dt = pre.dt();
    const Eigen::Matrix3d Rb0 = R_body(k0);

    // from dp:  s(p_c1 - p_c0) - v_0 dt - 1/2 g dt^2 = Rb0 dp - (R_c1 - R_c0) t_ci
    A.block<3, 3>(row, 3 * a) = -Eigen::Matrix3d::Identity() * dt;
    A.block<3, 3>(row, gi) = -0.5 * Eigen::Matrix3d::Identity() * dt * dt;
    A.block<3, 1>(row, si) = p_cam(k1) - p_cam(k0);
    b.segment<3>(row) = Rb0 * pre.dp() - (R_cam(k1) - R_cam(k0)) * t_ci;
    row += 3;

    // from dv:  -v_0 + v_1 - g dt = Rb0 dv
    A.block<3, 3>(row, 3 * a) = -Eigen::Matrix3d::Identity();
    A.block<3, 3>(row, 3 * (a + 1)) = Eigen::Matrix3d::Identity();
    A.block<3, 3>(row, gi) = -Eigen::Matrix3d::Identity() * dt;
    b.segment<3>(row) = Rb0 * pre.dv();
    row += 3;
    ++intervals;
  }

  out.align_intervals = intervals;
  if (intervals < p_.align_min_intervals) {
    return false;
  }
  A.conservativeResize(row, Eigen::NoChange);
  b.conservativeResize(row);

  const Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
  if (!x.allFinite()) {
    return false;
  }

  out.gravity_sfm = x.segment<3>(gi);
  out.scale = x(si);
  out.velocity_sfm.clear();
  for (int a = 0; a < n; ++a) {
    out.velocity_sfm.push_back(x.segment<3>(3 * a));
  }

  // THE ORACLE. |g| entered the solve as three free numbers -- nothing told it what gravity
  // weighs. If the formulation, the frames or the extrinsic were wrong there is no reason
  // for its magnitude to land near 9.80665, so this is a genuine test rather than a
  // tautology. It is also the ONLY self-check available without ground truth.
  const double g_err = std::abs(out.gravity_sfm.norm() - kGravity) / kGravity;
  if (g_err > p_.max_gravity_error_pct / 100.0) {
    return false;
  }

  // Scale observability.
  //
  // THE TEST IS THE SIGN, NOT THE MAGNITUDE, because the ruler is ARBITRARY. s is metres per
  // base-pair baseline, and that baseline was invented by declaring |t| = 1 -- so s is 2.43
  // on a car covering 2.4 m between keyframes and 0.059 on a MAV covering 6 cm. Any absolute
  // threshold encodes one dataset's speed. (This read `> 0.1` for a while, tuned on KITTI,
  // and duly reported EuRoC's perfectly good s = 0.0591 as unobservable.)
  //
  // A NEGATIVE s is physically impossible -- a baseline cannot have negative length -- so it
  // is proof the s/v_0 ridge won and the solve landed on an absurd point of an equal-cost
  // valley (KITTI 0117: s = -0.21 against a truth of +2.43).
  //
  // This is necessary, not sufficient: it catches the gross degeneracy, not a mildly
  // ill-conditioned one. |g| is no help here -- on KITTI it landed within 0.8% while s was
  // 108% wrong, because gravity is observable from the dv equations whether or not scale is.
  out.scale_observable = out.scale > 0.0;
  return true;
}

// =================================================================================
// The pipeline.
// =================================================================================

InitResult VioInitializer::run(
  const std::vector<SfmFrame> & frames, const ImuBuffer & imu, int begin) const
{
  InitResult out;
  if (begin < 0 || begin + 3 >= static_cast<int>(frames.size())) {
    return out;
  }

  // [2] Shape, in an invented ruler. Needs nothing, so it goes first and breaks the
  //     chicken-and-egg the metric stages are stuck in.
  out.sfm = reconstruct(frames, begin);
  if (!out.sfm.valid) {
    return out;
  }

  // [3] Gyro bias. Rotations are scale-free, so this is observable before the metre exists
  //     -- and it goes before [4] because [4] integrates the gyro.
  if (!estimateGyroBias(frames, imu, begin, out.gyro_bias, out.bias_pairs)) {
    return out;
  }

  // [4] The metre.
  if (!align(frames, imu, out.sfm, out.gyro_bias, out)) {
    return out;
  }

  out.ok = true;
  return out;
}

}  // namespace glassvio
