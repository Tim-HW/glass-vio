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

  // Unknowns: [v_0 .. v_n (3 each), g (3), s (1), db_a (3)]. Every equation below is LINEAR in
  // all of them, which is why this is one least-squares solve and not an optimisation.
  //
  // db_a STAYS LINEAR. The deltas are integrated at a FIXED bias b0 (p_.accel_bias), and
  // preintegration carries how they move when the bias does -- dp(b0 + db) = dp + dp_dba db, the
  // same first-order correction the tracker's IMU factor uses. So the accel bias is three more
  // columns, not an iteration. Without them b_a is pinned at b0, and whatever EuRoC's 0.55 m/s^2
  // does to the fit has nowhere to go but into s, g and v.
  const int dim = 3 * n + 3 + 1;   // without db_a
  const int gi = 3 * n;
  const int si = 3 * n + 3;
  const int bai = 3 * n + 4;
  const int dim_ba = dim + 3;      // with db_a

  Eigen::MatrixXd A = Eigen::MatrixXd::Zero(6 * (n - 1), dim_ba);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(6 * (n - 1));
  int row = 0;
  int intervals = 0;

  for (int a = 0; a + 1 < n; ++a) {
    const int k0 = out.frames[a], k1 = out.frames[a + 1];
    ImuPreintegration pre(gyro_bias, p_.accel_bias, calib_.gyro_noise, calib_.accel_noise);
    if (!imu.preintegrate(
        frames[k0].t, frames[k1].t, gyro_bias, p_.accel_bias, pre,
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
    A.block<3, 3>(row, bai) = -Rb0 * pre.dp_dba();   // Rb0 dp(b0 + db), moved to the left
    b.segment<3>(row) = Rb0 * pre.dp() - (R_cam(k1) - R_cam(k0)) * t_ci;
    row += 3;

    // from dv:  -v_0 + v_1 - g dt = Rb0 dv
    A.block<3, 3>(row, 3 * a) = -Eigen::Matrix3d::Identity();
    A.block<3, 3>(row, 3 * (a + 1)) = Eigen::Matrix3d::Identity();
    A.block<3, 3>(row, gi) = -Eigen::Matrix3d::Identity() * dt;
    A.block<3, 3>(row, bai) = -Rb0 * pre.dv_dba();   // Rb0 dv(b0 + db), moved to the left
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

  // One solve over the first `cols` columns, plus what the gates need: the residual variance and
  // (A^T A)^-1, the marginal covariance up to that variance.
  struct Solution
  {
    Eigen::VectorXd x;
    double sigma2 = 0.0;
    Eigen::MatrixXd N_inv;
  };
  const auto solve = [&](int cols) -> Solution {
      const Eigen::MatrixXd Ak = A.leftCols(cols);
      Solution s;
      s.x = Ak.colPivHouseholderQr().solve(b);
      s.sigma2 = (Ak * s.x - b).squaredNorm() / static_cast<double>(std::max(1, row - cols));
      s.N_inv = (Ak.transpose() * Ak).inverse();
      return s;
    };

  // B_A OBSERVABILITY -- the same kind of gate the scale gets below. b_a enters the dv rows as
  // ~ -R_b dt b_a and gravity as -dt g: without ROTATION in the window they are the same column
  // and b_a is indistinguishable from a tilt of g. Its marginal std catches exactly that ridge;
  // when it is too wide, drop the three columns and fall back to b0 rather than let a phantom
  // bias soak up gravity.
  Solution sol;
  out.accel_bias = p_.accel_bias;
  out.accel_bias_estimated = false;
  out.accel_bias_std = 0.0;
  if (p_.estimate_accel_bias) {
    sol = solve(dim_ba);
    out.accel_bias_std = std::sqrt(
      std::max(0.0, sol.sigma2 * sol.N_inv.diagonal().segment<3>(bai).maxCoeff()));
    if (sol.x.allFinite() && out.accel_bias_std < p_.max_accel_bias_std) {
      out.accel_bias = p_.accel_bias + sol.x.segment<3>(bai);
      out.accel_bias_estimated = true;
    }
  }
  if (!out.accel_bias_estimated) {
    sol = solve(dim);
  }
  const Eigen::VectorXd & x = sol.x;
  if (!x.allFinite()) {
    return false;
  }

  out.gravity_sfm = x.segment<3>(gi);
  out.scale = x(si);

  // SCALE OBSERVABILITY, the SUFFICIENT test the sign check is not. Scale reaches the solve
  // only through the accelerometer's non-gravity part, so a window without excitation leaves s
  // poorly determined -- positive but metrically meaningless (measured on EuRoC: a starved
  // window gave a 3 cm room). The MARGINAL RELATIVE UNCERTAINTY of s is what detects this:
  //
  //     cov(x) = sigma^2 (A^T A)^-1 ,   sigma_s / |s| = how uncertain the scale is, as a fraction
  //
  // Marginal (not the raw diagonal) because it accounts for s trading off against v and g --
  // the exact ridge that lets a bad window fake a fit. Dimensionless, so it needs no
  // per-dataset threshold. The raw condition number of A does NOT work here: its columns span
  // dt, dt^2 and ruler units, so its conditioning measures column scaling, not observability.
  const double var_s = sol.sigma2 * sol.N_inv(si, si);
  out.scale_uncertainty =
    std::sqrt(std::max(0.0, var_s)) / std::max(std::abs(out.scale), 1e-9);
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
  // Sign (necessary) AND relative uncertainty (sufficient). The sign catches the gross ridge
  // failure (negative baseline); the uncertainty catches the subtle one -- a positive but
  // poorly-excited scale, which is what silently gave a 3 cm room online.
  out.scale_observable =
    out.scale > 0.0 && out.scale_uncertainty < p_.max_scale_uncertainty;
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
  //
  // FROM 0, NOT `begin`. `begin` is stage [2]'s window -- deliberately the NEWEST frames, so
  // its landmarks are alive. The bias is a CONSTANT of the sensor, so every frame in the
  // stream is evidence and averaging is what pulls it below the vision noise floor of any one
  // pair (~0.28 deg). Passing `begin` here would hand it only the reconstruction's 30 frames,
  // i.e. 6 pairs against a required 8 -- which is exactly how the online node stalled forever
  // while the offline checks, loading whole bags, never noticed.
  if (!estimateGyroBias(frames, imu, 0, out.gyro_bias, out.bias_pairs)) {
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
