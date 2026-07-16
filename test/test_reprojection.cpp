// Self-check for the REPROJECTION FACTOR -- glassvio's half of the "one solver, two sensors"
// claim.
//
// THE ORACLE PRINCIPLE. The obvious way to test reprojectionJacobian() is to re-derive it by
// hand and compare. That is worthless: if the derivation was wrong, the check is wrong the
// same way. So the Jacobian is compared against FINITE DIFFERENCES -- a second source of
// truth that shares none of the first's assumptions, and knows nothing about chain rules,
// right perturbations, or hat operators.
//
// AND THE PERTURBATION MUST BE THE REAL ONE. The finite difference steps the state through
// glass_core's own boxplus(), not through naive addition. That is the entire point: the
// analytic Jacobian claims to be d r / d dx under the RIGHT retraction (R <- R Exp(dphi)),
// and only boxplus knows what that means. Difference the state additively and the test would
// pass for a LEFT-perturbation Jacobian too -- which still runs, still converges, and is
// wrong.

#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glass_core/nav_state.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/reprojection.hpp"

using glass_core::boxplus;
using glass_core::kNavDim;
using glass_core::NavState;
using glass_core::NavVec;

namespace
{

/// EuRoC's cam0, so the test exercises the numbers the pipeline actually runs on.
glassvio::CameraCalib makeCalib()
{
  glassvio::CameraCalib c;
  c.K << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1;
  // A realistic body->camera transform: ~90 deg of axis permutation plus a few cm.
  Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
  T.linear() = (Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()) *
    Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitX())).toRotationMatrix();
  T.translation() = Eigen::Vector3d(0.065, -0.021, -0.008);
  c.T_cam_imu = T;
  return c;
}

}  // namespace

int main()
{
  const glassvio::CameraCalib calib = makeCalib();
  const glassvio::ReprojectionParams params;
  std::mt19937 rng(7);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  double worst = 0.0;
  int cases = 0;

  for (int trial = 0; trial < 200; ++trial) {
    // A state well away from identity: a Jacobian that is only right at R = I is a Jacobian
    // that is wrong.
    NavState x;
    x.R = Sophus::SO3d::exp(Eigen::Vector3d(u(rng), u(rng), u(rng)) * 1.2);
    x.p = Eigen::Vector3d(u(rng), u(rng), u(rng)) * 2.0;
    x.v = Eigen::Vector3d(u(rng), u(rng), u(rng));
    x.bg = Eigen::Vector3d(u(rng), u(rng), u(rng)) * 0.01;
    x.ba = Eigen::Vector3d(u(rng), u(rng), u(rng)) * 0.05;

    // A landmark somewhere in front of the camera, at a plausible indoor depth.
    Eigen::Vector3d P_i_want(u(rng) * 2.0, u(rng) * 2.0, 0.0);
    P_i_want.z() = 2.0 + std::abs(u(rng)) * 4.0;
    const Eigen::Vector3d P_c_want = calib.T_cam_imu * P_i_want;
    if (P_c_want.z() < 0.5) {
      continue;
    }
    // Put it in the world so that this state sees it there.
    const Eigen::Vector3d P_w = x.R * P_i_want + x.p;

    Eigen::Vector3d P_i, P_c;
    if (!glassvio::landmarkInCamera(x, P_w, calib.T_cam_imu, params.min_depth, P_i, P_c)) {
      continue;
    }
    // An observation offset from the projection, so the residual is non-zero and the
    // Jacobian is evaluated somewhere real rather than at a perfect fit.
    const Eigen::Vector2d observed =
      glassvio::reprojectionResidual(P_c, Eigen::Vector2d::Zero(), calib) +
      Eigen::Vector2d(u(rng) * 3.0, u(rng) * 3.0);

    const glassvio::PixelJacobian J_analytic =
      glassvio::reprojectionJacobian(x, P_i, P_c, calib.T_cam_imu, calib);

    // --- The oracle: central differences THROUGH boxplus.
    glassvio::PixelJacobian J_numeric = glassvio::PixelJacobian::Zero();
    const double h = 1e-6;
    for (int c = 0; c < kNavDim; ++c) {
      NavVec dx = NavVec::Zero();
      dx(c) = h;
      Eigen::Vector3d Pi_p, Pc_p, Pi_m, Pc_m;
      const NavState xp = boxplus(x, dx);
      const NavState xm = boxplus(x, -dx);
      glassvio::landmarkInCamera(xp, P_w, calib.T_cam_imu, params.min_depth, Pi_p, Pc_p);
      glassvio::landmarkInCamera(xm, P_w, calib.T_cam_imu, params.min_depth, Pi_m, Pc_m);
      J_numeric.col(c) =
        (glassvio::reprojectionResidual(Pc_p, observed, calib) -
        glassvio::reprojectionResidual(Pc_m, observed, calib)) / (2.0 * h);
    }

    const double err = (J_analytic - J_numeric).cwiseAbs().maxCoeff();
    const double scale = std::max(1.0, J_numeric.cwiseAbs().maxCoeff());
    worst = std::max(worst, err / scale);
    ++cases;
  }

  assert(cases > 100 && "too few usable trials -- the generator is wrong, not the Jacobian");
  assert(worst < 1e-5 && "reprojection Jacobian disagrees with finite differences");

  // --- The structural claim: a camera says NOTHING about velocity or the biases. Those nine
  // columns must be EXACTLY zero, not merely small -- the same claim pointToPlaneJacobianNav
  // makes for the LiDAR, and the reason the IMU is the only thing that observes them.
  {
    NavState x;
    x.R = Sophus::SO3d::exp(Eigen::Vector3d(0.3, -0.2, 0.5));
    x.p = Eigen::Vector3d(0.4, -0.3, 0.2);
    const Eigen::Vector3d P_w = x.R * Eigen::Vector3d(0.3, 0.1, 3.0) + x.p;
    Eigen::Vector3d P_i, P_c;
    assert(glassvio::landmarkInCamera(x, P_w, calib.T_cam_imu, params.min_depth, P_i, P_c));
    const glassvio::PixelJacobian J =
      glassvio::reprojectionJacobian(x, P_i, P_c, calib.T_cam_imu, calib);
    // Parenthesised: assert is a macro, and the comma inside block<2, 3> would otherwise
    // read as a second macro argument.
    const bool vel_zero = (J.block<2, 3>(0, glass_core::kIdxVel).isZero());
    const bool bg_zero = (J.block<2, 3>(0, glass_core::kIdxBg).isZero());
    const bool ba_zero = (J.block<2, 3>(0, glass_core::kIdxBa).isZero());
    assert(vel_zero && "camera cannot see velocity");
    assert(bg_zero && "camera cannot see gyro bias");
    assert(ba_zero && "camera cannot see accel bias");
  }

  // --- Cheirality: a landmark BEHIND the camera must be refused, not projected. It would
  // otherwise produce a perfectly ordinary pixel with a sign-flipped Jacobian.
  {
    NavState x;
    const Eigen::Vector3d behind = calib.T_cam_imu.inverse() * Eigen::Vector3d(0.1, 0.1, -2.0);
    const Eigen::Vector3d P_w = x.R * behind + x.p;
    Eigen::Vector3d P_i, P_c;
    assert(
      !glassvio::landmarkInCamera(x, P_w, calib.T_cam_imu, params.min_depth, P_i, P_c) &&
      "a landmark behind the camera must be rejected");
  }

  std::printf(
    "ok: %d trials, worst Jacobian error %.2e (finite differences through boxplus);\n"
    "    vel/bg/ba columns exactly zero; cheirality rejected\n",
    cases, worst);
  return 0;
}
