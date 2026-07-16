#ifndef GLASSVIO_REPROJECTION_HPP
#define GLASSVIO_REPROJECTION_HPP

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glass_core/nav_state.hpp"
#include "glassvio/camera_calib.hpp"

namespace glassvio
{

using namespace glass_core;  // NOLINT(build/namespaces)

/// THE REPROJECTION FACTOR -- glassvio's answer to glasslio's point-to-plane.
///
/// This is the whole thesis of the two repos in one file: the same 15-DoF state, the same
/// Gauss-Newton, the same normal equations, the same retraction -- and a camera residual
/// where the LiDAR one was. Nothing in glass_core changes to accept it.
///
/// THE GEOMETRY. A landmark in the world, seen through the body pose and the extrinsic:
///
///     P_i = R_wi^T (P_w - p_wi)        landmark in the IMU frame
///     P_c = R_ci P_i + t_ci            landmark in the camera frame  (T_cam_imu)
///     r   = pi(P_c) - z_obs            pixel error
///
/// LANDMARKS ARE FIXED, exactly as the LiDAR map's planes are. `closestPlane` hands back a
/// centroid and a normal that the solve does not move; here P_w plays the same role. That is
/// what keeps the state at 15 and lets NormalEquationsN<15> be reused untouched. Putting
/// landmarks in the state is bundle adjustment: H grows to 15 + 3L, dense LDLT stops being
/// viable, and the Schur complement becomes mandatory. Not yet.
///
/// TWO THINGS THE LIDAR FACTOR NEVER HAD TO WORRY ABOUT:
///
///   * CHEIRALITY. pi() divides by Z. A landmark behind the camera does not produce a NaN --
///     it produces an ordinary-looking pixel (mirrored through the optical centre) with a
///     SIGN-FLIPPED Jacobian, so the solver confidently steps the wrong way. Point-to-plane
///     is affine in the transformed point and has no such failure. Always gate on Z.
///   * OUTLIERS. A KLT mismatch is exactly the "individually suspect measurement" addScalar
///     and its Huber weight exist for. addBlock has no Huber, deliberately -- a preintegrated
///     IMU delta is not an outlier candidate the way one tracked corner is.

/// Pixel noise sigma is isotropic for KLT, which is what lets the 2-vector residual split
/// into two independent whitened scalars -- see accumulateReprojection.
struct ReprojectionParams
{
  double sigma_px = 1.0;        ///< pixel noise standard deviation
  double huber_delta_px = 2.0;  ///< in PIXELS; whitened alongside the residual
  double min_depth = 0.1;       ///< m; below this the point is behind us or on the lens
};

using PixelResidual = Eigen::Matrix<double, 2, 1>;
using PixelJacobian = Eigen::Matrix<double, 2, kNavDim>;

/// Landmark in the camera frame. Returns false if it fails cheirality.
inline bool landmarkInCamera(
  const NavState & x, const Eigen::Vector3d & P_w, const Eigen::Isometry3d & T_cam_imu,
  double min_depth, Eigen::Vector3d & P_i, Eigen::Vector3d & P_c)
{
  P_i = x.R.inverse() * (P_w - x.p);
  P_c = T_cam_imu * P_i;
  return P_c.z() > min_depth;
}

/// r = pi(P_c) - z_obs, in pixels.
inline PixelResidual reprojectionResidual(
  const Eigen::Vector3d & P_c, const Eigen::Vector2d & observed, const CameraCalib & calib)
{
  const double inv_z = 1.0 / P_c.z();
  return PixelResidual(
    calib.fx() * P_c.x() * inv_z + calib.cx() - observed.x(),
    calib.fy() * P_c.y() * inv_z + calib.cy() - observed.y());
}

/// d r / d dx under the RIGHT perturbation -- the same retraction nav_state.hpp's boxplus
/// applies (R <- R Exp(dphi)). NOT interchangeable with optimizeSE3's left-perturbation
/// Jacobians: same residual, different manifold convention, and the mismatch is silent.
///
/// Pinned against finite differences in test_reprojection.cpp. Never check a derivation
/// against itself.
inline PixelJacobian reprojectionJacobian(
  const NavState & x, const Eigen::Vector3d & P_i, const Eigen::Vector3d & P_c,
  const Eigen::Isometry3d & T_cam_imu, const CameraCalib & calib)
{
  // d pi / d P_c. The 1/Z^2 terms are where all the nonlinearity lives.
  const double inv_z = 1.0 / P_c.z();
  const double inv_z2 = inv_z * inv_z;
  Eigen::Matrix<double, 2, 3> J_pi;
  J_pi << calib.fx() * inv_z, 0.0, -calib.fx() * P_c.x() * inv_z2,
    0.0, calib.fy() * inv_z, -calib.fy() * P_c.y() * inv_z2;

  const Eigen::Matrix3d R_ci = T_cam_imu.linear();
  const Eigen::Matrix3d R_iw = x.R.inverse().matrix();

  // RIGHT perturbation: R_wi <- R_wi Exp(dphi), so R_wi^T <- Exp(-dphi) R_wi^T ~= (I - dphi^)
  // R_wi^T. Then P_i = R_wi^T (P_w - p) picks up +hat(P_i) dphi:
  //     d P_i / d dphi = hat(P_i)
  //     d P_i / d dp   = -R_wi^T
  PixelJacobian J = PixelJacobian::Zero();
  J.block<2, 3>(0, kIdxPhi) = J_pi * R_ci * Sophus::SO3d::hat(P_i);
  J.block<2, 3>(0, kIdxPos) = -J_pi * R_ci * R_iw;
  // Velocity and both biases are STRUCTURALLY absent -- a camera cannot see them, exactly as
  // pointToPlaneJacobianNav says of the LiDAR. The IMU remains the only thing observing them,
  // which is why loose coupling could never estimate them.
  return J;
}

/// Fold one tracked feature into the normal equations.
///
/// TWO addScalar CALLS, NOT addBlock<2>, and it is an identity rather than a hack. The proper
/// 2D form contributes J^T Omega J with Omega = Sigma^-1; for isotropic pixel noise
/// Sigma = sigma^2 I, so Omega = (1/sigma^2) I is DIAGONAL and
///
///     J^T (1/sigma^2) I J  =  (1/sigma^2) (J_u^T J_u + J_v^T J_v)
///
/// which is exactly the sum of two independent whitened scalar rows. Same H, same b. The
/// cross-coupling addBlock exists to carry is zero here -- and splitting buys the HUBER
/// weight that addBlock deliberately omits, which a KLT mismatch very much needs.
///
/// Whitening follows the LiDAR path exactly: divide residual, Jacobian AND the Huber
/// threshold by sigma, so huber_delta_px keeps its natural units. Skip it and the camera and
/// the IMU are being compared in different units and the fusion is meaningless.
///
/// ponytail: Huber is applied per-AXIS, so a track gets w_u and w_v rather than one radial
/// w = huber(|r|). Axis-aligned robustness, not circular. A gross mismatch is bad in both
/// axes and dies either way; if this ever matters, the fix is a robust addBlock in glass_core.
///
/// Returns false if the landmark failed cheirality and contributed nothing.
template<typename Equations>
bool accumulateReprojection(
  Equations & eq, const NavState & x, const Eigen::Vector3d & P_w,
  const Eigen::Vector2d & observed, const Eigen::Isometry3d & T_cam_imu,
  const CameraCalib & calib, const ReprojectionParams & p)
{
  Eigen::Vector3d P_i, P_c;
  if (!landmarkInCamera(x, P_w, T_cam_imu, p.min_depth, P_i, P_c)) {
    return false;   // behind the camera: a plausible pixel with a sign-flipped Jacobian
  }

  const PixelResidual r = reprojectionResidual(P_c, observed, calib);
  const PixelJacobian J = reprojectionJacobian(x, P_i, P_c, T_cam_imu, calib);

  const double inv_sigma = 1.0 / p.sigma_px;
  const double huber_whitened = p.huber_delta_px * inv_sigma;
  eq.addScalar(r(0) * inv_sigma, J.row(0) * inv_sigma, huber_whitened);
  eq.addScalar(r(1) * inv_sigma, J.row(1) * inv_sigma, huber_whitened);
  return true;
}

}  // namespace glassvio

#endif  // GLASSVIO_REPROJECTION_HPP
