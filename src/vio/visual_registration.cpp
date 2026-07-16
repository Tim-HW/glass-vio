#include "glassvio/visual_registration.hpp"

#include <cmath>

#include "glass_core/gauss_newton.hpp"

namespace glassvio
{

using NavEquations = NormalEquationsN<kNavDim>;

VisualResult solveFrame(
  const std::unordered_map<long, Eigen::Vector3d> & landmarks,
  const std::unordered_map<long, cv::Point2f> & observations,
  const NavState & xi,
  const Eigen::Matrix<double, kNavDim, kNavDim> & P_i,
  const ImuPreintegration & pre,
  const Eigen::Vector3d & gravity,
  const NavState & guess,
  const Eigen::Matrix<double, 6, 6> & bias_information,
  const CameraCalib & calib,
  const VisualParams & params)
{
  VisualResult result;
  result.state = guess;
  if (landmarks.empty() || observations.empty()) {
    return result;
  }

  // --- Information, built once, at the guess.
  //
  // THE IMU'S RESIDUAL IS NOT ONLY AS UNCERTAIN AS THE DELTA. Weighting it by
  // pre.covariance()^-1 alone -- the delta's own uncertainty -- silently asserts that x_i is
  // PERFECT, because x_i enters imuResidual as a constant. It is not perfect, and over a short
  // interval the delta's covariance is tiny, so that assertion hands the IMU factor effectively
  // infinite information. It then overrules every pixel and the solve collapses into dead
  // reckoning. Measured on EuRoC V1_01 before this line existed: reprojection error grew
  // monotonically to 643 px while 135 landmarks sat in view, unrejected and simply outvoted.
  // glasslio's README records the same divergence from the LiDAR side.
  //
  //     Sigma_eff = Sigma_pre + J_i P_i J_i^T
  //
  // J_i = d r / d dx_i is the half of the IMU Jacobian neither front-end needed while x_i was
  // held fixed. Propagating x_i's uncertainty through it is what gives the previous state a
  // FINITE certainty -- and what lets the caller carry a posterior forward instead of moving
  // to a two-state solve. NormalEquationsN::H() has always existed for that.
  //
  // Linearised at the guess, once: J_i depends on x_j, but re-forming the information every
  // iteration would make the cost function change under the solver, which is the same thing
  // that makes ICP's re-association awkward to damp.
  const ImuJacobian J_i = imuJacobianI(xi, guess, pre, gravity);
  const Eigen::Matrix<double, 9, 9> sigma_eff =
    pre.covariance() + J_i * P_i * J_i.transpose();

  Eigen::Matrix<double, 9, 9> imu_information =
    sigma_eff.inverse() * params.imu_prior_weight;
  if (!imu_information.allFinite()) {
    imu_information.setZero();   // a singular covariance means we learned nothing
  }

  NavState x = guess;

  for (int iter = 0; iter < params.max_iterations; ++iter) {
    NavEquations eq;

    // --- 1. VISION: two scalar residuals per landmark in view, each Huber-weighted.
    //
    // No re-association: the ids ARE the correspondence. This loop is the whole of what
    // replaced ICP.
    double sq_px = 0.0;
    int n_feat = 0;
    int rejected = 0;
    for (const auto & entry : landmarks) {
      const auto obs = observations.find(entry.first);
      if (obs == observations.end()) {
        continue;   // this landmark is not tracked in this frame
      }
      const Eigen::Vector2d z(obs->second.x, obs->second.y);

      // Raw pixel error, kept for a reportable RMSE in pixels. Recomputed rather than taken
      // from the whitened residual so the number stays in units a human can judge.
      Eigen::Vector3d P_i, P_c;
      if (!landmarkInCamera(
          x, entry.second, calib.T_cam_imu, params.reproj.min_depth, P_i, P_c))
      {
        ++rejected;
        continue;   // behind the camera: a plausible pixel with a sign-flipped Jacobian
      }
      sq_px += reprojectionResidual(P_c, z, calib).squaredNorm();

      if (accumulateReprojection(
          eq, x, entry.second, z, calib.T_cam_imu, calib, params.reproj))
      {
        ++n_feat;
      }
    }

    result.features = n_feat;
    result.rejected_cheirality = rejected;
    if (n_feat < params.min_features) {
      // Under-constrained: refuse rather than invent a pose. With landmarks held fixed this
      // is the STARVATION signal -- the camera has flown past everything it was triangulated
      // against, and a sliding window is what fixes it, not a looser threshold.
      result.valid = false;
      return result;
    }

    // --- 2. The IMU factor: one 9-vector, weighted by its own information.
    eq.addBlock<9>(
      imuResidual(xi, x, pre, gravity), imuJacobian(xi, x, pre), imu_information);

    // --- 3. The bias random-walk prior: keeps the biases from absorbing real motion in the
    //        (common) case where they are unobservable.
    eq.addBlock<6>(biasResidual(xi, x), biasJacobian(), bias_information);

    // --- Solve and retract. Identical to the LiDAR case; only the factors differed.
    const NavVec dx = eq.solve();
    if (!dx.allFinite()) {
      result.valid = false;   // degenerate geometry; do NOT hand back a state
      return result;
    }

    x = boxplus(x, dx);   // RIGHT perturbation on R; additive elsewhere
    result.H = eq.H();
    result.iterations = iter + 1;
    result.valid = true;
    result.rmse_px = std::sqrt(sq_px / static_cast<double>(std::max(n_feat, 1)));

    if (dx.segment<3>(kIdxPos).norm() < params.eps_translation &&
      dx.segment<3>(kIdxPhi).norm() < params.eps_rotation)
    {
      result.converged = true;
      break;
    }
  }

  result.state = x;
  return result;
}

}  // namespace glassvio
