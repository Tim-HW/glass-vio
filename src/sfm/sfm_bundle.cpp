#include "glassvio/sfm_bundle.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include <Eigen/Dense>

#include "glass_core/gauss_newton.hpp"   // huberWeight
#include "glassvio/reprojection.hpp"

namespace glassvio
{

namespace
{

struct Obs
{
  int pose;   ///< index into the free-pose list, or -1 for a fixed frame
  int frame;
  int landmark;
  Eigen::Vector2d z;
};

}  // namespace

SfmBundleStats bundleAdjust(
  const std::vector<SfmFrame> & frames, SfmWindow & w, const CameraCalib & calib,
  int max_iterations, double huber_delta_px, bool fix_pair_frame)
{
  SfmBundleStats stats;
  if (w.second < 0 || !w.pose.count(w.base) || !w.pose.count(w.second)) {
    return stats;
  }

  // Cameras as NavStates: the "IMU" is the camera, so the extrinsic is identity.
  CameraCalib cam = calib;
  cam.T_cam_imu = Eigen::Isometry3d::Identity();
  constexpr double kMinDepth = 1e-6;   // ruler units; cheirality, not a distance gate

  std::vector<int> frame_ids;
  std::unordered_map<int, NavState> x;
  for (const auto & kv : w.pose) {
    const Eigen::Isometry3d T_c0_ck = kv.second.inverse();
    NavState s;
    s.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_c0_ck.linear()));
    s.p = T_c0_ck.translation();
    x.emplace(kv.first, s);
    frame_ids.push_back(kv.first);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  std::unordered_map<int, int> free_index;
  for (int k : frame_ids) {
    if (k != w.base && (k != w.second || !fix_pair_frame)) {
      const int i = static_cast<int>(free_index.size());
      free_index.emplace(k, i);
    }
  }

  std::vector<long> ids;
  std::vector<Eigen::Vector3d> X;
  for (const auto & kv : w.landmark) {
    ids.push_back(kv.first);
    X.push_back(kv.second);
  }
  std::vector<Obs> obs;
  std::vector<int> seen(ids.size(), 0);
  for (int k : frame_ids) {
    for (std::size_t l = 0; l < ids.size(); ++l) {
      const auto it = frames[k].by_id.find(ids[l]);
      if (it == frames[k].by_id.end()) {
        continue;
      }
      const auto fi = free_index.find(k);
      obs.push_back(
        {fi == free_index.end() ? -1 : fi->second, k, static_cast<int>(l),
          Eigen::Vector2d(it->second.x, it->second.y)});
      ++seen[l];
    }
  }
  const int P = static_cast<int>(free_index.size());
  const int L = static_cast<int>(ids.size());
  if (obs.empty()) {
    return stats;
  }
  stats.ran = true;
  stats.observations = static_cast<int>(obs.size());

  // Robust cost and the per-observation pixel errors, at a given state.
  const auto evaluate = [&](
    const std::unordered_map<int, NavState> & xs, const std::vector<Eigen::Vector3d> & Xs,
    std::vector<double> * px) {
      double cost = 0.0;
      for (const auto & o : obs) {
        Eigen::Vector3d P_i, P_c;
        if (!landmarkInCamera(xs.at(o.frame), Xs[o.landmark], cam.T_cam_imu, kMinDepth, P_i, P_c)) {
          cost += 2.0 * huber_delta_px * 1e3;   // behind the camera: large, finite
          continue;
        }
        const double r = reprojectionResidual(P_c, o.z, cam).norm();
        cost += r <= huber_delta_px ? r * r : 2.0 * huber_delta_px * r - huber_delta_px *
          huber_delta_px;
        if (px) {
          px->push_back(r);
        }
      }
      return cost;
    };
  const auto median = [](std::vector<double> v) {
      if (v.empty()) {return 0.0;}
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };

  std::vector<double> px;
  double cost = evaluate(x, X, &px);
  stats.median_px_before = median(px);
  stats.median_px_after = stats.median_px_before;
  double lambda = 1e-3;

  for (int it = 0; it < max_iterations; ++it) {
    // --- Normal equations. H_pp is block-diagonal: an observation touches ONE camera.
    std::vector<Eigen::Matrix<double, 6, 6>> H_pp(P, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_p(P, Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<Eigen::Matrix3d> H_ll(L, Eigen::Matrix3d::Zero());
    std::vector<Eigen::Vector3d> b_l(L, Eigen::Vector3d::Zero());
    std::vector<Eigen::Matrix<double, 6, 3>> H_pl(obs.size(), Eigen::Matrix<double, 6, 3>::Zero());

    for (std::size_t n = 0; n < obs.size(); ++n) {
      const Obs & o = obs[n];
      if (seen[o.landmark] < 2) {
        continue;
      }
      const NavState & s = x.at(o.frame);
      Eigen::Vector3d P_i, P_c;
      if (!landmarkInCamera(s, X[o.landmark], cam.T_cam_imu, kMinDepth, P_i, P_c)) {
        continue;
      }
      const PixelResidual r = reprojectionResidual(P_c, o.z, cam);
      const double wgt = huberWeight(r.norm(), huber_delta_px);
      const PixelJacobian J = reprojectionJacobian(s, P_i, P_c, cam.T_cam_imu, cam);
      Eigen::Matrix<double, 2, 6> Jp;
      Jp << J.block<2, 3>(0, kIdxPhi), J.block<2, 3>(0, kIdxPos);
      const Eigen::Matrix<double, 2, 3> Jl = reprojectionLandmarkJacobian(J);

      H_ll[o.landmark] += wgt * Jl.transpose() * Jl;
      b_l[o.landmark] -= wgt * Jl.transpose() * r;
      if (o.pose >= 0) {
        H_pp[o.pose] += wgt * Jp.transpose() * Jp;
        b_p[o.pose] -= wgt * Jp.transpose() * r;
        H_pl[n] = wgt * Jp.transpose() * Jl;
      }
    }

    // --- Levenberg-Marquardt damping, then eliminate each landmark: S dp = rhs.
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(6 * P, 6 * P);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(6 * P);
    for (int i = 0; i < P; ++i) {
      Eigen::Matrix<double, 6, 6> Hd = H_pp[i];
      Hd.diagonal() += lambda * (Hd.diagonal().array() + 1e-9).matrix();
      S.block<6, 6>(6 * i, 6 * i) = Hd;
      rhs.segment<6>(6 * i) = b_p[i];
    }
    std::vector<Eigen::Matrix3d> H_ll_inv(L, Eigen::Matrix3d::Zero());
    std::vector<std::vector<std::size_t>> obs_of(L);
    for (std::size_t n = 0; n < obs.size(); ++n) {
      if (obs[n].pose >= 0 && !H_pl[n].isZero()) {
        obs_of[obs[n].landmark].push_back(n);
      }
    }
    for (int l = 0; l < L; ++l) {
      Eigen::Matrix3d Hd = H_ll[l];
      if (Hd.isZero()) {
        continue;
      }
      Hd.diagonal() += lambda * (Hd.diagonal().array() + 1e-9).matrix();
      H_ll_inv[l] = Hd.inverse();
      if (!H_ll_inv[l].allFinite()) {
        H_ll_inv[l].setZero();
        continue;
      }
      for (std::size_t a : obs_of[l]) {
        const Eigen::Matrix<double, 6, 3> W = H_pl[a] * H_ll_inv[l];
        rhs.segment<6>(6 * obs[a].pose) -= W * b_l[l];
        for (std::size_t c : obs_of[l]) {
          S.block<6, 6>(6 * obs[a].pose, 6 * obs[c].pose) -= W * H_pl[c].transpose();
        }
      }
    }
    const Eigen::VectorXd dp = P > 0 ? Eigen::VectorXd(S.ldlt().solve(rhs)) : Eigen::VectorXd();
    if (!dp.allFinite()) {
      break;
    }

    // --- Back-substitute and try the step.
    std::unordered_map<int, NavState> x_new = x;
    for (const auto & kv : free_index) {
      NavVec dx = NavVec::Zero();
      dx.segment<3>(kIdxPhi) = dp.segment<3>(6 * kv.second);
      dx.segment<3>(kIdxPos) = dp.segment<3>(6 * kv.second + 3);
      x_new[kv.first] = boxplus(x.at(kv.first), dx);
    }
    std::vector<Eigen::Vector3d> X_new = X;
    for (int l = 0; l < L; ++l) {
      Eigen::Vector3d rl = b_l[l];
      for (std::size_t a : obs_of[l]) {
        rl -= H_pl[a].transpose() * dp.segment<6>(6 * obs[a].pose);
      }
      X_new[l] += H_ll_inv[l] * rl;
    }
    const double cost_new = evaluate(x_new, X_new, nullptr);
    ++stats.iterations;
    if (cost_new < cost) {
      x = std::move(x_new);
      X = std::move(X_new);
      const double gain = (cost - cost_new) / std::max(cost, 1e-12);
      cost = cost_new;
      lambda = std::max(lambda / 10.0, 1e-7);
      if (gain < 1e-6) {
        break;
      }
    } else {
      lambda *= 10.0;
      if (lambda > 1e6) {
        break;
      }
    }
  }

  // Free pair frame: the scale was left to the damping, so restore the ruler -- the pair's
  // distance from the base (which sits at the origin) -- by scaling about the origin.
  const double pair_before = w.pose.at(w.second).inverse().translation().norm();
  const double pair_after = x.at(w.second).p.norm();
  if (!fix_pair_frame && pair_after > 1e-12) {
    const double k = pair_before / pair_after;
    for (auto & kv : x) {
      kv.second.p *= k;
    }
    for (auto & Xl : X) {
      Xl *= k;
    }
  }

  px.clear();
  evaluate(x, X, &px);
  stats.median_px_after = median(px);

  for (const auto & kv : x) {
    Eigen::Isometry3d T_c0_ck = Eigen::Isometry3d::Identity();
    T_c0_ck.linear() = kv.second.R.matrix();
    T_c0_ck.translation() = kv.second.p;
    w.pose[kv.first] = T_c0_ck.inverse();
  }
  for (int l = 0; l < L; ++l) {
    w.landmark[ids[l]] = X[l];
  }
  return stats;
}

}  // namespace glassvio
