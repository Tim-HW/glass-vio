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

struct Observation
{
  int pose;   ///< index into the free-pose list, or -1 for a fixed frame
  int frame;
  int landmark;
  Eigen::Vector2d pixel;
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

  // 1. Pack the reconstruction into camera states and indexed pixel observations.
  // Reprojection treats each camera as a NavState with identity IMU extrinsic.
  CameraCalib cam = calib;
  cam.T_cam_imu = Eigen::Isometry3d::Identity();
  constexpr double kMinDepth = 1e-6;   // ruler units; cheirality, not a distance gate

  std::vector<int> frame_ids;
  std::unordered_map<int, NavState> camera_states;
  for (const auto & kv : w.pose) {
    const Eigen::Isometry3d T_c0_ck = kv.second.inverse();
    NavState s;
    s.R = Sophus::SO3d(Sophus::SO3d::fitToSO3(T_c0_ck.linear()));
    s.p = T_c0_ck.translation();
    camera_states.emplace(kv.first, s);
    frame_ids.push_back(kv.first);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  // The base pose is fixed; the pair pose is fixed unless its direction is being refined.
  std::unordered_map<int, int> free_index;
  for (int k : frame_ids) {
    if (k != w.base && (k != w.second || !fix_pair_frame)) {
      const int i = static_cast<int>(free_index.size());
      free_index.emplace(k, i);
    }
  }

  std::vector<long> ids;
  std::vector<Eigen::Vector3d> landmarks;
  for (const auto & kv : w.landmark) {
    ids.push_back(kv.first);
    landmarks.push_back(kv.second);
  }
  std::vector<Observation> obs;
  std::vector<int> observation_count(ids.size(), 0);
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
      ++observation_count[l];
    }
  }
  const int free_pose_count = static_cast<int>(free_index.size());
  const int landmark_count = static_cast<int>(ids.size());
  if (obs.empty()) {
    return stats;
  }
  stats.ran = true;
  stats.observations = static_cast<int>(obs.size());

  // 2. Evaluate robust reprojection cost and collect pixel errors for the summary.
  const auto evaluate = [&](
    const std::unordered_map<int, NavState> & states, const std::vector<Eigen::Vector3d> & points,
    std::vector<double> * pixel_errors) {
      double cost = 0.0;
      for (const auto & o : obs) {
        Eigen::Vector3d P_i, P_c;
        if (!landmarkInCamera(
            states.at(o.frame), points[o.landmark], cam.T_cam_imu, kMinDepth, P_i, P_c))
        {
          cost += 2.0 * huber_delta_px * 1e3;   // behind the camera: large, finite
          continue;
        }
        const double r = reprojectionResidual(P_c, o.pixel, cam).norm();
        cost += r <= huber_delta_px ? r * r :
          2.0 * huber_delta_px * r - huber_delta_px * huber_delta_px;
        if (pixel_errors) {
          pixel_errors->push_back(r);
        }
      }
      return cost;
    };
  const auto median = [](std::vector<double> v) {
      if (v.empty()) {return 0.0;}
      std::nth_element(v.begin(), v.begin() + v.size() / 2, v.end());
      return v[v.size() / 2];
    };

  std::vector<double> pixel_errors;
  double cost = evaluate(camera_states, landmarks, &pixel_errors);
  stats.median_px_before = median(pixel_errors);
  stats.median_px_after = stats.median_px_before;
  double lambda = 1e-3;

  // 3. Linearize, eliminate landmarks, then accept or reject each LM step.
  for (int it = 0; it < max_iterations; ++it) {
    // --- Normal equations. H_pp is block-diagonal: an observation touches ONE camera.
    std::vector<Eigen::Matrix<double, 6, 6>> H_pp(
      free_pose_count, Eigen::Matrix<double, 6, 6>::Zero());
    std::vector<Eigen::Matrix<double, 6, 1>> b_p(
      free_pose_count, Eigen::Matrix<double, 6, 1>::Zero());
    std::vector<Eigen::Matrix3d> H_ll(landmark_count, Eigen::Matrix3d::Zero());
    std::vector<Eigen::Vector3d> b_l(landmark_count, Eigen::Vector3d::Zero());
    std::vector<Eigen::Matrix<double, 6, 3>> H_pl(obs.size(), Eigen::Matrix<double, 6, 3>::Zero());

    for (std::size_t n = 0; n < obs.size(); ++n) {
      const Observation & o = obs[n];
      if (observation_count[o.landmark] < 2) {
        continue;
      }
      const NavState & s = camera_states.at(o.frame);
      Eigen::Vector3d P_i, P_c;
      if (!landmarkInCamera(s, landmarks[o.landmark], cam.T_cam_imu, kMinDepth, P_i, P_c)) {
        continue;
      }
      const PixelResidual r = reprojectionResidual(P_c, o.pixel, cam);
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

    // Dampen the blocks, then eliminate landmarks: Schur system S dp = rhs.
    Eigen::MatrixXd S = Eigen::MatrixXd::Zero(6 * free_pose_count, 6 * free_pose_count);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(6 * free_pose_count);
    for (int i = 0; i < free_pose_count; ++i) {
      Eigen::Matrix<double, 6, 6> Hd = H_pp[i];
      Hd.diagonal() += lambda * (Hd.diagonal().array() + 1e-9).matrix();
      S.block<6, 6>(6 * i, 6 * i) = Hd;
      rhs.segment<6>(6 * i) = b_p[i];
    }
    std::vector<Eigen::Matrix3d> H_ll_inv(landmark_count, Eigen::Matrix3d::Zero());
    std::vector<std::vector<std::size_t>> obs_of(landmark_count);
    for (std::size_t n = 0; n < obs.size(); ++n) {
      if (obs[n].pose >= 0 && !H_pl[n].isZero()) {
        obs_of[obs[n].landmark].push_back(n);
      }
    }
    for (int l = 0; l < landmark_count; ++l) {
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
    const Eigen::VectorXd dp = free_pose_count > 0 ?
      Eigen::VectorXd(S.ldlt().solve(rhs)) : Eigen::VectorXd();
    if (!dp.allFinite()) {
      break;
    }

    // --- Back-substitute and try the step.
    std::unordered_map<int, NavState> candidate_states = camera_states;
    for (const auto & kv : free_index) {
      NavVec dx = NavVec::Zero();
      dx.segment<3>(kIdxPhi) = dp.segment<3>(6 * kv.second);
      dx.segment<3>(kIdxPos) = dp.segment<3>(6 * kv.second + 3);
      candidate_states[kv.first] = boxplus(camera_states.at(kv.first), dx);
    }
    std::vector<Eigen::Vector3d> candidate_landmarks = landmarks;
    for (int l = 0; l < landmark_count; ++l) {
      Eigen::Vector3d rl = b_l[l];
      for (std::size_t a : obs_of[l]) {
        rl -= H_pl[a].transpose() * dp.segment<6>(6 * obs[a].pose);
      }
      candidate_landmarks[l] += H_ll_inv[l] * rl;
    }
    const double cost_new = evaluate(candidate_states, candidate_landmarks, nullptr);
    ++stats.iterations;
    if (cost_new < cost) {
      camera_states = std::move(candidate_states);
      landmarks = std::move(candidate_landmarks);
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

  // 4. If the pair pose moved, rescale about the fixed base to restore its original ruler.
  const double pair_before = w.pose.at(w.second).inverse().translation().norm();
  const double pair_after = camera_states.at(w.second).p.norm();
  if (!fix_pair_frame && pair_after > 1e-12) {
    const double k = pair_before / pair_after;
    for (auto & kv : camera_states) {
      kv.second.p *= k;
    }
    for (auto & Xl : landmarks) {
      Xl *= k;
    }
  }

  pixel_errors.clear();
  evaluate(camera_states, landmarks, &pixel_errors);
  stats.median_px_after = median(pixel_errors);

  // Write refined states back in SfM's base-camera pose convention.
  for (const auto & kv : camera_states) {
    Eigen::Isometry3d T_c0_ck = Eigen::Isometry3d::Identity();
    T_c0_ck.linear() = kv.second.R.matrix();
    T_c0_ck.translation() = kv.second.p;
    w.pose[kv.first] = T_c0_ck.inverse();
  }
  for (int l = 0; l < landmark_count; ++l) {
    w.landmark[ids[l]] = landmarks[l];
  }
  return stats;
}

}  // namespace glassvio
