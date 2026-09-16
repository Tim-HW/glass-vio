// Self-check for bundleAdjust: a synthetic reconstruction with exact pixels, its free poses and
// landmarks perturbed, must come back to zero reprojection error -- with the two gauge frames
// untouched, and the free poses back where the truth put them (the gauge fixes the similarity,
// so "back" is well defined).
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>

#include "glassvio/sfm_bundle.hpp"

int main()
{
  glassvio::CameraCalib calib;
  calib.K << 458.0, 0.0, 367.0, 0.0, 457.0, 248.0, 0.0, 0.0, 1.0;

  std::mt19937 rng(7);
  std::normal_distribution<double> n01(0.0, 1.0);
  std::uniform_real_distribution<double> u(-1.0, 1.0);

  // Ten cameras sliding along x and yawing a little; T_ck_c0 = pose[k].
  const int K = 10;
  std::vector<Eigen::Isometry3d> truth(K);
  for (int k = 0; k < K; ++k) {
    Eigen::Isometry3d T_c0_ck = Eigen::Isometry3d::Identity();
    T_c0_ck.linear() = Eigen::AngleAxisd(0.02 * k, Eigen::Vector3d::UnitY()).toRotationMatrix();
    T_c0_ck.translation() = Eigen::Vector3d(0.1 * k, 0.01 * k, 0.0);
    truth[k] = T_c0_ck.inverse();
  }
  std::vector<glassvio::SfmFrame> frames(K);
  glassvio::SfmWindow w;
  w.valid = true;
  w.base = 0;
  w.second = K - 1;
  for (long id = 0; id < 150; ++id) {
    const Eigen::Vector3d X(2.0 * u(rng), 1.5 * u(rng), 6.0 + 2.0 * u(rng));
    w.landmark[id] = X;
    for (int k = 0; k < K; ++k) {
      const Eigen::Vector3d P = truth[k] * X;
      frames[k].by_id[id] = cv::Point2f(
        static_cast<float>(calib.fx() * P.x() / P.z() + calib.cx()),
        static_cast<float>(calib.fy() * P.y() / P.z() + calib.cy()));
    }
  }
  for (int k = 0; k < K; ++k) {
    Eigen::Isometry3d T = truth[k];
    if (k != w.base && k != w.second) {   // perturb the free poses only
      T.linear() = T.linear() *
        Eigen::AngleAxisd(0.02, Eigen::Vector3d(n01(rng), n01(rng), n01(rng)).normalized())
        .toRotationMatrix();
      T.translation() += 0.03 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
    }
    w.pose[k] = T;
  }
  for (auto & lm : w.landmark) {
    lm.second += 0.05 * Eigen::Vector3d(n01(rng), n01(rng), n01(rng));
  }

  const glassvio::SfmBundleStats st = glassvio::bundleAdjust(frames, w, calib, 30);
  std::printf(
    "bundleAdjust: %d obs, %d iters, median %.3f -> %.5f px\n", st.observations, st.iterations,
    st.median_px_before, st.median_px_after);
  assert(st.ran);
  assert(st.median_px_before > 1.0 && "the perturbation must be visible");
  assert(st.median_px_after < 0.01 && "exact pixels: the error must vanish");

  double worst_rot = 0.0, worst_pos = 0.0;
  for (int k = 0; k < K; ++k) {
    const Eigen::Isometry3d E = w.pose[k] * truth[k].inverse();
    worst_rot = std::max(worst_rot, Eigen::AngleAxisd(E.linear()).angle());
    worst_pos = std::max(worst_pos, (w.pose[k].inverse().translation() -
        truth[k].inverse().translation()).norm());
  }
  std::printf("worst pose error: %.2e rad, %.2e\n", worst_rot, worst_pos);
  assert(
    (w.pose[w.base].matrix() - truth[w.base].matrix()).norm() < 1e-12 &&
    "the gauge frames stay fixed");
  assert(
    (w.pose[w.second].matrix() - truth[w.second].matrix()).norm() < 1e-12 &&
    "the gauge frames stay fixed");
  assert(worst_rot < 1e-3 && worst_pos < 1e-3 && "free poses must return to the truth");
  return 0;
}
