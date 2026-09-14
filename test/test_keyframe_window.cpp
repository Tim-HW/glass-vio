// Self-check for the window's SCHUR SOLVE -- the one piece of new linear algebra Stage A adds.
//
// Bundle adjustment's landmark block is block-DIAGONAL (3x3 per point), and schurSolve exploits
// it: eliminate each landmark on its own, solve the small keyframe system, back-substitute. The
// oracle is what it replaces. Assemble the SAME system densely and solve it with one LDLT; and,
// independently, reduce it with glass_core's pinned schurMarginalize. Three routes, one answer --
// or the elimination is wrong in a way the estimator would never report.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include <Eigen/Dense>

#include "glass_core/marginalization.hpp"
#include "glassvio/keyframe_window.hpp"

int main()
{
  constexpr int D = glass_core::kNavDim;
  const int K = 4;    // keyframes
  const int L = 40;   // landmarks
  std::mt19937 rng(11);
  std::normal_distribution<double> g(0.0, 1.0);
  std::uniform_int_distribution<int> pick(0, K - 1);

  // The structured system schurSolve takes, and the same system dense, ordered
  // [landmarks ; keyframes] because schurMarginalize eliminates the FIRST block.
  Eigen::MatrixXd H_pp = Eigen::MatrixXd::Identity(D * K, D * K);   // a prior: keyframes constrained
  Eigen::VectorXd b_p = Eigen::VectorXd::Zero(D * K);
  std::vector<glassvio::LandmarkBlock> blocks(L);
  const int N = 3 * L + D * K;
  Eigen::MatrixXd H = Eigen::MatrixXd::Zero(N, N);
  Eigen::VectorXd b = Eigen::VectorXd::Zero(N);
  const auto lm = [](int l) {return 3 * l;};
  const auto kf = [L](int k) {return 3 * L + D * k;};
  H.bottomRightCorner(D * K, D * K) = H_pp;

  for (int l = 0; l < L; ++l) {
    std::vector<int> seen;   // 2-4 distinct keyframes see each landmark
    while (static_cast<int>(seen.size()) < 2 + l % 3) {
      const int k = pick(rng);
      if (std::find(seen.begin(), seen.end(), k) == seen.end()) {
        seen.push_back(k);
      }
    }
    for (const int k : seen) {
      Eigen::Matrix<double, 2, D> Jp;
      Eigen::Matrix<double, 2, 3> Jl;
      Eigen::Vector2d r;
      for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < D; ++j) {
          Jp(i, j) = g(rng);
        }
        for (int j = 0; j < 3; ++j) {
          Jl(i, j) = g(rng);
        }
        r(i) = g(rng);
      }
      H_pp.block<D, D>(D * k, D * k) += Jp.transpose() * Jp;
      b_p.segment<D>(D * k) -= Jp.transpose() * r;
      glassvio::LandmarkBlock & B = blocks[l];
      B.H_ll += Jl.transpose() * Jl;
      B.b_l -= Jl.transpose() * r;
      B.H_kl.emplace_back(k, Eigen::Matrix<double, D, 3>(Jp.transpose() * Jl));

      H.block<D, D>(kf(k), kf(k)) += Jp.transpose() * Jp;
      H.block<3, 3>(lm(l), lm(l)) += Jl.transpose() * Jl;
      H.block<D, 3>(kf(k), lm(l)) += Jp.transpose() * Jl;
      H.block<3, D>(lm(l), kf(k)) += Jl.transpose() * Jp;
      b.segment<D>(kf(k)) -= Jp.transpose() * r;
      b.segment<3>(lm(l)) -= Jl.transpose() * r;
    }
  }

  // Route 1: the Schur solve under test.
  Eigen::VectorXd dp;
  std::vector<Eigen::Vector3d> dl;
  Eigen::MatrixXd S;
  const bool ok = glassvio::schurSolve(H_pp, b_p, blocks, 0.0, dp, dl, &S);
  assert(ok && "a well-posed system must solve");

  // Route 2: the whole system, dense, one LDLT.
  const Eigen::VectorXd x = H.ldlt().solve(b);
  const double scale = std::max(1.0, x.cwiseAbs().maxCoeff());
  double err = (dp - x.tail(D * K)).cwiseAbs().maxCoeff();
  for (int l = 0; l < L; ++l) {
    err = std::max(err, (dl[l] - x.segment<3>(lm(l))).cwiseAbs().maxCoeff());
  }
  err /= scale;
  assert(err < 1e-8 && "Schur solve disagrees with the dense solve");

  // Route 3: glass_core's pinned kernel, eliminating the landmark block densely.
  Eigen::MatrixXd H_marg;
  Eigen::VectorXd b_marg;
  glass_core::schurMarginalize(H, b, 3 * L, H_marg, b_marg);
  const double err_S = (S - H_marg).cwiseAbs().maxCoeff() / H_marg.cwiseAbs().maxCoeff();
  assert(err_S < 1e-10 && "reduced keyframe system disagrees with schurMarginalize");

  // A landmark seen from ONE direction has a rank-deficient H_ll: no depth. It must be SKIPPED --
  // inverting it would inject an arbitrary depth -- and must leave the solution untouched.
  {
    glassvio::LandmarkBlock one_view;
    Eigen::Matrix<double, 2, 3> Jl;
    Jl << 1.0, 0.0, 0.3, 0.0, 1.0, -0.2;   // rank 2
    one_view.H_ll = Jl.transpose() * Jl;
    one_view.b_l = Eigen::Vector3d(0.5, -0.5, 0.1);
    one_view.H_kl.emplace_back(0, Eigen::Matrix<double, D, 3>::Ones());
    std::vector<glassvio::LandmarkBlock> with = blocks;
    with.push_back(one_view);
    Eigen::VectorXd dp2;
    std::vector<Eigen::Vector3d> dl2;
    assert(glassvio::schurSolve(H_pp, b_p, with, 0.0, dp2, dl2));
    const bool skipped = dl2.back().isZero() && (dp2 - dp).cwiseAbs().maxCoeff() < 1e-12;
    assert(skipped && "a one-view landmark must be skipped, not inverted");
  }

  // --- Triangulation from the window's keyframe poses. Three keyframes 20 cm apart, and a track
  // for every gate: one that should pass, and one that should fail each check.
  double tri_err = 0.0;
  {
    glassvio::CameraCalib calib;
    calib.K << 458.654, 0, 367.215, 0, 457.296, 248.375, 0, 0, 1;
    Eigen::Isometry3d T_ci = Eigen::Isometry3d::Identity();
    T_ci.linear() = (Eigen::AngleAxisd(-M_PI / 2, Eigen::Vector3d::UnitZ()) *
      Eigen::AngleAxisd(0.03, Eigen::Vector3d::UnitX())).toRotationMatrix();
    T_ci.translation() = Eigen::Vector3d(0.065, -0.021, -0.008);
    calib.T_cam_imu = T_ci;

    std::vector<glass_core::NavState> xs(3);
    for (int k = 0; k < 3; ++k) {
      xs[k].R = Sophus::SO3d::exp(Eigen::Vector3d(0.0, 0.0, 0.02 * k));
      xs[k].p = Eigen::Vector3d(0.2 * k, 0.0, 0.0);
    }
    // A point in front of keyframe 0's camera, and its pixel in keyframe k.
    const auto point = [&](double depth, double lateral) -> Eigen::Vector3d {
        return xs[0].R * (T_ci.inverse() * Eigen::Vector3d(lateral, -0.1, depth)) + xs[0].p;
      };
    const auto pixel = [&](int k, const Eigen::Vector3d & X) -> cv::Point2f {
        const Eigen::Vector3d Pc = T_ci * (xs[k].R.inverse() * (X - xs[k].p));
        return cv::Point2f(
          static_cast<float>(calib.fx() * Pc.x() / Pc.z() + calib.cx()),
          static_cast<float>(calib.fy() * Pc.y() / Pc.z() + calib.cy()));
      };
    const Eigen::Vector3d X_good = point(3.0, 0.2);    // ~7.6 deg of parallax over 0.4 m
    const Eigen::Vector3d X_far = point(30.0, 0.2);    // ~0.76 deg: under the 1 deg gate
    const Eigen::Vector3d X_known = point(2.5, -0.3);  // already a landmark
    const Eigen::Vector3d X_bad = point(3.5, 0.0);     // one view mismatched by 20 px
    const Eigen::Vector3d X_gone = point(2.0, 0.1);    // not seen by the newest keyframe

    // Off by default in the pipeline (measured worse, doc/08 §6) -- ON here, because this pins the
    // code itself, whatever the default.
    glassvio::WindowParams wp;
    wp.triangulate = true;
    glassvio::KeyframeWindow win(wp);
    for (int k = 0; k < 3; ++k) {
      glassvio::Keyframe kf;
      kf.t = 0.2 * k;
      kf.x = xs[k];
      kf.obs.emplace(1, pixel(k, X_good));
      kf.obs.emplace(2, pixel(k, X_far));
      kf.obs.emplace(3, pixel(k, X_known));
      cv::Point2f bad = pixel(k, X_bad);
      if (k == 1) {
        bad.x += 20.0f;
      }
      kf.obs.emplace(4, bad);
      if (k < 2) {
        kf.obs.emplace(5, pixel(k, X_gone));
      }
      win.push(std::move(kf));
    }
    const std::unordered_map<long, Eigen::Vector3d> existing{{3, X_known}};
    const std::unordered_map<long, Eigen::Vector3d> fresh = win.triangulate(existing, calib, 0.1);
    const bool only_good = fresh.size() == 1 && fresh.count(1) == 1;
    assert(only_good && "only the well-conditioned, consistent, live, new track is triangulated");
    tri_err = (fresh.at(1) - X_good).norm();
    assert(tri_err < 1e-4 && "triangulated point is off");
  }

  // --- The gravity-DIRECTION Jacobian the window uses: glass_core's imuGravityJacobian (pinned
  // there) times the 3x2 tilt map. Pinned here by finite differences through tiltGravity itself.
  double grav_err = 0.0;
  {
    glass_core::ImuPreintegration pre(
      Eigen::Vector3d(0.01, -0.02, 0.005), Eigen::Vector3d(0.1, 0.2, -0.1), 1e-3, 1e-2);
    for (int s = 0; s < 40; ++s) {
      pre.integrate(Eigen::Vector3d(0.3, -0.1, 0.2), Eigen::Vector3d(0.5, 9.7, 0.3), 0.005);
    }
    glass_core::NavState xi;
    glass_core::NavState xj;
    xi.R = Sophus::SO3d::exp(Eigen::Vector3d(0.2, -0.4, 0.7));
    xi.p = Eigen::Vector3d(0.3, -0.2, 1.0);
    xi.v = Eigen::Vector3d(0.4, 0.1, -0.2);
    xj.R = Sophus::SO3d::exp(Eigen::Vector3d(0.25, -0.35, 0.75));
    xj.p = Eigen::Vector3d(0.38, -0.18, 0.98);
    xj.v = Eigen::Vector3d(0.45, 0.05, -0.25);
    const Eigen::Vector3d g0(0.3, -0.5, -9.79);   // already tilted, as a window hands it over
    const Eigen::Matrix<double, 9, 2> Ja = glassvio::imuGravityDirectionJacobian(xi, pre, g0);
    Eigen::Matrix<double, 9, 2> Jn;
    const double h = 1e-6;
    for (int c = 0; c < 2; ++c) {
      Eigen::Vector2d d = Eigen::Vector2d::Zero();
      d(c) = h;
      Jn.col(c) = (glass_core::imuResidual(xi, xj, pre, glassvio::tiltGravity(g0, d)) -
        glass_core::imuResidual(xi, xj, pre, glassvio::tiltGravity(g0, -d))) / (2.0 * h);
    }
    grav_err = (Ja - Jn).cwiseAbs().maxCoeff() / std::max(1.0, Jn.cwiseAbs().maxCoeff());
    assert(grav_err < 1e-6 && "gravity-direction Jacobian disagrees with finite differences");
    const double mag = glassvio::tiltGravity(g0, Eigen::Vector2d(0.1, -0.2)).norm();
    assert(std::abs(mag - g0.norm()) < 1e-12 && "a tilt must not change |g|");
  }

  std::printf(
    "ok: Schur solve matches the dense solve (%.1e) and schurMarginalize (%.1e) over %d "
    "keyframes, %d landmarks; a one-view landmark is skipped;\n"
    "    window triangulation recovers the point (%.1e m) and refuses thin parallax, a "
    "mismatched view, a known landmark, and a dead track;\n"
    "    gravity-direction Jacobian matches finite differences (%.1e), |g| preserved\n",
    err, err_S, K, L, tri_err, grav_err);
  return 0;
}
