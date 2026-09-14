#include "glassvio/keyframe_window.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Dense>

#include "glass_core/gauss_newton.hpp"   // huberWeight
#include "glass_core/nav_residual.hpp"

namespace glassvio
{

namespace
{

constexpr int D = kNavDim;
/// Information for a direction the anchor must not move in -- sigma 1e-4 in its own units.
constexpr double kHard = 1e8;

/// The Huber COST that huberWeight is the IRLS weight of: quadratic inside delta, linear out.
double huberCost(double r, double delta)
{
  const double a = std::abs(r);
  return a <= delta ? r * r : 2.0 * delta * a - delta * delta;
}

struct Track
{
  long id;
  Eigen::Vector3d X;
  std::vector<std::pair<int, Eigen::Vector2d>> obs;   ///< (keyframe index, pixel)
};

}  // namespace

bool schurSolve(
  const Eigen::MatrixXd & H_pp, const Eigen::VectorXd & b_p,
  const std::vector<LandmarkBlock> & landmarks, double lambda,
  Eigen::VectorXd & dp, std::vector<Eigen::Vector3d> & dl, Eigen::MatrixXd * S_out)
{
  Eigen::MatrixXd S = H_pp;
  S.diagonal() += lambda * H_pp.diagonal();
  Eigen::VectorXd bs = b_p;

  std::vector<Eigen::Matrix3d> Hll_inv(landmarks.size(), Eigen::Matrix3d::Zero());
  for (std::size_t l = 0; l < landmarks.size(); ++l) {
    const LandmarkBlock & B = landmarks[l];
    Eigen::Matrix3d Hll = B.H_ll;
    Hll.diagonal() += lambda * B.H_ll.diagonal();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(Hll);
    const Eigen::Vector3d ev = es.eigenvalues();   // ascending
    if (!(ev(2) > 0.0) || !(ev(0) > 1e-9 * ev(2))) {
      continue;   // seen from one direction only: no depth, constrains nothing
    }
    Hll_inv[l] =
      es.eigenvectors() * ev.cwiseInverse().asDiagonal() * es.eigenvectors().transpose();

    for (const auto & [k, Hkl] : B.H_kl) {
      const Eigen::Matrix<double, D, 3> HkW = Hkl * Hll_inv[l];   // H_kl H_ll^-1
      bs.segment<D>(D * k).noalias() -= HkW * B.b_l;
      for (const auto & [k2, Hk2l] : B.H_kl) {
        S.block<D, D>(D * k, D * k2).noalias() -= HkW * Hk2l.transpose();
      }
    }
  }
  if (S_out) {
    *S_out = S;
  }

  const Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
  if (ldlt.info() != Eigen::Success) {
    return false;
  }
  dp = ldlt.solve(bs);
  if (!dp.allFinite()) {
    return false;
  }

  // Back-substitute: dl = H_ll^-1 (b_l - H_lp dp). Zero for a skipped landmark.
  dl.assign(landmarks.size(), Eigen::Vector3d::Zero());
  for (std::size_t l = 0; l < landmarks.size(); ++l) {
    Eigen::Vector3d rhs = landmarks[l].b_l;
    for (const auto & [k, Hkl] : landmarks[l].H_kl) {
      rhs.noalias() -= Hkl.transpose() * dp.segment<D>(D * k);
    }
    dl[l] = Hll_inv[l] * rhs;
  }
  return true;
}

void KeyframeWindow::push(Keyframe kf)
{
  kfs_.push_back(std::move(kf));
  while (static_cast<int>(kfs_.size()) > p_.max_keyframes + 1) {
    kfs_.pop_front();
  }
}

void KeyframeWindow::forget(const std::vector<long> & ids)
{
  for (auto & kf : kfs_) {
    for (const long id : ids) {
      kf.obs.erase(id);
    }
  }
}

WindowResult KeyframeWindow::optimize(
  const std::unordered_map<long, Eigen::Vector3d> & landmarks, const ImuBuffer & imu,
  const CameraCalib & calib, const ReprojectionParams & reproj, const Eigen::Vector3d & gravity)
{
  WindowResult out;
  const int n = static_cast<int>(kfs_.size());
  if (n < 3) {
    return out;
  }

  // --- The IMU between consecutive keyframes, re-integrated from the buffer on every solve (at
  // the newer keyframe's bias, which is the one imuResidual corrects with). No merging of
  // per-frame deltas as VINS-Fusion does: the buffer keeps every sample, so one call is exact.
  std::vector<ImuPreintegration> pre;
  std::vector<Eigen::Matrix<double, 9, 9>> info_imu;
  std::vector<Eigen::Matrix<double, 6, 6>> info_bias;
  pre.reserve(n - 1);
  for (int k = 0; k + 1 < n; ++k) {
    const NavState & xj = kfs_[k + 1].x;
    pre.emplace_back(xj.bg, xj.ba, calib.gyro_noise, calib.accel_noise);
    if (!imu.preintegrate(
        kfs_[k].t, kfs_[k + 1].t, xj.bg, xj.ba, pre.back(), calib.gyro_noise,
        calib.accel_noise))
    {
      return out;   // a hole in the IMU inside the window: no chain to solve over
    }
    const Eigen::Matrix<double, 9, 9> I9 = pre.back().covariance().inverse();
    if (!I9.allFinite()) {
      return out;
    }
    info_imu.push_back(I9);
    // Biases may only drift as fast as their random walk allows over this interval.
    const double dt = pre.back().dt();
    Eigen::Matrix<double, 6, 1> w;
    w << Eigen::Vector3d::Constant(1.0 / (calib.gyro_random_walk * calib.gyro_random_walk * dt)),
      Eigen::Vector3d::Constant(1.0 / (calib.accel_random_walk * calib.accel_random_walk * dt));
    info_bias.push_back(w.asDiagonal());
  }

  // --- The landmarks worth solving for: in the map, and seen by at least two keyframes. One
  // view gives a ray, not a point.
  std::unordered_map<long, int> seen;
  for (const auto & kf : kfs_) {
    for (const auto & o : kf.obs) {
      if (landmarks.count(o.first)) {
        ++seen[o.first];
      }
    }
  }
  std::unordered_map<long, int> index;
  std::vector<Track> tracks;
  for (int k = 0; k < n; ++k) {
    for (const auto & o : kfs_[k].obs) {
      const auto s = seen.find(o.first);
      if (s == seen.end() || s->second < 2) {
        continue;
      }
      auto it = index.find(o.first);
      if (it == index.end()) {
        it = index.emplace(o.first, static_cast<int>(tracks.size())).first;
        tracks.push_back({o.first, landmarks.at(o.first), {}});
      }
      tracks[it->second].obs.emplace_back(k, Eigen::Vector2d(o.second.x, o.second.y));
    }
  }
  // --- Gross outliers, kept out of the solve entirely (WindowParams::outlier_px).
  if (p_.outlier_px > 0.0) {
    for (auto & tr : tracks) {
      const std::size_t before = tr.obs.size();
      tr.obs.erase(
        std::remove_if(
          tr.obs.begin(), tr.obs.end(),
          [&](const std::pair<int, Eigen::Vector2d> & o) {
            Eigen::Vector3d P_i, P_c;
            return !landmarkInCamera(
              kfs_[o.first].x, tr.X, calib.T_cam_imu, reproj.min_depth, P_i, P_c) ||
                   reprojectionResidual(P_c, o.second, calib).norm() > p_.outlier_px;
          }),
        tr.obs.end());
      out.outliers_removed += static_cast<int>(before - tr.obs.size());
    }
    tracks.erase(
      std::remove_if(
        tracks.begin(), tracks.end(), [](const Track & tr) {return tr.obs.size() < 2;}),
      tracks.end());
  }
  if (tracks.empty()) {
    return out;
  }

  // --- The anchor: the gauge, and the window's only memory of what came before it.
  const NavState anchor_ref = kfs_[0].x;
  Eigen::Matrix<double, D, D> info_anchor = Eigen::Matrix<double, D, D>::Zero();
  if (p_.anchor_full) {
    info_anchor.diagonal().setConstant(kHard);
  } else {
    // Yaw is rotation about WORLD z; under the right perturbation R Exp(dphi) that is the body
    // direction R^T e_z. Pin that one axis of the rotation and leave the other two -- the tilt,
    // which gravity observes -- to the soft prior.
    const Eigen::Vector3d yaw = anchor_ref.R.inverse() * Eigen::Vector3d::UnitZ();
    const Eigen::Matrix3d P_yaw = yaw * yaw.transpose();
    const double w_tilt = 1.0 / (p_.anchor_sigma_tilt_rad * p_.anchor_sigma_tilt_rad);
    info_anchor.block<3, 3>(kIdxPhi, kIdxPhi) =
      kHard * P_yaw + w_tilt * (Eigen::Matrix3d::Identity() - P_yaw);
    info_anchor.block<3, 3>(kIdxPos, kIdxPos).diagonal().setConstant(kHard);
    info_anchor.block<3, 3>(kIdxVel, kIdxVel).diagonal().setConstant(
      1.0 / (p_.anchor_sigma_velocity * p_.anchor_sigma_velocity));
    info_anchor.block<3, 3>(kIdxBg, kIdxBg).diagonal().setConstant(
      1.0 / (p_.anchor_sigma_gyro_bias * p_.anchor_sigma_gyro_bias));
    info_anchor.block<3, 3>(kIdxBa, kIdxBa).diagonal().setConstant(
      1.0 / (p_.anchor_sigma_accel_bias * p_.anchor_sigma_accel_bias));
  }

  const double inv_s = 1.0 / reproj.sigma_px;
  const double delta = reproj.huber_delta_px * inv_s;
  // A landmark that a step pushes behind a camera must COST something, or the solver would
  // happily "fix" a residual by making it disappear.
  const double behind_cost = 2.0 * huberCost(10.0 * delta, delta);

  // --- Every factor, once: the cost, and -- when H is given -- the linearization. The same
  // function for both, so the LM accept test can never disagree with what was linearized.
  // --- Gravity's DIRECTION, as two more unknowns after the keyframes (doc/08 §6): th tilts the
  // gravity the window was handed, g(th) = Exp([th_x, th_y, 0]) g_ref. |g| is known; only the tilt
  // the bootstrap got wrong is free. No landmark couples to it, so schurSolve is unchanged.
  const int gi = D * n;
  const int np = D * n + (p_.estimate_gravity ? 2 : 0);
  const double w_grav = p_.gravity_sigma_deg > 0.0 ?
    1.0 / std::pow(p_.gravity_sigma_deg * M_PI / 180.0, 2) : 0.0;

  double c_prior = 0.0;   // the last build()'s cost, by factor (WindowResult::cost_before_*)
  double c_imu = 0.0;
  double c_vis = 0.0;
  const auto build = [&](
    const std::vector<NavState> & xs, const std::vector<Eigen::Vector3d> & Xs,
    const Eigen::Vector2d & th, Eigen::MatrixXd * H, Eigen::VectorXd * b,
    std::vector<LandmarkBlock> * blocks) -> double {
      const bool lin = H != nullptr;
      if (lin) {
        H->setZero(np, np);
        b->setZero(np);
        blocks->assign(tracks.size(), LandmarkBlock());
      }
      double cost = 0.0;
      c_prior = c_imu = c_vis = 0.0;
      const Eigen::Vector3d g = tiltGravity(gravity, th);

      if (p_.estimate_gravity) {   // the gravity prior, toward the direction this window was handed
        cost += w_grav * th.squaredNorm();
        c_prior += w_grav * th.squaredNorm();
        if (lin) {
          H->block<2, 2>(gi, gi).diagonal().array() += w_grav;
          b->segment<2>(gi) -= w_grav * th;
        }
      }

      {   // the anchor prior
        const PriorResidual r = priorResidual(xs[0], anchor_ref);
        cost += r.dot(info_anchor * r);
        c_prior += r.dot(info_anchor * r);
        if (lin) {
          const PriorJacobian J = priorJacobian(xs[0], anchor_ref);
          H->block<D, D>(0, 0) += J.transpose() * info_anchor * J;
          b->segment<D>(0) -= J.transpose() * info_anchor * r;
        }
      }

      for (int k = 0; k + 1 < n; ++k) {   // IMU + bias random walk, keyframe k -> k+1
        const int i0 = D * k;
        const int j0 = D * (k + 1);
        const ImuResidual r = imuResidual(xs[k], xs[k + 1], pre[k], g);
        const Eigen::Matrix<double, 6, 1> rb = biasResidual(xs[k], xs[k + 1]);
        cost += r.dot(info_imu[k] * r) + rb.dot(info_bias[k] * rb);
        c_imu += r.dot(info_imu[k] * r) + rb.dot(info_bias[k] * rb);
        if (!lin) {
          continue;
        }
        // BOTH halves of the IMU Jacobian -- the reason imuJacobianI was built.
        const ImuJacobian Ji = imuJacobianI(xs[k], xs[k + 1], pre[k], g);
        const ImuJacobian Jj = imuJacobian(xs[k], xs[k + 1], pre[k]);
        const Eigen::Matrix<double, D, 9> JiT_O = Ji.transpose() * info_imu[k];
        const Eigen::Matrix<double, D, 9> JjT_O = Jj.transpose() * info_imu[k];
        H->block<D, D>(i0, i0) += JiT_O * Ji;
        H->block<D, D>(i0, j0) += JiT_O * Jj;
        H->block<D, D>(j0, i0) += JjT_O * Ji;
        H->block<D, D>(j0, j0) += JjT_O * Jj;
        b->segment<D>(i0) -= JiT_O * r;
        b->segment<D>(j0) -= JjT_O * r;
        if (p_.estimate_gravity) {   // every IMU factor also sees gravity -- linearly
          const Eigen::Matrix<double, 9, 2> Jg = imuGravityDirectionJacobian(xs[k], pre[k], g);
          const Eigen::Matrix<double, D, 2> HiG = JiT_O * Jg;
          const Eigen::Matrix<double, D, 2> HjG = JjT_O * Jg;
          H->block<D, 2>(i0, gi) += HiG;
          H->block<2, D>(gi, i0) += HiG.transpose();
          H->block<D, 2>(j0, gi) += HjG;
          H->block<2, D>(gi, j0) += HjG.transpose();
          H->block<2, 2>(gi, gi) += Jg.transpose() * info_imu[k] * Jg;
          b->segment<2>(gi) -= Jg.transpose() * info_imu[k] * r;
        }
        // r_b = b_j - b_i: d/dx_j = biasJacobian(), d/dx_i = -biasJacobian().
        const Eigen::Matrix<double, 6, D> Bj = biasJacobian();
        const Eigen::Matrix<double, D, D> BOB = Bj.transpose() * info_bias[k] * Bj;
        const Eigen::Matrix<double, D, 1> BOr = Bj.transpose() * info_bias[k] * rb;
        H->block<D, D>(i0, i0) += BOB;
        H->block<D, D>(i0, j0) -= BOB;
        H->block<D, D>(j0, i0) -= BOB;
        H->block<D, D>(j0, j0) += BOB;
        b->segment<D>(i0) += BOr;
        b->segment<D>(j0) -= BOr;
      }

      for (std::size_t l = 0; l < tracks.size(); ++l) {   // reprojection, keyframe <-> landmark
        for (const auto & [k, z] : tracks[l].obs) {
          Eigen::Vector3d P_i, P_c;
          if (!landmarkInCamera(xs[k], Xs[l], calib.T_cam_imu, reproj.min_depth, P_i, P_c)) {
            cost += behind_cost;
            c_vis += behind_cost;
            continue;
          }
          // Whitened, exactly as accumulateReprojection whitens the tracker's rows.
          const PixelResidual r = reprojectionResidual(P_c, z, calib) * inv_s;
          cost += huberCost(r(0), delta) + huberCost(r(1), delta);
          c_vis += huberCost(r(0), delta) + huberCost(r(1), delta);
          if (!lin) {
            continue;
          }
          const PixelJacobian Jp =
            reprojectionJacobian(xs[k], P_i, P_c, calib.T_cam_imu, calib) * inv_s;
          const Eigen::Matrix<double, 2, 3> Jl = reprojectionLandmarkJacobian(Jp);
          const Eigen::Matrix2d W =
            Eigen::Vector2d(huberWeight(r(0), delta), huberWeight(r(1), delta)).asDiagonal();
          H->block<D, D>(D * k, D * k) += Jp.transpose() * W * Jp;
          b->segment<D>(D * k) -= Jp.transpose() * W * r;
          LandmarkBlock & B = (*blocks)[l];
          B.H_ll += Jl.transpose() * W * Jl;
          B.b_l -= Jl.transpose() * W * r;
          B.H_kl.emplace_back(k, Eigen::Matrix<double, D, 3>(Jp.transpose() * W * Jl));
        }
      }
      return cost;
    };

  // --- Levenberg-Marquardt. Damped because the window starts where the TRACKER left it, and
  // that is exactly the state whose map has drifted -- not always Gauss-Newton's sweet spot.
  std::vector<NavState> xs;
  for (const auto & kf : kfs_) {
    xs.push_back(kf.x);
  }
  std::vector<Eigen::Vector3d> Xs;
  for (const auto & t : tracks) {
    Xs.push_back(t.X);
  }

  Eigen::MatrixXd H;
  Eigen::VectorXd b;
  std::vector<LandmarkBlock> blocks;
  Eigen::Vector2d th = Eigen::Vector2d::Zero();
  double cost = build(xs, Xs, th, nullptr, nullptr, nullptr);
  out.cost_before = cost;
  out.cost_before_prior = c_prior;
  out.cost_before_imu = c_imu;
  out.cost_before_vis = c_vis;
  out.keyframes = n;
  for (int k = 0; k + 1 < n; ++k) {   // the worst IMU factor, before the solve moves anything
    const ImuResidual r = imuResidual(xs[k], xs[k + 1], pre[k], tiltGravity(gravity, th));
    const double c = r.dot(info_imu[k] * r);
    if (c > out.worst_imu_cost) {
      out.worst_imu_cost = c;
      out.worst_imu_k = k;
      out.worst_imu_dt = pre[k].dt();
      out.worst_imu_rot_deg = r.segment<3>(0).norm() * 180.0 / M_PI;
      out.worst_imu_vel = r.segment<3>(3).norm();
      out.worst_imu_pos = r.segment<3>(6).norm();
    }
  }
  out.vis_cost_before_kf.assign(n, 0.0);
  for (std::size_t l = 0; l < tracks.size(); ++l) {   // vision, per keyframe, before the solve
    for (const auto & [k, z] : tracks[l].obs) {
      ++out.vis_obs_before;
      Eigen::Vector3d P_i, P_c;
      if (!landmarkInCamera(xs[k], Xs[l], calib.T_cam_imu, reproj.min_depth, P_i, P_c)) {
        out.vis_cost_before_kf[k] += behind_cost;
        ++out.vis_bad_obs_before;
        continue;
      }
      const PixelResidual r = reprojectionResidual(P_c, z, calib);
      out.vis_cost_before_kf[k] +=
        huberCost(r(0) * inv_s, delta) + huberCost(r(1) * inv_s, delta);
      out.vis_bad_obs_before += r.norm() > 20.0;
    }
  }
  double lambda = 1e-4;
  for (int it = 0; it < p_.iterations; ++it) {
    build(xs, Xs, th, &H, &b, &blocks);
    Eigen::VectorXd dp;
    std::vector<Eigen::Vector3d> dl;
    if (!schurSolve(H, b, blocks, lambda, dp, dl)) {
      lambda *= 10.0;
      continue;
    }
    std::vector<NavState> xs_new(n);
    for (int k = 0; k < n; ++k) {
      xs_new[k] = boxplus(xs[k], dp.segment<D>(D * k));
    }
    std::vector<Eigen::Vector3d> Xs_new(Xs.size());
    for (std::size_t l = 0; l < Xs.size(); ++l) {
      Xs_new[l] = Xs[l] + dl[l];
    }
    const Eigen::Vector2d th_new =
      p_.estimate_gravity ? Eigen::Vector2d(th + dp.segment<2>(gi)) : th;
    const double cost_new = build(xs_new, Xs_new, th_new, nullptr, nullptr, nullptr);
    ++out.iterations;
    if (cost_new < cost) {
      xs = std::move(xs_new);
      Xs = std::move(Xs_new);
      th = th_new;
      cost = cost_new;
      lambda = std::max(lambda / 10.0, 1e-9);
      if (dp.lpNorm<Eigen::Infinity>() < 1e-6) {
        break;
      }
    } else {
      lambda *= 10.0;
    }
  }
  out.cost_after = cost;
  out.newest_shift_m = (xs[n - 1].p - kfs_[n - 1].x.p).norm();

  for (int k = 0; k < n; ++k) {
    kfs_[k].x = xs[k];
  }
  for (std::size_t l = 0; l < tracks.size(); ++l) {
    out.refined.emplace(tracks[l].id, Xs[l]);
  }
  out.gravity = tiltGravity(gravity, th);
  out.landmarks = static_cast<int>(tracks.size());
  out.ok = true;
  return out;
}

std::unordered_map<long, Eigen::Vector3d> KeyframeWindow::triangulate(
  const std::unordered_map<long, Eigen::Vector3d> & existing, const CameraCalib & calib,
  double min_depth) const
{
  std::unordered_map<long, Eigen::Vector3d> fresh;
  if (!p_.triangulate || kfs_.size() < 2) {
    return fresh;
  }

  // World -> camera for every keyframe, and its optical centre, once.
  const std::size_t n = kfs_.size();
  std::vector<Eigen::Isometry3d> T_cw(n);
  std::vector<Eigen::Vector3d> centre(n);
  for (std::size_t k = 0; k < n; ++k) {
    Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
    T_wb.linear() = kfs_[k].x.R.matrix();
    T_wb.translation() = kfs_[k].x.p;
    const Eigen::Isometry3d T_wc = T_wb * calib.T_cam_imu.inverse();
    T_cw[k] = T_wc.inverse();
    centre[k] = T_wc.translation();
  }
  const double cos_min = std::cos(p_.tri_min_parallax_deg * M_PI / 180.0);

  for (const auto & o : kfs_.back().obs) {
    const long id = o.first;
    if (existing.count(id)) {
      continue;
    }
    std::vector<std::size_t> views;
    for (std::size_t k = 0; k < n; ++k) {
      if (kfs_[k].obs.count(id)) {
        views.push_back(k);
      }
    }
    if (views.size() < 2) {
      continue;
    }

    // Linear triangulation over EVERY view (DLT): each observation says the point projects onto
    // its ray, x * P.row(2) = P.row(0) and y * P.row(2) = P.row(1) in normalized coordinates, and
    // the point is the null vector of the stacked rows. More views than the map's first-and-last
    // pair, and all of them at poses the window has just agreed with.
    Eigen::MatrixXd A(2 * views.size(), 4);
    for (std::size_t i = 0; i < views.size(); ++i) {
      const cv::Point2f & px = kfs_[views[i]].obs.at(id);
      const double u = (px.x - calib.cx()) / calib.fx();
      const double v = (px.y - calib.cy()) / calib.fy();
      const Eigen::Matrix<double, 3, 4> P = T_cw[views[i]].matrix().topRows<3>();
      A.row(2 * i) = u * P.row(2) - P.row(0);
      A.row(2 * i + 1) = v * P.row(2) - P.row(1);
    }
    const Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Eigen::Vector4d h = svd.matrixV().col(3);
    if (std::abs(h(3)) < 1e-12) {
      continue;   // at infinity: no baseline reached it
    }
    const Eigen::Vector3d X = h.head<3>() / h(3);

    // Parallax of the WIDEST pair of rays -- the same angle gate the map uses, for the same
    // reason: a rotating camera gives plenty of pixel flow and nothing to triangulate.
    double cos_widest = 1.0;
    for (std::size_t i = 0; i < views.size(); ++i) {
      for (std::size_t j = i + 1; j < views.size(); ++j) {
        cos_widest = std::min(
          cos_widest, (X - centre[views[i]]).normalized().dot(
            (X - centre[views[j]]).normalized()));
      }
    }
    if (cos_widest > cos_min) {
      continue;
    }

    // In front of every camera, and consistent with every observation.
    bool ok = true;
    for (const std::size_t k : views) {
      const Eigen::Vector3d Pc = T_cw[k] * X;
      if (Pc.z() < min_depth || Pc.z() > p_.tri_max_depth) {
        ok = false;
        break;
      }
      const cv::Point2f & px = kfs_[k].obs.at(id);
      const Eigen::Vector2d proj(
        calib.fx() * Pc.x() / Pc.z() + calib.cx(), calib.fy() * Pc.y() / Pc.z() + calib.cy());
      if ((proj - Eigen::Vector2d(px.x, px.y)).norm() > p_.tri_max_reproj_px) {
        ok = false;
        break;
      }
    }
    if (ok) {
      fresh.emplace(id, X);
    }
  }
  return fresh;
}

}  // namespace glassvio
