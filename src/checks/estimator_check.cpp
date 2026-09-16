// Deterministic offline drive of the real VioEstimator, with no frame drops.
// Compares its trajectory with ground truth after one rigid alignment at bootstrap.
// Quality thresholds target the default EuRoC V1_01_easy run; experiments may override
// them or explicitly choose --report-only. Input and output failures always exit nonzero.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <memory>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <sensor_msgs/msg/imu.hpp>

#include "glassvio/camera_calib.hpp"
#include "glassvio/dataset.hpp"
#include "glassvio/estimator_regression.hpp"
#include "glassvio/types.hpp"
#include "glassvio/vio_estimator.hpp"

static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

// Round the fractional nanoseconds and carry across the seconds boundary.
builtin_interfaces::msg::Time toStamp(double t)
{
  builtin_interfaces::msg::Time stamp;
  const auto sec = static_cast<int64_t>(std::floor(t));
  const auto ns = static_cast<int64_t>(std::llround((t - std::floor(t)) * 1e9));
  stamp.sec = static_cast<int32_t>(sec + ns / 1000000000);
  stamp.nanosec = static_cast<uint32_t>(ns % 1000000000);
  return stamp;
}

// Every numeric option must consume its entire value; NaN must never disable a gate.
double nonnegativeNumber(const std::string & value)
{
  std::size_t used = 0;
  const double number = std::stod(value, &used);
  if (used != value.size() || !std::isfinite(number) || number < 0.0) {
    throw std::invalid_argument("expected a finite nonnegative number: " + value);
  }
  return number;
}

int nonnegativeInteger(const std::string & value)
{
  std::size_t used = 0;
  const int number = std::stoi(value, &used);
  if (used != value.size() || number < 0) {
    throw std::invalid_argument("expected a nonnegative integer: " + value);
  }
  return number;
}

/// StampedImu -> the sensor_msgs the MeasureGroup carries. The estimator converts straight
/// back at its boundary; this just satisfies the type.
sensor_msgs::msg::Imu::ConstSharedPtr toMsg(const glassvio::StampedImu & s)
{
  auto m = std::make_shared<sensor_msgs::msg::Imu>();
  m->header.stamp = toStamp(s.t);
  m->angular_velocity.x = s.gyro.x();
  m->angular_velocity.y = s.gyro.y();
  m->angular_velocity.z = s.gyro.z();
  m->linear_acceleration.x = s.accel.x();
  m->linear_acceleration.y = s.accel.y();
  m->linear_acceleration.z = s.accel.z();
  return m;
}

}  // namespace

int main(int argc, char ** argv)
{
  // Positional [bag] [config] [out.csv], plus one optional flag for the accel-bias experiment
  // (doc/08-sliding-window.md §5):
  //   (none)       the node's defaults: b_a estimated in the alignment when observable
  //   --no-est-ba  b_a pinned at 0 through the bootstrap -- the old behaviour, the baseline
  //   --oracle-ba  b_a pinned at the dataset's TRUE value: what the scale would be if b_a were
  //                known. An oracle, not a mode to ship -- it splits "b_a is the cause" from
  //                "b_a is a bystander".
  // and, independently, for where the scale goes during TRACKING:
  //   --oracle-map    triangulate new landmarks from GROUND-TRUTH poses (carried into the
  //                   estimator's world by T_align) instead of solved ones
  //   --parallax=DEG  the map's minimum triangulation parallax (default 1.0 deg)
  // and the camera-vs-IMU weighting (doc/08-sliding-window.md §6, Step 0):
  //   --px-sigma=PX   pixel noise the camera rows are whitened by (default 1.0)
  //   --imu-weight=W  scale on the IMU factor's information (default 1.0)
  //   --imu-noise-x=K multiply the datasheet IMU noise densities (EuRoC's MAV vibrates)
  // and Stage A, the keyframe window (doc/08-sliding-window.md §6):
  //   --no-window     the per-frame tracker alone, as before Stage A
  //   --anchor-gauge  VINS's anchor (position + yaw pinned, tilt soft) instead of ORB-SLAM3's
  //   --kf-every=N    tracked frames between keyframes (default 4)
  //   --window=K      keyframes in the window (default 10)
  //   --window-tri    the window ALSO triangulates new landmarks, from its optimized poses (off
  //                   by default: measured worse, doc/08 §6)
  //   --no-map-tri    the map does not triangulate either -- new landmarks from the window alone
  //   --gravity       re-estimate gravity's direction in the window (off by default: it halves
  //                   the tilt but gains no position accuracy, doc/08 §6)
  //   --gravity-sigma=DEG  the window's prior on gravity's direction (default 1 deg; 0 = none)
  //   --inlier-fraction=F  refuse a tracker solve when fewer than F of its observations agree
  //                   with it (default 0.5; 0 = accept every converged solve, as before)
  //   --no-refused-insert  while coasting, do not insert a refused solve's frame at the
  //                   coasted pose (starved frames still are)
  //   --coast-on-pnp  a coast frame adopts the PnP-replaced seed, as before, instead of the IMU
  //                   prediction
  //   --no-recycle    a track dropped as an outlier is never triangulated again
  //   --no-forget     the window keeps its views of a landmark the map dropped as an outlier
  //   --init-log      one line per bootstrap attempt: the stage that refused, and its numbers
  //   --sfm-window=N  frames the bootstrap reconstructs and aligns over (the node uses 30)
  //   --refine-gravity  the alignment fixes |g| and re-solves gravity's direction (VINS-Fusion)
  //   --no-sfm-ba     the bootstrap reconstruction goes to the alignment without bundle adjustment
  //   --oracle-sfm=rot|pos|both  the bootstrap reconstruction's rotations and/or positions from
  //                   ground truth, in the reconstruction's own ruler -- is stage [4] fed badly?
  //   --fast=N        the tracker's FAST corner threshold (default 20)
  //   --window-outlier-px=PX  the window drops observations more than PX off before solving
  std::vector<std::string> pos;
  std::string mode = "est-ba";
  bool oracle_map = false;
  double parallax_deg = 0.0;   // 0 = the node's default, for this and the two below
  double px_sigma = 0.0;
  double imu_weight = 0.0;
  double imu_noise_x = 0.0;
  bool no_window = false;
  bool anchor_gauge = false;
  int kf_every = 0;
  int window_k = 0;
  bool window_tri = false;
  bool no_map_tri = false;
  bool gravity = false;
  double gravity_sigma = -1.0;   // < 0 = the default
  double inlier_fraction = -1.0;   // < 0 = the default
  bool no_refused_insert = false;
  bool coast_on_pnp = false;
  bool no_recycle = false;
  bool no_forget = false;
  bool init_log = false;
  int fast_threshold = 20;
  int sfm_window = 0;   // 0 = the initializer default
  bool refine_gravity = false;
  bool no_sfm_ba = false;
  std::string oracle_sfm;   // "", "rot", "pos" or "both"
  double window_outlier_px = -1.0;   // < 0 = the default
  glassvio::EstimatorRegressionLimits limits;
  bool report_only = false;
  std::string gt_path;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string a = argv[i];
      if (a == "--help") {
        std::printf(
          "Usage: estimator_check [bag] [config] [out.csv] [options]\n"
          "  --gt=PATH                  required for a nondefault bag\n"
          "  --min-tracked-seconds=S     default 60, measured from bootstrap\n"
          "  --max-median-error=M        default 0.75 metres\n"
          "  --min-below-1m-seconds=S    default 30, continuous since bootstrap\n"
          "  --report-only              print quality failures but exit 0\n"
          "Experiment options (see doc/08-sliding-window.md):\n"
          "  --no-est-ba --oracle-ba --oracle-map --parallax=DEG --px-sigma=PX\n"
          "  --imu-weight=W --imu-noise-x=K --no-window --anchor-gauge\n"
          "  --kf-every=N --window=K --window-tri --no-map-tri --gravity\n"
          "  --gravity-sigma=DEG --inlier-fraction=F\n"
          "Exit codes: 0 quality pass (or report-only), 1 quality failure, 2 input/I/O error.\n");
        return 0;
      } else if (a.rfind("--gt=", 0) == 0) {
        gt_path = a.substr(5);
        if (gt_path.empty()) {
          throw std::invalid_argument("--gt requires a path");
        }
      } else if (a == "--report-only") {
        report_only = true;
      } else if (a.rfind("--min-tracked-seconds=", 0) == 0) {
        limits.min_tracked_seconds = nonnegativeNumber(a.substr(22));
      } else if (a.rfind("--max-median-error=", 0) == 0) {
        limits.max_median_error = nonnegativeNumber(a.substr(19));
      } else if (a.rfind("--min-below-1m-seconds=", 0) == 0) {
        limits.min_below_1m_seconds = nonnegativeNumber(a.substr(23));
      } else if (a == "--no-est-ba" || a == "--oracle-ba") {
        mode = a.substr(2);
      } else if (a == "--oracle-map") {
        oracle_map = true;
      } else if (a.rfind("--parallax=", 0) == 0) {
        parallax_deg = nonnegativeNumber(a.substr(11));
      } else if (a.rfind("--px-sigma=", 0) == 0) {
        px_sigma = nonnegativeNumber(a.substr(11));
      } else if (a.rfind("--imu-weight=", 0) == 0) {
        imu_weight = nonnegativeNumber(a.substr(13));
      } else if (a.rfind("--imu-noise-x=", 0) == 0) {
        imu_noise_x = nonnegativeNumber(a.substr(14));
      } else if (a == "--no-window") {
        no_window = true;
      } else if (a == "--anchor-gauge") {
        anchor_gauge = true;
      } else if (a.rfind("--kf-every=", 0) == 0) {
        kf_every = nonnegativeInteger(a.substr(11));
      } else if (a.rfind("--window=", 0) == 0) {
        window_k = nonnegativeInteger(a.substr(9));
      } else if (a == "--window-tri") {
        window_tri = true;
      } else if (a == "--no-map-tri") {
        no_map_tri = true;
      } else if (a == "--gravity") {
        gravity = true;
      } else if (a.rfind("--gravity-sigma=", 0) == 0) {
        gravity_sigma = nonnegativeNumber(a.substr(16));
      } else if (a.rfind("--inlier-fraction=", 0) == 0) {
        inlier_fraction = nonnegativeNumber(a.substr(18));
      } else if (a == "--no-refused-insert") {
        no_refused_insert = true;
      } else if (a == "--coast-on-pnp") {
        coast_on_pnp = true;
      } else if (a == "--no-recycle") {
        no_recycle = true;
      } else if (a == "--no-forget") {
        no_forget = true;
      } else if (a == "--init-log") {
        init_log = true;
      } else if (a.rfind("--oracle-sfm=", 0) == 0) {
        oracle_sfm = a.substr(13);
        if (oracle_sfm != "rot" && oracle_sfm != "pos" && oracle_sfm != "both") {
          throw std::invalid_argument("--oracle-sfm takes rot, pos or both");
        }
      } else if (a == "--no-sfm-ba") {
        no_sfm_ba = true;
      } else if (a == "--refine-gravity") {
        refine_gravity = true;
      } else if (a.rfind("--sfm-window=", 0) == 0) {
        sfm_window = static_cast<int>(nonnegativeNumber(a.substr(13)));
      } else if (a.rfind("--fast=", 0) == 0) {
        fast_threshold = static_cast<int>(nonnegativeNumber(a.substr(7)));
      } else if (a.rfind("--window-outlier-px=", 0) == 0) {
        window_outlier_px = nonnegativeNumber(a.substr(20));
      } else if (!a.empty() && a.front() == '-') {
        throw std::invalid_argument("unknown option: " + a);
      } else {
        pos.push_back(a);
      }
    }
    if (pos.size() > 3) {
      throw std::invalid_argument("expected at most [bag] [config] [out.csv]");
    }
    if (gt_path.empty()) {
      if (!pos.empty() && pos[0] != kDefaultBag) {
        throw std::invalid_argument("a nondefault bag requires --gt=PATH");
      }
      gt_path = kDefaultGt;
    }
  } catch (const std::exception & e) {
    std::fprintf(stderr, "invalid arguments: %s\n", e.what());
    return 2;
  }
  const std::string bag_path = pos.size() > 0 ? pos[0] : kDefaultBag;
  const std::string calib_dir = pos.size() > 1 ? pos[1] : "config";
  const std::string csv_path = pos.size() > 2 ? pos[2] : "/tmp/glassvio_run.csv";

  glassvio::CameraCalib calib;
  glassvio::EurocDataset bag;
  try {
    calib = glassvio::loadEurocCalib(calib_dir);
    glassvio::DatasetOptions opts;
    opts.track_images = true;
    opts.fast_threshold = fast_threshold;
    bag = glassvio::EurocDataset::load(bag_path, gt_path, calib, opts);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (imu_noise_x > 0.0) {
    calib.gyro_noise *= imu_noise_x;
    calib.accel_noise *= imu_noise_x;
    std::printf(
      "IMU noise densities x%.1f: gyro %.3g rad/s/rtHz, accel %.3g m/s^2/rtHz\n",
      imu_noise_x, calib.gyro_noise, calib.accel_noise);
  }
  if (bag.frames.size() < 100 || bag.imu.empty() || bag.gt.empty()) {
    std::fprintf(stderr, "need image, imu and ground-truth streams\n");
    return 2;
  }
  std::printf(
    "deterministic drive: %zu frames, %zu imu, %zu gt\n",
    bag.frames.size(), bag.imu.size(), bag.gt.size());

  glassvio::EstimatorParams ep;   // exactly the node's defaults, unless a mode flag says otherwise
  if (mode != "est-ba") {
    ep.init.estimate_accel_bias = false;
  }
  if (mode == "oracle-ba") {
    // EuRoC's b_a is near-constant over the sequence; take it where the bootstrap will fire.
    const std::size_t kb = std::min<std::size_t>(ep.bootstrap_frames, bag.frames.size() - 1);
    ep.init.accel_bias = bag.gt.accelBias(bag.frames[kb].t);
  }
  if (parallax_deg > 0.0) {
    ep.map.min_parallax_deg = parallax_deg;
  }
  if (px_sigma > 0.0) {
    ep.visual.reproj.sigma_px = px_sigma;
  }
  if (imu_weight > 0.0) {
    ep.visual.imu_prior_weight = imu_weight;
  }
  ep.window.enabled = !no_window;
  ep.window.anchor_full = !anchor_gauge;
  if (kf_every > 0) {
    ep.window.keyframe_every = kf_every;
  }
  if (window_k > 0) {
    ep.window.max_keyframes = window_k;
  }
  ep.window.triangulate = window_tri;
  ep.map.triangulate = !no_map_tri;
  ep.window.estimate_gravity = gravity;
  if (gravity_sigma >= 0.0) {
    ep.window.gravity_sigma_deg = gravity_sigma;
  }
  if (inlier_fraction >= 0.0) {
    ep.visual.min_inlier_fraction = inlier_fraction;
  }
  ep.coast_insert_refused = !no_refused_insert;
  ep.coast_on_pnp = coast_on_pnp;
  ep.map.recycle_outliers = !no_recycle;
  ep.window.forget_outliers = !no_forget;
  if (sfm_window > 0) {
    ep.init.window_frames = sfm_window;
  }
  ep.init.refine_gravity = refine_gravity;
  ep.init.sfm_bundle_adjust = !no_sfm_ba;
  if (window_outlier_px >= 0.0) {
    ep.window.outlier_px = window_outlier_px;
  }
  std::printf(
    "tracker solves refused below an inlier fraction of %.2f\n", ep.visual.min_inlier_fraction);
  std::printf(
    "gravity direction: %s (prior %.2f deg)\n",
    ep.window.enabled && ep.window.estimate_gravity ? "re-estimated by the window" : "frozen",
    ep.window.gravity_sigma_deg);
  std::printf(
    "new landmarks triangulated by: window %s, map %s\n",
    ep.window.enabled && ep.window.triangulate ? "yes" : "no", ep.map.triangulate ? "yes" : "no");
  std::printf(
    "window: %s, %d keyframes, one every %d frames, anchor %s\n",
    ep.window.enabled ? "ON" : "off", ep.window.max_keyframes, ep.window.keyframe_every,
    ep.window.anchor_full ? "fully fixed (ORB-SLAM3)" : "gauge only (VINS)");
  std::printf(
    "mode: b_a %s, map from %s, min parallax %.1f deg, sigma_px %.1f, imu weight %.1f\n",
    mode.c_str(), oracle_map ? "GROUND-TRUTH poses" : "solved poses", ep.map.min_parallax_deg,
    ep.visual.reproj.sigma_px, ep.visual.imu_prior_weight);

  const auto & imu = bag.imu.samples();
  std::size_t imu_cursor = 0;

  // Per-frame CSV, so a run can be plotted rather than squinted at. One row per frame.
  std::ofstream csv(csv_path);
  if (!csv) {
    std::fprintf(stderr, "cannot open output CSV: %s\n", csv_path.c_str());
    return 2;
  }
  csv <<
    "t,stage,feats,map,pending,rmse_px,"
    "pos_err,vel_err,"
    "px,py,pz,gx,gy,gz,vx,vy,vz,gvx,gvy,gvz,"
    "bgx,bgy,bgz,gbgx,gbgy,gbgz,bax,bay,baz,gbax,gbay,gbaz,"
    "win_lm,win_cost0,win_cost1,win_iter,win_shift,win_tri,"
    "tri_try,tri_inf,tri_depth,tri_par,pend_exp,grav_err_deg,med_px,inliers,"
    "win_c0_prior,win_c0_imu,win_c0_vis,"
    "win_kfs,imu_worst_k,imu_worst_dt,imu_worst_rot_deg,imu_worst_vel,imu_worst_pos,"
    "rot_err_deg,pnp_jump_deg,pnp_jump_m,win_vis_obs,win_vis_bad,win_vis_kf,"
    "tri_recycled,win_outl\n";
  csv.setf(std::ios::fixed);
  csv.precision(6);

  int bootstrapped_at = -1;
  int tracked = 0;
  int lost_at = -1;
  double boot_depth_median = 0.0;
  std::size_t boot_landmarks = 0;
  std::vector<glassvio::EstimatorRegressionSample> regression_samples;
  double bootstrap_time = std::numeric_limits<double>::quiet_NaN();
  bool finite_state = true;
  std::vector<double> speed_ratios;
  double first_over_1m = -1.0;   // the honest survival score, see the verdict
  double last_grav_err = 0.0;

  // ALIGNMENT AT BOOTSTRAP, so the error is honest. The estimator DEFINES its own world:
  // origin at the first body pose, +Z along gravity, yaw arbitrary. That frame is NOT the
  // ground-truth frame, so a raw |p_est - p_gt| conflates real drift with a fixed origin/yaw
  // offset -- which is exactly what inflated the earlier "3.4 m". T_align is the one rigid
  // transform between the two worlds, fixed at the bootstrap instant; everything after is the
  // drift THROUGH it.
  Eigen::Isometry3d T_align = Eigen::Isometry3d::Identity();
  const double t0 = bag.frames.front().t;

  // --oracle-map: ground truth, carried into the estimator's world through T_align. Only after
  // the bootstrap -- before it there is no T_align, and no map to grow.
  if (oracle_map) {
    ep.oracle_insert_pose = [&](double t, Eigen::Isometry3d & T_world_body) {
        if (bootstrapped_at < 0) {
          return false;
        }
        T_world_body = T_align.inverse() * bag.gt.at(t);
        return true;
      };
  }
  if (!oracle_sfm.empty()) {
    const bool sub_rot = oracle_sfm != "pos";
    const bool sub_pos = oracle_sfm != "rot";
    ep.init.oracle_sfm = [&, sub_rot, sub_pos](
      const std::vector<glassvio::SfmFrame> & fr, glassvio::SfmWindow & w) {
        if (w.second < 0 || !w.pose.count(w.second)) {
          return;
        }
        const Eigen::Isometry3d T_ci = calib.T_cam_imu;
        const auto truth_c0_ck = [&](int k) -> Eigen::Isometry3d {   // metric
            return T_ci * bag.gt.at(fr[w.base].t).inverse() * bag.gt.at(fr[k].t) * T_ci.inverse();
          };
        // Keep the reconstruction's ruler: metres per unit, fixed by its own base pair.
        const double ruler = truth_c0_ck(w.second).translation().norm() /
          std::max(1e-9, w.pose.at(w.second).inverse().translation().norm());
        for (auto & kv : w.pose) {
          Eigen::Isometry3d T = kv.second.inverse();   // T_c0_ck, ruler units
          const Eigen::Isometry3d G = truth_c0_ck(kv.first);
          if (sub_rot) {
            T.linear() = G.linear();
          }
          if (sub_pos) {
            T.translation() = G.translation() / ruler;
          }
          kv.second = T.inverse();
        }
      };
  }
  glassvio::VioEstimator est(calib, ep);

  for (std::size_t k = 0; k < bag.frames.size(); ++k) {
    const auto & f = bag.frames[k];

    glassvio::MeasureGroup g;
    g.header.stamp = toStamp(f.t);
    for (const auto & entry : f.by_id) {
      g.features.ids.push_back(entry.first);
      g.features.points.push_back(entry.second);
    }
    // Feed an unbroken IMU chain including the first sample at/after the image.
    // The estimator retains that endpoint for the next interval's interpolation.
    while (imu_cursor < imu.size() && imu[imu_cursor].t < f.t) {
      g.imu.push_back(toMsg(imu[imu_cursor++]));
    }
    if (imu_cursor < imu.size() && (imu_cursor == 0 || imu[imu_cursor - 1].t < f.t)) {
      g.imu.push_back(toMsg(imu[imu_cursor++]));
    }

    const int attempts_before = est.bootstrapAttempts();
    const glassvio::FrameResult r = est.process(g);
    if (init_log && est.bootstrapAttempts() != attempts_before) {
      const glassvio::InitResult & ir = est.lastInit();
      std::printf(
        "init %6.2f s  shared %3d cand %2d best_lm %3d lm %3zu pairs %3d intervals %3d "
        "sfm_ba %.2f->%.2f px |g| err %5.2f%% s %.4f sigma_s/s %.3f ba_std %.2f  %s\n",
        f.t - t0, ir.sfm.max_shared, ir.sfm.candidates_tried, ir.sfm.max_trial_landmarks,
        ir.sfm.landmark.size(), ir.bias_pairs, ir.align_intervals, ir.sfm_ba.median_px_before,
        ir.sfm_ba.median_px_after,
        100.0 * std::abs(ir.gravity_sfm.norm() - 9.80665) / 9.80665, ir.scale,
        ir.scale_uncertainty, ir.accel_bias_std,
        est.lastFailure().empty() ? "OK" : est.lastFailure().c_str());
      // The reconstruction against the truth, scale-free: every posed frame relative to the
      // base -- body rotation error, translation DIRECTION error, and the metres per ruler unit
      // the truth implies (median), next to the s the alignment solved.
      if (ir.sfm.pose.count(ir.sfm.base) && est.lastInitTimes().count(ir.sfm.base)) {
        const Eigen::Isometry3d T_ci = calib.T_cam_imu;
        const Eigen::Isometry3d G0 = bag.gt.at(est.lastInitTimes().at(ir.sfm.base));
        std::vector<double> rot, dir, s_true;
        double worst_rot = -1.0;
        int worst_k = -1;
        for (const auto & kv : ir.sfm.pose) {
          if (kv.first == ir.sfm.base) {
            continue;
          }
          // body k in body 0: SfM gives T_ck_c0 = pose[k]; T_b0_bk = T_ic * T_c0_ck * T_ci.
          const Eigen::Isometry3d S = T_ci.inverse() * kv.second.inverse() * T_ci;
          const Eigen::Isometry3d G = G0.inverse() * bag.gt.at(est.lastInitTimes().at(kv.first));
          rot.push_back(
            Eigen::AngleAxisd(S.linear().transpose() * G.linear()).angle() * 180.0 / M_PI);
          if (rot.back() > worst_rot) {
            worst_rot = rot.back();
            worst_k = kv.first;
          }
          // camera-centre translation is what SfM measures; compare c0->ck in the base camera.
          const Eigen::Vector3d ts = kv.second.inverse().translation();
          const Eigen::Vector3d tg = (T_ci * G * T_ci.inverse()).translation();
          if (ts.norm() > 1e-9 && tg.norm() > 0.01) {
            dir.push_back(
              std::acos(std::clamp(ts.normalized().dot(tg.normalized()), -1.0, 1.0)) * 180.0 /
              M_PI);
            s_true.push_back(tg.norm() / ts.norm());
          }
        }
        const auto med = [](std::vector<double> v) {
            if (v.empty()) {return std::nan("");}
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
          };
        std::printf(
          "     vs truth: %zu poses, rot err med %.2f max %.2f deg, dir err med %.1f deg, "
          "s_true %.4f (solved %.4f)  worst: frame %+d from base%s\n", rot.size(), med(rot),
          rot.empty() ? std::nan("") : *std::max_element(rot.begin(), rot.end()), med(dir),
          med(s_true), ir.scale, worst_k - ir.sfm.base,
          worst_k == ir.sfm.second ? " (the base pair)" : " (PnP)");
      }
    }

    const bool tracking_now = r.stage == glassvio::FrameResult::Stage::Bootstrapped ||
      r.stage == glassvio::FrameResult::Stage::Tracking;
    // The ground truth ENDING while the estimator still tracks is not an input error: EuRoC's
    // images outlast its ground truth, so any run that survives the sequence gets here. Scoring
    // stops where the truth does. A tracked frame BEFORE the truth begins still is an error.
    if (tracking_now && f.t > bag.gt.t_end()) {
      std::printf("\nground truth ends at t = %.1f s; scoring stops there\n", f.t - t0);
      break;
    }
    if (tracking_now && f.t < bag.gt.t_begin()) {
      std::fprintf(stderr, "ground truth does not cover estimated frame at %.9f\n", f.t);
      return 2;
    }

    // The estimator body pose in ITS world, this frame.
    Eigen::Isometry3d T_wb = Eigen::Isometry3d::Identity();
    T_wb.linear() = est.state().R.matrix();
    T_wb.translation() = est.state().p;

    if (r.stage == glassvio::FrameResult::Stage::Bootstrapped) {
      bootstrapped_at = static_cast<int>(k);
      bootstrap_time = f.t;
      boot_landmarks = est.landmarks().size();
      // The one rigid transform between est-world and gt-world, fixed here forever.
      T_align = bag.gt.at(f.t) * T_wb.inverse();

      const Eigen::Isometry3d T_cw = (T_wb * calib.T_cam_imu.inverse()).inverse();
      std::vector<double> depths;
      for (const auto & lm : est.landmarks()) {
        depths.push_back((T_cw * lm.second).z());
      }
      std::sort(depths.begin(), depths.end());
      boot_depth_median = depths.empty() ? 0.0 : depths[depths.size() / 2];

      std::printf(
        "\nBOOTSTRAP at frame %d (%.1f s): %zu landmarks, median depth %.3f m, "
        "|v|=%.2f\n  bg = [%+.4f %+.4f %+.4f]  (truth [%+.4f %+.4f %+.4f])\n",
        bootstrapped_at, f.t - t0, boot_landmarks, boot_depth_median, est.state().v.norm(),
        est.state().bg.x(), est.state().bg.y(), est.state().bg.z(),
        bag.gt.gyroBias(f.t).x(), bag.gt.gyroBias(f.t).y(), bag.gt.gyroBias(f.t).z());

      // Gravity TILT at bootstrap. The estimator defines +Z along ITS gravity, so a roll/pitch
      // error lands in T_align as a rotation that moves Z; yaw is free and does not.
      const Eigen::Vector3d gba = bag.gt.accelBias(f.t);
      const double tilt_deg = std::acos(std::clamp(
            (T_align.linear() * Eigen::Vector3d::UnitZ()).z(), -1.0, 1.0)) * 180.0 / M_PI;
      std::printf(
        "  ba = [%+.3f %+.3f %+.3f]  (truth [%+.3f %+.3f %+.3f])\n"
        "  gravity tilt %.2f deg, |v|/|v_gt| = %.3f\n",
        est.state().ba.x(), est.state().ba.y(), est.state().ba.z(), gba.x(), gba.y(), gba.z(),
        tilt_deg, est.state().v.norm() / std::max(bag.gt.velocity(f.t).norm(), 1e-9));
    } else if (r.stage == glassvio::FrameResult::Stage::Lost) {
      if (lost_at < 0 && bootstrapped_at >= 0) {
        lost_at = static_cast<int>(k);
      }
    }

    // --- CSV row, and the tracked-error accumulation, both off the ALIGNED pose.
    if (bootstrapped_at >= 0 &&
      (r.stage == glassvio::FrameResult::Stage::Tracking ||
      r.stage == glassvio::FrameResult::Stage::Bootstrapped))
    {
      const Eigen::Isometry3d est_in_gt = T_align * T_wb;
      const Eigen::Vector3d p = est_in_gt.translation();
      const Eigen::Vector3d gp = bag.gt.at(f.t).translation();
      const Eigen::Vector3d v = T_align.linear() * est.state().v;
      const Eigen::Vector3d gv = bag.gt.velocity(f.t);
      const double pos_err = (p - gp).norm();
      const double vel_err = (v - gv).norm();
      if (first_over_1m < 0.0 && pos_err > 1.0) {
        first_over_1m = f.t - t0;
      }
      // Gravity's DIRECTION against the truth, in the estimator's own world: the true "down",
      // carried in by T_align. At the bootstrap this is the tilt printed above.
      const double grav_err = std::acos(
        std::clamp(
          est.gravity().normalized().dot(
            T_align.linear().transpose() * Eigen::Vector3d(0.0, 0.0, -1.0)), -1.0, 1.0)) *
        180.0 / M_PI;
      last_grav_err = grav_err;
      // The tracker's ATTITUDE against the truth, through the same T_align -- the bootstrap tilt
      // is its floor. A step here is the pose itself turning, whatever the pixels say.
      const double rot_err = Eigen::AngleAxisd(
        (T_align.linear() * est.state().R.matrix()).transpose() *
        bag.gt.at(f.t).linear()).angle() * 180.0 / M_PI;
      if (r.stage == glassvio::FrameResult::Stage::Tracking) {
        ++tracked;
        regression_samples.push_back({f.t, pos_err});
        // THE SCALE, read off the speed. Skip near-hover frames, where the ratio is noise.
        if (gv.norm() > 0.3) {
          speed_ratios.push_back(v.norm() / gv.norm());
        }
      }

      const auto & s = est.state();
      finite_state = finite_state && s.R.matrix().allFinite() && s.p.allFinite() &&
        s.v.allFinite() && s.bg.allFinite() && s.ba.allFinite() &&
        std::isfinite(pos_err) && std::isfinite(vel_err) && std::isfinite(grav_err);
      const Eigen::Vector3d gbg = bag.gt.gyroBias(f.t);
      const Eigen::Vector3d gba = bag.gt.accelBias(f.t);
      // The window's vision cost per keyframe, anchor first, as one ';'-joined field.
      std::string vis_kf;
      for (const double c : est.lastWindow().vis_cost_before_kf) {
        vis_kf += (vis_kf.empty() ? "" : ";") + std::to_string(static_cast<long>(c));
      }
      csv << (f.t - t0) << ","
          << (r.stage == glassvio::FrameResult::Stage::Bootstrapped ? "boot" : "track") << ","
          << r.features << "," << est.map().size() << "," << est.map().pending() << ","
          << r.rmse_px << "," << pos_err << "," << vel_err << ","
          << p.x() << "," << p.y() << "," << p.z() << ","
          << gp.x() << "," << gp.y() << "," << gp.z() << ","
          << v.x() << "," << v.y() << "," << v.z() << ","
          << gv.x() << "," << gv.y() << "," << gv.z() << ","
          << s.bg.x() << "," << s.bg.y() << "," << s.bg.z() << ","
          << gbg.x() << "," << gbg.y() << "," << gbg.z() << ","
          << s.ba.x() << "," << s.ba.y() << "," << s.ba.z() << ","
          << gba.x() << "," << gba.y() << "," << gba.z() << ","
        // The LAST window solve (it repeats between keyframes): how it went, and how far it
        // moved the state the tracker resumes from.
          << est.lastWindow().landmarks << "," << est.lastWindow().cost_before << ","
          << est.lastWindow().cost_after << "," << est.lastWindow().iterations << ","
          << est.lastWindow().newest_shift_m << "," << est.lastWindow().triangulated << ","
        // Why the map's pending tracks did NOT mature this frame (LandmarkMap::lastStats).
          << est.map().lastStats().attempts << "," << est.map().lastStats().at_infinity << ","
          << est.map().lastStats().depth << "," << est.map().lastStats().parallax << ","
          << est.map().lastStats().expired << "," << grav_err << ","
          << r.median_px << "," << r.inliers << ","
          << est.lastWindow().cost_before_prior << "," << est.lastWindow().cost_before_imu << ","
          << est.lastWindow().cost_before_vis << "," << est.lastWindow().keyframes << ","
          << est.lastWindow().worst_imu_k << "," << est.lastWindow().worst_imu_dt << ","
          << est.lastWindow().worst_imu_rot_deg << "," << est.lastWindow().worst_imu_vel << ","
          << est.lastWindow().worst_imu_pos << "," << rot_err << "," << r.pnp_jump_deg << ","
          << r.pnp_jump_m << "," << est.lastWindow().vis_obs_before << ","
          << est.lastWindow().vis_bad_obs_before << "," << vis_kf << ","
          << est.map().lastStats().recycled << "," << est.lastWindow().outliers_removed << "\n";
    }
    if (r.stage == glassvio::FrameResult::Stage::Lost && bootstrapped_at >= 0) {
      break;
    }
  }

  csv.close();
  if (!csv) {
    std::fprintf(stderr, "failed to write output CSV: %s\n", csv_path.c_str());
    return 2;
  }
  std::printf("\nper-frame CSV -> %s\n\n=== VERDICT ===\n", csv_path.c_str());
  auto verdict = glassvio::estimatorRegression(bootstrap_time, regression_samples, limits);
  if (!std::isfinite(boot_depth_median) || boot_depth_median < 0.3) {
    verdict.failures.push_back("bootstrap median landmark depth is below 0.3 m or nonfinite");
  }
  if (!finite_state) {
    verdict.failures.push_back("a tracked state or ground-truth error is nonfinite");
  }
  std::printf(
    "bootstrapped at frame %d; tracked %d frames (%.2f s) before %s\n"
    "bootstrap median landmark depth: %.3f m (minimum 0.300 m)\n"
    "tracked duration: %.2f s (minimum %.2f s)\n"
    "median position error vs ground truth: %.3f m (maximum %.3f m)\n"
    "continuous duration at or below 1 m: %.2f s (minimum %.2f s)\n",
    bootstrapped_at, tracked, verdict.tracked_seconds,
    lost_at < 0 ? "the sequence ended" : "losing the scene",
    boot_depth_median, verdict.tracked_seconds, limits.min_tracked_seconds,
    verdict.median_error, limits.max_median_error,
    verdict.below_1m_seconds, limits.min_below_1m_seconds);
  if (first_over_1m < 0.0) {
    std::printf("no position error above 1 m was observed\n");
  } else {
    std::printf("position error first passed 1 m at t = %.1f s\n", first_over_1m);
  }
  std::printf("gravity direction error at the last tracked frame: %.2f deg\n", last_grav_err);
  if (!speed_ratios.empty()) {
    std::sort(speed_ratios.begin(), speed_ratios.end());
    std::printf(
      "median |v|/|v_gt| while tracking: %.3f  (1.0 = metric scale held)\n",
      speed_ratios[speed_ratios.size() / 2]);
  }
  std::printf(
    "keyframe window: %d solves, %d refused, %d landmarks triangulated\n", est.windowSolves(),
    est.windowRefusals(), est.windowTriangulated());
  for (const auto & failure : verdict.failures) {
    std::printf("FAIL: %s\n", failure.c_str());
  }
  std::printf("REGRESSION %s%s\n", verdict.failures.empty() ? "PASS" : "FAIL",
    report_only ? " (report-only: quality does not affect exit status)" : "");
  return report_only || verdict.failures.empty() ? 0 : 1;
}
