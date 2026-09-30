// Deterministic offline drive of the real VioEstimator, with no frame drops.
// Compares its trajectory with ground truth after one rigid alignment at bootstrap.
// Quality thresholds target the default EuRoC V1_01_easy run; experiments may override
// them or explicitly choose --report-only. Input and output failures always exit nonzero.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
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


/// Everything the command line can change, written straight into the structs the run uses. A
/// flag is ONE entry in flags() below -- it used to be a comment, a sentinel variable, a parse
/// branch and an apply-to-params block, in four different places, for each of forty flags.
struct Options
{
  glassvio::EstimatorParams ep;   // exactly the node's defaults, unless a flag says otherwise
  glassvio::DatasetOptions data;
  glassvio::EstimatorRegressionLimits limits;
  std::vector<std::string> pos;   // [bag] [config] [out.csv]
  std::string gt_path;
  std::string mode = "est-ba";    // est-ba | no-est-ba | oracle-ba
  std::string oracle_sfm;         // "", rot, pos or both
  double imu_noise_x = 0.0;       // 0 = the datasheet densities
  bool oracle_map = false;
  bool oracle_bg = false;
  bool init_log = false;
  bool report_only = false;
};

struct Flag
{
  const char * name;    ///< "--window"; one that takes a value is written --window=K
  const char * value;   ///< nullptr for a switch, else the placeholder --help shows
  const char * help;
  std::function<void(Options &, const std::string &)> set;
};

/// A count or a weight where 0 means "leave the default alone".
template<typename T>
void setIfPositive(T & field, double v)
{
  if (v > 0.0) {
    field = static_cast<T>(v);
  }
}

const std::vector<Flag> & flags()
{
  using O = Options;
  using S = const std::string &;
  static const std::vector<Flag> table = {
    // --- inputs and the quality gate
    {"--gt", "PATH", "ground-truth CSV; required for a nondefault bag",
      [](O & o, S v) {
        if (v.empty()) {throw std::invalid_argument("--gt requires a path");}
        o.gt_path = v;
      }},
    {"--min-tracked-seconds", "S", "default 60, measured from bootstrap",
      [](O & o, S v) {o.limits.min_tracked_seconds = nonnegativeNumber(v);}},
    {"--max-median-error", "M", "default 0.75 metres",
      [](O & o, S v) {o.limits.max_median_error = nonnegativeNumber(v);}},
    {"--min-below-1m-seconds", "S", "default 30, continuous since bootstrap",
      [](O & o, S v) {o.limits.min_below_1m_seconds = nonnegativeNumber(v);}},
    {"--report-only", nullptr, "print quality failures but exit 0",
      [](O & o, S) {o.report_only = true;}},

    // --- the accel-bias experiment and the oracles (doc/08 §4-§6). Oracles are TEST-ONLY:
    //     they split "X is the cause" from "X is a bystander", they are not modes to ship.
    {"--no-est-ba", nullptr, "b_a pinned at 0 through the bootstrap (the old baseline)",
      [](O & o, S) {o.mode = "no-est-ba";}},
    {"--oracle-ba", nullptr, "b_a pinned at the dataset's TRUE value",
      [](O & o, S) {o.mode = "oracle-ba";}},
    {"--oracle-bg", nullptr, "the bootstrap's gyro bias from ground truth",
      [](O & o, S) {o.oracle_bg = true;}},
    {"--oracle-map", nullptr, "triangulate new landmarks from GROUND-TRUTH poses",
      [](O & o, S) {o.oracle_map = true;}},
    {"--oracle-sfm", "rot|pos|both",
      "the bootstrap reconstruction's rotations and/or positions from ground truth",
      [](O & o, S v) {
        if (v != "rot" && v != "pos" && v != "both") {
          throw std::invalid_argument("--oracle-sfm takes rot, pos or both");
        }
        o.oracle_sfm = v;
      }},

    // --- camera-vs-IMU weighting (doc/08 §6, Step 0)
    {"--parallax", "DEG", "the map's minimum triangulation parallax (default 1.0)",
      [](O & o, S v) {setIfPositive(o.ep.map.min_parallax_deg, nonnegativeNumber(v));}},
    {"--px-sigma", "PX", "pixel noise the camera rows are whitened by (default 1.0)",
      [](O & o, S v) {setIfPositive(o.ep.visual.reproj.sigma_px, nonnegativeNumber(v));}},
    {"--imu-weight", "W", "scale on the IMU factor's information (default 1.0)",
      [](O & o, S v) {setIfPositive(o.ep.visual.imu_prior_weight, nonnegativeNumber(v));}},
    {"--imu-noise-x", "K", "multiply the datasheet IMU noise densities",
      [](O & o, S v) {o.imu_noise_x = nonnegativeNumber(v);}},
    {"--sigma-bg", "X", "the bootstrap's prior std on the gyro bias, rad/s (default 2e-3)",
      [](O & o, S v) {o.ep.sigma_gyro_bias = nonnegativeNumber(v);}},

    // --- Stage A, the keyframe window
    {"--no-window", nullptr, "the per-frame tracker alone, as before Stage A",
      [](O & o, S) {o.ep.window.enabled = false;}},
    {"--anchor-gauge", nullptr, "VINS's anchor (position + yaw pinned) instead of ORB-SLAM3's",
      [](O & o, S) {o.ep.window.anchor_full = false;}},
    {"--kf-every", "N", "tracked frames between keyframes (default 4)",
      [](O & o, S v) {setIfPositive(o.ep.window.keyframe_every, nonnegativeInteger(v));}},
    {"--window", "K", "keyframes in the window (default 10)",
      [](O & o, S v) {setIfPositive(o.ep.window.max_keyframes, nonnegativeInteger(v));}},
    {"--window-tri", nullptr, "the window also triangulates new landmarks (measured worse)",
      [](O & o, S) {o.ep.window.triangulate = true;}},
    {"--no-map-tri", nullptr, "the map does not triangulate; new landmarks from the window only",
      [](O & o, S) {o.ep.map.triangulate = false;}},
    {"--gravity", nullptr, "re-estimate gravity's direction in the window (no position gain)",
      [](O & o, S) {o.ep.window.estimate_gravity = true;}},
    {"--gravity-sigma", "DEG", "the window's prior on gravity's direction (default 1; 0 = none)",
      [](O & o, S v) {o.ep.window.gravity_sigma_deg = nonnegativeNumber(v);}},
    {"--window-outlier-px", "PX", "the window drops views more than PX off before solving",
      [](O & o, S v) {o.ep.window.outlier_px = nonnegativeNumber(v);}},
    {"--no-forget", nullptr, "the window keeps its views of a landmark the map dropped",
      [](O & o, S) {o.ep.window.forget_outliers = false;}},

    // --- the tracker solve and the map
    {"--inlier-fraction", "F", "refuse a tracker solve below this inlier fraction (default 0.5)",
      [](O & o, S v) {o.ep.visual.min_inlier_fraction = nonnegativeNumber(v);}},
    {"--no-refused-insert", nullptr, "while coasting, do not insert a refused solve's frame",
      [](O & o, S) {o.ep.coast_insert_refused = false;}},
    {"--coast-on-pnp", nullptr, "a coast adopts the PnP seed instead of the IMU prediction",
      [](O & o, S) {o.ep.coast_on_pnp = true;}},
    {"--no-recycle", nullptr, "a track dropped as an outlier is never triangulated again",
      [](O & o, S) {o.ep.map.recycle_outliers = false;}},

    // --- the bootstrap
    {"--init-log", nullptr, "one line per bootstrap attempt: the stage that refused, and why",
      [](O & o, S) {o.init_log = true;}},
    {"--sfm-window", "N", "frames the bootstrap reconstructs and aligns over (default 20)",
      [](O & o, S v) {setIfPositive(o.ep.init.window_frames, nonnegativeNumber(v));}},
    {"--no-sfm-ba", nullptr, "skip the bundle adjustment of the bootstrap reconstruction",
      [](O & o, S) {o.ep.init.sfm_bundle_adjust = false;}},
    {"--sfm-ba-free-pair", nullptr, "that adjustment fixes only the base frame",
      [](O & o, S) {o.ep.init.sfm_ba_fix_pair = false;}},
    {"--sfm-tri", nullptr, "also triangulate the rest of the window's tracks (measured worse)",
      [](O & o, S) {o.ep.init.sfm.triangulate_window = true;}},
    {"--refine-gravity", nullptr, "fix |g| and re-solve gravity's direction (VINS-Fusion)",
      [](O & o, S) {o.ep.init.refine_gravity = true;}},
    {"--align-stride", "N", "the alignment uses every Nth posed frame",
      [](O & o, S v) {setIfPositive(o.ep.init.align_stride, nonnegativeInteger(v));}},
    {"--max-scale-unc", "X", "the bootstrap's scale gate, sigma_s/s (default 0.06)",
      [](O & o, S v) {o.ep.init.max_scale_uncertainty = nonnegativeNumber(v);}},
    {"--visual-init", "S", "track vision-only after a refused alignment, realign from S s on",
      [](O & o, S v) {o.ep.visual_init_seconds = nonnegativeNumber(v);}},

    // --- the front end
    {"--fast", "N", "the tracker's FAST corner threshold (default 20)",
      [](O & o, S v) {o.data.fast_threshold = static_cast<int>(nonnegativeNumber(v));}},
    {"--top-up", "N", "refill to N live tracks on every frame (0 = only below 150)",
      [](O & o, S v) {o.data.top_up_target = nonnegativeInteger(v);}},
    {"--min-spacing", "PX", "minimum distance between a new corner and a live track (15)",
      [](O & o, S v) {o.data.min_spacing_px = static_cast<float>(nonnegativeNumber(v));}},
    {"--flow-back", "PX", "keep a track only if flowing it back lands within PX (0 = off)",
      [](O & o, S v) {o.data.flow_back_px = nonnegativeNumber(v);}},
    {"--flow-back-keep", "N", "skip that check on a frame where fewer than N tracks pass it",
      [](O & o, S v) {o.data.flow_back_min_keep = nonnegativeInteger(v);}},
  };
  return table;
}

void printHelp()
{
  std::printf(
    "Usage: estimator_check [bag] [config] [out.csv] [options]\n"
    "Experiments are described in doc/08-sliding-window.md.\n");
  for (const Flag & f : flags()) {
    const std::string left = std::string(f.name) + (f.value ? std::string("=") + f.value : "");
    std::printf("  %-28s %s\n", left.c_str(), f.help);
  }
  std::printf(
    "Exit codes: 0 quality pass (or report-only), 1 quality failure, 2 input/I/O error.\n");
}

/// False when --help was asked for and printed. Throws std::invalid_argument on a bad one.
bool parseArgs(int argc, char ** argv, Options & o)
{
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--help") {
      printHelp();
      return false;
    }
    if (a.empty() || a.front() != '-') {
      o.pos.push_back(a);
      continue;
    }
    bool known = false;
    for (const Flag & f : flags()) {
      const std::string name = f.name;
      if (f.value == nullptr ? a == name : a.rfind(name + "=", 0) == 0) {
        f.set(o, f.value == nullptr ? std::string() : a.substr(name.size() + 1));
        known = true;
        break;
      }
    }
    if (!known) {
      throw std::invalid_argument("unknown option: " + a);
    }
  }
  if (o.pos.size() > 3) {
    throw std::invalid_argument("expected at most [bag] [config] [out.csv]");
  }
  if (o.gt_path.empty()) {
    if (!o.pos.empty() && o.pos[0] != kDefaultBag) {
      throw std::invalid_argument("a nondefault bag requires --gt=PATH");
    }
    o.gt_path = kDefaultGt;
  }
  return true;
}

}  // namespace

int main(int argc, char ** argv)
{
  Options o;
  try {
    if (!parseArgs(argc, argv, o)) {
      return 0;
    }
  } catch (const std::exception & e) {
    std::fprintf(stderr, "invalid arguments: %s\n", e.what());
    return 2;
  }
  const std::string bag_path = o.pos.size() > 0 ? o.pos[0] : kDefaultBag;
  const std::string calib_dir = o.pos.size() > 1 ? o.pos[1] : "config";
  const std::string csv_path = o.pos.size() > 2 ? o.pos[2] : "/tmp/glassvio_run.csv";
  glassvio::EstimatorParams & ep = o.ep;
  const std::string & mode = o.mode;
  const std::string & oracle_sfm = o.oracle_sfm;
  const glassvio::EstimatorRegressionLimits & limits = o.limits;
  const bool oracle_map = o.oracle_map;
  const bool init_log = o.init_log;
  const bool report_only = o.report_only;

  glassvio::CameraCalib calib;
  glassvio::EurocDataset bag;
  try {
    calib = glassvio::loadEurocCalib(calib_dir);
    o.data.track_images = true;
    bag = glassvio::EurocDataset::load(bag_path, o.gt_path, calib, o.data);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
  if (o.imu_noise_x > 0.0) {
    calib.gyro_noise *= o.imu_noise_x;
    calib.accel_noise *= o.imu_noise_x;
    std::printf(
      "IMU noise densities x%.1f: gyro %.3g rad/s/rtHz, accel %.3g m/s^2/rtHz\n",
      o.imu_noise_x, calib.gyro_noise, calib.accel_noise);
  }
  if (bag.frames.size() < 100 || bag.imu.empty() || bag.gt.empty()) {
    std::fprintf(stderr, "need image, imu and ground-truth streams\n");
    return 2;
  }
  std::printf(
    "deterministic drive: %zu frames, %zu imu, %zu gt\n",
    bag.frames.size(), bag.imu.size(), bag.gt.size());

  // What needs the loaded bag: the oracles read the ground truth.
  if (mode != "est-ba") {
    ep.init.estimate_accel_bias = false;
  }
  if (mode == "oracle-ba") {
    // EuRoC's b_a is near-constant over the sequence; take it where the bootstrap will fire.
    const std::size_t kb = std::min<std::size_t>(ep.bootstrap_frames, bag.frames.size() - 1);
    ep.init.accel_bias = bag.gt.accelBias(bag.frames[kb].t);
  }
  if (o.oracle_bg) {
    ep.init.oracle_gyro_bias = [&](double t, Eigen::Vector3d & bg) {bg = bag.gt.gyroBias(t);};
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
        "tri +%d sfm_ba %.2f->%.2f px |g| err %5.2f%% s %.4f sigma_s/s %.3f ba_std %.2f  %s\n",
        f.t - t0, ir.sfm.max_shared, ir.sfm.candidates_tried, ir.sfm.max_trial_landmarks,
        ir.sfm.landmark.size(), ir.bias_pairs, ir.align_intervals, ir.sfm.window_triangulated,
        ir.sfm_ba.median_px_before,
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
