// PHASE 2b-2: up-to-scale structure from motion over a window.
//
// WHAT "UP TO SCALE" MEANS, since it is the whole point. A monocular camera cannot know
// scale -- scaling the world and the baseline together produces pixel-identical images. So
// we do not recover a scale here: we INVENT one. The base pair's essential matrix returns a
// unit translation, we declare that baseline to be exactly 1, and everything downstream is
// measured against that arbitrary ruler. The reconstruction that comes out is rigid and
// internally correct, in a world uniformly stretched by one unknown factor s. Phase 2b-3
// asks the IMU -- the only component with physical units -- what s is in metres.
//
// PnP IS WHAT CARRIES THE SCALE. Chaining essential matrices cannot work: each pair returns
// its own |t| = 1, and gluing rulers of unknown relative length gives nothing. PnP instead
// consumes 3D landmarks that ALREADY carry the ruler and returns a pose in those same units.
// Shared landmarks are the thread that stitches the window into one consistent scale.
//
// TWO HAZARDS, both measured on this bag rather than assumed:
//
//   * The base pair must be chosen on PARALLAX, not taken as whatever comes next. Where the
//     baseline collapsed to 0.11 m, epipolar error rose 6x (see epipolar_check). No parallax
//     means no triangulation, and a near-degenerate E on top.
//   * KITTI drives FORWARD, so features near the focus of expansion barely move and their
//     depth is ill-conditioned. Hence landmarks are gated on the PARALLAX ANGLE between
//     their two viewing rays -- not on depth, which is the symptom rather than the cause.
//
// The check: fit the single scale s that best maps this reconstruction onto ground truth,
// then report what is left. A good up-to-scale SfM leaves almost nothing -- and the s it
// reports is exactly the number 2b-3 has to recover from the accelerometer.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include "glassvio/dataset.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/sfm_window.hpp"

/// EuRoC V1_01_easy: the ROS2 bag for sensors, the ASL CSV for ground truth. The CSV is not
/// in the bag -- it is the dataset's own batch solution over VICON and IMU, and it states
/// velocity and both biases outright, which the /vicon topic does not.
static const char * kDefaultBag = "data/vicon_room1/V1_01_easy/V1_01_easy_ros2";
static const char * kDefaultGt = "data/vicon_room1/V1_01_easy/gt/data.csv";

namespace
{

int g_window = 20;             ///< frames in the reconstruction
int g_window_start = 40;       ///< WHERE matters: a stationary MAV has no parallax
constexpr double kMaxRelErrPct = 5.0;        ///< the gate: residual after fitting s

}  // namespace

int main(int argc, char ** argv)
{
  const std::string bag_path = (argc > 1) ? argv[1] : kDefaultBag;
  const std::string calib_dir = (argc > 2) ? argv[2] : "config";
  if (argc > 3) {g_window_start = std::atoi(argv[3]);}
  if (argc > 4) {g_window = std::atoi(argv[4]);}

  glassvio::CameraCalib calib;
  try {
    calib = glassvio::loadEurocCalib(calib_dir);
  } catch (const std::exception & e) {
    std::fprintf(stderr, "calib: %s\n", e.what());
    return 2;
  }

  glassvio::EurocDataset bag;
  try {
    bag = glassvio::EurocDataset::load(bag_path, kDefaultGt, calib,
        {true, g_window_start + g_window});
  } catch (const std::exception & e) {
    std::fprintf(stderr, "cannot read bag: %s\n", e.what());
    return 2;
  }
  if (static_cast<int>(bag.frames.size()) < g_window_start + g_window || bag.gt.empty()) {
    std::fprintf(stderr, "not enough frames\n");
    return 2;
  }

  // The reconstruction itself now lives in glassvio/sfm_window.hpp -- vi_align_check needs
  // exactly the same thing, and two copies of an SfM would be two copies to get wrong.
  const glassvio::SfmWindow sfm =
    glassvio::buildSfmWindow(bag.frames, g_window_start, g_window_start + g_window, calib);
  if (!sfm.valid) {
    std::fprintf(stderr, "SfM failed: no parallax, too few landmarks, or too few poses\n");
    return 2;
  }
  const auto & pose = sfm.pose;
  std::printf(
    "base pair: frames %d -> %d   median parallax %.1f px\n"
    "  recoverPose: %d inliers, |t| = 1 by construction (the ruler)\n"
    "  triangulated %zu landmarks (%d rejected: low parallax, %d rejected: cheirality)\n"
    "  %zu poses recovered by PnP propagating that ruler\n",
    sfm.base, sfm.second, sfm.base_parallax_px, sfm.inliers,
    sfm.landmark.size(), sfm.rejected_parallax, sfm.rejected_cheirality, pose.size());

  // --- 5. Score: fit the ONE scale that maps this onto ground truth, report the remainder.
  std::vector<int> ks;
  for (const auto & [k, T] : pose) {
    ks.push_back(k);
  }
  std::sort(ks.begin(), ks.end());

    // Ground-truth CAMERA pose = the OXTS body pose composed with the extrinsic.
  const auto gt_cam = [&](int k) -> Eigen::Isometry3d {
      return bag.gt.at(bag.frames[k].t) * calib.T_cam_imu.inverse();
    };
  const Eigen::Isometry3d gt0 = gt_cam(g_window_start);
  std::vector<Eigen::Vector3d> p_sfm, p_gt;
  double rot_err = 0.0;
  for (int k : ks) {
    // camera k's position, expressed in the base camera's frame
    const Eigen::Isometry3d T_c0_ck = pose.at(k).inverse();
    p_sfm.push_back(T_c0_ck.translation());
    const Eigen::Isometry3d G = gt0.inverse() * gt_cam(k);
    p_gt.push_back(G.translation());
    const Eigen::Matrix3d dR = T_c0_ck.linear().transpose() * G.linear();
    rot_err = std::max(rot_err, Eigen::AngleAxisd(dR).angle() * 180.0 / M_PI);
  }

  // s = argmin sum || s*p_sfm - p_gt ||^2  -- one unknown, closed form.
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < p_sfm.size(); ++i) {
    num += p_sfm[i].dot(p_gt[i]);
    den += p_sfm[i].squaredNorm();
  }
  if (den < 1e-12) {std::fprintf(stderr, "degenerate reconstruction\n"); return 2;}
  const double s = num / den;

  std::vector<double> resid;
  double extent = 0.0;
  for (std::size_t i = 0; i < p_sfm.size(); ++i) {
    resid.push_back((s * p_sfm[i] - p_gt[i]).norm());
    extent = std::max(extent, p_gt[i].norm());
  }
  std::sort(resid.begin(), resid.end());
  const double med = resid[resid.size() / 2];
  const double rel = 100.0 * med / std::max(extent, 1e-9);

  std::printf(
    "\nfitted scale s = %.4f m per ruler unit\n"
    "  (i.e. the base pair's invented |t|=1 was really %.3f m of travel)\n"
    "  <- THIS is the number phase 2b-3 must recover from the accelerometer alone\n",
    s, s);
  std::printf(
    "\nafter fitting that single scale:\n"
    "  median position residual: %.4f m over a %.2f m window  (%.2f%%)\n"
    "  worst rotation error:     %.3f deg\n"
    "  gate: < %.1f%%\n",
    med, extent, rel, rot_err, kMaxRelErrPct);

  if (rel > kMaxRelErrPct) {
    std::printf("\nFAIL: the reconstruction is not rigid -- one scale does not explain it\n");
    return 1;
  }
  std::printf(
    "\nok: one scale factor explains the whole window, so the structure is rigid and\n"
    "    correct up to that factor. Vision has done all it can; the metre must come from\n"
    "    the IMU.\n");
  return 0;
}
