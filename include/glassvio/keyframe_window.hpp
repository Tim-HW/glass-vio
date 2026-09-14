#ifndef GLASSVIO_KEYFRAME_WINDOW_HPP
#define GLASSVIO_KEYFRAME_WINDOW_HPP

#include <deque>
#include <unordered_map>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include "glass_core/nav_residual.hpp"
#include "glass_core/nav_state.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/dataset.hpp"
#include "glassvio/reprojection.hpp"

namespace glassvio
{

using namespace glass_core;  // NOLINT(build/namespaces)

/// STAGE A of doc/08-sliding-window.md: a fixed-lag window of keyframes, solved JOINTLY with the
/// landmarks they see.
///
/// WHY. The per-frame tracker holds landmarks fixed and triangulates new ones from its own
/// solved poses, so any pose error is written into the map and read back as ground truth: the
/// loop that shrinks the scale (measured: pure vision contracts 0.86 -> 0.55 in 20 s). The IMU
/// could resist it, but in the tracker it only ever touches the pose. Here the landmarks are
/// STATE, so the IMU factors between keyframes pull the whole structure -- poses and points -- to
/// the accelerometer's metre.
///
/// THE DESIGN, borrowed rather than invented:
///   * ORB-SLAM3's LocalInertialBA for the shape: ~10 keyframes on a temporal chain, landmarks as
///     XYZ points eliminated by the Schur complement, per-keyframe biases tied by random-walk
///     factors, no marginalization (the oldest is simply dropped).
///   * The anchor, both ways: ORB-SLAM3 fixes the whole oldest state; VINS-Fusion pins only the
///     four gauge DoF (position, yaw) and leaves roll and pitch free to RE-LEVEL against
///     gravity. The second should matter here -- gravity_world_ is frozen at the bootstrap's
///     estimate (doc/08 §6) -- but measured, the first wins, so it is the default.
struct WindowParams
{
  bool enabled = true;
  int keyframe_every = 4;   ///< tracked frames between keyframes (0.2 s at EuRoC's 20 Hz)
  int max_keyframes = 10;   ///< optimized keyframes; the anchor sits in front of them
  int iterations = 8;       ///< Levenberg-Marquardt steps per solve (VINS-Fusion uses 8)
  /// true:  ORB-SLAM3-style -- the whole anchor state is fixed.
  /// false: VINS-style -- pin the anchor's position and yaw, soft prior on the rest, so the
  ///        window may re-level against gravity.
  /// MEASURED, not argued: on EuRoC V1_01 the fixed anchor won -- 80 s tracked at 0.46 m
  /// median against 76 s at 0.68 m -- although only the soft one can re-level.
  /// (estimator_check --anchor-gauge to compare.)
  bool anchor_full = true;
  /// The soft half of the anchor: what the previous windows already knew about it.
  double anchor_sigma_tilt_rad = 2.0 * M_PI / 180.0;
  double anchor_sigma_velocity = 0.10;
  double anchor_sigma_gyro_bias = 2.0e-3;
  double anchor_sigma_accel_bias = 0.2;

  /// Triangulate new landmarks from the window's OPTIMIZED keyframe poses (triangulate()), as
  /// VINS-Fusion and ORB-SLAM3 do. OFF, because it was measured and lost. The hypothesis was that
  /// the map's pending tracks never mature because their stored tracker poses are stale -- but
  /// with GROUND-TRUTH poses the map matures about as many (doc/08 §6): supply was never short.
  /// Turning this on pulled the first >1 m error in from 89 s to 40 s. Kept, pinned and
  /// switchable (estimator_check --window-tri) for when the poses it trusts beat the map's.
  bool triangulate = false;
  double tri_min_parallax_deg = 1.0;   ///< the map's gate: a small baseline means WAIT
  double tri_max_reproj_px = 3.0;      ///< in EVERY observing keyframe (VINS-Fusion's bound)
  double tri_max_depth = 40.0;         ///< m; an indoor room has no 40 m walls

  /// Re-estimate gravity's DIRECTION in the window -- two more unknowns; |g| is known. The
  /// bootstrap fixes gravity once, and gravity_world_ used to keep it forever: from the b_a = 0
  /// bootstrap that is a 4 deg tilt, a permanent 0.69 m/s^2 error the accel bias chased without
  /// ever converging (doc/08 §6, Step 0). With the anchor fixed, the tilt is observable against
  /// it -- the quantity ORB-SLAM3's inertial initialization refines as Rwg.
  /// OFF, because measured it halves the tilt (4.0 -> 1.8-2.7 deg while tracking) but moves
  /// neither the position error nor the first >1 m error (87.9 s vs 89.0 s). ORB-SLAM3 instead
  /// re-levels the WHOLE map from a global inertial optimization at 2, 5 and 15 s. Kept, pinned
  /// and switchable (estimator_check --gravity).
  bool estimate_gravity = false;
  /// Soft prior toward the direction the previous window left, degrees; 0 = none. glass-lio's
  /// divergence was exactly such a prior with nothing to restore it (doc/08 §3). Here every IMU
  /// factor in the window is a restoring force -- and the value is measured, not argued.
  double gravity_sigma_deg = 1.0;
};

struct Keyframe
{
  double t = 0.0;
  NavState x;
  std::unordered_map<long, cv::Point2f> obs;   ///< undistorted pixels, by track id
};

struct WindowResult
{
  bool ok = false;
  int landmarks = 0;
  int iterations = 0;
  double cost_before = 0.0;
  double cost_after = 0.0;
  /// cost_before split by factor -- the anchor (and gravity) prior, the IMU with its bias random
  /// walk, and vision. A spike in one says which input the window was handed wrong.
  double cost_before_prior = 0.0;
  double cost_before_imu = 0.0;
  double cost_before_vis = 0.0;
  /// The single worst IMU factor as the window was handed it: which interval (0 = anchor to the
  /// next, keyframes - 2 = the newest), how long, and which residual carries the disagreement.
  int keyframes = 0;
  int worst_imu_k = -1;
  double worst_imu_cost = 0.0;
  double worst_imu_dt = 0.0;
  double worst_imu_rot_deg = 0.0;
  double worst_imu_vel = 0.0;   ///< m/s
  double worst_imu_pos = 0.0;   ///< m
  /// How far the solve moved the NEWEST keyframe, m. The tracker resumes from it, so a large
  /// value is a jump the tracker has to survive -- the first thing to look at when it doesn't.
  double newest_shift_m = 0.0;
  /// New landmarks triangulate() produced after this solve (filled in by the caller).
  int triangulated = 0;
  /// Gravity, world frame, as this solve left it -- re-estimated when estimate_gravity is on.
  Eigen::Vector3d gravity = Eigen::Vector3d::Zero();
  /// The re-estimated landmarks, world frame, metres. The map adopts them.
  std::unordered_map<long, Eigen::Vector3d> refined;
};

/// One landmark's slice of the bundle-adjustment normal equations: its own 3x3 block, its part
/// of b, and its coupling H_kl (15x3) to every keyframe that sees it.
struct LandmarkBlock
{
  Eigen::Matrix3d H_ll = Eigen::Matrix3d::Zero();
  Eigen::Vector3d b_l = Eigen::Vector3d::Zero();
  std::vector<std::pair<int, Eigen::Matrix<double, kNavDim, 3>>> H_kl;
};

/// Solve  [H_pp H_pl ; H_lp H_ll] [dp ; dl] = [b_p ; b_l]  by eliminating the landmarks.
///
/// H_ll is block-DIAGONAL -- two landmarks never share a residual -- so each 3x3 block inverts
/// on its own, the reduced keyframe system S = H_pp - H_pl H_ll^-1 H_lp is only 15K wide, and
/// the landmarks come back by substitution. glass_core's schurMarginalize is the same algebra
/// for a DENSE marginalized block; test_keyframe_window.cpp pins this one against it and
/// against a plain dense solve.
///
/// `lambda` is Levenberg-Marquardt damping, scaled by each diagonal. A landmark whose H_ll is
/// rank-deficient (seen from one direction: no depth) is skipped -- it gets dl = 0 and adds
/// nothing to S. `S_out`, if given, receives the reduced matrix.
bool schurSolve(
  const Eigen::MatrixXd & H_pp, const Eigen::VectorXd & b_p,
  const std::vector<LandmarkBlock> & landmarks, double lambda,
  Eigen::VectorXd & dp, std::vector<Eigen::Vector3d> & dl, Eigen::MatrixXd * S_out = nullptr);

/// Gravity tilted by th, a world-frame rotation about x and y: g(th) = Exp([th_x, th_y, 0]) g_ref.
/// Two DoF with the magnitude preserved -- the window's parametrization of gravity's direction.
inline Eigen::Vector3d tiltGravity(const Eigen::Vector3d & g_ref, const Eigen::Vector2d & th)
{
  return Sophus::SO3d::exp(Eigen::Vector3d(th.x(), th.y(), 0.0)) * g_ref;
}

/// d(imuResidual) / d(th) around `g`: glass_core's imuGravityJacobian (d r / d g, pinned there)
/// times d g / d th = -[g]x [e_x e_y]. Exact at th = 0; the window re-evaluates it at the current
/// g each iteration, first-order in the tilt, which is a few degrees. Pinned against finite
/// differences through tiltGravity in test_keyframe_window.cpp.
inline Eigen::Matrix<double, 9, 2> imuGravityDirectionJacobian(
  const NavState & xi, const ImuPreintegration & pre, const Eigen::Vector3d & g)
{
  Eigen::Matrix<double, 3, 2> B = Eigen::Matrix<double, 3, 2>::Zero();
  B(0, 0) = 1.0;
  B(1, 1) = 1.0;
  return imuGravityJacobian(xi, pre) * (-Sophus::SO3d::hat(g) * B);
}

class KeyframeWindow
{
public:
  explicit KeyframeWindow(const WindowParams & params = {})
  : p_(params) {}

  /// Append the newest keyframe; drop the oldest once there are more than max_keyframes plus
  /// the anchor. Dropped outright, as ORB-SLAM3 does -- no marginalization prior (Stage B).
  void push(Keyframe kf);
  void clear() {kfs_.clear();}
  const std::deque<Keyframe> & keyframes() const {return kfs_;}

  /// Jointly re-estimate every keyframe state and every landmark two or more keyframes see.
  /// `landmarks` supplies the starting points; the result carries the refined ones. False
  /// (result.ok) when there is nothing to solve or the IMU has a hole inside the window.
  WindowResult optimize(
    const std::unordered_map<long, Eigen::Vector3d> & landmarks, const ImuBuffer & imu,
    const CameraCalib & calib, const ReprojectionParams & reproj, const Eigen::Vector3d & gravity);

  /// New landmarks from the keyframes' CURRENT poses -- call after optimize(). Every track the
  /// newest keyframe sees (so it is alive) that is not in `existing` and that two or more
  /// keyframes observe, triangulated linearly over ALL its observations. Kept only if its widest
  /// pair of rays clears tri_min_parallax_deg, it lies in front of every observing camera, and
  /// it reprojects within tri_max_reproj_px in each -- one mismatched view rejects the track.
  std::unordered_map<long, Eigen::Vector3d> triangulate(
    const std::unordered_map<long, Eigen::Vector3d> & existing, const CameraCalib & calib,
    double min_depth) const;

private:
  WindowParams p_;
  std::deque<Keyframe> kfs_;
};

}  // namespace glassvio

#endif  // GLASSVIO_KEYFRAME_WINDOW_HPP
