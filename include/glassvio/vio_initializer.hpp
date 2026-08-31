#ifndef GLASSVIO_VIO_INITIALIZER_HPP
#define GLASSVIO_VIO_INITIALIZER_HPP

#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "glass_core/imu_init.hpp"   // kGravity
#include "glass_core/nav_state.hpp"
#include "glass_core/preintegration.hpp"
#include "glassvio/dataset.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/sfm_window.hpp"

namespace glassvio
{

// The estimation math lives in glass_core. glassvio is the VISUAL front-end built on top of
// it -- pull the engine's names into this namespace, exactly as glasslio does.
using namespace glass_core;  // NOLINT(build/namespaces)

/// Everything the bootstrap needs to be built. Parsed from ROS parameters by the node --
/// which is the node's ONLY remaining say in how initialisation behaves.
struct InitializerParams
{
  /// [2] Up-to-scale reconstruction.
  SfmParams sfm;
  int window_frames = 20;

  /// [3] Gyro bias. Frames apart for each essential-matrix pair: the bias signal grows with
  /// the interval while the vision rotation error does not, so wider is better -- bounded by
  /// how long a track survives.
  int bias_frame_gap = 5;
  int bias_min_pairs = 8;
  /// Frames the BIAS may look at, which is deliberately NOT `window_frames`. The bias is a
  /// constant of the sensor, so every pair in the stream is evidence and averaging is what
  /// pulls the estimate below the vision noise floor of any single pair (~0.11 deg). The SfM
  /// window, by contrast, must stay short or its landmarks die. Tying the two together
  /// starves the bias solve: a 20-frame window yields 3 pairs.
  /// 0 = every frame from `begin` to the end of the stream.
  int bias_window_frames = 0;

  /// [4] Visual-inertial alignment.
  //
  // NO gyro_noise/accel_noise HERE. They are properties of the SENSOR, not of this stage's
  // tuning, and they live in CameraCalib -- which reads them from the dataset's own Kalibr
  // file rather than guessing (EuRoC: 1.6968e-4 and 2.0e-3). Having them in both places means
  // having two answers to one question, and the wrong one silently winning: these defaulted
  // to 1e-3/1e-2, the numbers guessed for KITTI before any datasheet was available.
  int align_min_intervals = 4;
  /// |g| is never constrained in the solve, so how far it lands from 9.80665 is a direct
  /// test of the formulation, the frames, and the extrinsic. Above this, refuse.
  double max_gravity_error_pct = 5.0;
  /// The SUFFICIENT scale gate: reject a window whose scale is uncertain by more than this
  /// fraction (sigma_s / |s|). This is what turns the standing KITTI lesson -- scale needs
  /// excitation -- into a real gate, and it is what lets the estimator retry past a
  /// poorly-excited window instead of bootstrapping at a metrically-wrong scale. Measured on
  /// EuRoC V1_01: 0.02 gave 2%-accurate scale, 0.10 gave 32% off, 0.2+ degenerate. 0.15 let a
  /// 3x-biased-scale window through (est velocity 30% of truth, drift); 0.06 admits only
  /// genuinely well-excited windows.
  double max_scale_uncertainty = 0.06;
};

/// What the bootstrap produced. Everything spatial is in the SFM FRAME -- the base camera --
/// because that is the only frame the reconstruction has. The caller composes it into world.
struct InitResult
{
  /// Every stage ran and produced something. NOT a trust signal on its own: see
  /// `scale_observable`.
  bool ok = false;

  /// THE ONE THAT MATTERS ON A CAR. Scale reaches the estimator only through the
  /// accelerometer's non-gravity part; a vehicle at constant velocity has almost none, and
  /// then s and v_0 slide against each other along a valley of equal cost. When this is
  /// false the reconstruction, gravity and the gyro bias are still good -- only `scale` and
  /// `velocity_sfm` are meaningless. Measured on KITTI 0117: gravity lands within 0.8%,
  /// scale is off by 100%.
  bool scale_observable = false;

  /// [2]
  SfmWindow sfm;
  /// [3] rad/s, body frame.
  Eigen::Vector3d gyro_bias = Eigen::Vector3d::Zero();
  /// [4] m/s^2, in the SfM frame. Its MAGNITUDE is the oracle -- nothing told the solve.
  Eigen::Vector3d gravity_sfm = Eigen::Vector3d::Zero();
  /// [4] metres per ruler unit: what the base pair's invented |t| = 1 was really worth.
  double scale = 0.0;
  /// Marginal RELATIVE uncertainty of the scale, sigma_s / |s|. Large = scale poorly excited =
  /// s unreliable even when positive. The sufficient half of scale_observable, and
  /// dimensionless so it needs no per-dataset threshold.
  double scale_uncertainty = 0.0;
  /// [4] The frames the alignment solved for, sorted, and their velocities in the SfM frame.
  std::vector<int> frames;
  std::vector<Eigen::Vector3d> velocity_sfm;

  int bias_pairs = 0;
  int align_intervals = 0;
};

/// THE INITIALISER -- stages [2] reconstruct, [3] gyro bias, [4] align.
///
/// This is the monocular bootstrap, and nothing else. It has no subscriptions, no publishers,
/// no parameters and no threads: the caller feeds it tracked frames and an IMU buffer and
/// asks what came out. Same contract as glasslio's LioEstimator, and for the same reason --
/// it is what makes the pipeline DRIVABLE FROM A TEST.
///
/// WHY A BOOTSTRAP EXISTS AT ALL. The steady-state estimator needs a metric NavState to start
/// from, and no single sensor can supply one:
///
///   the camera sees landmarks but has no units;
///   the accelerometer has units but has never seen a landmark.
///
/// So the stages are ordered by what each one needs from the last:
///
///   [2] reconstruct  needs nothing    -> shape, in an INVENTED ruler (|t| = 1)
///   [3] gyro bias    needs rotations  -> rotations are scale-free, so this runs early
///   [4] align        needs [2] + IMU  -> the metre: gravity, scale, velocity
///
/// LOOSE COUPLING LIVES HERE, AND IS CORRECT HERE. glass_core's nav_state.hpp records loose
/// coupling as the origin of the constant-velocity runaway -- true of a steady-state
/// estimator, which has no velocity state and finite-differences it from the very poses it is
/// helping produce. None of that applies to a bootstrap: velocity is an explicit unknown,
/// solved once, from a reconstruction that is not being updated in the loop. Loose to
/// bootstrap, tight to run.
class VioInitializer
{
public:
  VioInitializer(const CameraCalib & calib, const InitializerParams & params);

  /// Run [2] -> [4] over frames [begin, begin + window_frames).
  InitResult run(
    const std::vector<SfmFrame> & frames, const ImuBuffer & imu, int begin) const;

  // --- The stages, public because each is independently verifiable and each has a check
  //     tool that scores exactly one of them against an oracle. That is not incidental:
  //     a bootstrap whose stages could only be run together would be a bootstrap whose
  //     failures could only be guessed at.

  /// [2] Poses and landmarks, up to scale. Knows no metre.
  SfmWindow reconstruct(
    const std::vector<SfmFrame> & frames, int begin) const;

  /// [3] Gyro bias from the essential matrix's rotations. Needs no scale, so it runs before
  /// the metre exists -- and every later stage integrates the gyro, so it runs first.
  bool estimateGyroBias(
    const std::vector<SfmFrame> & frames, const ImuBuffer & imu, int begin,
    Eigen::Vector3d & bias, int & pairs) const;

  /// [4] The metre. Solves gravity, scale and per-frame velocity in ONE linear least
  /// squares -- no iteration, no initial guess, no local minimum.
  bool align(
    const std::vector<SfmFrame> & frames, const ImuBuffer & imu,
    const SfmWindow & sfm, const Eigen::Vector3d & gyro_bias, InitResult & out) const;

  const CameraCalib & calib() const {return calib_;}
  const InitializerParams & params() const {return p_;}

private:
  CameraCalib calib_;
  InitializerParams p_;
};

}  // namespace glassvio

#endif  // GLASSVIO_VIO_INITIALIZER_HPP
