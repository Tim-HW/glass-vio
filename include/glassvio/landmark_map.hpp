#ifndef GLASSVIO_LANDMARK_MAP_HPP
#define GLASSVIO_LANDMARK_MAP_HPP

#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>

#include "glassvio/camera_calib.hpp"
#include "glassvio/feature_tracker.hpp"

namespace glassvio
{

struct LandmarkMapParams
{
  /// Ray angle below which a triangulated depth is noise. 1.0 is what ORB-SLAM3 hardcodes at
  /// its Reconstruct() call site, and what stage [2] uses. Not a knob to loosen when the
  /// baseline is small -- a small baseline means WAIT, not guess.
  double min_parallax_deg = 1.0;
  /// m. Cheirality guard: a landmark behind the camera projects to an ordinary-looking pixel
  /// with a sign-flipped Jacobian, so it must be refused rather than fitted.
  double min_depth = 0.1;
  double max_depth = 40.0;      ///< m; an indoor room has no 40 m walls, so this is noise
  /// Observations a pending track keeps. Only the first and last are triangulated (the widest
  /// baseline available), so this bounds memory, not accuracy.
  std::size_t max_observations = 12;
  /// Frames a landmark may go unseen before it is dropped. The camera has flown past it; a
  /// landmark that cannot be observed cannot constrain anything, and keeping it only grows
  /// the map.
  int max_unseen_frames = 8;
  /// Reprojection error (px) above which an existing landmark is dropped as an outlier. A KLT
  /// mismatch that survives Huber still poisons the map if it stays.
  double max_reprojection_px = 8.0;
};

/// THE LOCAL MAP, for landmarks -- glasslio's LocalMap, with points instead of planes.
///
/// WHY THIS IS THE SLIDING WINDOW, and why it is NOT bundle adjustment. glasslio does not put
/// its map in the state: it maintains a voxel hash of planes, inserts each new scan, and
/// registers against it with the map held FIXED. The state stays 6/15 DoF and the solver never
/// grows. This is the same trade, for a camera:
///
///   * landmarks are triangulated as tracks mature, and held fixed during a solve;
///   * they are dropped when they leave view or stop fitting;
///   * NormalEquationsN<15> is reused untouched.
///
/// Full bundle adjustment would put the landmarks in the state -- H grows to 15N + 3L, dense
/// LDLT dies, and the Schur complement becomes mandatory. That is a real thing to want; it is
/// not what fixes starvation, and starvation is what we measured (~4 s, then 0 landmarks in
/// view).
///
/// THE RULER IS ALREADY DEAD HERE. Stage [2] had to invent a baseline because a monocular
/// camera has no units. Once the bootstrap runs, the state is METRIC -- so new landmarks
/// triangulate from metric poses and come out in metres directly. No scale, no alignment, no
/// gauge. The hard part happened once, at the start, and never again.
class LandmarkMap
{
public:
  explicit LandmarkMap(const CameraCalib & calib, const LandmarkMapParams & params = {})
  : calib_(calib), p_(params) {}

  /// Take the bootstrap's landmarks as the starting map.
  void seed(const std::unordered_map<long, Eigen::Vector3d> & landmarks);

  /// Fold in one frame: match, prune, and triangulate whatever has matured.
  ///
  /// `T_world_cam` must be METRIC -- it is the estimator's own state composed with the
  /// extrinsic. Feeding it a pose from before the bootstrap would triangulate in ruler units
  /// and poison the map with points that are geometrically fine and physically meaningless.
  void insert(const FeatureTracker::Result & features, const Eigen::Isometry3d & T_world_cam);

  /// Landmarks the reprojection factor may use: metric, world frame, FIXED during a solve.
  const std::unordered_map<long, Eigen::Vector3d> & landmarks() const {return landmarks_;}

  void clear();

  std::size_t size() const {return landmarks_.size();}
  std::size_t pending() const {return pending_.size();}
  /// Landmarks triangulated on the last insert(). The signal that the map is keeping up:
  /// if this is persistently zero while size() falls, the solve is about to starve.
  int lastTriangulated() const {return last_triangulated_;}
  int lastDropped() const {return last_dropped_;}

private:
  struct Observation
  {
    Eigen::Isometry3d T_world_cam;
    cv::Point2f px;
  };

  /// Two views of one track, in METRIC world poses -> a metric landmark. False if the parallax
  /// is too thin, the point is behind either camera, or the depth is absurd.
  bool triangulate(
    const Observation & a, const Observation & b, Eigen::Vector3d & out) const;

  CameraCalib calib_;
  LandmarkMapParams p_;

  std::unordered_map<long, Eigen::Vector3d> landmarks_;
  std::unordered_map<long, std::vector<Observation>> pending_;
  std::unordered_map<long, int> last_seen_;
  int frame_ = 0;
  int last_triangulated_ = 0;
  int last_dropped_ = 0;
};

}  // namespace glassvio

#endif  // GLASSVIO_LANDMARK_MAP_HPP
