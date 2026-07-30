#include "glassvio/landmark_map.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/calib3d.hpp>

namespace glassvio
{

void LandmarkMap::seed(const std::unordered_map<long, Eigen::Vector3d> & landmarks)
{
  landmarks_ = landmarks;
  for (const auto & lm : landmarks_) {
    last_seen_[lm.first] = frame_;
  }
}

void LandmarkMap::clear()
{
  landmarks_.clear();
  pending_.clear();
  last_seen_.clear();
  frame_ = 0;
  last_triangulated_ = 0;
  last_dropped_ = 0;
}

bool LandmarkMap::triangulate(
  const Observation & a, const Observation & b, Eigen::Vector3d & out) const
{
  // world -> cam, which is what the projection matrices need.
  const Eigen::Isometry3d T_ca_w = a.T_world_cam.inverse();
  const Eigen::Isometry3d T_cb_w = b.T_world_cam.inverse();

  const auto proj = [this](const Eigen::Isometry3d & T) {
      cv::Mat P(3, 4, CV_64F);
      const Eigen::Matrix<double, 3, 4> Rt =
        calib_.K * T.matrix().topRows<3>();
      for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
          P.at<double>(i, j) = Rt(i, j);
        }
      }
      return P;
    };

  std::vector<cv::Point2f> pa{a.px}, pb{b.px};
  cv::Mat X4f;
  cv::triangulatePoints(proj(T_ca_w), proj(T_cb_w), pa, pb, X4f);
  // CV_32F for Point2f input. Reading it as double does not throw -- it reinterprets two
  // floats as one double and yields values that look exactly like points at infinity.
  cv::Mat X4;
  X4f.convertTo(X4, CV_64F);

  const double w = X4.at<double>(3, 0);
  if (std::abs(w) < 1e-12) {
    return false;   // at infinity: no baseline reached it
  }
  const Eigen::Vector3d X(
    X4.at<double>(0, 0) / w, X4.at<double>(1, 0) / w, X4.at<double>(2, 0) / w);

  // Cheirality: in front of BOTH cameras.
  const double za = (T_ca_w * X).z();
  const double zb = (T_cb_w * X).z();
  if (za < p_.min_depth || zb < p_.min_depth || za > p_.max_depth || zb > p_.max_depth) {
    return false;
  }

  // Parallax ANGLE between the viewing rays. This is the test a pixel-flow threshold cannot
  // make: a rotating camera produces plenty of flow with parallel rays and nothing to
  // triangulate. Measured from the two camera CENTRES, so it is a property of the geometry
  // rather than of the image.
  const Eigen::Vector3d ra = (X - a.T_world_cam.translation()).normalized();
  const Eigen::Vector3d rb = (X - b.T_world_cam.translation()).normalized();
  const double ang = std::acos(std::clamp(ra.dot(rb), -1.0, 1.0)) * 180.0 / M_PI;
  if (ang < p_.min_parallax_deg) {
    return false;   // WAIT for a wider baseline; do not guess a depth
  }

  out = X;
  return true;
}

void LandmarkMap::insert(
  const FeatureTracker::Result & features, const Eigen::Isometry3d & T_world_cam)
{
  ++frame_;
  last_triangulated_ = 0;
  last_dropped_ = 0;

  const Eigen::Isometry3d T_cam_w = T_world_cam.inverse();

  for (std::size_t i = 0; i < features.ids.size(); ++i) {
    const long id = features.ids[i];
    const cv::Point2f & px = features.points[i];

    // --- Already a landmark: mark it seen, and check it still FITS.
    const auto known = landmarks_.find(id);
    if (known != landmarks_.end()) {
      last_seen_[id] = frame_;

      // An outlier that survives Huber still poisons the map if it stays -- Huber downweights
      // it in the solve but never removes it, and it is re-offered every frame thereafter.
      const Eigen::Vector3d Pc = T_cam_w * known->second;
      if (Pc.z() > p_.min_depth) {
        const Eigen::Vector2d proj(
          calib_.fx() * Pc.x() / Pc.z() + calib_.cx(),
          calib_.fy() * Pc.y() / Pc.z() + calib_.cy());
        if ((proj - Eigen::Vector2d(px.x, px.y)).norm() > p_.max_reprojection_px) {
          landmarks_.erase(known);
          last_seen_.erase(id);
          ++last_dropped_;
        }
      }
      continue;
    }

    // --- Not a landmark yet: accumulate observations until the baseline is worth using.
    auto & obs = pending_[id];
    obs.push_back({T_world_cam, px});
    last_seen_[id] = frame_;   // pending tracks age too -- see the prune below
    if (obs.size() > p_.max_observations) {
      // Keep the FIRST and the recent ones: the first gives the widest baseline available,
      // and the widest baseline is the whole point.
      obs.erase(obs.begin() + 1);
    }
    if (obs.size() < 2) {
      continue;
    }

    // First against last: the widest baseline this track has ever offered. Triangulating
    // adjacent frames instead would ask for depth from a 5 cm baseline and get noise -- which
    // is exactly the failure the min_parallax_deg gate exists to refuse.
    Eigen::Vector3d X;
    if (triangulate(obs.front(), obs.back(), X)) {
      landmarks_.emplace(id, X);
      last_seen_[id] = frame_;
      pending_.erase(id);
      ++last_triangulated_;
    }
  }

  // --- Prune what the camera has flown past. A landmark that cannot be observed cannot
  // constrain anything; keeping it only grows the map and the loop over it.
  for (auto it = landmarks_.begin(); it != landmarks_.end(); ) {
    const auto seen = last_seen_.find(it->first);
    if (seen == last_seen_.end() || frame_ - seen->second > p_.max_unseen_frames) {
      last_seen_.erase(it->first);
      it = landmarks_.erase(it);
      ++last_dropped_;
    } else {
      ++it;
    }
  }

  // Pending tracks die too, and by LAST SEEN -- a track the tracker dropped will never mature,
  // and its observations would otherwise leak for the length of the run. They are given a
  // longer grace than landmarks: a pending track is waiting for parallax, and the whole point
  // of waiting is that it takes time.
  for (auto it = pending_.begin(); it != pending_.end(); ) {
    const auto seen = last_seen_.find(it->first);
    if (seen == last_seen_.end() || frame_ - seen->second > p_.max_unseen_frames * 4) {
      last_seen_.erase(it->first);
      it = pending_.erase(it);
    } else {
      ++it;
    }
  }
}

}  // namespace glassvio
