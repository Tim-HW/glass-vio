#ifndef GLASSVIO_FEATURE_TRACKER_HPP
#define GLASSVIO_FEATURE_TRACKER_HPP

#include <algorithm>
#include <vector>

#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

namespace glassvio
{

/// FAST + KLT feature tracker. Harvested from super_visual_odometry's front-end, minus the
/// essential-matrix pose step (the IMU-coupled solver replaces that) and minus the Python /
/// LightGlue path. What survives is the genuinely reusable core: detect corners, follow them
/// frame to frame with optical flow, and re-detect to top up when tracks die off.
///
/// EVERY TRACK CARRIES A PERSISTENT ID. That is the whole reason this is a "front-end" and
/// not just a motion estimator: the reprojection factor (next phase) needs to know that the
/// pixel here in frame N is the SAME landmark as the pixel there in frame N+1. Without stable
/// ids there is nothing to triangulate and nothing to reproject. Carrying them costs one
/// parallel vector; omitting them would force a rewrite the moment the back-end lands.
class FeatureTracker
{
public:
  struct Result
  {
    std::vector<cv::Point2f> points;   ///< current pixel location of each live track
    std::vector<long> ids;             ///< persistent landmark id, parallel to points
  };

  /// `max_features` : cap on live tracks (detection stops adding past this)
  /// `min_features` : re-detect to top up once live tracks fall below this
  /// `fast_threshold` : FAST corner-response threshold
  /// `flow_back_px` : keep a track only if flowing it BACK lands within this many pixels of where
  ///                  it started (VINS-Fusion's flow_back, 0.5 px); 0 = off
  /// `top_up_target` : when > 0, re-detect on EVERY frame to refill the live tracks to this count
  ///                   (VINS-Fusion: 150) instead of waiting for them to fall below min_features
  /// `min_spacing_px` : minimum distance between a new corner and any live track
  FeatureTracker(
    int max_features = 1000, int min_features = 150, int fast_threshold = 20,
    double flow_back_px = 0.0, int top_up_target = 0, float min_spacing_px = 15.0f,
    int flow_back_min_keep = 0)
  : max_features_(max_features),
    min_features_(min_features),
    flow_back_px_(flow_back_px),
    top_up_target_(top_up_target),
    min_spacing_px_(min_spacing_px),
    flow_back_min_keep_(flow_back_min_keep),
    detector_(cv::FastFeatureDetector::create(fast_threshold, true)),
    lk_criteria_(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01)
  {
  }

  /// Feed one GRAYSCALE frame. Returns the live tracks (points + ids) in this frame.
  Result track(const cv::Mat & gray)
  {
    if (prev_gray_.empty() || prev_pts_.empty()) {
      // Cold start: nothing to flow. NOT "too few survivors" -- that once came here too, and
      // detectInto APPENDS, so the survivors were handed back at the PREVIOUS frame's pixels
      // under their old ids. Low texture (EuRoC V1_01's mattress wall, 87.5-91 s) kept the
      // count under min_features for 22 frames running: ~140 points each, 6-14 px stale, and
      // every tracker solve refused (doc/08 §6).
      detectInto(gray, prev_pts_, ids_, cap());
    } else {
      std::vector<cv::Point2f> tracked;
      std::vector<long> tracked_ids;
      flow(prev_gray_, gray, prev_pts_, ids_, tracked, tracked_ids);

      // Top up only once the survivors fall below the floor. Re-detecting EVERY frame was
      // tried and MEASURABLY broke structure-from-motion: sfm_check's rigidity residual at
      // EuRoC window 100 went from 2.64% to 76%, and the bootstrap reconstructed the room at
      // ~3 cm instead of metres. The exact mechanism is still open (detectInto appends, so it
      // does not churn ids), but the empirical result is unambiguous, deterministic, and the
      // same tracked frames feed SfM -- so the corruption here is the corruption there.
      //
      // The sliding-window map does need a supply of young tracks, and gating starves it
      // (~13 pending against ~11 landmarks lost per frame). That is a real problem, but the
      // fix is NOT here -- it is a higher floor or a map-driven re-detect that leaves the
      // reconstruction's tracks intact. Supply the map without breaking the geometry it is
      // built on.
      const std::size_t floor = static_cast<std::size_t>(
        top_up_target_ > 0 ? top_up_target_ : min_features_);
      if (tracked.size() < floor) {
        detectInto(gray, tracked, tracked_ids, cap());
      }
      prev_pts_ = std::move(tracked);
      ids_ = std::move(tracked_ids);
    }

    prev_gray_ = gray.clone();
    return {prev_pts_, ids_};
  }

private:
  /// KLT prev -> curr, keeping only tracks that survive with low error and stay in bounds.
  void flow(
    const cv::Mat & prev, const cv::Mat & curr,
    const std::vector<cv::Point2f> & prev_pts, const std::vector<long> & prev_ids,
    std::vector<cv::Point2f> & out_pts, std::vector<long> & out_ids) const
  {
    std::vector<cv::Point2f> next;
    std::vector<unsigned char> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(
      prev, curr, prev_pts, next, status, err, cv::Size(21, 21), 3, lk_criteria_);

    // FORWARD-BACKWARD CONSISTENCY. KLT's own error only says the patch matched SOMETHING; flowing
    // the result back and landing elsewhere says it matched the wrong thing.
    std::vector<cv::Point2f> back;
    std::vector<unsigned char> back_status;
    if (flow_back_px_ > 0.0 && !next.empty()) {
      std::vector<float> back_err;
      back = prev_pts;   // seed the reverse flow at the origin, as VINS-Fusion does
      cv::calcOpticalFlowPyrLK(
        curr, prev, next, back, back_status, back_err, cv::Size(21, 21), 1, lk_criteria_,
        cv::OPTFLOW_USE_INITIAL_FLOW);
    }

    // ...unless it would leave almost nothing. In a near-blind stretch the last few marginal tracks
    // are what keep the solve alive; dropping them measurably lost V1_03 (flow_back_min_keep).
    if (!back.empty()) {
      int consistent = 0;
      for (std::size_t i = 0; i < prev_pts.size(); ++i) {
        consistent += status[i] && back_status[i] &&
          cv::norm(back[i] - prev_pts[i]) <= flow_back_px_;
      }
      if (consistent < flow_back_min_keep_) {
        back.clear();
      }
    }

    constexpr float kMaxError = 20.0f;
    constexpr int kBorder = 5;
    for (std::size_t i = 0; i < prev_pts.size(); ++i) {
      if (!status[i] || err[i] >= kMaxError) {
        continue;
      }
      if (!back.empty() &&
        (!back_status[i] || cv::norm(back[i] - prev_pts[i]) > flow_back_px_))
      {
        continue;
      }
      const cv::Point2f & p = next[i];
      if (p.x < kBorder || p.y < kBorder ||
        p.x >= curr.cols - kBorder || p.y >= curr.rows - kBorder)
      {
        continue;
      }
      out_pts.push_back(p);
      out_ids.push_back(prev_ids[i]);
    }
  }

  /// Detect FAST corners in `gray`, keep the strongest, and append those that are not within
  /// kMinSpacing of a feature already in `pts` (surviving tracks plus corners accepted so far
  /// this call). Each kept corner gets a fresh id.
  std::size_t cap() const
  {
    return static_cast<std::size_t>(top_up_target_ > 0 ? top_up_target_ : max_features_);
  }

  void detectInto(
    const cv::Mat & gray,
    std::vector<cv::Point2f> & pts, std::vector<long> & ids, std::size_t limit)
  {
    std::vector<cv::KeyPoint> kps;
    detector_->detect(gray, kps);
    std::sort(
      kps.begin(), kps.end(),
      [](const cv::KeyPoint & a, const cv::KeyPoint & b) {return a.response > b.response;});

    for (const auto & kp : kps) {
      if (pts.size() >= limit) {
        break;
      }
      bool too_close = false;
      for (const auto & q : pts) {
        if (cv::norm(kp.pt - q) < min_spacing_px_) {
          too_close = true;
          break;
        }
      }
      if (!too_close) {
        pts.push_back(kp.pt);
        ids.push_back(next_id_++);
      }
    }
  }

  int max_features_;
  int min_features_;
  double flow_back_px_;
  int top_up_target_;
  float min_spacing_px_;
  int flow_back_min_keep_;
  cv::Ptr<cv::FastFeatureDetector> detector_;
  cv::TermCriteria lk_criteria_;

  cv::Mat prev_gray_;
  std::vector<cv::Point2f> prev_pts_;
  std::vector<long> ids_;
  long next_id_ = 0;
};

}  // namespace glassvio

#endif  // GLASSVIO_FEATURE_TRACKER_HPP
