#ifndef GLASSVIO_SYNC_HPP
#define GLASSVIO_SYNC_HPP

#include <deque>
#include <utility>
#include <vector>

#include <sensor_msgs/msg/imu.hpp>
#include <std_msgs/msg/header.hpp>

#include "glassvio/feature_tracker.hpp"
#include "glassvio/types.hpp"

namespace glassvio
{

/// [1] SYNC -- pair a frame's TRACKS with the IMU samples spanning the gap since the last one.
///
/// It syncs tracks rather than images because the tracker runs upstream of here, in the
/// callback, where nothing is ever dropped -- see MeasureGroup for why a sequential KLT
/// cannot survive glasslio's drop-oldest policy.
///
/// The LiDAR sibling of this (glasslio's MeasureSync) brackets a SWEEP on both sides, because
/// deskew needs R(t) at every point's acquisition time. VIO's requirement is different and
/// simpler to state, and easier to get subtly wrong:
///
///   preintegration must cover [t_prev_image, t_image] EXACTLY.
///
///     IMU:    *----*----*----*----*----*----*----*
///                  ^                         ^
///              at/before t_prev          at/after t_cur
///     images:      |                         |
///                t_prev                    t_cur
///
/// THE RELEASE RULE. An image is held until the IMU stream has a sample at or past its stamp.
/// Release earlier and preintegration silently covers only PART of the interval: dt comes out
/// short, the delta is missing its tail, and the factor is wrong in a way that looks like a
/// slightly worse estimator rather than an error. ImuPreintegration cannot notice -- it
/// integrates what it is given.
///
/// CONSUMED IMU IS NOT EAGERLY DROPPED, for the same reason it is not in glasslio: the NEXT
/// group needs the sample at or before t_cur, which is a sample this group also used. Trim to
/// the last released image's stamp, never past it.
///
/// The first image has no predecessor, so it is released with an empty IMU list and only
/// establishes t_prev. There is no interval before the first frame; pretending otherwise
/// would preintegrate from an arbitrary start.
class MeasureSync
{
public:
  /// Bound on retained IMU. Purely a leak guard for the case where images stop arriving --
  /// never reached in normal operation, where the trim keeps this at one image gap.
  static constexpr std::size_t kMaxImuBuffer = 20000;

  void pushImu(const sensor_msgs::msg::Imu::ConstSharedPtr & msg)
  {
    imu_.push_back(msg);
    if (imu_.size() > kMaxImuBuffer) {
      imu_.pop_front();
    }
  }

  /// The tracker's output for one frame. Already extracted: the image is gone.
  void pushFrame(const std_msgs::msg::Header & header, FeatureTracker::Result features)
  {
    frames_.push_back({header, std::move(features)});
  }

  /// Pop one group if the IMU covers the oldest pending frame. False means "not yet".
  bool next(MeasureGroup & out)
  {
    if (frames_.empty() || imu_.empty()) {
      return false;
    }
    const double t_cur = stamp(frames_.front().first);

    // The release rule: hold until the IMU has crossed this frame's stamp.
    if (stamp(imu_.back()->header) < t_cur) {
      return false;
    }

    out.header = frames_.front().first;
    out.features = std::move(frames_.front().second);
    out.imu.clear();
    frames_.pop_front();

    if (!have_prev_) {
      // No interval exists before the first image. Establish the origin and hand back an
      // empty group rather than inventing a start time.
      have_prev_ = true;
      t_prev_ = t_cur;
      trim(t_prev_);
      return true;
    }

    // Everything in [t_prev, t_cur], inclusive of the brackets: preintegration steps between
    // consecutive samples, so it needs the one at/before t_prev to have a left edge.
    for (const auto & m : imu_) {
      const double t = stamp(m->header);
      if (t > t_cur) {
        break;
      }
      out.imu.push_back(m);
    }

    t_prev_ = t_cur;
    trim(t_prev_);
    return true;
  }

  void clear()
  {
    imu_.clear();
    frames_.clear();
    have_prev_ = false;
  }

  std::size_t frame_count() const {return frames_.size();}
  std::size_t imu_count() const {return imu_.size();}

private:
  template<typename Header>
  static double stamp(const Header & h)
  {
    return static_cast<double>(h.stamp.sec) + h.stamp.nanosec * 1e-9;
  }

  /// Drop IMU strictly older than the last sample at/before `t`. That sample STAYS: it is the
  /// left edge of the next interval.
  void trim(double t)
  {
    while (imu_.size() >= 2 && stamp(imu_[1]->header) <= t) {
      imu_.pop_front();
    }
  }

  std::deque<sensor_msgs::msg::Imu::ConstSharedPtr> imu_;
  std::deque<std::pair<std_msgs::msg::Header, FeatureTracker::Result>> frames_;
  double t_prev_ = 0.0;
  bool have_prev_ = false;
};

}  // namespace glassvio

#endif  // GLASSVIO_SYNC_HPP
