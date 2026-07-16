#ifndef GLASSVIO_TYPES_HPP
#define GLASSVIO_TYPES_HPP

#include <vector>

#include <std_msgs/msg/header.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "glassvio/feature_tracker.hpp"

namespace glassvio
{

/// One frame's TRACKED FEATURES, and the IMU samples carrying the state up to it.
///
/// The unit of work handed from the callbacks to the worker -- glasslio's MeasureGroup, with
/// two deliberate differences.
///
/// [1] IT CARRIES FEATURES, NOT AN IMAGE, and that is not a memory optimisation (though it is
///     19x smaller: 23 KB of tracks against a 455 KB frame). KLT is SEQUENTIAL -- the
///     tracker's entire state is the previous frame. glasslio's queue drops the oldest scan
///     when the worker falls behind, which is safe there because ICP registers against the
///     MAP and nothing remembers the scan that was skipped. Drop an IMAGE and the tracker
///     goes N -> N+2: the flow doubles, KLT fails its error threshold, tracks die, and IDs
///     churn -- which is the one thing FeatureTracker exists to prevent. So tracking happens
///     in the callback, where nothing is ever dropped, and what queues up is its OUTPUT. A
///     dropped MeasureGroup then costs observations, not the tracker's state: the surviving
///     frames still share IDs across the gap.
///
/// [2] THE IMU SPANS THE GAP BETWEEN FRAMES, not the frame itself. A LiDAR scan is a ~100 ms
///     SWEEP and must be bracketed on both sides, because deskew needs R(t) at every point's
///     acquisition time. An image is an INSTANT. What VIO needs is the interval
///     [t_prev, t_cur] -- exactly what preintegration turns into the factor tying pose i to
///     pose j.
struct MeasureGroup
{
  /// The image's header: stamp and frame_id. The pixels are gone by now, by design.
  std_msgs::msg::Header header;
  /// Live tracks at this frame, with the persistent IDs that make them correspondences.
  FeatureTracker::Result features;
  /// Samples covering [previous frame, this frame], in time order. The first is at or before
  /// the previous frame's stamp -- preintegration steps BETWEEN samples, so it needs that one
  /// as the interval's left edge.
  std::vector<sensor_msgs::msg::Imu::ConstSharedPtr> imu;
};

}  // namespace glassvio

#endif  // GLASSVIO_TYPES_HPP
