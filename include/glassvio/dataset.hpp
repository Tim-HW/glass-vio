#ifndef GLASSVIO_DATASET_HPP
#define GLASSVIO_DATASET_HPP

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <cv_bridge/cv_bridge.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "glass_core/preintegration.hpp"
#include "glassvio/feature_tracker.hpp"
#include "glassvio/camera_calib.hpp"
#include "glassvio/sfm_window.hpp"

namespace glassvio
{

/// One IMU reading with its stamp. glass_core's own ImuSample is deliberately timeless --
/// the engine never sees a clock -- so the time lives out here, with the buffer.
struct StampedImu
{
  double t = 0.0;
  Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
  Eigen::Vector3d accel = Eigen::Vector3d::Zero();
};

/// Ground-truth poses, with lookup by time.
///
/// The IMU pose in the world frame. Both datasets hand this over directly -- KITTI because
/// its OXTS stamps and orientations are bit-identical to its pose messages, EuRoC because its
/// ground-truth T_BS is identity. Neither needs an extrinsic to reach the body frame.
class GroundTruth
{
public:
  /// Pose only. Velocity then has to be finite-differenced -- see velocity().
  void add(double t, const Eigen::Isometry3d & T)
  {
    t_.push_back(t);
    T_.push_back(T);
    v_.emplace_back();
    bg_.emplace_back();
    ba_.emplace_back();
  }

  /// Pose AND the states a differentiator can only approximate. EuRoC's batch ground truth
  /// states velocity and both biases outright, which is what lets stage [3] be scored against
  /// a TRUE bias instead of against zero.
  void add(
    double t, const Eigen::Isometry3d & T, const Eigen::Vector3d & v,
    const Eigen::Vector3d & bg, const Eigen::Vector3d & ba)
  {
    t_.push_back(t);
    T_.push_back(T);
    v_.push_back(v);
    bg_.push_back(bg);
    ba_.push_back(ba);
    exact_ = true;
  }

  /// True when the source stated velocity/biases rather than leaving them to be inferred.
  bool exact() const {return exact_;}

  /// Nearest pose to `t`.
  Eigen::Isometry3d at(double t) const {return T_[index(t)];}

  /// Velocity, stated if the source knew it and differentiated otherwise.
  ///
  /// The fallback exists because KITTI had no velocity to give. Its positions are GPS-derived
  /// and quantised, so the obvious (p[i+1]-p[i])/dt at 100 Hz is dominated by quantisation
  /// rather than motion (measured: ~11 m/s^2 of apparent acceleration against a ~0.5 m/s^2
  /// signal) -- hence the WIDE baseline. It is still only an approximation, and it is biased
  /// exactly when acceleration is changing, which is exactly when it matters.
  Eigen::Vector3d velocity(double t, double half_window = 0.25) const
  {
    if (exact_) {
      return v_[index(t)];
    }
    return (at(t + half_window).translation() - at(t - half_window).translation()) /
           (2.0 * half_window);
  }

  /// True biases, where the source states them. Zero otherwise -- check exact() first.
  Eigen::Vector3d gyroBias(double t) const {return bg_[index(t)];}
  Eigen::Vector3d accelBias(double t) const {return ba_[index(t)];}

  bool empty() const {return t_.empty();}
  std::size_t size() const {return t_.size();}
  double t_begin() const {return t_.front();}
  double t_end() const {return t_.back();}

private:
  /// Index of the sample nearest `t`. Binary search, not a scan: callers ask once per frame
  /// over tens of thousands of samples, and the linear version was quietly quadratic.
  std::size_t index(double t) const
  {
    if (t_.empty()) {
      throw std::runtime_error("GroundTruth queried on an empty track");
    }
    const auto it = std::lower_bound(t_.begin(), t_.end(), t);
    if (it == t_.begin()) {
      return 0;
    }
    if (it == t_.end()) {
      return t_.size() - 1;
    }
    const std::size_t hi = static_cast<std::size_t>(it - t_.begin());
    return (t - t_[hi - 1] <= t_[hi] - t) ? hi - 1 : hi;
  }

  std::vector<double> t_;
  std::vector<Eigen::Isometry3d> T_;
  std::vector<Eigen::Vector3d> v_, bg_, ba_;
  bool exact_ = false;
};

/// IMU samples, and preintegration over an interval.
///
/// The gap check is the reason this is a class rather than a vector. ImuPreintegration holds
/// a sample constant over dt, so handing it the 1.67 s hole this bag contains would invent
/// 1.67 s of fictitious constant acceleration -- silently. glass_core cannot police that
/// itself (it has no way to tell a dropout from a slow sensor), so the policing lives here,
/// once, instead of in every caller.
class ImuBuffer
{
public:
  static constexpr double kDefaultMaxGap = 0.02;   ///< s; larger dt means the stream dropped

  void add(const StampedImu & s) {samples_.push_back(s);}

  /// Preintegrate [t0, t1) at the given bias. False if the stream has a hole inside, or if
  /// the interval is empty.
  bool preintegrate(
    double t0, double t1,
    const Eigen::Vector3d & bias_gyro, const Eigen::Vector3d & bias_accel,
    glass_core::ImuPreintegration & out,
    double gyro_noise = 1e-3, double accel_noise = 1e-2,
    double max_gap = kDefaultMaxGap) const
  {
    out = glass_core::ImuPreintegration(bias_gyro, bias_accel, gyro_noise, accel_noise);

    auto it = std::lower_bound(
      samples_.begin(), samples_.end(), t0,
      [](const StampedImu & s, double v) {return s.t < v;});

    for (; it != samples_.end() && (it + 1) != samples_.end() && it->t < t1; ++it) {
      const double dt = (it + 1)->t - it->t;
      if (dt > max_gap) {
        return false;
      }
      out.integrate(it->gyro, it->accel, dt);
    }
    return out.dt() > 1e-9;
  }

  /// True if [t0, t1) is free of dropouts -- for callers that want to reject a window
  /// before doing the work.
  bool continuous(double t0, double t1, double max_gap = kDefaultMaxGap) const
  {
    auto it = std::lower_bound(
      samples_.begin(), samples_.end(), t0,
      [](const StampedImu & s, double v) {return s.t < v;});
    for (; it != samples_.end() && (it + 1) != samples_.end() && it->t < t1; ++it) {
      if ((it + 1)->t - it->t > max_gap) {
        return false;
      }
    }
    return true;
  }

  const std::vector<StampedImu> & samples() const {return samples_;}
  std::size_t size() const {return samples_.size();}
  bool empty() const {return samples_.empty();}

  /// Mean sample rate, Hz. Worth printing: every "num_samples" parameter in this system is a
  /// COUNT, and its meaning in seconds depends entirely on this number.
  double rate() const
  {
    if (samples_.size() < 2) {
      return 0.0;
    }
    return static_cast<double>(samples_.size() - 1) / (samples_.back().t - samples_.front().t);
  }

private:
  std::vector<StampedImu> samples_;
};

/// At namespace scope, not nested: a nested struct with default member initializers is not
/// complete inside its own enclosing class, so it cannot be a default argument there.
struct DatasetOptions
{
  bool track_images = false;   ///< run the FeatureTracker over the image stream
  int max_frames = -1;         ///< stop after this many images (-1 = all)
  std::string imu_topic = "/imu0";
  std::string image_topic = "/cam0/image_raw";
};

/// A EuRoC MAV sequence: the ROS2 bag for sensors, and the ASL CSV for ground truth.
///
/// TWO SOURCES, DELIBERATELY. The converted bag carries /vicon/firefly_sbx -- the raw VICON
/// MARKER pose, which would need T_BS to reach the IMU and offers no velocity. The ASL
/// `state_groundtruth_estimate0/data.csv` is the dataset's own batch solution over VICON and
/// IMU, and gives p, q, v, b_w AND b_a directly in the body frame (its T_BS is identity). So
/// it turns two of our checks from approximations into real comparisons: velocity stops being
/// a finite difference, and the gyro bias finally has a TRUE value to be scored against
/// rather than "is it near zero".
///
/// WHICH IMU. The bag has two: /imu0 is the ADIS16448 the calibration describes, and /fcu/imu
/// belongs to the flight controller. They are different sensors; the calibration only fits
/// one of them.
class EurocDataset
{
public:
  using Options = DatasetOptions;

  ImuBuffer imu;
  GroundTruth gt;
  std::vector<SfmFrame> frames;   ///< populated only when Options::track_images

  /// `gt_csv` may be empty -- the sensors load without it, for a run with no oracle.
  static EurocDataset load(
    const std::string & bag_path, const std::string & gt_csv,
    const CameraCalib & calib, const Options & opts = Options())
  {
    EurocDataset out;
    if (!gt_csv.empty()) {
      out.loadGroundTruth(gt_csv);
    }

    rosbag2_cpp::Reader reader;
    reader.open(bag_path);   // throws; the caller reports it

    rclcpp::Serialization<sensor_msgs::msg::Imu> imu_codec;
    rclcpp::Serialization<sensor_msgs::msg::Image> image_codec;
    FeatureTracker tracker;
    int n_frames = 0;

    while (reader.has_next()) {
      const auto msg = reader.read_next();
      if (msg->topic_name != opts.imu_topic && msg->topic_name != opts.image_topic) {
        continue;   // the bag also holds /cam1, /fcu/imu, /fcu/motor_speed, /vicon
      }
      rclcpp::SerializedMessage raw(*msg->serialized_data);

      if (msg->topic_name == opts.imu_topic) {
        sensor_msgs::msg::Imu m;
        imu_codec.deserialize_message(&raw, &m);
        out.imu.add(
          {stamp(m.header),
            {m.angular_velocity.x, m.angular_velocity.y, m.angular_velocity.z},
            {m.linear_acceleration.x, m.linear_acceleration.y, m.linear_acceleration.z}});

      } else {
        if (opts.max_frames >= 0 && n_frames >= opts.max_frames) {
          continue;
        }
        sensor_msgs::msg::Image m;
        image_codec.deserialize_message(&raw, &m);
        const auto img = cv_bridge::toCvCopy(m, "mono8");

        // TRACK ON THE RAW FRAME, UNDISTORT THE OUTPUT. KLT is a local search and distortion
        // is a smooth warp, so tracking is correct on the pixels the sensor produced. Only
        // the geometry needs true rays -- so the points are corrected here, at the boundary,
        // and every stage downstream sees a clean pinhole. A no-op on a rectified stream.
        const auto r = tracker.track(img->image);
        const auto undistorted = calib.undistort(r.points);

        SfmFrame f;
        f.t = stamp(m.header);
        for (std::size_t i = 0; i < r.ids.size(); ++i) {
          f.by_id.emplace(r.ids[i], undistorted[i]);
        }
        out.frames.push_back(std::move(f));
        ++n_frames;
      }
    }
    return out;
  }

private:
  /// `#timestamp [ns], p_RS_R xyz, q_RS wxyz, v_RS_R xyz, b_w xyz, b_a xyz`
  void loadGroundTruth(const std::string & path)
  {
    std::ifstream in(path);
    if (!in) {
      throw std::runtime_error("cannot open ground truth: " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
      if (line.empty() || line[0] == '#') {
        continue;
      }
      std::istringstream ss(line);
      std::string tok;
      std::vector<double> v;
      while (std::getline(ss, tok, ',')) {
        try {
          v.push_back(std::stod(tok));
        } catch (...) {
          break;
        }
      }
      if (v.size() < 17) {
        continue;
      }
      Eigen::Isometry3d T = Eigen::Isometry3d::Identity();
      // q_RS is stored w, x, y, z.
      T.linear() = Eigen::Quaterniond(v[4], v[5], v[6], v[7]).normalized().toRotationMatrix();
      T.translation() = Eigen::Vector3d(v[1], v[2], v[3]);
      gt.add(
        v[0] * 1e-9, T,
        Eigen::Vector3d(v[8], v[9], v[10]),      // velocity, world frame
        Eigen::Vector3d(v[11], v[12], v[13]),    // gyro bias, body frame
        Eigen::Vector3d(v[14], v[15], v[16]));   // accel bias, body frame
    }
  }

  template<typename Header>
  static double stamp(const Header & h)
  {
    return static_cast<double>(h.stamp.sec) + h.stamp.nanosec * 1e-9;
  }
};


}  // namespace glassvio

#endif  // GLASSVIO_DATASET_HPP
