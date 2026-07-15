// GlassVIO -- visual-inertial odometry sharing GlassLIO's estimation engine.
//
// This is the FRONT-END SKELETON. What runs today is the IMU-initialization stage, on the
// exact same glass_core::ImuInit that glasslio uses: estimate the gyro bias and gravity
// from a static window, and seed a gravity-aligned world frame. VIO needs this bootstrap
// just as much as LIO does -- the world's +Z has to be real before any camera pose means
// anything.
//
// What is deliberately NOT here yet (the next phases, see the repo plan):
//   * feature detection + KLT tracking on the image stream,
//   * a reprojection residual (pixel error of a projected 3D landmark) folded into the
//     SAME glass_core Gauss-Newton solver the LiDAR path uses,
//   * landmark triangulation + a sliding window.
//
// The point of this file right now is to prove the engine imports and drives from a second,
// independent front-end -- one solver, two sensors, zero copied Jacobians.

#include <cmath>
#include <memory>
#include <string>

#include <Eigen/Core>
#include <cv_bridge/cv_bridge.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>

#include "glass_core/imu_init.hpp"
#include "glass_core/nav_state.hpp"
#include "glassvio/feature_tracker.hpp"

namespace glassvio
{

using namespace glass_core;  // the shared engine (NOLINT: build/namespaces)

class GlassVioNode : public rclcpp::Node
{
public:
  GlassVioNode()
  : Node("glassvio_node")
  {
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/livox/imu");
    const int num_samples = declare_parameter<int>("imu.init.num_samples", 200);
    const double max_gyro = declare_parameter<double>("imu.init.max_gyro", 0.1);
    const double max_accel_sd = declare_parameter<double>("imu.init.max_accel_sd", 0.5);
    // Livox publishes linear_acceleration in g, not m/s^2 -- convert at the boundary.
    const bool accel_in_g = declare_parameter<bool>("imu.accel_in_g", true);
    const double accel_scale = accel_in_g ? kGravity : 1.0;

    imu_init_ = std::make_unique<ImuInit>(num_samples, max_gyro, max_accel_sd, accel_scale);

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::SharedPtr msg) {onImu(msg);});

    // Visual front-end, phase 1: FAST + KLT feature tracking. No pose out of it yet -- it
    // just proves the tracker drives on real images. The reprojection factor that folds
    // these tracks into the same glass_core solver as the IMU is the next phase.
    const std::string image_topic = declare_parameter<std::string>("image_topic", "/camera/image_raw");
    const int max_features = declare_parameter<int>("features.max", 1000);
    const int min_features = declare_parameter<int>("features.min", 150);
    const int fast_threshold = declare_parameter<int>("features.fast_threshold", 20);
    tracker_ = std::make_unique<FeatureTracker>(max_features, min_features, fast_threshold);

    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Image::SharedPtr msg) {onImage(msg);});
    pub_features_ = create_publisher<sensor_msgs::msg::Image>("~/features", 1);

    RCLCPP_INFO(
      get_logger(),
      "glassvio up -- IMU init from '%s', feature tracking on '%s'",
      imu_topic.c_str(), image_topic.c_str());
  }

private:
  void onImu(const sensor_msgs::msg::Imu::SharedPtr msg)
  {
    if (imu_init_->initialized()) {
      return;   // nothing else to do until the visual front-end lands
    }

    // Same ROS -> engine boundary conversion as glasslio: glass_core never sees sensor_msgs.
    const ImuSample sample{
      {msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z},
      {msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z}};

    if (!imu_init_->add(sample)) {
      return;
    }

    // Seed the 15-DoF nav state from the init: gravity-aligned rotation, gyro bias.
    // (Yaw is unobservable from gravity and left at zero.) This is the state the visual
    // factors will later update.
    state_.R = imu_init_->initial_rotation();
    state_.bg = imu_init_->gyro_bias();

    const Eigen::Vector3d g = imu_init_->gravity();
    const double tilt_deg = std::acos(g.normalized().z()) * 180.0 / M_PI;
    RCLCPP_INFO(
      get_logger(),
      "IMU initialized (%d window(s) rejected). |g|=%.3f m/s^2, mount tilt %.2f deg, "
      "gyro bias [%.4f %.4f %.4f]. Awaiting visual front-end.",
      imu_init_->rejected_windows(), g.norm(), tilt_deg,
      state_.bg.x(), state_.bg.y(), state_.bg.z());
  }

  void onImage(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    // ROS -> engine boundary: hand OpenCV a grayscale view. mono8 shares the message buffer;
    // a color image is converted once here so the tracker only ever sees one channel.
    cv_bridge::CvImageConstPtr cv;
    try {
      cv = cv_bridge::toCvShare(msg, "mono8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    const FeatureTracker::Result r = tracker_->track(cv->image);

    if (pub_features_->get_subscription_count() > 0) {
      cv::Mat vis;
      cv::cvtColor(cv->image, vis, cv::COLOR_GRAY2BGR);
      for (const auto & p : r.points) {
        cv::circle(vis, p, 2, cv::Scalar(0, 255, 0), -1);
      }
      cv_bridge::CvImage out(msg->header, "bgr8", vis);
      pub_features_->publish(*out.toImageMsg());
    }

    // Cheap heartbeat so a bag replay visibly shows the tracker living/dying.
    RCLCPP_INFO_THROTTLE(
      get_logger(), *get_clock(), 1000, "tracking %zu features", r.points.size());
  }

  std::unique_ptr<ImuInit> imu_init_;
  std::unique_ptr<FeatureTracker> tracker_;
  NavState state_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_features_;
};

}  // namespace glassvio

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<glassvio::GlassVioNode>());
  rclcpp::shutdown();
  return 0;
}
