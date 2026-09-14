// ROS adapter: image callbacks track features, MeasureQueue synchronizes and orders
// measurements, and one worker owns VioEstimator. The numerical engine is glass_core;
// the estimator uses ROS message types but has no subscriptions or threads.
//
// Tracking stays before the bounded worker queue so dropped observations do not skip
// KLT input frames. IMU and image callbacks have separate mutually exclusive groups.
// MeasureQueue holds one mutex across synchronization and enqueue, then releases it
// before estimation. Dropped observations retain their IMU samples.

#include <cmath>
#include <memory>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <cv_bridge/cv_bridge.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <tf2_ros/transform_broadcaster.h>

#include "glassvio/camera_calib.hpp"
#include "glassvio/feature_tracker.hpp"
#include "glassvio/measure_queue.hpp"
#include "glassvio/types.hpp"
#include "glassvio/vio_estimator.hpp"

namespace glassvio
{

class GlassVioNode : public rclcpp::Node
{
public:
  GlassVioNode()
  : Node("glassvio_node")
  {
    const std::string imu_topic = declare_parameter<std::string>("imu_topic", "/imu0");
    const std::string image_topic =
      declare_parameter<std::string>("image_topic", "/cam0/image_raw");
    const std::string calib_dir = declare_parameter<std::string>("calib_dir", "config");
    world_frame_ = declare_parameter<std::string>("world_frame", "odom");
    body_frame_ = declare_parameter<std::string>("body_frame", "imu");

    // THE CALIBRATION IS NOT OPTIONAL and there is no sane default: K, the extrinsic and the
    // noise densities all come from the dataset's own Kalibr files. Guessing any of them is
    // how the estimator ends up plausible and wrong, so fail loudly instead.
    try {
      calib_ = loadEurocCalib(calib_dir);
    } catch (const std::exception & e) {
      RCLCPP_FATAL(get_logger(), "cannot load calibration from '%s': %s", calib_dir.c_str(),
        e.what());
      throw;
    }
    RCLCPP_INFO(
      get_logger(),
      "calib: fx=%.2f fy=%.2f cx=%.2f cy=%.2f, %s, gyro_noise=%.3e accel_noise=%.3e",
      calib_.fx(), calib_.fy(), calib_.cx(), calib_.cy(),
      calib_.rectified() ? "rectified" : "radtan (points undistorted at the boundary)",
      calib_.gyro_noise, calib_.accel_noise);

    const int max_features = declare_parameter<int>("features.max", 1000);
    const int min_features = declare_parameter<int>("features.min", 150);
    const int fast_threshold = declare_parameter<int>("features.fast_threshold", 20);
    tracker_ = std::make_unique<FeatureTracker>(max_features, min_features, fast_threshold);

    EstimatorParams ep;
    // TWO SPANS, NOT ONE. `bootstrap_frames` is how much to collect (stage [3]'s bias wants
    // as many pairs as it can get); `sfm_window_frames` is stage [2]'s reconstruction window,
    // which must stay short or its landmarks die. Setting them equal starves the bias solve --
    // see EstimatorParams::bootstrap_frames.
    ep.bootstrap_frames = declare_parameter<int>("bootstrap_frames", 120);
    ep.init.window_frames = declare_parameter<int>("sfm_window_frames", 30);
    estimator_ = std::make_unique<VioEstimator>(calib_, ep);

    // Bounded: if the worker falls behind, drop the OLDEST group rather than let latency grow
    // without bound. A stale pose is useless.
    measurements_ = std::make_unique<MeasureQueue>(declare_parameter<int>("max_queue_size", 3));

    // SEPARATE, MUTUALLY EXCLUSIVE CALLBACK GROUPS. This is what lets the tracker run in the
    // image callback without starving the IMU: on the default single-threaded executor the two
    // callbacks share one thread and the ~1.3 ms track() would sit in front of every IMU
    // sample. Mutually exclusive rather than reentrant: each callback is still serialised WITH
    // ITSELF, so tracker_ needs no lock and IMU order is preserved.
    imu_cbg_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    image_cbg_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions imu_opts;
    imu_opts.callback_group = imu_cbg_;
    rclcpp::SubscriptionOptions image_opts;
    image_opts.callback_group = image_cbg_;

    sub_imu_ = create_subscription<sensor_msgs::msg::Imu>(
      imu_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Imu::ConstSharedPtr msg) {imuCallback(msg);}, imu_opts);
    sub_image_ = create_subscription<sensor_msgs::msg::Image>(
      image_topic, rclcpp::SensorDataQoS(),
      [this](const sensor_msgs::msg::Image::ConstSharedPtr msg) {imageCallback(msg);},
      image_opts);

    pub_features_ = create_publisher<sensor_msgs::msg::Image>("~/features", 1);
    pub_odom_ = create_publisher<nav_msgs::msg::Odometry>("~/odom", 10);
    tf_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    worker_ = std::thread(&GlassVioNode::workerLoop, this);
    RCLCPP_INFO(
      get_logger(), "glassvio up -- imu '%s', camera '%s'; collecting %d frames to bootstrap",
      imu_topic.c_str(), image_topic.c_str(), ep.bootstrap_frames);
  }

  ~GlassVioNode() override
  {
    measurements_->stop();
    if (worker_.joinable()) {
      worker_.join();
    }
  }

private:
  // =========================================================================
  // Callback threads (one group each): buffer, track, sync. No ESTIMATION here.
  // =========================================================================

  void imuCallback(const sensor_msgs::msg::Imu::ConstSharedPtr & msg)
  {
    warnDropped(measurements_->pushImu(msg));
  }

  /// Tracks HERE, not in the worker: the queue may drop a group, and a sequential KLT cannot
  /// survive a dropped frame. Nothing drops before this point.
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr & msg)
  {
    cv_bridge::CvImageConstPtr cv;
    try {
      cv = cv_bridge::toCvShare(msg, "mono8");
    } catch (const cv_bridge::Exception & e) {
      RCLCPP_WARN(get_logger(), "cv_bridge: %s", e.what());
      return;
    }

    // tracker_ is touched only here, and this group is MutuallyExclusive, so no lock.
    FeatureTracker::Result r = tracker_->track(cv->image);
    publishFeatures(msg->header, cv->image, r);

    // UNDISTORT AT THE BOUNDARY. KLT is a local search and distortion is a smooth warp, so
    // tracking is correct on the pixels the sensor produced -- but every geometric stage
    // downstream needs true rays. EuRoC's leading radtan coefficient is -0.283: a corner
    // feature moves 155 px. A no-op on a rectified stream.
    r.points = calib_.undistort(r.points);

    warnDropped(measurements_->pushFrame(msg->header, std::move(r)));
  }

  void publishFeatures(
    const std_msgs::msg::Header & header, const cv::Mat & gray,
    const FeatureTracker::Result & r)
  {
    if (pub_features_->get_subscription_count() == 0) {
      return;
    }
    cv::Mat vis;
    cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
    for (const auto & p : r.points) {
      cv::circle(vis, p, 2, cv::Scalar(0, 255, 0), -1);
    }
    cv_bridge::CvImage out(header, "bgr8", vis);
    pub_features_->publish(*out.toImageMsg());
  }

  void warnDropped(std::size_t dropped)
  {
    if (dropped > 0) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 2000,
        "worker behind -- dropped %zu old observations, keeping their IMU", dropped);
    }
  }

  // Estimator state belongs exclusively to this worker.
  void workerLoop()
  {
    MeasureGroup group;
    while (measurements_->waitPop(group)) {
      handleFrame(group);
    }
  }

  void handleFrame(const MeasureGroup & group)
  {
    const FrameResult r = estimator_->process(group);

    switch (r.stage) {
      case FrameResult::Stage::Collecting:
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 2000, "collecting... last attempt: %s (%d landmarks, "
          "%d bias pairs)",
          estimator_->lastFailure().empty() ? "still filling the window" :
          estimator_->lastFailure().c_str(),
          estimator_->lastLandmarks(), estimator_->lastBiasPairs());
        return;

      case FrameResult::Stage::Bootstrapped:
        RCLCPP_INFO(
          get_logger(),
          "BOOTSTRAPPED: %zu landmarks, |v0| = %.2f m/s, bg = [%.4f %.4f %.4f]. "
          "World frame defined: origin here, +Z along measured gravity, yaw arbitrary.",
          estimator_->landmarks().size(), r.velocity.norm(),
          estimator_->state().bg.x(), estimator_->state().bg.y(), estimator_->state().bg.z());
        break;

      case FrameResult::Stage::Lost:
        // Loss is terminal for this run; restart the node to collect a new bootstrap.
        RCLCPP_WARN_THROTTLE(
          get_logger(), *get_clock(), 2000,
          "LOST -- only %d landmarks in view. With the map maintained this means the tracker "
          "lost the scene, not that we outlived a frozen set.", r.features);
        return;

      case FrameResult::Stage::Tracking:
        RCLCPP_INFO_THROTTLE(
          get_logger(), *get_clock(), 1000,
          "tracking: %d/%zu landmarks, rmse %.2f px, |v| = %.2f m/s "
          "| map: +%d -%d, %zu pending",
          r.features, estimator_->map().size(), r.rmse_px, r.velocity.norm(),
          estimator_->map().lastTriangulated(), estimator_->map().lastDropped(),
          estimator_->map().pending());
        break;
    }

    if (r.pose_trusted) {
      publishOdom(group.header, r);
    }
  }

  void publishOdom(const std_msgs::msg::Header & header, const FrameResult & r)
  {
    const Eigen::Quaterniond q(r.pose.linear());

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = header.stamp;
    odom.header.frame_id = world_frame_;
    odom.child_frame_id = body_frame_;
    odom.pose.pose.position.x = r.pose.translation().x();
    odom.pose.pose.position.y = r.pose.translation().y();
    odom.pose.pose.position.z = r.pose.translation().z();
    odom.pose.pose.orientation.w = q.w();
    odom.pose.pose.orientation.x = q.x();
    odom.pose.pose.orientation.y = q.y();
    odom.pose.pose.orientation.z = q.z();
    // Velocity is reported in the WORLD frame, which is what the estimator holds. Odometry's
    // twist is conventionally in child_frame_id -- rotate it rather than mislabel it.
    const Eigen::Vector3d v_body = r.pose.linear().transpose() * r.velocity;
    odom.twist.twist.linear.x = v_body.x();
    odom.twist.twist.linear.y = v_body.y();
    odom.twist.twist.linear.z = v_body.z();
    pub_odom_->publish(odom);

    geometry_msgs::msg::TransformStamped tf;
    tf.header.stamp = header.stamp;
    tf.header.frame_id = world_frame_;
    tf.child_frame_id = body_frame_;
    tf.transform.translation.x = r.pose.translation().x();
    tf.transform.translation.y = r.pose.translation().y();
    tf.transform.translation.z = r.pose.translation().z();
    tf.transform.rotation = odom.pose.pose.orientation;
    tf_->sendTransform(tf);
  }

  // --- image-callback-owned. Never touched by the worker: that is the whole point.
  std::unique_ptr<FeatureTracker> tracker_;
  CameraCalib calib_;

  std::unique_ptr<MeasureQueue> measurements_;
  std::thread worker_;

  // --- worker-owned. Touched by nothing else.
  std::unique_ptr<VioEstimator> estimator_;

  std::string world_frame_, body_frame_;
  rclcpp::CallbackGroup::SharedPtr imu_cbg_, image_cbg_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr sub_imu_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_image_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr pub_features_;
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr pub_odom_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_;
};

}  // namespace glassvio

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  // MultiThreadedExecutor, so the IMU and image callback groups actually run on different
  // threads. With the default single-threaded executor the groups exist but share one thread,
  // and the tracker would sit in front of every IMU sample -- the bug this is here to avoid,
  // silently reintroduced.
  rclcpp::executors::MultiThreadedExecutor executor;
  auto node = std::make_shared<glassvio::GlassVioNode>();
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
