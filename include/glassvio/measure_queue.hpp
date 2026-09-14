#ifndef GLASSVIO_MEASURE_QUEUE_HPP
#define GLASSVIO_MEASURE_QUEUE_HPP

#include <algorithm>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "glassvio/sync.hpp"

namespace glassvio
{

/// Synchronize the two callback streams and hand ordered groups to one worker.
/// One mutex covers release AND enqueue; separate locks allow producers to swap frames.
class MeasureQueue
{
public:
  explicit MeasureQueue(int capacity)
  {
    if (capacity < 1) {
      throw std::invalid_argument("max_queue_size must be positive");
    }
    capacity_ = static_cast<std::size_t>(capacity);
  }

  /// Return the number of old observations dropped while releasing ready groups.
  std::size_t pushImu(const sensor_msgs::msg::Imu::ConstSharedPtr & msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {return 0;}
    sync_.pushImu(msg);
    return release();
  }

  std::size_t pushFrame(const std_msgs::msg::Header & header, FeatureTracker::Result features)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (stopped_) {return 0;}
    sync_.pushFrame(header, std::move(features));
    return release();
  }

  /// Wait for work, or return false once shutdown has drained the queue.
  bool waitPop(MeasureGroup & out)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait(lock, [this] {return stopped_ || !queue_.empty();});
    if (queue_.empty()) {return false;}
    out = std::move(queue_.front());
    queue_.pop_front();
    return true;
  }

  void stop()
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stopped_ = true;
    ready_.notify_all();
  }

private:
  // Called with mutex_ held. Dropping observations must not punch a hole in the IMU chain.
  std::size_t release()
  {
    std::size_t dropped_count = 0;
    MeasureGroup group;
    while (sync_.next(group)) {
      queue_.push_back(std::move(group));
      if (queue_.size() > capacity_) {
        auto dropped = std::move(queue_.front());
        queue_.pop_front();
        auto & next = queue_.front();
        if (!dropped.imu.empty()) {
          const auto end = dropped.imu.back()->header.stamp;
          const auto begin = std::find_if(next.imu.begin(), next.imu.end(),
              [&end](const auto & m) {
                const auto t = m->header.stamp;
                return t.sec > end.sec || (t.sec == end.sec && t.nanosec > end.nanosec);
            });
          dropped.imu.insert(dropped.imu.end(), begin, next.imu.end());
          next.imu = std::move(dropped.imu);
        }
        ++dropped_count;
      }
      ready_.notify_one();
    }
    return dropped_count;
  }

  MeasureSync sync_;
  std::deque<MeasureGroup> queue_;
  std::size_t capacity_;
  bool stopped_ = false;
  std::mutex mutex_;
  std::condition_variable ready_;
};

}  // namespace glassvio

#endif  // GLASSVIO_MEASURE_QUEUE_HPP
