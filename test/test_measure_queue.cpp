#include <cassert>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <thread>

#include "glassvio/measure_queue.hpp"

namespace
{
std_msgs::msg::Header header(int milliseconds)
{
  std_msgs::msg::Header h;
  h.stamp.sec = milliseconds / 1000;
  h.stamp.nanosec = (milliseconds % 1000) * 1000000;
  return h;
}

sensor_msgs::msg::Imu::ConstSharedPtr sample(int milliseconds)
{
  auto m = std::make_shared<sensor_msgs::msg::Imu>();
  m->header = header(milliseconds);
  return m;
}
}  // namespace

int main()
{
  bool refused = false;
  try {
    glassvio::MeasureQueue invalid(0);
  } catch (const std::invalid_argument &) {
    refused = true;
  }
  assert(refused && "zero capacity must not pop then access an empty queue");

  // One worker, two independent callback producers. Every frame must arrive in order.
  constexpr int count = 20000;
  glassvio::MeasureQueue queue(count);
  std::thread camera([&] {
      for (int k = 0; k < count; ++k) {
        queue.pushFrame(header(k), {});
        std::this_thread::yield();
      }
    });
  std::thread imu([&] {
      for (int k = 0; k <= count; ++k) {
        queue.pushImu(sample(k));
        std::this_thread::yield();
      }
    });
  for (int k = 0; k < count; ++k) {
    glassvio::MeasureGroup group;
    const bool ok1 = queue.waitPop(group);
    assert(ok1);
    assert(group.header.stamp == header(k).stamp && "callbacks reordered frames");
  }
  camera.join();
  imu.join();
  queue.stop();
  glassvio::MeasureGroup group;
  const bool ok2 = queue.waitPop(group);
  assert(!ok2);

  // A stalled worker drops observations but keeps a complete, ordered IMU chain.
  glassvio::MeasureQueue bounded(1);
  bounded.pushImu(sample(0));
  bounded.pushFrame(header(0), {});
  const bool ok3 = bounded.waitPop(group);
  assert(ok3);
  for (int k = 1; k <= 5; ++k) {
    bounded.pushFrame(header(k * 10 - 5), {});
    bounded.pushImu(sample(k * 10));
  }
  bounded.stop();   // shutdown drains the queued work before returning false.
  const bool ok4 = bounded.waitPop(group);
  assert(ok4);
  assert(group.header.stamp == header(45).stamp);
  assert(group.imu.size() == 6);
  for (int k = 0; k <= 5; ++k) {
    assert(group.imu[k]->header.stamp == header(k * 10).stamp);
  }
  const bool ok5 = bounded.waitPop(group);
  assert(!ok5);

  // A bag restart discards queued and synchronized observations from the old timeline.
  glassvio::MeasureQueue restarted(3);
  restarted.pushImu(sample(1500));
  restarted.pushFrame(header(1500), {});
  const auto old_epoch = restarted.epoch();
  restarted.reset();
  assert(restarted.epoch() != old_epoch);
  restarted.pushImu(sample(0));
  restarted.pushFrame(header(0), {});
  std::size_t group_epoch = old_epoch;
  assert(restarted.waitPop(group, &group_epoch));
  assert(group_epoch == restarted.epoch());
  assert(group.header.stamp == header(0).stamp);
  restarted.stop();
  assert(!restarted.waitPop(group));
  std::puts("measurement queue: ok");
}
