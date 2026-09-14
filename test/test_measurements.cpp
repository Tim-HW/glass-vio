#include <cassert>
#include <cmath>
#include <cstdio>
#include <memory>

#include "glassvio/dataset.hpp"
#include "glassvio/sync.hpp"

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
  m->linear_acceleration.x = 2.0;
  return m;
}
}  // namespace

int main()
{
  const Eigen::Vector3d zero = Eigen::Vector3d::Zero();
  glassvio::ImuBuffer imu;
  for (double t : {0.0, 0.01, 0.02, 0.03}) {
    imu.add({t, zero, Eigen::Vector3d(2.0, 0.0, 0.0)});
  }
  glass_core::ImuPreintegration pre(zero, zero, 1e-3, 1e-2);
  assert(imu.preintegrate(0.005, 0.024, zero, zero, pre));
  assert(std::abs(pre.dt() - 0.019) < 1e-12 && "clip both IMU interval boundaries");
  assert(std::abs(pre.dv().x() - 0.038) < 1e-12);
  assert(std::abs(pre.dp().x() - 0.000361) < 1e-12);
  assert(imu.continuous(0.005, 0.024));
  assert(!imu.continuous(-0.001, 0.02) && "missing left bracket");
  assert(!imu.preintegrate(0.0, 0.04, zero, zero, pre) && "missing right bracket");
  assert(!imu.continuous(0.0, 0.04));
  assert(!imu.continuous(0.01, 0.01));
  assert(!imu.continuous(0.02, 0.01));

  glassvio::ImuBuffer gap;
  for (double t : {0.0, 0.01, 0.05, 0.06}) {
    gap.add({t, zero, zero});
  }
  assert(!gap.preintegrate(0.005, 0.055, zero, zero, pre));
  assert(!gap.continuous(0.005, 0.055));
  assert(!gap.preintegrate(0.02, 0.03, zero, zero, pre) && "gap straddles the query");

  // Overlapping groups replay bracket samples; the stored stream must stay sorted and unique.
  imu.add({0.02, zero, zero});
  imu.add({0.03, zero, zero});
  assert(imu.size() == 4);

  glassvio::MeasureSync sync;
  sync.pushImu(sample(0));
  sync.pushImu(sample(10));
  sync.pushFrame(header(5), {});
  glassvio::MeasureGroup group;
  assert(sync.next(group));
  assert(group.imu.empty());
  sync.pushFrame(header(24), {});
  sync.pushImu(sample(20));
  assert(!sync.next(group) && "wait for the right bracket");
  sync.pushImu(sample(30));
  assert(sync.next(group));
  assert(group.imu.size() == 4 && "include left AND right brackets");
  assert(group.imu.front()->header.stamp.nanosec == 0);
  assert(group.imu.back()->header.stamp.nanosec == 30000000);
  glassvio::ImuBuffer online;
  for (const auto & m : group.imu) {
    online.add({m->header.stamp.nanosec * 1e-9, zero, Eigen::Vector3d(2.0, 0.0, 0.0)});
  }
  assert(online.preintegrate(0.005, 0.024, zero, zero, pre));
  assert(std::abs(pre.dt() - 0.019) < 1e-12);
  std::puts("measurement intervals: ok");
}
