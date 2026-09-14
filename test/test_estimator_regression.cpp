#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>

#include "glassvio/estimator_regression.hpp"

int main()
{
  const auto check = [](bool ok, const char * message) {
      if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        assert(ok);
      }
    };
  using glassvio::estimatorRegression;
  using glassvio::EstimatorRegressionSample;
  const double nan = std::numeric_limits<double>::quiet_NaN();
  // Sparse, irregular timestamps catch any conversion from frame counts / nominal FPS.
  const std::vector<EstimatorRegressionSample> good = {
    {1001.0, 0.1}, {1035.0, 0.4}, {1080.0, 0.6}};
  const auto pass = estimatorRegression(1000.0, good);
  check(pass.failures.empty(), "healthy trajectory passes");
  check(pass.tracked_seconds == 80.0, "duration uses actual timestamps since bootstrap");
  check(pass.median_error == 0.4, "median uses aligned tracked position errors");
  check(pass.below_1m_seconds == 80.0, "good duration ends at last tracked sample");

  check(!estimatorRegression(1000.0, {{1001.0, 0.1}, {1020.0, 0.1}}).failures.empty(),
    "early tracking loss fails despite small errors");
  check(!estimatorRegression(1000.0, {{1001.0, 0.8}, {1035.0, 0.9}, {1080.0, 0.95}})
    .failures.empty(), "excessive drift fails despite sufficient tracking time");
  const auto early_drift = estimatorRegression(1000.0,
      {{1001.0, 0.1}, {1002.0, 1.1}, {1035.0, 0.1}, {1080.0, 0.1}});
  check(!early_drift.failures.empty(), "early transient divergence fails despite good median");
  check(early_drift.below_1m_seconds == 1.0,
    "good duration stops at last sample before first divergence");
  check(!estimatorRegression(1000.0, {}).failures.empty(), "no tracking data fails");
  check(!estimatorRegression(1000.0, {}, {0.0, 0.0, 0.0}).failures.empty(),
    "zero thresholds never turn missing data into a pass");
  check(!estimatorRegression(nan, good).failures.empty(), "missing bootstrap fails");
  check(!estimatorRegression(1000.0, {{1001.0, nan}, {1080.0, 0.1}}).failures.empty(),
    "a single nonfinite error fails even if median would be finite");
  check(!estimatorRegression(1000.0, {{nan, 0.1}}).failures.empty(), "nonfinite time fails");
  check(!estimatorRegression(1000.0, {{1080.0, 0.1}, {1001.0, 0.1}}).failures.empty(),
    "out of order timestamps fail");
  check(!estimatorRegression(1000.0, {{999.0, 0.1}, {1080.0, 0.1}}).failures.empty(),
    "tracking timestamp before bootstrap fails");
  check(estimatorRegression(1000.0, {{1060.0, 0.75}}, {60.0, 0.75, 60.0})
    .failures.empty(), "threshold equality passes");
  check(estimatorRegression(1000.0, {{1001.0, 0.8}}, {0.0, 1.0, 0.0})
    .failures.empty(), "explicit experimental thresholds are respected");
  check(estimatorRegression(1000.0, {{1001.0, 0.2}, {1080.0, 1.0}}, {60.0, 0.65, 30.0})
    .failures.empty(), "even-sized median averages the two middle errors");
  check(!estimatorRegression(1000.0, good, {-1.0, 1.0, 0.0}).failures.empty(),
    "negative limits fail");
  check(!estimatorRegression(1000.0, good, {0.0, nan, 0.0}).failures.empty(),
    "nonfinite limits fail");
  return 0;
}
