#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace glassvio
{
struct EstimatorRegressionLimits
{
  double min_tracked_seconds = 60.0;
  double max_median_error = 0.75;
  double min_below_1m_seconds = 30.0;
};

struct EstimatorRegressionSample
{
  double t;
  double position_error;
};

struct EstimatorRegressionVerdict
{
  double tracked_seconds = 0.0;
  double median_error = std::numeric_limits<double>::quiet_NaN();
  double below_1m_seconds = 0.0;
  std::vector<std::string> failures;
};

inline EstimatorRegressionVerdict estimatorRegression(
  double bootstrap_time, const std::vector<EstimatorRegressionSample> & samples,
  const EstimatorRegressionLimits & limits = {})
{
  EstimatorRegressionVerdict result;
  for (double limit : {limits.min_tracked_seconds, limits.max_median_error,
      limits.min_below_1m_seconds})
  {
    if (!std::isfinite(limit) || limit < 0.0) {
      result.failures.push_back("limits must be finite and nonnegative");
      return result;
    }
  }
  if (!std::isfinite(bootstrap_time) || samples.empty()) {
    result.failures.push_back("no bootstrap or no tracked ground-truth samples");
    return result;
  }
  std::vector<double> errors;
  double previous_time = bootstrap_time;
  bool below_1m = true;
  for (const auto & sample : samples) {
    if (!std::isfinite(sample.t) || sample.t <= previous_time ||
      !std::isfinite(sample.position_error) || sample.position_error < 0.0)
    {
      result.failures.push_back("nonfinite error or invalid tracking timestamp");
      return result;
    }
    previous_time = sample.t;
    errors.push_back(sample.position_error);
    below_1m = below_1m && sample.position_error <= 1.0;
    if (below_1m) {
      result.below_1m_seconds = sample.t - bootstrap_time;
    }
  }
  result.tracked_seconds = samples.back().t - bootstrap_time;
  std::sort(errors.begin(), errors.end());
  result.median_error = errors[errors.size() / 2];
  if (errors.size() % 2 == 0) {
    result.median_error = 0.5 * errors[errors.size() / 2 - 1] + 0.5 * result.median_error;
  }
  if (result.tracked_seconds < limits.min_tracked_seconds) {
    result.failures.push_back("tracked duration is below the minimum");
  }
  if (result.median_error > limits.max_median_error) {
    result.failures.push_back("median position error exceeds the maximum");
  }
  if (result.below_1m_seconds < limits.min_below_1m_seconds) {
    result.failures.push_back("duration before the first position error above 1 m is too short");
  }
  return result;
}
}  // namespace glassvio
