// Self-check for the FAST+KLT FeatureTracker. The claim under test is the one thing the
// back-end will depend on: a feature keeps its ID as it flows across frames. If IDs churn
// every frame there is nothing stable to triangulate, and the reprojection factor has
// nothing to attach to.
//
// build+run (no framework, no ROS):
//   g++ -std=c++17 -I include test/test_feature_tracker.cpp \
//       $(pkg-config --cflags --libs opencv4) -o /tmp/tt && /tmp/tt
#include <cassert>
#include <cmath>
#include <cstdio>
#include <set>

#include <opencv2/imgproc.hpp>

#include "glassvio/feature_tracker.hpp"

// A real textured scene FAST can bite on, that genuinely TRANSLATES between frames: a fixed
// random-noise field, cropped through a window offset by shift_x. (FAST does not fire on a
// convex white corner -- no 9-pixel-contiguous arc -- so a grid of blobs detects nothing;
// broadband noise gives corners everywhere, and a cropped shift gives true correspondence.)
static cv::Mat makeScene(int shift_x)
{
  static const cv::Mat base = [] {
      cv::Mat b(480, 720, CV_8UC1);
      cv::randu(b, 0, 256);
      return b;
    }();
  return base(cv::Rect(shift_x, 0, 640, 480)).clone();
}

int main()
{
  glassvio::FeatureTracker tracker(1000, 150, 20);

  const auto a = tracker.track(makeScene(0));   // cold start: detect
  assert(a.points.size() == a.ids.size());
  assert(a.points.size() > 150 && "cold start should seed plenty of corners");

  // Shift the whole scene by 3px; KLT should follow, carrying IDs forward.
  const auto b = tracker.track(makeScene(3));
  assert(b.points.size() == b.ids.size());

  const std::set<long> ids_a(a.ids.begin(), a.ids.end());
  std::size_t survived = 0;
  for (long id : b.ids) {
    if (ids_a.count(id)) {
      ++survived;
    }
  }
  // Most of the first frame's IDs must still be alive after the flow step.
  assert(survived > a.ids.size() / 2 && "IDs must persist across a small motion");

  // IDs are unique within a frame (no duplicate landmark labels).
  const std::set<long> ids_b(b.ids.begin(), b.ids.end());
  assert(ids_b.size() == b.ids.size() && "IDs must be unique per frame");

  std::printf("ok: %zu seeded, %zu/%zu survived flow\n",
    a.ids.size(), survived, b.ids.size());

  // BELOW THE FLOOR, survivors must still FLOW. A floor no scene can meet keeps the count under
  // min_features every frame; the old re-seed branch then handed the survivors back at the
  // PREVIOUS frame's pixels -- 0 px of motion reported for a 3 px shift.
  glassvio::FeatureTracker starved(1000, 100000, 20);
  const auto c = starved.track(makeScene(0));
  const auto d = starved.track(makeScene(3));
  std::size_t moved = 0, common = 0;
  for (std::size_t i = 0; i < d.ids.size(); ++i) {
    for (std::size_t j = 0; j < c.ids.size(); ++j) {
      if (c.ids[j] == d.ids[i]) {
        ++common;
        moved += std::abs(d.points[i].x - (c.points[j].x - 3.0f)) < 0.5f;
        break;
      }
    }
  }
  assert(common > c.ids.size() / 2 && "IDs must persist below the floor too");
  assert(moved > common * 9 / 10 && "survivors below the floor must be FLOWED, not frozen");
  std::printf("ok: below the floor, %zu/%zu survivors followed the 3 px shift\n", moved, common);
  return 0;
}
