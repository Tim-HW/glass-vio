cmake_minimum_required(VERSION 3.8)
project(glassvio)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosbag2_cpp REQUIRED)

# The shared estimation engine. It is NOT a colcon package -- it is a pure-CMake
# subproject that physically lives in the glass-lio repo (glasslio/glass_core). glassvio
# is a sibling checkout, so we reach into that tree and add_subdirectory it, exactly as
# glasslio itself does. This is the whole point of glass_core: two front-ends, one engine,
# no copy-paste of Jacobians.
set(GLASS_CORE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../glasslio/glass_core")
if(NOT EXISTS "${GLASS_CORE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR
    "glass_core not found at ${GLASS_CORE_DIR}.\n"
    "glassvio shares the glass-lio repo's engine -- check out glass-lio beside glassvio "
    "so the path src/glasslio/glass_core exists.")
endif()
add_subdirectory("${GLASS_CORE_DIR}" glass_core)

# THE PIPELINE, built once as a library so the node and every check tool share the same
# objects rather than recompiling them per target -- the same arrangement glasslio uses for
# its `lio` library, and for the same reason: a stage that is compiled twice is a stage that
# can be tested in one build and shipped from the other.
add_library(vio
  src/vio/vio_initializer.cpp
  src/vio/visual_registration.cpp
  src/vio/vio_estimator.cpp
  src/vio/landmark_map.cpp)
target_include_directories(vio PUBLIC include)
ament_target_dependencies(vio
  rclcpp sensor_msgs geometry_msgs rosbag2_cpp cv_bridge OpenCV)
target_link_libraries(vio glass_core::glass_core)

# The node. Stages [2]-[4] live in VioInitializer and know nothing about ROS; the node
# parses parameters, subscribes, and publishes.
add_executable(glassvio_node src/glassvio_node.cpp)
target_include_directories(glassvio_node PRIVATE include)
ament_target_dependencies(glassvio_node
  rclcpp sensor_msgs geometry_msgs nav_msgs tf2_ros cv_bridge OpenCV)
target_link_libraries(glassvio_node vio)

# =============================================================================
# src/checks -- THE GATES. One executable per pipeline stage, each scoring exactly that stage
# against an independent oracle (ground truth, or gravity's own magnitude).
#
# These are NOT the node, and NOT unit tests. They cannot be unit tests: each needs a real
# bag, which is not in the repo, so ctest could never run them. And they answer a different
# question than a unit test does -- glass_core's suites pin the MATHS against finite
# differences; these ask whether the DATA can support it. Both failed at least once here:
# the maths held and KITTI's cruising car still made scale unobservable.
#
# Each exits non-zero when its stage stops holding, so they are runnable regression gates:
#
#   imu_dead_reckon_check   [1] can the IMU dead-reckon at all?      -> 0.24 m / 2 s
#   epipolar_check          [-] are K and T_cam_imu right?           -> 0.26 px Sampson
#   gyro_bias_check         [3] bias from vision, no scale needed    -> beats ImuInit 3.9x
#   sfm_check               [2] is the reconstruction rigid?         -> 1.06% after one scale
#   vi_align_check          [4] does the IMU supply the metre?       -> gravity yes, scale no
# =============================================================================
add_executable(imu_dead_reckon_check src/checks/imu_dead_reckon_check.cpp)
target_include_directories(imu_dead_reckon_check PRIVATE include)
ament_target_dependencies(imu_dead_reckon_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)
target_link_libraries(imu_dead_reckon_check vio)

# Phase-2 gate: prove K and T_cam_imu against ground-truth epipolar geometry before the
# essential matrix and the alignment get built on top of them.
add_executable(epipolar_check src/checks/epipolar_check.cpp)
target_include_directories(epipolar_check PRIVATE include)
ament_target_dependencies(epipolar_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)
# Links glass_core: kitti_bag.hpp preintegrates, so every tool now shares that dependency.
target_link_libraries(epipolar_check vio)

# Phase 2b-1: gyro bias from the essential matrix's rotations. The first initialisation
# stage that is observable, because rotation needs no scale.
add_executable(gyro_bias_check src/checks/gyro_bias_check.cpp)
target_include_directories(gyro_bias_check PRIVATE include)
ament_target_dependencies(gyro_bias_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)
target_link_libraries(gyro_bias_check vio)

# Phase 2b-2: up-to-scale SfM. Invents a ruler at the base pair and propagates it with PnP.
add_executable(sfm_check src/checks/sfm_check.cpp)
target_include_directories(sfm_check PRIVATE include)
ament_target_dependencies(sfm_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)

# Phase 2b-3: the linear visual-inertial alignment. Recovers the metre.
add_executable(vi_align_check src/checks/vi_align_check.cpp)
target_include_directories(vi_align_check PRIVATE include)
ament_target_dependencies(vi_align_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)
target_link_libraries(vi_align_check vio)
target_link_libraries(sfm_check vio)

# THE GATE FOR THE THESIS: bootstrap, then solve frames with the reprojection factor and the
# IMU in ONE NormalEquationsN<15>. Also the honest test of whether tight coupling buys
# anything -- argv[4] is imu_prior_weight, and 0 makes it pure vision.
add_executable(vio_check src/checks/vio_check.cpp)
target_include_directories(vio_check PRIVATE include)
ament_target_dependencies(vio_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)
target_link_libraries(vio_check vio)

# Deterministic offline drive of the real VioEstimator -- splits "the node fails" into
# whether bootstrap()'s LOGIC or the online plumbing is at fault.
add_executable(estimator_check src/checks/estimator_check.cpp)
target_include_directories(estimator_check PRIVATE include)
ament_target_dependencies(estimator_check rosbag2_cpp sensor_msgs geometry_msgs cv_bridge OpenCV)
target_link_libraries(estimator_check vio)

install(TARGETS vio glassvio_node imu_dead_reckon_check epipolar_check gyro_bias_check sfm_check vi_align_check vio_check estimator_check
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()

  # Plain assert-based self-checks, matching glasslio and glass_core: no test framework.
  #
  # CRITICAL: -UNDEBUG. This project defaults to CMAKE_BUILD_TYPE=Release (line 8), and Release
  # defines NDEBUG, which compiles every assert() to NOTHING. Without this, these suites run,
  # print "ok", exit 0, and verify ABSOLUTELY NOTHING -- which is exactly what happened in
  # glasslio until it was caught (see its doc/testing.md: one of three landmines that cost real
  # time, all of which produced plausible output rather than an error).
  foreach(t test_feature_tracker test_reprojection)
    add_executable(${t} test/${t}.cpp)
    target_include_directories(${t} PRIVATE include)
    target_compile_options(${t} PRIVATE -UNDEBUG)
    target_link_libraries(${t} ${OpenCV_LIBS} glass_core::glass_core)
    add_test(NAME ${t} COMMAND ${t})
  endforeach()
endif()

ament_package()
