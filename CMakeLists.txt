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

# The node. For now it runs the IMU-initialization stage on the shared engine -- the same
# gravity/bias/world-frame bootstrap glasslio uses, which VIO needs just as much. The
# visual front-end (feature tracking + a reprojection factor folded into the same
# Gauss-Newton solver) is the next phase.
add_executable(glassvio_node src/glassvio_node.cpp)
target_include_directories(glassvio_node PRIVATE include)
ament_target_dependencies(glassvio_node
  rclcpp sensor_msgs geometry_msgs nav_msgs tf2_ros cv_bridge OpenCV)
target_link_libraries(glassvio_node glass_core::glass_core)

install(TARGETS glassvio_node
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
