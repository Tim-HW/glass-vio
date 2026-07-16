#!/usr/bin/env bash
# Replay the KITTI bag through glassvio_node. Args are passed to the launch file,
# e.g. ./run_kitti.sh rate:=0.5 init_samples:=100
# No `set -u`: ROS's own setup.bash reads unbound vars (AMENT_TRACE_SETUP_FILES) and dies.
set -eo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "${HERE}/../.." && pwd)"

source /opt/ros/jazzy/setup.bash
if [ -f "${WS}/install/setup.bash" ]; then
  source "${WS}/install/setup.bash"
else
  echo "No ${WS}/install -- run: cd ${WS} && colcon build --packages-select glassvio" >&2
  exit 1
fi

# Launched by path, not `ros2 launch glassvio kitti.launch.py`: CMakeLists.txt does not
# install launch/ into the package share, so the name-based form would not resolve.
exec ros2 launch "${HERE}/launch/kitti.launch.py" "$@"
