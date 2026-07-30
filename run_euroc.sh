#!/usr/bin/env bash
# Replay a EuRoC MAV sequence through glassvio_node, WITH RViz by default.
#   ./run_euroc.sh                       # V1_01_easy at 1x, RViz on
#   ./run_euroc.sh rate:=0.5             # slower, if the worker falls behind
#   ./run_euroc.sh rviz:=false           # headless (RViz competes for CPU; use this for numbers)
#   ./run_euroc.sh bag:=/path/to/other_ros2
#
# RViz shows the odom trajectory, the TF (gravity-aligned world frame), and the feature
# overlay. It is DEFAULT ON here because this script is the interactive/watch entry point; the
# launch file itself defaults RViz off, so a headless or CI `ros2 launch` stays clean.
#
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

# Default rviz:=true unless the caller already passed an rviz argument -- so `rviz:=false`
# still turns it off, and we never set it twice (ros2 launch rejects duplicate args).
RVIZ_ARG="rviz:=true"
for a in "$@"; do
  case "$a" in
    rviz:=*) RVIZ_ARG="" ;;
  esac
done

# Launched by path, not `ros2 launch glassvio euroc.launch.py`: CMakeLists.txt does not
# install launch/ into the package share, so the name-based form would not resolve.
exec ros2 launch "${HERE}/launch/euroc.launch.py" ${RVIZ_ARG} "$@"
