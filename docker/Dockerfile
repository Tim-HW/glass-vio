# glassvio dev image.
#
# Pinned to ROS 2 Jazzy on Ubuntu Noble, which is what the project is developed and
# tested against. OpenCV and Eigen 3.4 come from that distro -- pinning the base
# image is what makes "works on my machine" reproducible for everyone else.
FROM ros:jazzy-ros-base

ARG DEBIAN_FRONTEND=noninteractive

# Runtime + build deps.
#
#   rviz2 / rosbag2      -- ros-base does not include either, and run_euroc.sh needs both
#   sqlite3 + mcap       -- rosbags-convert may emit either storage format depending on its
#                           version, and an unknown-storage failure reads as a corrupt bag
#   curl                 -- download_bag.sh
#   pipx                 -- likewise: Noble's Python is externally managed (PEP 668), so the
#                           `rosbags` converter cannot be installed with a bare pip
#   uncrustify           -- the lint the test suite actually runs
#
# The package-specific deps (pcl_conversions, tf2_eigen, ...) are installed below by
# rosdep, straight from package.xml -- so this list never drifts from the manifest.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      curl \
      wget \
      unzip \
      sudo \
      gdb \
      python3-colcon-common-extensions \
      python3-rosdep \
      pipx \
      ros-jazzy-rviz2 \
      ros-jazzy-rosbag2 \
      ros-jazzy-rosbag2-storage-default-plugins \
      ros-jazzy-rosbag2-storage-mcap \
      ros-jazzy-ament-uncrustify \
      ros-jazzy-ament-lint-auto \
      ros-jazzy-ament-lint-common \
      libgl1-mesa-dri \
      libglx-mesa0 \
      mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# Resolve the package's own dependencies FROM package.xml, so this image cannot drift
# out of sync with the manifest. Copying only the manifest keeps this layer cached
# across source edits.
WORKDIR /ws
COPY package.xml /tmp/glassvio/package.xml
RUN apt-get update \
    && rosdep update --rosdistro jazzy \
    && rosdep install --from-paths /tmp/glassvio --ignore-src -y --rosdistro jazzy \
    && rm -rf /var/lib/apt/lists/* /tmp/glassvio

# The `ubuntu` user (uid 1000) ships with the Noble base image and normally matches the
# host user, so files created in the bind-mounted workspace are not left root-owned.
RUN echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu \
    && chmod 0440 /etc/sudoers.d/ubuntu \
    && mkdir -p /ws && chown -R ubuntu:ubuntu /ws

# --- GUI environment ------------------------------------------------------------------
#
# Two distinct failures show up here, and they look alike but are not:
#
#   "QStandardPaths: XDG_RUNTIME_DIR not set"
#       Qt wants a private runtime dir. Harmless in isolation, but it is noise that
#       hides the real error below. Give it one, mode 0700 as Qt requires.
#
#   "MESA: failed to query drm device / failed to load driver: iris"
#       The container cannot see the host GPU (/dev/dri), so Mesa tries the hardware
#       driver, fails, and RViz has no GL at all. THE IMAGE CANNOT FIX THIS -- it is a
#       `docker run` decision. docker/run.sh passes the device through when the host has
#       one; the software fallback below is what makes RViz work when it does not.
ENV XDG_RUNTIME_DIR=/tmp/runtime-ubuntu
RUN mkdir -p /tmp/runtime-ubuntu && chown ubuntu:ubuntu /tmp/runtime-ubuntu \
    && chmod 0700 /tmp/runtime-ubuntu

# Old MIT-SHM breaks under X11 forwarding on some setups; costs nothing to disable.
ENV QT_X11_NO_MITSHM=1

USER ubuntu

# The ROS 1 -> ROS 2 bag converter scripts/download_bag.sh needs. Pure Python, no ROS 1
# install: EuRoC ships ROS 1 bags and this is a ROS 2 package. pipx rather than pip because
# Noble marks its Python externally managed and refuses a bare `pip install`.
RUN pipx install rosbags
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Auto-source ROS and, once it exists, this workspace's overlay -- so `ros2 run glassvio`
# and the `vio` shared library resolve without the user having to think about it.
RUN echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc \
    && echo '[ -f /ws/install/setup.bash ] && source /ws/install/setup.bash' >> ~/.bashrc \
    && echo 'cd /ws' >> ~/.bashrc

WORKDIR /ws
CMD ["/bin/bash"]
