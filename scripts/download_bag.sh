#!/usr/bin/env bash
#
# Fetch the EuRoC sequence the course's Labs are scored against, into data/.
#
#   ./scripts/download_bag.sh              download, convert to ROS 2, verify
#   ./scripts/download_bag.sh --force      redo even if the converted bag is here
#   ./scripts/download_bag.sh --keep-ros1  keep the original ROS 1 .bag afterwards
#
# ~1.2 GB downloaded, and about the same again converted, which is why data/ is
# gitignored rather than committed.
#
# Source: Burri et al. (2016), "The EuRoC micro aerial vehicle datasets",
#         IJRR 35(10). https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets
#         -- CC-BY-3.0.
#
# WHY V1_01_easy AND NOT KITTI: monocular scale reaches the estimator only through the
# accelerometer's non-gravity part, and a car at constant velocity has almost none. See the
# docstring in launch/euroc.launch.py for the measured numbers (KITTI: scale 108% wrong,
# wrong sign; EuRoC: 1.8%). The dataset is a constraint of the problem, not a preference.
#
# WHY A CONVERSION STEP: EuRoC ships ROS 1 bags; this is a ROS 2 package. `rosbags-convert`
# (pip package `rosbags`, pure Python, no ROS 1 install needed) does the translation. The
# node reads message stamps throughout and never asks the ROS clock for anything, so the
# absence of /clock in the converted bag costs nothing.
#
# Safe to re-run: if the converted bag is present and carries both topics, this does nothing.

set -euo pipefail

readonly URL="http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.bag"
readonly ROS1_BAG="V1_01_easy.bag"
readonly ROS2_DIR="V1_01_easy_ros2"

# The two topics the node actually subscribes to. /imu0 is the ADIS16448 the calibration in
# config/ describes -- the bag also carries the flight controller's IMU, a DIFFERENT sensor
# the calibration does not fit. See launch/euroc.launch.py.
readonly TOPIC_CAM="/cam0/image_raw"
readonly TOPIC_IMU="/imu0"

readonly REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Mirrors the upstream server's own layout, which is what launch/euroc.launch.py expects.
readonly DEST_DIR="${REPO_DIR}/data/vicon_room1/V1_01_easy"

FORCE=0
KEEP_ROS1=0
for arg in "$@"; do
  case "$arg" in
    --force)     FORCE=1 ;;
    --keep-ros1) KEEP_ROS1=1 ;;
    -h|--help)   sed -n '3,10p' "${BASH_SOURCE[0]}" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown option: $arg (try --help)" >&2; exit 2 ;;
  esac
done

log()  { printf '\033[1;34m[bag]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[bag]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[bag]\033[0m %s\n' "$*" >&2; exit 1; }

mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

# Integrity check. There is deliberately NO published checksum here: EuRoC does not ship one,
# and a checksum invented on a machine that never downloaded the file would fail for everyone.
# So we verify the property that actually matters instead -- that the converted bag carries
# the two topics the estimator subscribes to. A truncated download fails the conversion long
# before this, which is the failure mode this is really guarding against.
verify() {
  [[ -f "${ROS2_DIR}/metadata.yaml" ]] || return 1
  grep -q -- "$TOPIC_CAM" "${ROS2_DIR}/metadata.yaml" || return 1
  grep -q -- "$TOPIC_IMU" "${ROS2_DIR}/metadata.yaml" || return 1
}

if [[ "$FORCE" -eq 0 ]] && verify; then
  log "converted bag already present and carries both topics. Nothing to do."
  log "run:  ./run_euroc.sh"
  exit 0
fi

# --- fetch -----------------------------------------------------------------
# curl or wget, whichever exists. Both RESUME a partial transfer, which matters for a
# gigabyte on a flaky connection: an interrupted run is fixed by re-running this script,
# not by starting over.
if [[ "$FORCE" -eq 1 || ! -f "$ROS1_BAG" ]]; then
  if command -v curl >/dev/null 2>&1; then
    log "downloading ~1.2 GB from ETH Zurich (resumable -- re-run if interrupted)..."
    curl -L --fail --progress-bar -C - -o "$ROS1_BAG" "$URL" \
      || die "download failed. Re-run to resume from where it stopped."
  elif command -v wget >/dev/null 2>&1; then
    log "downloading ~1.2 GB from ETH Zurich (resumable -- re-run if interrupted)..."
    wget --continue --show-progress -O "$ROS1_BAG" "$URL" \
      || die "download failed. Re-run to resume from where it stopped."
  else
    die "need either curl or wget."
  fi
else
  log "ROS 1 bag already downloaded; converting it."
fi

# --- convert ---------------------------------------------------------------
command -v rosbags-convert >/dev/null 2>&1 || die \
  "need 'rosbags-convert' to translate the ROS 1 bag.
    pipx install rosbags        # or: pip install --user rosbags
  It is pure Python and needs no ROS 1 installation."

if [[ -d "$ROS2_DIR" ]]; then
  log "removing a previous ${ROS2_DIR}/ (rosbags-convert refuses an existing destination)"
  rm -rf "$ROS2_DIR"
fi

log "converting to ROS 2 (a few minutes)..."
rosbags-convert --src "$ROS1_BAG" --dst "$ROS2_DIR" \
  || die "conversion failed. If the download was interrupted the .bag is truncated -- re-run with --force."

# --- verify ----------------------------------------------------------------
verify || die \
  "converted bag is missing ${TOPIC_CAM} or ${TOPIC_IMU}. Re-run with --force; if it persists,
  check the upstream file has not changed layout."

if [[ "$KEEP_ROS1" -eq 0 ]]; then
  rm -f "$ROS1_BAG"
  log "removed the ROS 1 bag (keep it with --keep-ros1)"
fi

log "done. $(du -sh "$ROS2_DIR" | cut -f1) at ${DEST_DIR#"${REPO_DIR}/"}/${ROS2_DIR}"
log "run:  ./run_euroc.sh          (the node, live, with RViz)"
log "or:   ./build/glassvio/estimator_check   (the deterministic harness the Labs score)"
