#!/usr/bin/env bash
# Exercise fresh download, cached-bag GT repair and interrupted-transfer recovery
# without the multi-gigabyte network transfer or ROS conversion dependency.
set -euo pipefail
test_dir="$(mktemp -d)"
trap 'rm -rf "$test_dir"' EXIT
repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$test_dir/repo/scripts" "$test_dir/bin"
cp "$repo_dir/scripts/download_bag.sh" "$test_dir/repo/scripts/"
export FIXTURE="$test_dir/fixture.zip" CALL_LOG="$test_dir/calls"
export PATH="$test_dir/bin:$PATH"
python3 - <<'PY'
import os
import zipfile
with zipfile.ZipFile(os.environ['FIXTURE'], 'w') as archive:
    archive.writestr('mav0/state_groundtruth_estimate0/data.csv',
                     '#timestamp,states\n' + ','.join(['1'] * 17) + '\n')
PY
cat > "$test_dir/bin/curl" <<'SH'
#!/usr/bin/env bash
set -eu
printf 'fetch\n' >> "$CALL_LOG"
while [[ "$1" != '-o' ]]; do shift; done
output="$2"
url="$3"
if [[ "${FAIL_FETCH:-0}" == 1 ]]; then
  printf 'partial' > "$output"
  exit 22
fi
if [[ "$url" == *.zip ]]; then
  cp "$FIXTURE" "$output"
else
  printf 'bag' > "$output"
fi
SH
cat > "$test_dir/bin/rosbags-convert" <<'SH'
#!/usr/bin/env bash
set -eu
printf 'convert\n' >> "$CALL_LOG"
[[ -f "$2" ]]
mkdir -p "$4"
printf '/cam0/image_raw\n/imu0\n' > "$4/metadata.yaml"
SH
chmod +x "$test_dir/bin/"*
download="$test_dir/repo/scripts/download_bag.sh"
data="$test_dir/repo/data/vicon_room1/V1_01_easy"

bash "$download" > /dev/null
[[ -s "$data/gt/data.csv" && -f "$data/V1_01_easy_ros2/metadata.yaml" ]]
[[ ! -f "$data/V1_01_easy.bag" && ! -f "$data/V1_01_easy.zip" ]]
[[ "$(wc -l < "$CALL_LOG")" == 3 ]]
bash "$download" > /dev/null
[[ "$(wc -l < "$CALL_LOG")" == 3 ]]

# An existing sensor cache still needs GT; a failed fetch must not be reused as complete.
rm "$data/gt/data.csv"
if FAIL_FETCH=1 bash "$download" > /dev/null 2>&1; then
  echo 'interrupted download unexpectedly succeeded' >&2
  exit 1
fi
[[ -f "$data/V1_01_easy.zip.part" && ! -f "$data/V1_01_easy.zip" ]]
bash "$download" > /dev/null
[[ -s "$data/gt/data.csv" && ! -f "$data/V1_01_easy.zip.part" ]]
[[ "$(grep -c convert "$CALL_LOG")" == 1 ]]

# Force repairs both sources and preserves ROS 1 only when explicitly requested.
bash "$download" --force --keep-ros1 > /dev/null
[[ -s "$data/V1_01_easy.bag" && -s "$data/gt/data.csv" ]]
[[ "$(grep -c convert "$CALL_LOG")" == 2 ]]
echo 'dataset downloader checks passed'
