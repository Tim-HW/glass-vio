#ifndef GLASSVIO_CAMERA_CALIB_HPP
#define GLASSVIO_CAMERA_CALIB_HPP

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>

namespace glassvio
{

/// The only calibration the pipeline needs: how the camera projects, where it sits relative
/// to the IMU, and how much the IMU is trusted.
///
/// ONE STRUCT, TWO LOADERS (KITTI and EuRoC), because SfmWindow and the initialiser must not
/// care which dataset they are looking at -- the moment they do, every stage needs a second
/// copy and the copies drift.
///
/// DISTORTION IS NOT OPTIONAL HERE, unlike on KITTI. KITTI ships images already rectified, so
/// P_rect_00's intrinsics ARE the projection and there is nothing to undo. EuRoC ships raw
/// frames with a radial-tangential model whose leading coefficient is -0.283 -- a strong
/// barrel. Ignore it and a corner feature's ray is tens of pixels off, which the essential
/// matrix will happily absorb into a slightly wrong rotation rather than reject.
struct CameraCalib
{
  Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
  /// X_cam = T_cam_imu * X_imu. NOTE THE DIRECTION: EuRoC's Kalibr file stores the INVERSE
  /// of this ("T_imu_cam: rotation from camera to IMU"), and the two compile identically.
  Eigen::Isometry3d T_cam_imu = Eigen::Isometry3d::Identity();
  /// radtan [k1, k2, p1, p2]. All zeros for an already-rectified stream (KITTI).
  Eigen::Vector4d distortion = Eigen::Vector4d::Zero();

  /// Continuous-time noise DENSITIES, straight from the sensor's datasheet where the dataset
  /// provides them. These set how much the IMU is trusted against the camera; we were
  /// guessing on KITTI, and EuRoC states them.
  double gyro_noise = 1.0e-3;    ///< rad/s/sqrt(Hz)
  double accel_noise = 1.0e-2;   ///< m/s^2/sqrt(Hz)
  double gyro_random_walk = 1.0e-5;
  double accel_random_walk = 1.0e-3;

  int width = 0;
  int height = 0;

  double fx() const {return K(0, 0);}
  double fy() const {return K(1, 1);}
  double cx() const {return K(0, 2);}
  double cy() const {return K(1, 2);}

  bool rectified() const {return distortion.isZero();}

  cv::Mat cvK() const
  {
    return  cv::Mat_<double>(3, 3) <<
           K(0, 0), 0, K(0, 2), 0, K(1, 1), K(1, 2), 0, 0, 1;
  }

  cv::Mat cvDistortion() const
  {
    return  cv::Mat_<double>(1, 4) <<
           distortion(0), distortion(1), distortion(2), distortion(3);
  }

  /// Map distorted pixels to the pixels an ideal pinhole with this K would have produced.
  ///
  /// POINTS, NOT IMAGES, and that is deliberate. KLT is a LOCAL search and distortion is a
  /// smooth warp, so tracking on the raw frame is correct and costs nothing. Only the
  /// geometry -- the essential matrix, triangulation, PnP -- needs true rays. Undistorting
  /// ~800 points beats remapping a 752x480 frame every time, and it leaves the tracker
  /// looking at the pixels the sensor actually produced.
  ///
  /// A no-op on an already-rectified stream, so callers need not ask which dataset they have.
  std::vector<cv::Point2f> undistort(const std::vector<cv::Point2f> & pts) const
  {
    if (pts.empty() || rectified()) {
      return pts;
    }
    std::vector<cv::Point2f> out;
    // P = K, so what comes back is PIXELS in this same K rather than normalised coordinates.
    // Everything downstream then treats the camera as a clean pinhole and never learns that
    // distortion existed.
    cv::undistortPoints(pts, out, cvK(), cvDistortion(), cv::noArray(), cvK());
    return out;
  }
};

namespace detail
{

/// Read a file, drop `#` comments, return the text between `from:` and `to:` (or EOF).
inline std::string section(
  const std::string & path, const std::string & from, const std::string & to)
{
  std::ifstream in(path);
  if (!in) {
    throw std::runtime_error("cannot open " + path);
  }
  std::string out, line;
  bool inside = false;
  while (std::getline(in, line)) {
    const auto hash = line.find('#');
    if (hash != std::string::npos) {
      line.erase(hash);
    }
    if (line.rfind(from + ":", 0) == 0) {
      inside = true;
      continue;
    }
    if (inside && !to.empty() && line.rfind(to + ":", 0) == 0) {
      break;
    }
    if (inside) {
      out += line + "\n";
    }
  }
  return out;
}

/// Every number following `key:`, stopping at the next key.
///
/// Kalibr's dialect nests a matrix as `key:` then four `- [a, b, c, d]` lines, and writes
/// scalars as `key: [a, b, c]` on one line. Both are "the numbers after the key, until a line
/// with another colon" -- which is the whole rule, and why this needs no YAML library for
/// four fixed-arity fields.
inline std::vector<double> numbersAfter(
  const std::string & text, const std::string & key, std::size_t expect)
{
  std::istringstream in(text);
  std::string line;
  std::vector<double> out;
  bool started = false;

  while (std::getline(in, line)) {
    const auto pos = line.find(key + ":");
    if (!started) {
      if (pos == std::string::npos) {
        continue;
      }
      started = true;
      line = line.substr(pos + key.size() + 1);   // the remainder of the key's own line
    } else if (line.find(':') != std::string::npos) {
      break;                                      // a new key ends the block
    }

    for (char & c : line) {
      if (c == '[' || c == ']' || c == ',') {
        c = ' ';
      }
    }
    // A leading '-' is YAML's list bullet, not a minus, only when followed by a space.
    for (std::size_t i = 0; i + 1 < line.size(); ++i) {
      if (line[i] == '-' && line[i + 1] == ' ') {
        line[i] = ' ';
      }
    }
    std::istringstream ss(line);
    std::string tok;
    while (ss >> tok) {
      try {
        out.push_back(std::stod(tok));
      } catch (...) {
        // A non-numeric value (`pinhole`, `/cam0/image_raw`) is not ours; ignore it.
      }
    }
    if (out.size() >= expect) {
      break;
    }
  }
  if (out.size() < expect) {
    throw std::runtime_error("calib key '" + key + "' has too few values");
  }
  out.resize(expect);
  return out;
}

}  // namespace detail

/// Load EuRoC's Kalibr chain files: `kalibr_imucam_chain.yaml` + `kalibr_imu_chain.yaml`.
inline CameraCalib loadEurocCalib(const std::string & dir, const std::string & cam = "cam0")
{
  CameraCalib c;

  const std::string cam_text =
    detail::section(dir + "/kalibr_imucam_chain.yaml", cam, cam == "cam0" ? "cam1" : "");

  const auto intr = detail::numbersAfter(cam_text, "intrinsics", 4);   // fu, fv, cu, cv
  c.K << intr[0], 0, intr[2], 0, intr[1], intr[3], 0, 0, 1;

  const auto dist = detail::numbersAfter(cam_text, "distortion_coeffs", 4);
  c.distortion << dist[0], dist[1], dist[2], dist[3];

  const auto res = detail::numbersAfter(cam_text, "resolution", 2);
  c.width = static_cast<int>(res[0]);
  c.height = static_cast<int>(res[1]);

  // THE DIRECTION TRAP. Kalibr's `T_imu_cam` is X_imu = T * X_cam ("rotation from camera to
  // IMU, position of camera in IMU"). CameraCalib::T_cam_imu is the OTHER WAY. Both are 4x4
  // rigid transforms, so using the wrong one compiles, runs, and quietly puts the camera
  // 2 cm and a few degrees from where it is.
  const auto T = detail::numbersAfter(cam_text, "T_imu_cam", 16);
  Eigen::Matrix4d M;
  M << T[0], T[1], T[2], T[3],
    T[4], T[5], T[6], T[7],
    T[8], T[9], T[10], T[11],
    T[12], T[13], T[14], T[15];
  Eigen::Isometry3d T_imu_cam = Eigen::Isometry3d::Identity();
  T_imu_cam.linear() = M.topLeftCorner<3, 3>();
  T_imu_cam.translation() = M.topRightCorner<3, 1>();
  c.T_cam_imu = T_imu_cam.inverse();

  // Datasheet noise, stated by the dataset rather than guessed. These are what weigh the IMU
  // against the camera in the solver.
  const std::string imu_text = detail::section(dir + "/kalibr_imu_chain.yaml", "imu0", "");
  c.gyro_noise = detail::numbersAfter(imu_text, "gyroscope_noise_density", 1)[0];
  c.accel_noise = detail::numbersAfter(imu_text, "accelerometer_noise_density", 1)[0];
  c.gyro_random_walk = detail::numbersAfter(imu_text, "gyroscope_random_walk", 1)[0];
  c.accel_random_walk = detail::numbersAfter(imu_text, "accelerometer_random_walk", 1)[0];

  return c;
}

}  // namespace glassvio

#endif  // GLASSVIO_CAMERA_CALIB_HPP
