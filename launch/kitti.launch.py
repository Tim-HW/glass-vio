"""
Replay the KITTI bag in data/ through glassvio_node.

KITTI is not Livox, and the node's defaults are Livox-shaped. Two of them are not
cosmetic -- get either wrong and the node fails silently rather than loudly:

  * accel_in_g=False : the Livox driver reports linear_acceleration in g, KITTI's OXTS
    reports m/s^2 (measured |a| over this bag: 9.843). Leaving the default True scales
    gravity to ~96 m/s^2 and the gravity-aligned world frame is garbage.
  * init.num_samples=200 : this one is a COUNT, not a duration, and that is the trap.
    200 is 1 s at Livox's 200 Hz; it was 20 s on the old 9.7 Hz OXTS bag (which contained
    no such static window, so the node waited forever, and this file used to say 50).
    On the 100 Hz re-export, 200 is ~2.2 s and correct again. Re-derive it if the IMU
    rate ever changes: at 50 samples (0.55 s) the motion check accepts 91 of 126 windows
    and has effectively stopped checking; at 200 it accepts 15 and rejects 16.

CAVEAT that no window length fixes: ImuInit assumes STATIC, and this car never stops --
it starts the bag at 5.05 m/s. A vehicle cruising in a straight line passes both motion
checks at any window length, so the accepted window is quiet motion, not rest, and some
real rotation lands in the gyro bias. Worse, NavState.v is left at zero. That is the
initialiser's problem, not the integrator's -- see src/imu_dead_reckon_check.cpp.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

BAG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'data', 'kitti', 'kitti_0117_faithful',
)


def generate_launch_description():
    bag = LaunchConfiguration('bag')
    rate = LaunchConfiguration('rate')
    # Real IMUs need tuning the dataset can't tell you about -- KITTI 0117 is a moving
    # vehicle, so the "static" window is only ever approximately static. Keep the knob.
    # value_type is load-bearing: a substitution is a string, and the node declares these
    # as int/double -- an unwrapped LaunchConfiguration throws InvalidParameterType.
    num_samples = ParameterValue(LaunchConfiguration('init_samples'), value_type=int)
    max_gyro = ParameterValue(LaunchConfiguration('init_max_gyro'), value_type=float)

    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value=BAG),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument('init_samples', default_value='200'),
        DeclareLaunchArgument('init_max_gyro', default_value='0.1'),
        Node(
            package='glassvio',
            executable='glassvio_node',
            name='glassvio_node',
            output='screen',
            parameters=[{
                'use_sim_time': True,
                'imu_topic': '/kitti/imu',
                'image_topic': '/kitti/camera/image_raw',
                'imu.accel_in_g': False,
                # This car never stops, so ImuInit's "static" window is cruising, and the
                # gyro mean it returns is the vehicle's yaw rate (bg_z = 0.0188 rad/s), not
                # a bias. Using it is 3.9x worse against ground truth than using zero --
                # measured, see src/gyro_bias_check.cpp.
                'imu.init.trust_gyro_bias': False,
                'imu.init.num_samples': num_samples,
                'imu.init.max_gyro': max_gyro,
            }],
        ),
        # --clock drives use_sim_time above; the bag carries 1299 /clock messages.
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag, '--clock', '--rate', rate],
            output='screen',
        ),
    ])
