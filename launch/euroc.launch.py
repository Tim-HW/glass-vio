"""
Replay a EuRoC MAV sequence in data/ through glassvio_node.

WHY EuRoC AND NOT KITTI. Monocular scale reaches the estimator only through the
accelerometer's NON-GRAVITY part, and a car cruising at constant velocity has almost none:
on KITTI 0117 scale came out at s = -0.21 against a truth of +2.43 (108% wrong, wrong sign),
while gravity was recovered to 0.8%. Not a bug -- s and v_0 slide along a valley of equal
cost when there is no excitation. A hand-held MAV shakes the IMU constantly, and the same
code then recovers s to 1.8%. The dataset was the constraint, not the estimator.

THE BAG IS NOT PLAYED WITH --clock HERE. EuRoC's stamps are 2014 wall-clock; the node uses
message stamps throughout and never asks the ROS clock for anything that matters, so sim time
buys nothing and the /clock topic does not exist in the converted bag.
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BAG = os.path.join(HERE, 'data', 'vicon_room1', 'V1_01_easy', 'V1_01_easy_ros2')
CALIB = os.path.join(HERE, 'config')
RVIZ = os.path.join(HERE, 'rviz', 'glassvio.rviz')


def generate_launch_description():
    bag = LaunchConfiguration('bag')
    rate = LaunchConfiguration('rate')
    # value_type is load-bearing: a substitution is a string, and the node declares this as an
    # int -- an unwrapped LaunchConfiguration throws InvalidParameterType.
    bootstrap = ParameterValue(LaunchConfiguration('bootstrap_frames'), value_type=int)

    return LaunchDescription([
        DeclareLaunchArgument('bag', default_value=BAG),
        DeclareLaunchArgument('rate', default_value='1.0'),
        DeclareLaunchArgument(
            'rviz', default_value='false',
            description='Open RViz with rviz/glassvio.rviz (odom trajectory, TF, feature '
                        'overlay). Off by default: it competes for CPU with a worker that '
                        'already drops frames during bootstrap attempts.'),
        DeclareLaunchArgument(
            'bootstrap_frames', default_value='120',
            description='Frames collected before attempting the bootstrap. Must span real '
                        'translation: a rotating MAV gives stage [2] no baseline, and the '
                        'window slides on until it finds one.'),
        Node(
            package='glassvio',
            executable='glassvio_node',
            name='glassvio_node',
            output='screen',
            parameters=[{
                # /imu0 is the ADIS16448 the calibration describes. The bag also carries
                # /fcu/imu -- the flight controller's, a DIFFERENT sensor the calibration does
                # not fit.
                'imu_topic': '/imu0',
                'image_topic': '/cam0/image_raw',
                'calib_dir': CALIB,
                'bootstrap_frames': bootstrap,
                'world_frame': 'odom',
                'body_frame': 'imu',
            }],
        ),
        ExecuteProcess(
            cmd=['ros2', 'bag', 'play', bag, '--rate', rate],
            output='screen',
        ),
        # RViz is OPTIONAL and off by default: it is a second process competing for the CPU,
        # and the worker already falls behind on a bootstrap attempt (which drops frames --
        # see the queue's splice). Turn it on when you want to look, not when you want numbers.
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', RVIZ],
            output='log',
            condition=IfCondition(LaunchConfiguration('rviz')),
        ),
    ])
