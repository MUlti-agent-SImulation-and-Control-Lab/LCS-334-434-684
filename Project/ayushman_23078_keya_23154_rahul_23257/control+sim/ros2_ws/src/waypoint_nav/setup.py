# ~/waypoint_nav_ws/src/waypoint_nav/setup.py
from setuptools import setup
import os
from glob import glob

package_name = 'waypoint_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'path_publisher   = waypoint_nav.path_publisher:main',
            'waypoint_navigator = waypoint_nav.waypoint_navigator:main',
            'mpc_controller_node = waypoint_nav.mpc_controller_node:main',
            'odom_tf_broadcaster = waypoint_nav.odom_tf_broadcaster:main',
            'gps_covariance_fix = waypoint_nav.gps_covariance_fix:main',
        ],
    },
)