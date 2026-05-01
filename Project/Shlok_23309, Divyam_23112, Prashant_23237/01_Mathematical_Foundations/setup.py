import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'lcs'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', [
            'launch/standard_mpc.launch.py',
            'launch/tube_mpc.launch.py',
            'launch/et_tube_mpc.launch.py',
            'launch/crazyswarm_bridge.launch.py',
            'launch/waypoint_publisher.launch.py',
            'launch/simple_gazebo.launch.py',
        ]),
        ('share/' + package_name + '/urdf', ['urdf/simple_drone.urdf']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ptc',
    maintainer_email='ptc@todo.todo',
    description='MPC Controller Nodes for Gazebo/Crazyswarm Integration',
    license='MIT',
    entry_points={
        'console_scripts': [
            'standard_mpc_node = lcs.standard_mpc_node:main',
            'tube_mpc_node = lcs.tube_mpc_node:main',
            'et_tube_mpc_node = lcs.et_tube_mpc_node:main',
            'crazyswarm_bridge_node = lcs.crazyswarm_bridge_node:main',
            'waypoint_publisher_node = lcs.waypoint_publisher_node:main',
            'gazebo_bridge_node = lcs.gazebo_bridge_node:main',
        ],
    },
)
