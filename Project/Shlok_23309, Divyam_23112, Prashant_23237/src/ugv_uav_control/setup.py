import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'ugv_uav_control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        
        # Ground Plane Model
        (os.path.join('share', package_name, 'models/path_ground_plane_v2'), glob('models/path_ground_plane_v2/model.*')),
        (os.path.join('share', package_name, 'models/path_ground_plane_v2/materials/scripts'), glob('models/path_ground_plane_v2/materials/scripts/*')),
        (os.path.join('share', package_name, 'models/path_ground_plane_v2/materials/textures'), glob('models/path_ground_plane_v2/materials/textures/*')),
        (os.path.join('share', package_name, 'models/path_ground_plane_v2'), glob('models/path_ground_plane_v2/*.png')),
        
        # NEW: Custom Robot Model (This was missing!)
        (os.path.join('share', package_name, 'models/turtlebot3_burger_aruco'), glob('models/turtlebot3_burger_aruco/model.*')),
        (os.path.join('share', package_name, 'models/turtlebot3_burger_aruco/materials/textures'), glob('models/turtlebot3_burger_aruco/materials/textures/*')),

        # NEW: X500 Drone Model
        (os.path.join('share', package_name, 'models/x500_downward_camera'), glob('models/x500_downward_camera/model.*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='shlok-mehndiratta',
    maintainer_email='shlokm23@iiserb.ac.in',
    description='UGV UAV Control Project',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'middle_path_follower = ugv_uav_control.middle_path_follower:main',
            'stanley_node = ugv_uav_control.stanley_node:main',
            'pure_pursuit_node = ugv_uav_control.pure_pursuit_node:main',
            'mpc_node = ugv_uav_control.mpc_node:main', 
            'ekf_node = ugv_uav_control.ekf_node:main',
            'belief_mpc_node = ugv_uav_control.belief_mpc_node:main',
            'uav_mpc_node = ugv_uav_control.uav_mpc_node:main',
            'uav_ancillary_node = ugv_uav_control.uav_ancillary_node:main',
            'experiment_logger = ugv_uav_control.experiment_logger:main',
        ],
    },
)
