FROM osrf/ros:jazzy-desktop-full

# Install pip
RUN apt-get update && apt-get install -y python3-pip

# Install PyTorch
RUN pip3 install torch --index-url https://download.pytorch.org/whl/cu121 --break-system-packages

# Install CasADi
RUN pip3 install casadi --break-system-packages

# Optional useful ROS tools
RUN apt-get install -y python3-colcon-common-extensions

# Install ROS Gazebo integration
RUN apt-get install -y ros-jazzy-ros-gz

# Install SLAM Toolbox
RUN apt-get install -y ros-jazzy-slam-toolbox

# Install PointCloud to LaserScan
RUN apt-get install -y ros-jazzy-pointcloud-to-laserscan

RUN apt-get install -y ros-jazzy-tf-transformations
RUN apt-get install -y ros-jazzy-topic-tools
RUN apt-get install -y ros-jazzy-robot-localization

# Environment variables
ENV LIBGL_ALWAYS_SOFTWARE=0
ENV GZ_IP=127.0.0.1

COPY /ros2_ws /root/ros2_ws/

RUN /bin/bash -c "source /opt/ros/jazzy/setup.bash && \
    cd /root/ros2_ws && \
    rosdep install -r --from-paths src -i -y && \
    colcon build --symlink-install"

RUN echo "source /opt/ros/jazzy/setup.bash" >> /root/.bashrc && \
    echo "source /root/ros2_ws/install/setup.bash" >> /root/.bashrc
WORKDIR /root/ros2_ws
