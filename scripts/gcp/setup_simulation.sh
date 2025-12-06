#!/bin/bash
# Setup catkin workspace with Kinova ROS and RealSense plugin
# Must be run after setup_ros.sh and setup_gazebo_deps.sh

set -e

echo "=========================================="
echo "  Setting Up Simulation Workspace"
echo "=========================================="

# Source ROS
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
else
    echo "ERROR: ROS Noetic not found. Please run setup_ros.sh first."
    exit 1
fi

# Create catkin workspace
echo "[1/5] Creating catkin workspace..."
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/src

# Clone kinova-ros (noetic-devel branch)
echo "[2/5] Cloning kinova-ros..."
if [ ! -d "kinova-ros" ]; then
    git clone -b noetic-devel https://github.com/Kinovarobotics/kinova-ros.git
else
    echo "kinova-ros already exists, pulling latest..."
    cd kinova-ros && git pull && cd ..
fi

# Clone RealSense Gazebo plugin (melodic-devel works with Noetic)
echo "[3/5] Cloning realsense_gazebo_plugin..."
if [ ! -d "realsense_gazebo_plugin" ]; then
    git clone -b melodic-devel https://github.com/pal-robotics/realsense_gazebo_plugin.git
else
    echo "realsense_gazebo_plugin already exists, pulling latest..."
    cd realsense_gazebo_plugin && git pull && cd ..
fi

# Clone RealSense Gazebo description (URDF models)
echo "[4/5] Cloning realsense_gazebo_description..."
if [ ! -d "realsense_gazebo_description" ]; then
    git clone https://github.com/m-tartari/realsense_gazebo_description.git
else
    echo "realsense_gazebo_description already exists, pulling latest..."
    cd realsense_gazebo_description && git pull && cd ..
fi

# Clone the project repository (contains kinova_llm_control package)
echo "[5/6] Cloning deep-research project..."
if [ ! -d "deep-research" ]; then
    git clone https://github.com/kkrishnan90/deep-research.git
    # Create symlink to the ROS package in catkin workspace
    ln -sf ~/catkin_ws/src/deep-research/ros_ws/src/kinova_llm_control ~/catkin_ws/src/kinova_llm_control
else
    echo "deep-research already exists, pulling latest..."
    cd deep-research && git pull && cd ..
fi

# Install dependencies with rosdep
echo "[6/6] Installing ROS dependencies..."
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y

# Initialize and configure catkin workspace
echo "Initializing catkin workspace..."
catkin init 2>/dev/null || true
catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release

# Build workspace
echo "Building catkin workspace..."
catkin build

# Add workspace to bashrc
if ! grep -q "source ~/catkin_ws/devel/setup.bash" ~/.bashrc; then
    echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
fi

# Source for current session
source ~/catkin_ws/devel/setup.bash

echo ""
echo "=========================================="
echo "  Simulation Workspace Setup Complete"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "  - kinova-ros (Kinova Mico arm simulation)"
echo "  - realsense_gazebo_plugin (RealSense camera plugin)"
echo "  - realsense_gazebo_description (Camera URDF models)"
echo ""
echo "To test the simulation:"
echo "  roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=m1n6s300"
echo "=========================================="
