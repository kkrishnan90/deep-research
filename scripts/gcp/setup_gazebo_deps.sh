#!/bin/bash
# Install Gazebo dependencies and ROS control packages
# Must be run after setup_ros.sh

set -e

echo "=========================================="
echo "  Installing Gazebo and ROS Control"
echo "=========================================="

# Source ROS
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
else
    echo "ERROR: ROS Noetic not found. Please run setup_ros.sh first."
    exit 1
fi

# Update package index
echo "[1/2] Updating package index..."
sudo apt-get update

# Install Gazebo ROS packages and controllers
echo "[2/2] Installing ROS control and Gazebo packages..."
sudo apt-get install -y \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-ros-control \
    ros-noetic-effort-controllers \
    ros-noetic-position-controllers \
    ros-noetic-velocity-controllers \
    ros-noetic-joint-state-controller \
    ros-noetic-joint-trajectory-controller \
    ros-noetic-trac-ik-kinematics-plugin \
    ros-noetic-moveit \
    ros-noetic-moveit-ros-planning-interface \
    ros-noetic-moveit-ros-visualization \
    ros-noetic-moveit-simple-controller-manager \
    ros-noetic-ddynamic-reconfigure \
    ros-noetic-robot-state-publisher \
    ros-noetic-joint-state-publisher \
    ros-noetic-joint-state-publisher-gui \
    ros-noetic-xacro

echo ""
echo "=========================================="
echo "  Gazebo Dependencies Installed"
echo "=========================================="
echo ""
echo "Installed packages:"
echo "  - gazebo-ros-pkgs (Gazebo ROS integration)"
echo "  - ros-control (Robot control framework)"
echo "  - moveit (Motion planning)"
echo "  - Various controllers (effort, position, velocity, trajectory)"
echo "=========================================="
