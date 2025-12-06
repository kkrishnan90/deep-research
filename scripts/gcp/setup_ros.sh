#!/bin/bash
# Install ROS Noetic on Ubuntu 20.04
# Reference: http://wiki.ros.org/noetic/Installation/Ubuntu

set -e

echo "=========================================="
echo "  Installing ROS Noetic"
echo "=========================================="

# Check if ROS is already installed
if [ -d "/opt/ros/noetic" ]; then
    echo "ROS Noetic already installed at /opt/ros/noetic"
    source /opt/ros/noetic/setup.bash
    echo "ROS Version: $(rosversion -d)"
    exit 0
fi

# Setup locale
echo "[1/6] Setting up locale..."
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# Setup sources.list
echo "[2/6] Setting up ROS repository..."
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Setup keys
echo "[3/6] Setting up ROS keys..."
sudo apt-get install -y curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Update package index
echo "[4/6] Updating package index..."
sudo apt-get update

# Install ROS Noetic Desktop Full (includes Gazebo 11)
echo "[5/6] Installing ROS Noetic Desktop Full..."
echo "This includes: ROS, rqt, rviz, robot-generic libraries, 2D/3D simulators, navigation and 2D/3D perception"
sudo apt-get install -y ros-noetic-desktop-full

# Install additional ROS tools
echo "[6/6] Installing additional ROS tools..."
sudo apt-get install -y \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    python3-osrf-pycommon \
    build-essential

# Initialize rosdep
echo "Initializing rosdep..."
if [ ! -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then
    sudo rosdep init
fi
rosdep update

# Setup environment
echo "Setting up environment..."
if ! grep -q "source /opt/ros/noetic/setup.bash" ~/.bashrc; then
    echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
fi

# Source for current session
source /opt/ros/noetic/setup.bash

echo ""
echo "=========================================="
echo "  ROS Noetic Installation Complete"
echo "=========================================="
echo "ROS Version: $(rosversion -d)"
echo "Gazebo Version: $(gz --version 2>/dev/null | head -1 || echo 'Gazebo 11')"
echo ""
echo "Environment has been added to ~/.bashrc"
echo "Run 'source ~/.bashrc' or open a new terminal to use ROS"
echo "=========================================="
