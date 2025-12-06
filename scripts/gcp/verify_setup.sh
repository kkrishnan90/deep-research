#!/bin/bash
# Verify all components are properly installed
# Run this after master_setup.sh completes

set -e

echo "=========================================="
echo "  Verification Script"
echo "=========================================="

PASS_COUNT=0
FAIL_COUNT=0

# Function to check a component
check_component() {
    local name="$1"
    local command="$2"

    echo -n "[$name] "
    if eval "$command" &>/dev/null; then
        echo "PASS"
        ((PASS_COUNT++))
    else
        echo "FAIL"
        ((FAIL_COUNT++))
    fi
}

# Source ROS if available
if [ -f /opt/ros/noetic/setup.bash ]; then
    source /opt/ros/noetic/setup.bash
fi

if [ -f ~/catkin_ws/devel/setup.bash ]; then
    source ~/catkin_ws/devel/setup.bash
fi

echo ""
echo "--- System Checks ---"
check_component "NVIDIA Driver" "nvidia-smi"
check_component "GPU Memory" "nvidia-smi --query-gpu=memory.total --format=csv,noheader | grep -q 'MiB'"

echo ""
echo "--- ROS Checks ---"
check_component "ROS Noetic" "rosversion -d | grep -q noetic"
check_component "roscore" "which roscore"
check_component "roslaunch" "which roslaunch"

echo ""
echo "--- Gazebo Checks ---"
check_component "gzserver" "which gzserver"
check_component "gzclient" "which gzclient"
check_component "Gazebo ROS" "rospack find gazebo_ros"

echo ""
echo "--- Kinova Checks ---"
check_component "kinova_driver" "rospack find kinova_driver"
check_component "kinova_gazebo" "rospack find kinova_gazebo"
check_component "kinova_description" "rospack find kinova_description"
check_component "m1n6s300_moveit_config" "rospack find m1n6s300_moveit_config"

echo ""
echo "--- RealSense Checks ---"
check_component "realsense_gazebo_plugin" "rospack find realsense_gazebo_plugin"

echo ""
echo "--- MoveIt Checks ---"
check_component "moveit_commander" "rospack find moveit_commander"
check_component "moveit_ros_planning" "rospack find moveit_ros_planning"

echo ""
echo "--- Python Checks ---"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
fi
check_component "Python 3" "python3 --version"
check_component "google-genai" "python3 -c 'from google import genai'"
check_component "numpy" "python3 -c 'import numpy'"
check_component "opencv" "python3 -c 'import cv2'"

echo ""
echo "=========================================="
echo "  Results: $PASS_COUNT passed, $FAIL_COUNT failed"
echo "=========================================="

if [ $FAIL_COUNT -eq 0 ]; then
    echo ""
    echo "All checks passed! The system is ready."
    echo ""
    echo "To launch the simulation:"
    echo "  roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=m1n6s300"
    exit 0
else
    echo ""
    echo "Some checks failed. Please review the output above."
    exit 1
fi
