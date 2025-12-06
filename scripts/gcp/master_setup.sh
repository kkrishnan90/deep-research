#!/bin/bash
# Master setup script for Kinova Mico ROS Simulation
# Run this after creating the GCP instance
# Usage: ./master_setup.sh [--skip-reboot]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SKIP_REBOOT=false

# Parse arguments
if [ "$1" == "--skip-reboot" ]; then
    SKIP_REBOOT=true
fi

echo "=========================================="
echo "  Kinova Mico ROS Simulation Setup"
echo "=========================================="
echo "Script directory: $SCRIPT_DIR"
echo "=========================================="

# Step 1: Update system
echo ""
echo "[1/8] Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Step 2: Check NVIDIA drivers
echo ""
echo "[2/8] Checking NVIDIA drivers..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA drivers not found. Installing..."
    bash "$SCRIPT_DIR/setup_nvidia.sh"
    if [ "$SKIP_REBOOT" = false ]; then
        echo ""
        echo "=========================================="
        echo "  REBOOT REQUIRED"
        echo "=========================================="
        echo "Run this script again after reboot:"
        echo "  ./master_setup.sh --skip-reboot"
        exit 0
    fi
else
    echo "NVIDIA drivers already installed:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi

# Step 3: Verify GPU
echo ""
echo "[3/8] Verifying GPU..."
nvidia-smi || {
    echo "ERROR: GPU not accessible. Please check driver installation."
    exit 1
}

# Step 4: Install ROS Noetic
echo ""
echo "[4/8] Installing ROS Noetic..."
bash "$SCRIPT_DIR/setup_ros.sh"

# Source ROS for subsequent steps
source /opt/ros/noetic/setup.bash

# Step 5: Install Gazebo dependencies
echo ""
echo "[5/8] Installing Gazebo and ROS Control..."
bash "$SCRIPT_DIR/setup_gazebo_deps.sh"

# Step 6: Setup simulation packages
echo ""
echo "[6/8] Setting up simulation workspace..."
bash "$SCRIPT_DIR/setup_simulation.sh"

# Step 7: Setup Python environment
echo ""
echo "[7/8] Setting up Python environment..."
bash "$SCRIPT_DIR/setup_python.sh"

# Step 8: Setup Chrome Remote Desktop
echo ""
echo "[8/8] Setting up Chrome Remote Desktop..."
bash "$SCRIPT_DIR/setup_desktop.sh"

echo ""
echo "=========================================="
echo "       SETUP COMPLETE!"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Complete Chrome Remote Desktop setup:"
echo "   https://remotedesktop.google.com/headless"
echo ""
echo "2. Connect via Chrome Remote Desktop:"
echo "   https://remotedesktop.google.com/access"
echo ""
echo "3. Open a terminal and test the simulation:"
echo "   source ~/catkin_ws/devel/setup.bash"
echo "   roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=m1n6s300"
echo ""
echo "4. Run the verification script:"
echo "   bash $SCRIPT_DIR/verify_setup.sh"
echo ""
echo "=========================================="
