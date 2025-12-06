#!/bin/bash
# Sync kinova_llm_control package from deep-research to catkin_ws
# This creates a symlink so changes are automatically reflected

set -e

DEEP_RESEARCH_DIR="$HOME/deep-research"
CATKIN_WS_DIR="$HOME/catkin_ws"
PACKAGE_NAME="kinova_llm_control"

echo "=== Syncing $PACKAGE_NAME package ==="

# First, pull latest changes
cd "$DEEP_RESEARCH_DIR"
echo "Pulling latest changes..."
git pull

# Check if package exists in deep-research
SOURCE_PKG="$DEEP_RESEARCH_DIR/ros_ws/src/$PACKAGE_NAME"
if [ ! -d "$SOURCE_PKG" ]; then
    echo "ERROR: Package not found at $SOURCE_PKG"
    exit 1
fi

# Check catkin_ws structure
TARGET_DIR="$CATKIN_WS_DIR/src"
if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: Catkin workspace not found at $CATKIN_WS_DIR"
    exit 1
fi

TARGET_PKG="$TARGET_DIR/$PACKAGE_NAME"

# Remove existing package (copy or symlink)
if [ -e "$TARGET_PKG" ] || [ -L "$TARGET_PKG" ]; then
    echo "Removing existing package at $TARGET_PKG..."
    rm -rf "$TARGET_PKG"
fi

# Create symlink
echo "Creating symlink: $TARGET_PKG -> $SOURCE_PKG"
ln -s "$SOURCE_PKG" "$TARGET_PKG"

# Verify symlink
if [ -L "$TARGET_PKG" ]; then
    echo "Symlink created successfully!"
    ls -la "$TARGET_PKG"
else
    echo "ERROR: Failed to create symlink"
    exit 1
fi

# Rebuild catkin workspace
echo ""
echo "=== Rebuilding catkin workspace ==="
cd "$CATKIN_WS_DIR"
source /opt/ros/noetic/setup.bash

# Use catkin build if available, otherwise catkin_make
if command -v catkin &> /dev/null; then
    catkin build $PACKAGE_NAME
else
    catkin_make --only-pkg-with-deps $PACKAGE_NAME
fi

echo ""
echo "=== Done! ==="
echo "Package synced and rebuilt. Now run:"
echo "  source $CATKIN_WS_DIR/devel/setup.bash"
echo "  roslaunch kinova_llm_control tabletop_simulation.launch"
