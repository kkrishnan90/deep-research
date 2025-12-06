#!/bin/bash
# Wrapper script to run llm_controller.py with the llm_venv Python
# while also including ROS Python packages

# Source ROS environment
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# Set PYTHONPATH to include both venv and ROS packages
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
export PYTHONPATH=~/catkin_ws/devel/lib/python3/dist-packages:$PYTHONPATH

# Run the LLM controller with venv Python
exec ~/llm_venv/bin/python3 $(rospack find kinova_llm_control)/scripts/llm_controller.py "$@"
