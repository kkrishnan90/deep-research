#!/bin/bash
# Wrapper script to run llm_controller.py with the venv Python that has google-genai installed
# Also adds ROS Python packages to PYTHONPATH so rospy works

VENV_PYTHON="/home/krishnan_kkrish_altostrat_com/llm_venv/bin/python3"
SOURCE_SCRIPT="/home/krishnan_kkrish_altostrat_com/catkin_ws/src/kinova_llm_control/scripts/llm_controller.py"

# Add ROS and catkin Python paths for rospy, moveit_commander, etc.
export PYTHONPATH="/home/krishnan_kkrish_altostrat_com/catkin_ws/devel/lib/python3/dist-packages:/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH"

exec "$VENV_PYTHON" "$SOURCE_SCRIPT" "$@"
