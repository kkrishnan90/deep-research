cmake_minimum_required(VERSION 3.0.2)
project(kinova_llm_control)

## Find catkin and required components
find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  sensor_msgs
  geometry_msgs
  trajectory_msgs
  actionlib
  actionlib_msgs
  moveit_commander
  moveit_msgs
  kinova_msgs
  cv_bridge
  image_transport
)

## Declare catkin package
catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
    sensor_msgs
    geometry_msgs
    trajectory_msgs
    actionlib
    actionlib_msgs
    moveit_commander
    moveit_msgs
    kinova_msgs
    cv_bridge
    image_transport
)

## Install Python scripts
catkin_install_python(PROGRAMS
  scripts/llm_controller.py
  scripts/command_interface.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install shell wrapper script (uses venv Python with google-genai)
install(PROGRAMS
  scripts/llm_controller_wrapper.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Install launch files, config, and urdf
install(DIRECTORY launch config urdf
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)
