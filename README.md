# Kinova Mico LLM-Controlled Pick-and-Place Simulation

A ROS Noetic + Gazebo simulation of a Kinova Mico M1N6S300 robotic arm with Intel RealSense D435 depth camera, controlled via natural language commands using Google's Gemini LLM.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Command (text) ──► Gemini LLM ──► Motion Planning ──► Robotic Arm    │
│                              ▲                                              │
│                              │                                              │
│               RGB + Depth Camera (D435) Sensor Data                         │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Components:                                                                │
│  • Kinova Mico M1N6S300 (6 DOF, 3-finger gripper)                          │
│  • Intel RealSense D435 depth camera (simulated)                           │
│  • Gemini 2.0 Flash via Vertex AI                                          │
│  • ROS Noetic + Gazebo 11                                                  │
│  • MoveIt for motion planning                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM (for Gazebo rendering)
- **RAM**: 16GB+ recommended
- **Disk**: 50GB+ free space

### Software Requirements
- Ubuntu 20.04 LTS
- ROS Noetic
- Gazebo 11
- Python 3.8+
- NVIDIA drivers (for GPU acceleration)

## Installation

### 1. Install ROS Noetic

```bash
# Setup sources
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt install curl
curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

# Install ROS Noetic Desktop Full
sudo apt update
sudo apt install ros-noetic-desktop-full

# Setup environment
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

# Install dependencies
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
sudo rosdep init
rosdep update
```

### 2. Install Additional ROS Packages

```bash
sudo apt install \
    ros-noetic-moveit \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    ros-noetic-ros-control \
    ros-noetic-ros-controllers \
    ros-noetic-joint-state-publisher \
    ros-noetic-robot-state-publisher \
    ros-noetic-tf2-ros \
    ros-noetic-tf2-geometry-msgs \
    python3-catkin-tools
```

### 3. Create Catkin Workspace

```bash
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws
catkin init
```

### 4. Clone Required Repositories

```bash
cd ~/catkin_ws/src

# Kinova ROS packages
git clone -b noetic-devel https://github.com/Kinovarobotics/kinova-ros.git

# RealSense Gazebo plugin
git clone -b melodic-devel https://github.com/pal-robotics/realsense_gazebo_plugin.git

# RealSense Gazebo description (URDF models)
git clone https://github.com/m-tartari/realsense_gazebo_description.git
```

### 5. Copy Project Files

Clone this repository and copy the ROS package:

```bash
# Clone this repo
git clone https://github.com/kkrishnan90/deep-research.git
cd deep-research

# Copy the kinova_llm_control package to catkin workspace
cp -r ros_ws/src/kinova_llm_control ~/catkin_ws/src/
```

### 6. Install Python Dependencies

Create a Python virtual environment with required packages:

```bash
# Create virtual environment
python3 -m venv ~/llm_venv
source ~/llm_venv/bin/activate

# Install dependencies
pip install google-genai opencv-python numpy pillow
```

### 7. Build the Workspace

```bash
cd ~/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin build
source devel/setup.bash
```

### 8. Configure Google Cloud / Vertex AI

The LLM controller uses Gemini via Vertex AI. Set up authentication:

```bash
# Install gcloud CLI if not already installed
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Authenticate
gcloud auth application-default login

# Set project
export GOOGLE_CLOUD_PROJECT=your-project-id
```

## Project Structure

```
deep-research/
├── README.md                           # This file
├── CLAUDE.md                           # Claude Code instructions
├── PRD.md                              # Product requirements
├── .env.example                        # Environment variables template
├── requirements.txt                    # Python dependencies
│
├── scripts/
│   └── gcp/                            # GCP automation scripts
│       ├── create_instance.sh          # Create VM with L4 GPU
│       ├── setup_nvidia.sh             # Install NVIDIA drivers
│       ├── setup_ros.sh                # Install ROS Noetic
│       ├── setup_gazebo_deps.sh        # Install Gazebo + MoveIt
│       ├── setup_simulation.sh         # Clone repos, build workspace
│       ├── setup_python.sh             # Python venv with google-genai
│       ├── master_setup.sh             # Run all setup scripts
│       └── verify_setup.sh             # Verify installation
│
└── ros_ws/
    └── src/
        └── kinova_llm_control/         # Main ROS package
            ├── CMakeLists.txt
            ├── package.xml
            ├── config/
            │   └── robot_config.yaml   # Robot configuration
            ├── launch/
            │   ├── full_simulation.launch      # Complete simulation
            │   └── tabletop_simulation.launch  # Alternative launch
            ├── models/
            │   └── ...                 # Gazebo models
            ├── scripts/
            │   ├── llm_controller.py           # Main LLM controller
            │   ├── llm_controller_wrapper.sh   # Wrapper script
            │   └── command_interface.py        # CLI interface
            ├── urdf/
            │   ├── m1n6s300_with_camera.xacro  # Robot + camera URDF
            │   └── camera_stand.urdf.xacro     # Camera stand model
            └── worlds/
                └── tabletop_workspace.world    # Gazebo world file
```

## File Descriptions

### Launch Files

| File | Description |
|------|-------------|
| `full_simulation.launch` | Main launch file - starts Gazebo, robot, camera, MoveIt, and LLM controller |
| `tabletop_simulation.launch` | Alternative launch with different configuration |

### URDF Files

| File | Description |
|------|-------------|
| `m1n6s300_with_camera.xacro` | Combined URDF with Kinova arm on table + RealSense D435 on floor stand |
| `camera_stand.urdf.xacro` | Standalone camera stand model |

### Scripts

| File | Description |
|------|-------------|
| `llm_controller.py` | Main controller - receives commands, calls Gemini, executes motions |
| `llm_controller_wrapper.sh` | Wrapper to run controller with correct Python environment |
| `command_interface.py` | Simple CLI to send commands to the robot |

### World Files

| File | Description |
|------|-------------|
| `tabletop_workspace.world` | Gazebo world with table (1.2m x 0.8m) and colored balls |

## Running the Simulation

### Method 1: Full Launch (Recommended)

```bash
# Terminal 1: Set environment and launch
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$HOME/catkin_ws/devel/lib

roslaunch kinova_llm_control full_simulation.launch
```

### Method 2: Step-by-Step Launch

```bash
# Terminal 1: Start ROS Master
roscore

# Terminal 2: Start Gazebo with world
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash
roslaunch gazebo_ros empty_world.launch world_name:=$(rospack find kinova_llm_control)/worlds/tabletop_workspace.world

# Terminal 3: Spawn robot
roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=m1n6s300

# Terminal 4: Start MoveIt
roslaunch m1n6s300_moveit_config m1n6s300_gazebo_demo.launch

# Terminal 5: Start LLM Controller
source ~/llm_venv/bin/activate
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
python3 $(rospack find kinova_llm_control)/scripts/llm_controller.py
```

## Sending Commands

### Via ROS Topic

```bash
# Send a text command
rostopic pub /user_command std_msgs/String "data: 'Pick up the red ball'" --once

# Monitor LLM status
rostopic echo /llm_status

# Monitor LLM responses
rostopic echo /llm_response
```

### Via Command Interface

```bash
source ~/catkin_ws/devel/setup.bash
rosrun kinova_llm_control command_interface.py
```

## ROS Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/user_command` | `std_msgs/String` | Input: User commands to LLM |
| `/llm_status` | `std_msgs/String` | Output: LLM controller status |
| `/llm_response` | `std_msgs/String` | Output: LLM response text |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB camera image |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth camera image |
| `/camera/aligned_depth_to_color/image_raw` | `sensor_msgs/Image` | Aligned depth image |
| `/m1n6s300/joint_states` | `sensor_msgs/JointState` | Robot joint positions |

## Supported Commands

The LLM controller supports natural language commands like:

- "Pick up the red ball"
- "Move the green object to the drop zone"
- "What objects do you see on the table?"
- "Wave hello"
- "Open/close the gripper"
- "Move to home position"

## Troubleshooting

### Camera Not Publishing Depth Data

Ensure `GAZEBO_PLUGIN_PATH` includes the RealSense plugin:

```bash
export GAZEBO_PLUGIN_PATH=$GAZEBO_PLUGIN_PATH:$HOME/catkin_ws/devel/lib
```

Verify plugin is built:

```bash
ls ~/catkin_ws/devel/lib/librealsense_gazebo_plugin.so
```

### LLM Controller Import Errors

If you see `ModuleNotFoundError: No module named 'PyKDL'`, ensure the wrapper script sets PYTHONPATH:

```bash
export PYTHONPATH=/opt/ros/noetic/lib/python3/dist-packages:$PYTHONPATH
```

### MoveIt Action Server Not Connected

Wait for all controllers to initialize (can take 30-60 seconds). Check:

```bash
rostopic list | grep follow_joint_trajectory
rosnode list | grep move_group
```

### Gazebo Crashes on Startup

- Check GPU drivers: `nvidia-smi`
- Try headless mode: `roslaunch kinova_llm_control full_simulation.launch gui:=false`
- Check disk space: `df -h`

## GCP Deployment (Optional)

For running on Google Cloud with GPU:

```bash
# Create instance
./scripts/gcp/create_instance.sh

# SSH into instance
gcloud compute ssh ros-kinova-sim --zone=us-central1-a

# Run master setup
cd ~/deep-research && bash scripts/gcp/master_setup.sh

# Verify
bash scripts/gcp/verify_setup.sh
```

Use Chrome Remote Desktop for GUI access to Gazebo.

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_CLOUD_PROJECT` | GCP project ID | `account-pocs` |
| `GOOGLE_CLOUD_LOCATION` | Vertex AI location | `us-central1` |
| `GAZEBO_PLUGIN_PATH` | Path to Gazebo plugins | (add catkin devel/lib) |

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Kinova Robotics](https://github.com/Kinovarobotics/kinova-ros) for the Kinova ROS packages
- [PAL Robotics](https://github.com/pal-robotics/realsense_gazebo_plugin) for the RealSense Gazebo plugin
- [Google](https://cloud.google.com/vertex-ai) for Vertex AI and Gemini
