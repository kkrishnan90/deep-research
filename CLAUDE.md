# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Robotics simulation project for a pick-and-place system using:
- **Robotic Arm**: Kinova Mico 2 M1N6S300 (6 DOF, 3-finger gripper)
- **Depth Camera**: Intel RealSense D435
- **LLM**: Gemini 3 Pro Preview via Vertex AI (model: `gemini-3-pro-preview`, package: `google-genai`)
- **Simulation**: Gazebo Classic 11
- **Framework**: ROS Noetic

The LLM receives user text commands and depth camera data, then generates motion plans to control the robotic arm for pick-and-place operations.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                    ┌─────────────────────────────────┐ │
│  │    MacBook      │      Git Push      │   GCP Compute Engine (Ubuntu)   │ │
│  │  (Development)  │ ─────────────────► │      (Simulation Runtime)       │ │
│  │                 │                    │                                 │ │
│  │  - Code editing │    SSH / Chrome    │  - Ubuntu 20.04 LTS             │ │
│  │  - Git commits  │ ◄─────────────────►│  - ROS Noetic                   │ │
│  │  - Python (uv)  │   Remote Desktop   │  - Gazebo + GPU rendering       │ │
│  │                 │                    │  - NVIDIA L4 (24GB VRAM)        │ │
│  └─────────────────┘                    └─────────────────────────────────┘ │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                           SIMULATION PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Command (text) → Gemini LLM → Motion Planning → Robotic Arm          │
│                            ↑                                                │
│               Depth Camera (D435) Sensor Data                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
deep-research/
├── CLAUDE.md                    # This file
├── PRD.md                       # Product requirements
├── .gitignore                   # Git ignore patterns
├── .env.example                 # Environment variables template
├── requirements.txt             # Python dependencies
├── scripts/
│   ├── gcp/                     # GCP automation scripts
│   │   ├── create_instance.sh   # Create VM with L4 GPU
│   │   ├── setup_nvidia.sh      # Install NVIDIA drivers
│   │   ├── setup_ros.sh         # Install ROS Noetic
│   │   ├── setup_gazebo_deps.sh # Install Gazebo + MoveIt
│   │   ├── setup_simulation.sh  # Clone repos, build workspace
│   │   ├── setup_desktop.sh     # Chrome Remote Desktop
│   │   ├── setup_python.sh      # Python venv with google-genai
│   │   ├── master_setup.sh      # Run all setup scripts
│   │   └── verify_setup.sh      # Verify installation
│   └── local/
│       └── sync_to_gcp.sh       # Git push helper
└── ros_ws/
    └── src/
        └── kinova_llm_control/  # Custom ROS package
            ├── CMakeLists.txt
            ├── package.xml
            ├── scripts/
            │   ├── llm_controller.py      # Gemini integration
            │   └── command_interface.py   # CLI for commands
            ├── launch/
            │   ├── full_simulation.launch # Full sim + LLM
            │   └── llm_only.launch        # LLM controller only
            └── config/
                └── robot_config.yaml      # Configuration
```

## Infrastructure

### GCP Compute Engine Instance

| Component | Specification |
|-----------|---------------|
| Machine Type | `g2-standard-8` (8 vCPU, 32GB RAM, 1x L4 GPU) |
| GPU | NVIDIA L4 (24GB VRAM, Ada Lovelace) |
| OS | Ubuntu 20.04 LTS |
| Disk | 200GB SSD |
| Zone | `us-central1-a` |
| Project | `account-pocs` |

## Commands

### Local Development (MacBook)

```bash
# Setup virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Test Python imports
python3 -c "from google import genai; print('OK')"

# Sync to GCP
./scripts/local/sync_to_gcp.sh "commit message"
```

### GCP Instance Management

```bash
# Create instance
./scripts/gcp/create_instance.sh

# SSH into instance
gcloud compute ssh ros-kinova-sim --zone=us-central1-a --project=account-pocs

# Run master setup (on GCP VM)
cd ~/deep-research && bash scripts/gcp/master_setup.sh

# Verify installation (on GCP VM)
bash scripts/gcp/verify_setup.sh
```

### ROS Simulation (on GCP VM)

```bash
# Source environment
source /opt/ros/noetic/setup.bash
source ~/catkin_ws/devel/setup.bash

# Launch full simulation
roslaunch kinova_llm_control full_simulation.launch

# Launch Kinova arm only
roslaunch kinova_gazebo robot_launch.launch kinova_robotType:=m1n6s300

# Send command to LLM controller
rostopic pub /user_command std_msgs/String "data: 'pick up the red cube'"
```

### Build ROS Workspace

```bash
cd ~/catkin_ws
catkin build
source devel/setup.bash
```

## Key ROS Packages

| Package | Source | Purpose |
|---------|--------|---------|
| `kinova-ros` | [GitHub](https://github.com/Kinovarobotics/kinova-ros) (noetic-devel) | Kinova arm drivers + Gazebo |
| `realsense_gazebo_plugin` | [GitHub](https://github.com/pal-robotics/realsense_gazebo_plugin) (melodic-devel) | D435 simulation |
| `realsense_gazebo_description` | [GitHub](https://github.com/m-tartari/realsense_gazebo_description) | Camera URDF models |
| `kinova_llm_control` | Local (ros_ws/src/) | Custom LLM integration |

## ROS Topics

| Topic | Type | Description |
|-------|------|-------------|
| `/user_command` | `std_msgs/String` | User commands to LLM |
| `/llm_status` | `std_msgs/String` | LLM controller status |
| `/llm_response` | `std_msgs/String` | LLM response text |
| `/camera/depth/image_raw` | `sensor_msgs/Image` | Depth image |
| `/camera/color/image_raw` | `sensor_msgs/Image` | RGB image |
| `/camera/depth/points` | `sensor_msgs/PointCloud2` | Point cloud |

## Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
GCP_PROJECT_ID=account-pocs
GCP_ZONE=us-central1-a
GOOGLE_CLOUD_PROJECT=account-pocs
GOOGLE_CLOUD_LOCATION=us-central1
GOOGLE_GENAI_USE_VERTEXAI=true
```

## Remote Access

Chrome Remote Desktop provides GUI access to Gazebo on the GCP instance.

Setup: https://remotedesktop.google.com/headless
Access: https://remotedesktop.google.com/access

## Testing

```bash
# Verify all components (on GCP VM)
bash scripts/gcp/verify_setup.sh

# Test LLM connection
python3 -c "
from google import genai
client = genai.Client(vertexai=True, project='account-pocs', location='us-central1')
response = client.models.generate_content(model='gemini-3-pro-preview', contents='Hello')
print(response.text)
"
```
