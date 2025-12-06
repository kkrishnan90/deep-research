#!/home/krishnan_kkrish_altostrat_com/llm_venv/bin/python3
"""
LLM Controller for Kinova Mico arm using Gemini API.

This node:
1. Subscribes to depth camera data from RealSense D435
2. Receives user commands via /user_command topic
3. Uses Gemini to interpret commands and generate motion plans
4. Sends commands to MoveIt for execution

Author: Developer
License: MIT
"""

import rospy
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2
from geometry_msgs.msg import Pose, PoseStamped
from cv_bridge import CvBridge

# Gemini API import
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    rospy.logwarn("google-genai not installed. LLM features disabled.")

# MoveIt import
try:
    import moveit_commander
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False
    rospy.logwarn("moveit_commander not available.")


class LLMController:
    """
    Main controller class that integrates Gemini LLM with ROS/MoveIt.
    """

    def __init__(self):
        """Initialize the LLM controller node."""
        rospy.init_node('llm_controller', anonymous=False)

        # Configuration - using account-pocs for Gemini API access
        self.project_id = rospy.get_param('~project_id', 'account-pocs')
        self.location = rospy.get_param('~location', 'global')
        self.model_name = rospy.get_param('~model_name', 'gemini-3-pro-preview')

        # Initialize Gemini client
        self.gemini_client = None
        if GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                rospy.loginfo(f"Gemini client initialized with project: {self.project_id}")
            except Exception as e:
                rospy.logerr(f"Failed to initialize Gemini client: {e}")

        # Initialize MoveIt
        self.move_group = None
        if MOVEIT_AVAILABLE:
            try:
                moveit_commander.roscpp_initialize([])
                self.robot = moveit_commander.RobotCommander()
                self.scene = moveit_commander.PlanningSceneInterface()
                self.move_group = moveit_commander.MoveGroupCommander("arm")
                self.gripper_group = moveit_commander.MoveGroupCommander("gripper")
                rospy.loginfo("MoveIt commander initialized")
            except Exception as e:
                rospy.logwarn(f"MoveIt initialization failed: {e}")

        # CV Bridge for image processing
        self.bridge = CvBridge()

        # Current sensor data
        self.current_depth_image = None
        self.current_rgb_image = None
        self.current_pointcloud = None

        # Subscribers
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_raw',
            Image,
            self.depth_callback,
            queue_size=1
        )
        self.rgb_sub = rospy.Subscriber(
            '/camera/color/image_raw',
            Image,
            self.rgb_callback,
            queue_size=1
        )
        self.pointcloud_sub = rospy.Subscriber(
            '/camera/depth/points',
            PointCloud2,
            self.pointcloud_callback,
            queue_size=1
        )
        self.command_sub = rospy.Subscriber(
            '/user_command',
            String,
            self.command_callback,
            queue_size=10
        )

        # Publishers
        self.status_pub = rospy.Publisher(
            '/llm_status',
            String,
            queue_size=10
        )
        self.response_pub = rospy.Publisher(
            '/llm_response',
            String,
            queue_size=10
        )

        rospy.loginfo("LLM Controller initialized and ready")
        self.publish_status("ready")

    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='passthrough'
            )
        except Exception as e:
            rospy.logwarn(f"Failed to convert depth image: {e}")

    def rgb_callback(self, msg):
        """Store latest RGB image."""
        try:
            self.current_rgb_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding='bgr8'
            )
        except Exception as e:
            rospy.logwarn(f"Failed to convert RGB image: {e}")

    def pointcloud_callback(self, msg):
        """Store latest pointcloud."""
        self.current_pointcloud = msg

    def command_callback(self, msg):
        """Process user command through Gemini LLM."""
        command = msg.data.strip()
        if not command:
            return

        rospy.loginfo(f"Received command: {command}")
        self.publish_status("processing")

        try:
            # Generate response from Gemini
            response = self.process_command(command)
            self.response_pub.publish(String(data=response))
            self.publish_status("completed")
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            rospy.logerr(error_msg)
            self.response_pub.publish(String(data=error_msg))
            self.publish_status("error")

    def process_command(self, command):
        """
        Process a natural language command using Gemini LLM.

        Args:
            command: Natural language command from user

        Returns:
            Response string from the LLM
        """
        if not self.gemini_client:
            return "Gemini client not available. Please check configuration."

        # Build context about the robot state
        context = self.build_context()

        # Create prompt for Gemini
        prompt = f"""
You are controlling a Kinova Mico 2 robotic arm with the following specifications:
- Model: M1N6S300 (Mico v1, 6 DOF, 3-finger gripper)
- Workspace: Table with depth camera (Intel RealSense D435) at one end
- The arm is mounted at the opposite end of the table
- Objects can be placed between the camera and the arm

Current robot state:
{context}

User command: {command}

Based on this command, provide:
1. A brief interpretation of what the user wants
2. The sequence of actions needed to accomplish this task
3. Any safety considerations

If the command requires motion planning, describe the waypoints and gripper actions needed.
If the command is unclear or unsafe, explain why and ask for clarification.
"""

        try:
            response = self.gemini_client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return response.text
        except Exception as e:
            rospy.logerr(f"Gemini API error: {e}")
            return f"Failed to generate response: {e}"

    def build_context(self):
        """Build context string about current robot state."""
        context_parts = []

        # Add MoveIt state if available
        if self.move_group:
            try:
                current_pose = self.move_group.get_current_pose().pose
                context_parts.append(
                    f"End effector position: x={current_pose.position.x:.3f}, "
                    f"y={current_pose.position.y:.3f}, z={current_pose.position.z:.3f}"
                )
                joint_values = self.move_group.get_current_joint_values()
                context_parts.append(f"Joint values: {[f'{v:.3f}' for v in joint_values]}")
            except Exception as e:
                context_parts.append(f"Robot state unavailable: {e}")

        # Add sensor data availability
        if self.current_depth_image is not None:
            h, w = self.current_depth_image.shape[:2]
            context_parts.append(f"Depth image available: {w}x{h}")
        else:
            context_parts.append("Depth image: not available")

        if self.current_rgb_image is not None:
            h, w = self.current_rgb_image.shape[:2]
            context_parts.append(f"RGB image available: {w}x{h}")
        else:
            context_parts.append("RGB image: not available")

        return "\n".join(context_parts) if context_parts else "No state information available"

    def publish_status(self, status):
        """Publish current status."""
        self.status_pub.publish(String(data=status))

    def execute_motion(self, target_pose):
        """
        Execute a motion to a target pose using MoveIt.

        Args:
            target_pose: geometry_msgs/Pose target
        """
        if not self.move_group:
            rospy.logwarn("MoveIt not available, cannot execute motion")
            return False

        try:
            self.move_group.set_pose_target(target_pose)
            plan = self.move_group.go(wait=True)
            self.move_group.stop()
            self.move_group.clear_pose_targets()
            return plan
        except Exception as e:
            rospy.logerr(f"Motion execution failed: {e}")
            return False

    def open_gripper(self):
        """Open the gripper."""
        if self.gripper_group:
            try:
                self.gripper_group.set_named_target("open")
                self.gripper_group.go(wait=True)
                return True
            except Exception as e:
                rospy.logerr(f"Failed to open gripper: {e}")
        return False

    def close_gripper(self):
        """Close the gripper."""
        if self.gripper_group:
            try:
                self.gripper_group.set_named_target("close")
                self.gripper_group.go(wait=True)
                return True
            except Exception as e:
                rospy.logerr(f"Failed to close gripper: {e}")
        return False

    def run(self):
        """Main run loop."""
        rospy.loginfo("LLM Controller running. Waiting for commands on /user_command")
        rospy.spin()

    def shutdown(self):
        """Clean shutdown."""
        rospy.loginfo("Shutting down LLM Controller")
        if MOVEIT_AVAILABLE:
            moveit_commander.roscpp_shutdown()


def main():
    """Main entry point."""
    try:
        controller = LLMController()
        rospy.on_shutdown(controller.shutdown)
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"LLM Controller failed: {e}")


if __name__ == '__main__':
    main()
