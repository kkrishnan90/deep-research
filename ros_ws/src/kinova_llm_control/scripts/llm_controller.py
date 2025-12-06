#!/home/krishnan_kkrish_altostrat_com/llm_venv/bin/python3
"""
LLM Controller for Kinova Mico arm using Gemini API with Function Calling.

This node uses Gemini's function calling and multimodal vision to control the robot.
The LLM receives camera images, understands the scene, and executes robot actions.

Features:
- Multimodal vision: Send RGB images to Gemini for scene understanding
- Depth perception: Convert 2D pixel locations to 3D world coordinates
- Function calling: Execute robot actions through structured tool calls
- Composite actions: Pick, place, and other multi-step operations

Author: Developer
License: MIT
"""

# Add ROS Python paths for PyKDL when running in venv
import sys
import os
ros_python_paths = [
    '/opt/ros/noetic/lib/python3/dist-packages',
    os.path.expanduser('~/catkin_ws/devel/lib/python3/dist-packages'),
]
for path in ros_python_paths:
    expanded = os.path.expanduser(path)
    if expanded not in sys.path and os.path.exists(expanded):
        sys.path.insert(0, expanded)

import rospy
import numpy as np
import cv2
import base64
import actionlib
import tf2_ros
# tf2_geometry_msgs is optional - requires PyKDL
try:
    import tf2_geometry_msgs
    TF2_GEOMETRY_AVAILABLE = True
except ImportError:
    TF2_GEOMETRY_AVAILABLE = False
from std_msgs.msg import String
from sensor_msgs.msg import Image, PointCloud2, JointState, CameraInfo
from geometry_msgs.msg import Pose, PoseStamped, Point, PointStamped
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from moveit_msgs.srv import GetCartesianPath, GetCartesianPathRequest
from moveit_msgs.msg import RobotState
from cv_bridge import CvBridge

# Gemini API import
try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    rospy.logwarn("google-genai not installed. LLM features disabled.")

# MoveIt import
try:
    import moveit_commander
    from moveit_msgs.msg import DisplayTrajectory, RobotTrajectory
    MOVEIT_AVAILABLE = True
except ImportError:
    MOVEIT_AVAILABLE = False


class LLMController:
    """
    Main controller class that integrates Gemini LLM with ROS using function calling
    and multimodal vision for intelligent robot control.
    """

    def __init__(self):
        """Initialize the LLM controller node."""
        rospy.init_node('llm_controller', anonymous=False)

        # Configuration
        self.project_id = rospy.get_param('~project_id', 'account-pocs')
        self.location = rospy.get_param('~location', 'us-central1')
        self.model_name = rospy.get_param('~model_name', 'gemini-2.0-flash')

        # Initialize Gemini client
        self.gemini_client = None
        if GEMINI_AVAILABLE:
            try:
                self.gemini_client = genai.Client(
                    vertexai=True,
                    project=self.project_id,
                    location=self.location
                )
                rospy.loginfo(f"Gemini client initialized with model: {self.model_name}")
            except Exception as e:
                rospy.logerr(f"Failed to initialize Gemini client: {e}")

        # Define tools for Gemini
        self.tools = self._create_tools()

        # CV Bridge
        self.bridge = CvBridge()

        # Sensor data
        self.current_depth_image = None
        self.current_rgb_image = None
        self.current_pointcloud = None
        self.camera_info = None

        # Joint state
        self.current_joint_names = []
        self.current_joint_positions = []
        self.joint_state_received = False

        # TF Buffer for coordinate transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # Arm joint names for Kinova Mico M1N6S300
        self.arm_joint_names = [
            'm1n6s300_joint_1', 'm1n6s300_joint_2', 'm1n6s300_joint_3',
            'm1n6s300_joint_4', 'm1n6s300_joint_5', 'm1n6s300_joint_6'
        ]
        self.finger_joint_names = [
            'm1n6s300_joint_finger_1', 'm1n6s300_joint_finger_2', 'm1n6s300_joint_finger_3'
        ]

        # End effector link
        self.end_effector_link = 'm1n6s300_end_effector'
        self.planning_group = 'arm'

        # Trajectory action clients
        self.arm_client = actionlib.SimpleActionClient(
            '/m1n6s300/effort_joint_trajectory_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        self.finger_client = actionlib.SimpleActionClient(
            '/m1n6s300/effort_finger_trajectory_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )

        rospy.loginfo("Waiting for trajectory action servers...")
        arm_connected = self.arm_client.wait_for_server(timeout=rospy.Duration(5.0))
        finger_connected = self.finger_client.wait_for_server(timeout=rospy.Duration(5.0))

        if arm_connected:
            rospy.loginfo("Arm trajectory server connected!")
        else:
            rospy.logwarn("Arm trajectory server not available")

        if finger_connected:
            rospy.loginfo("Finger trajectory server connected!")
        else:
            rospy.logwarn("Finger trajectory server not available")

        # MoveIt Cartesian path service
        self.cartesian_path_service = None
        try:
            rospy.loginfo("Waiting for /compute_cartesian_path service...")
            rospy.wait_for_service('/compute_cartesian_path', timeout=5.0)
            self.cartesian_path_service = rospy.ServiceProxy('/compute_cartesian_path', GetCartesianPath)
            rospy.loginfo("Cartesian path service connected!")
        except rospy.ROSException:
            rospy.logwarn("Cartesian path service not available - Cartesian motion disabled")

        # Subscribers
        self.joint_state_sub = rospy.Subscriber(
            '/m1n6s300/joint_states', JointState, self.joint_state_callback, queue_size=1
        )
        self.depth_sub = rospy.Subscriber(
            '/camera/depth/image_raw', Image, self.depth_callback, queue_size=1
        )
        self.rgb_sub = rospy.Subscriber(
            '/camera/color/image_raw', Image, self.rgb_callback, queue_size=1
        )
        self.camera_info_sub = rospy.Subscriber(
            '/camera/color/camera_info', CameraInfo, self.camera_info_callback, queue_size=1
        )
        self.command_sub = rospy.Subscriber(
            '/user_command', String, self.command_callback, queue_size=10
        )

        # Publishers
        self.status_pub = rospy.Publisher('/llm_status', String, queue_size=10)
        self.response_pub = rospy.Publisher('/llm_response', String, queue_size=10)

        rospy.loginfo("LLM Controller with Vision + Function Calling initialized")
        self.publish_status("ready")

    def _create_tools(self):
        """Create function declarations for Gemini with vision and motion capabilities."""
        tools = [
            types.Tool(function_declarations=[
                # Scene Understanding
                types.FunctionDeclaration(
                    name='analyze_scene',
                    description='Analyze the current camera view to detect and describe objects. Returns object names, colors, and approximate positions in the scene.',
                    parameters=types.Schema(type='OBJECT', properties={})
                ),
                types.FunctionDeclaration(
                    name='get_object_position',
                    description='Get the 3D position of an object at a specific pixel location using depth data. Returns x, y, z coordinates in robot base frame.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'pixel_x': types.Schema(type='INTEGER', description='X coordinate in image (0-640)'),
                            'pixel_y': types.Schema(type='INTEGER', description='Y coordinate in image (0-480)'),
                        },
                        required=['pixel_x', 'pixel_y']
                    )
                ),

                # Basic Arm Motions
                types.FunctionDeclaration(
                    name='wave_gesture',
                    description='Perform a friendly wave gesture. The robot arm moves to a visible position and oscillates the wrist to wave.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'num_waves': types.Schema(type='INTEGER', description='Number of wave oscillations (default: 3)')
                        }
                    )
                ),
                types.FunctionDeclaration(
                    name='move_to_home',
                    description='Move the robot arm to its home/neutral position with arm raised.',
                    parameters=types.Schema(type='OBJECT', properties={})
                ),
                types.FunctionDeclaration(
                    name='move_arm_to_joints',
                    description='Move the arm to specific joint positions in degrees.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'joint1_deg': types.Schema(type='NUMBER', description='Base rotation (degrees)'),
                            'joint2_deg': types.Schema(type='NUMBER', description='Shoulder angle (degrees)'),
                            'joint3_deg': types.Schema(type='NUMBER', description='Elbow angle (degrees)'),
                            'joint4_deg': types.Schema(type='NUMBER', description='Wrist 1 angle (degrees)'),
                            'joint5_deg': types.Schema(type='NUMBER', description='Wrist 2 angle (degrees)'),
                            'joint6_deg': types.Schema(type='NUMBER', description='Wrist 3 angle (degrees)'),
                            'duration_sec': types.Schema(type='NUMBER', description='Movement duration in seconds (default: 3.0)')
                        },
                        required=['joint1_deg', 'joint2_deg', 'joint3_deg', 'joint4_deg', 'joint5_deg', 'joint6_deg']
                    )
                ),

                # Cartesian Motion
                types.FunctionDeclaration(
                    name='move_to_position',
                    description='Move the end effector to a 3D position in robot base frame using Cartesian motion planning.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'x': types.Schema(type='NUMBER', description='X position in meters (forward/back)'),
                            'y': types.Schema(type='NUMBER', description='Y position in meters (left/right)'),
                            'z': types.Schema(type='NUMBER', description='Z position in meters (up/down)'),
                        },
                        required=['x', 'y', 'z']
                    )
                ),
                types.FunctionDeclaration(
                    name='move_relative',
                    description='Move the end effector relative to its current position.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'dx': types.Schema(type='NUMBER', description='Delta X in meters (default: 0)'),
                            'dy': types.Schema(type='NUMBER', description='Delta Y in meters (default: 0)'),
                            'dz': types.Schema(type='NUMBER', description='Delta Z in meters (default: 0)'),
                        }
                    )
                ),

                # Gripper Control
                types.FunctionDeclaration(
                    name='open_gripper',
                    description='Open the robot gripper/hand to release an object or prepare to grasp.',
                    parameters=types.Schema(type='OBJECT', properties={})
                ),
                types.FunctionDeclaration(
                    name='close_gripper',
                    description='Close the robot gripper/hand to grasp an object.',
                    parameters=types.Schema(type='OBJECT', properties={})
                ),

                # Composite Actions
                types.FunctionDeclaration(
                    name='pick_at_position',
                    description='Pick up an object at a specified 3D position. This is a composite action: approach from above, open gripper, descend, close gripper, lift.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'x': types.Schema(type='NUMBER', description='X position of object in meters'),
                            'y': types.Schema(type='NUMBER', description='Y position of object in meters'),
                            'z': types.Schema(type='NUMBER', description='Z position of object in meters'),
                            'approach_height': types.Schema(type='NUMBER', description='Height above object to approach from (default: 0.1m)'),
                        },
                        required=['x', 'y', 'z']
                    )
                ),
                types.FunctionDeclaration(
                    name='place_at_position',
                    description='Place an object at a specified 3D position. This is a composite action: move above target, descend, open gripper, retract upward.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'x': types.Schema(type='NUMBER', description='X position to place object in meters'),
                            'y': types.Schema(type='NUMBER', description='Y position to place object in meters'),
                            'z': types.Schema(type='NUMBER', description='Z position to place object in meters'),
                            'retract_height': types.Schema(type='NUMBER', description='Height to retract to after placing (default: 0.1m)'),
                        },
                        required=['x', 'y', 'z']
                    )
                ),

                # Directional Motion
                types.FunctionDeclaration(
                    name='point_at_direction',
                    description='Point the robot arm in a specified direction.',
                    parameters=types.Schema(
                        type='OBJECT',
                        properties={
                            'direction': types.Schema(type='STRING', description='Direction: "forward", "left", "right", "up", "down"')
                        },
                        required=['direction']
                    )
                ),

                # Status
                types.FunctionDeclaration(
                    name='get_robot_state',
                    description='Get the current state of the robot including joint positions, end effector pose, and sensor status.',
                    parameters=types.Schema(type='OBJECT', properties={})
                ),
            ])
        ]
        return tools

    def joint_state_callback(self, msg):
        """Store latest joint state."""
        self.current_joint_names = list(msg.name)
        self.current_joint_positions = list(msg.position)
        self.joint_state_received = True

    def depth_callback(self, msg):
        """Store latest depth image."""
        try:
            self.current_depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            rospy.logwarn(f"Failed to convert depth image: {e}")

    def rgb_callback(self, msg):
        """Store latest RGB image."""
        try:
            self.current_rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logwarn(f"Failed to convert RGB image: {e}")

    def camera_info_callback(self, msg):
        """Store camera intrinsics."""
        self.camera_info = msg

    def command_callback(self, msg):
        """Process user command through Gemini with function calling and vision."""
        command = msg.data.strip()
        if not command:
            return

        rospy.loginfo(f"Received command: {command}")
        self.publish_status("processing")

        try:
            response = self.process_command_with_vision(command)
            self.response_pub.publish(String(data=response))
            self.publish_status("completed")
        except Exception as e:
            error_msg = f"Error: {e}"
            rospy.logerr(error_msg)
            import traceback
            traceback.print_exc()
            self.response_pub.publish(String(data=error_msg))
            self.publish_status("error")

    def process_command_with_vision(self, command):
        """Process command using Gemini with function calling and optional vision."""
        if not self.gemini_client:
            return "Gemini client not available."

        # Build context
        context = self._build_context()

        # System prompt
        system_prompt = f"""You are a robot controller for a Kinova Mico 2 arm (M1N6S300) with a 3-finger gripper and a depth camera.

Your capabilities:
1. Scene understanding: You can analyze the camera image to detect objects
2. Depth perception: You can get 3D positions from depth data
3. Motion control: You can move the arm to joint positions or Cartesian positions
4. Composite actions: You can perform pick-and-place operations

Current robot state:
{context}

IMPORTANT INSTRUCTIONS:
- Always use the available functions to execute actions
- For pick-up tasks: 
    1. analyze_scene (to find object)
    2. get_object_position (with pixel coordinates from analysis)
    3. pick_at_position (with 3D coordinates from get_object_position)
- For place tasks: Use place_at_position with the target location
- For multi-step tasks, call multiple functions in sequence
- Be concise in your responses - just describe what you did

Available function patterns:
- "wave", "hello", "hi" → wave_gesture
- "home", "reset" → move_to_home
- "open hand/gripper" → open_gripper
- "close hand/gripper", "grab", "grasp" → close_gripper
- "point left/right/up/down" → point_at_direction
- "pick up X", "grab X" → analyze_scene -> get_object_position -> pick_at_position
- "place X at Y", "put down" → place_at_position
- "move to X,Y,Z" → move_to_position
- "what do you see" → analyze_scene
"""

        # Build initial content with optional image
        contents = []

        # Add image if available and command seems to need vision
        vision_keywords = ['see', 'look', 'pick', 'grab', 'find', 'detect', 'object', 'what', 'where', 'color', 'table']
        needs_vision = any(kw in command.lower() for kw in vision_keywords)

        if needs_vision and self.current_rgb_image is not None:
            # Encode image as base64
            _, buffer = cv2.imencode('.jpg', self.current_rgb_image, [cv2.IMWRITE_JPEG_QUALITY, 80])
            image_base64 = base64.b64encode(buffer).decode('utf-8')

            contents.append(types.Part.from_bytes(
                data=base64.b64decode(image_base64),
                mime_type='image/jpeg'
            ))
            rospy.loginfo("Sending image to Gemini for analysis")

        contents.append(types.Part.from_text(text=command))

        # Multi-turn loop
        max_turns = 10
        turn = 0
        final_response_text = ""

        try:
            while turn < max_turns:
                rospy.loginfo(f"Turn {turn+1}: Sending request to Gemini...")
                
                # Call Gemini with tools
                response = self.gemini_client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=self.tools,
                        temperature=0.2,
                    )
                )

                # Check for function calls
                if response.function_calls:
                    # Append model's function call part to history
                    # We need to construct the Part manually if the SDK doesn't do it automatically for history
                    # The SDK usually handles this if we append the response.candidates[0].content.parts
                    
                    # For safety, we iterate and execute
                    function_calls = response.function_calls
                    
                    # Add the model's response (which contains function calls) to history
                    contents.append(response.candidates[0].content)

                    # Collect all function responses in a single Content object
                    # (Gemini requires all responses in one turn)
                    function_response_parts = []
                    for fc in function_calls:
                        func_name = fc.name
                        func_args = fc.args if fc.args else {}

                        rospy.loginfo(f"Executing function: {func_name} with args: {func_args}")

                        # Execute the function
                        result_text = str(self._execute_function(func_name, func_args))
                        rospy.loginfo(f"Function result: {result_text}")

                        # Collect function response part
                        function_response_parts.append(types.Part.from_function_response(
                            name=func_name,
                            response={"result": result_text}
                        ))

                    # Add all function responses as a single Content
                    contents.append(types.Content(
                        role="tool",
                        parts=function_response_parts
                    ))
                    
                    turn += 1
                    # Loop continues to send the updated history back to Gemini
                
                else:
                    # No function call, this is the final text response
                    final_response_text = response.text if response.text else "Action completed."
                    rospy.loginfo(f"Final response: {final_response_text}")
                    return final_response_text

            return "Max turns reached. Stopping."

        except Exception as e:
            rospy.logerr(f"Gemini API error: {e}")
            import traceback
            traceback.print_exc()
            return f"API Error: {e}"

    def _execute_function(self, func_name, args):
        """Execute a robot function by name."""
        try:
            # Scene Understanding
            if func_name == 'analyze_scene':
                return self._analyze_scene()

            elif func_name == 'get_object_position':
                pixel_x = int(args.get('pixel_x', 320))
                pixel_y = int(args.get('pixel_y', 240))
                return self._get_3d_position(pixel_x, pixel_y)

            # Basic Arm Motions
            elif func_name == 'wave_gesture':
                num_waves = int(args.get('num_waves', 3))
                success = self._wave(num_waves)
                return "Wave completed!" if success else "Wave failed"

            elif func_name == 'move_to_home':
                success = self._move_to_home()
                return "Moved to home!" if success else "Home motion failed"

            elif func_name == 'move_arm_to_joints':
                joints_deg = [
                    float(args.get('joint1_deg', 0)),
                    float(args.get('joint2_deg', 180)),
                    float(args.get('joint3_deg', 180)),
                    float(args.get('joint4_deg', 0)),
                    float(args.get('joint5_deg', 0)),
                    float(args.get('joint6_deg', 0)),
                ]
                duration = float(args.get('duration_sec', 3.0))
                success = self._move_to_joints(joints_deg, duration)
                return "Moved to position!" if success else "Motion failed"

            # Cartesian Motion
            elif func_name == 'move_to_position':
                x = float(args.get('x', 0.3))
                y = float(args.get('y', 0.0))
                z = float(args.get('z', 0.3))
                success = self._move_to_cartesian(x, y, z)
                return f"Moved to ({x:.2f}, {y:.2f}, {z:.2f})!" if success else "Cartesian motion failed"

            elif func_name == 'move_relative':
                dx = float(args.get('dx', 0))
                dy = float(args.get('dy', 0))
                dz = float(args.get('dz', 0))
                success = self._move_relative(dx, dy, dz)
                return f"Moved relative ({dx:.2f}, {dy:.2f}, {dz:.2f})!" if success else "Relative motion failed"

            # Gripper Control
            elif func_name == 'open_gripper':
                success = self._open_gripper()
                return "Gripper opened!" if success else "Gripper open failed"

            elif func_name == 'close_gripper':
                success = self._close_gripper()
                return "Gripper closed!" if success else "Gripper close failed"

            # Composite Actions
            elif func_name == 'pick_at_position':
                x = float(args.get('x', 0.3))
                y = float(args.get('y', 0.0))
                z = float(args.get('z', 0.1))
                approach_height = float(args.get('approach_height', 0.1))
                success = self._pick_at_position(x, y, z, approach_height)
                return f"Picked at ({x:.2f}, {y:.2f}, {z:.2f})!" if success else "Pick failed"

            elif func_name == 'place_at_position':
                x = float(args.get('x', 0.3))
                y = float(args.get('y', 0.0))
                z = float(args.get('z', 0.1))
                retract_height = float(args.get('retract_height', 0.1))
                success = self._place_at_position(x, y, z, retract_height)
                return f"Placed at ({x:.2f}, {y:.2f}, {z:.2f})!" if success else "Place failed"

            # Directional Motion
            elif func_name == 'point_at_direction':
                direction = args.get('direction', 'forward')
                success = self._point(direction)
                return f"Pointing {direction}!" if success else "Point motion failed"

            # Status
            elif func_name == 'get_robot_state':
                return self._build_context()

            else:
                return f"Unknown function: {func_name}"

        except Exception as e:
            rospy.logerr(f"Function execution error: {e}")
            import traceback
            traceback.print_exc()
            return f"Execution error: {e}"

    def _analyze_scene(self):
        """Analyze the current camera view to detect objects."""
        if self.current_rgb_image is None:
            return "No camera image available"

        # For now, return basic image analysis
        # In a full implementation, this would use object detection
        h, w = self.current_rgb_image.shape[:2]

        # Simple color-based detection for demonstration
        hsv = cv2.cvtColor(self.current_rgb_image, cv2.COLOR_BGR2HSV)

        # Detect red objects
        red_lower = np.array([0, 100, 100])
        red_upper = np.array([10, 255, 255])
        red_mask = cv2.inRange(hsv, red_lower, red_upper)

        # Detect blue objects
        blue_lower = np.array([100, 100, 100])
        blue_upper = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

        # Detect green objects
        green_lower = np.array([40, 100, 100])
        green_upper = np.array([80, 255, 255])
        green_mask = cv2.inRange(hsv, green_lower, green_upper)

        objects = []

        for color, mask in [('red', red_mask), ('blue', blue_mask), ('green', green_mask)]:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # Filter small noise
                    M = cv2.moments(cnt)
                    if M['m00'] > 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        objects.append(f"{color} object at pixel ({cx}, {cy})")

        if objects:
            return "Detected: " + ", ".join(objects)
        else:
            return "Scene analyzed - no distinct colored objects detected. Image size: {}x{}".format(w, h)

    def _get_3d_position(self, pixel_x, pixel_y):
        """Convert pixel coordinates to 3D position using depth data."""
        if self.current_depth_image is None:
            return "No depth data available"

        if self.camera_info is None:
            # Use default camera parameters
            fx, fy = 615.0, 615.0  # Focal lengths
            cx, cy = 320.0, 240.0  # Principal point
        else:
            fx = self.camera_info.K[0]
            fy = self.camera_info.K[4]
            cx = self.camera_info.K[2]
            cy = self.camera_info.K[5]

        # Get depth at pixel
        h, w = self.current_depth_image.shape[:2]
        pixel_x = max(0, min(pixel_x, w-1))
        pixel_y = max(0, min(pixel_y, h-1))

        depth = self.current_depth_image[pixel_y, pixel_x]

        if np.isnan(depth) or depth == 0:
            return f"Invalid depth at pixel ({pixel_x}, {pixel_y})"

        # Convert to meters if in mm
        if depth > 10:  # Likely in mm
            depth = depth / 1000.0

        # Convert to 3D camera coordinates
        x_cam = (pixel_x - cx) * depth / fx
        y_cam = (pixel_y - cy) * depth / fy
        z_cam = depth

        # Create PointStamped in camera frame
        # Note: We assume the depth image frame is 'camera_depth_optical_frame'
        # If not, we might need to verify the frame_id from the depth message header
        point_cam = PointStamped()
        point_cam.header.frame_id = "camera_depth_optical_frame" 
        point_cam.header.stamp = rospy.Time(0) # Get latest transform
        point_cam.point.x = x_cam
        point_cam.point.y = y_cam
        point_cam.point.z = z_cam

        try:
            # Transform to robot base frame (world)
            # The target frame should be 'world' or 'm1n6s300_link_base'
            target_frame = "world"
            if not self.tf_buffer.can_transform(target_frame, point_cam.header.frame_id, rospy.Time(0), rospy.Duration(1.0)):
                rospy.logwarn(f"Cannot transform from {point_cam.header.frame_id} to {target_frame}")
                return f"3D position (camera frame): x={x_cam:.3f}m, y={y_cam:.3f}m, z={z_cam:.3f}m (Transform failed)"

            point_world = self.tf_buffer.transform(point_cam, target_frame)
            
            x_world = point_world.point.x
            y_world = point_world.point.y
            z_world = point_world.point.z
            
            rospy.loginfo(f"Transformed point: ({x_cam:.3f}, {y_cam:.3f}, {z_cam:.3f})_cam -> ({x_world:.3f}, {y_world:.3f}, {z_world:.3f})_world")
            
            return f"3D position (world frame): x={x_world:.3f}m, y={y_world:.3f}m, z={z_world:.3f}m"

        except Exception as e:
            rospy.logerr(f"Transform error: {e}")
            return f"3D position (camera frame): x={x_cam:.3f}m, y={y_cam:.3f}m, z={z_cam:.3f}m (Transform error)"

    def _build_context(self):
        """Build context string about current robot state."""
        parts = []

        if self.joint_state_received and self.current_joint_positions:
            arm_positions = self.current_joint_positions[:6]
            arm_deg = [np.degrees(p) for p in arm_positions]
            parts.append("Arm joints (deg): " + str([f"{d:.1f}" for d in arm_deg]))

            if len(self.current_joint_positions) > 6:
                finger_positions = self.current_joint_positions[6:9]
                parts.append("Fingers: " + str([f"{p:.2f}" for p in finger_positions]))
        else:
            parts.append("Joint state: waiting for data")

        parts.append(f"Depth camera: {'available' if self.current_depth_image is not None else 'not available'}")
        parts.append(f"RGB camera: {'available' if self.current_rgb_image is not None else 'not available'}")
        parts.append(f"Cartesian path planning: {'available' if self.cartesian_path_service else 'not available'}")

        return "\n".join(parts)

    # ==================== Motion Execution Methods ====================

    def _execute_arm_trajectory(self, waypoints, durations):
        """Execute arm trajectory."""
        if not self.arm_client.wait_for_server(timeout=rospy.Duration(2.0)):
            rospy.logwarn("Arm action server not available")
            return False

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.arm_joint_names

        for positions, duration in zip(waypoints, durations):
            point = JointTrajectoryPoint()
            point.positions = list(positions)
            point.velocities = [0.0] * 6
            point.time_from_start = rospy.Duration(duration)
            goal.trajectory.points.append(point)

        rospy.loginfo(f"Executing arm trajectory with {len(waypoints)} waypoints")
        self.arm_client.send_goal(goal)
        return self.arm_client.wait_for_result(timeout=rospy.Duration(30.0))

    def _execute_finger_trajectory(self, positions, duration):
        """Execute finger trajectory."""
        if not self.finger_client.wait_for_server(timeout=rospy.Duration(2.0)):
            rospy.logwarn("Finger action server not available")
            return False

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = self.finger_joint_names

        point = JointTrajectoryPoint()
        point.positions = list(positions)
        point.velocities = [0.0] * 3
        point.time_from_start = rospy.Duration(duration)
        goal.trajectory.points.append(point)

        rospy.loginfo("Executing finger trajectory")
        self.finger_client.send_goal(goal)
        return self.finger_client.wait_for_result(timeout=rospy.Duration(10.0))

    def _wave(self, num_waves=3):
        """Execute wave gesture."""
        rospy.loginfo(f"Executing wave with {num_waves} oscillations")

        current = self.current_joint_positions[:6] if self.joint_state_received else [0]*6

        # Wave position: arm up
        wave_up = [current[0], np.radians(180), np.radians(180), np.radians(180), np.radians(180), 0]
        wave_left = wave_up.copy()
        wave_left[5] = np.radians(30)
        wave_right = wave_up.copy()
        wave_right[5] = np.radians(-30)

        # Build trajectory
        waypoints = [wave_up]
        durations = [2.0]

        for i in range(num_waves):
            waypoints.append(wave_left)
            durations.append(durations[-1] + 0.5)
            waypoints.append(wave_right)
            durations.append(durations[-1] + 0.5)

        waypoints.append(wave_up)
        durations.append(durations[-1] + 0.5)

        return self._execute_arm_trajectory(waypoints, durations)

    def _move_to_home(self):
        """Move to home position."""
        rospy.loginfo("Moving to home position")
        home = [np.radians(270), np.radians(180), np.radians(180), 0, 0, 0]
        return self._execute_arm_trajectory([home], [3.0])

    def _open_gripper(self):
        """Open gripper."""
        rospy.loginfo("Opening gripper")
        return self._execute_finger_trajectory([0.0, 0.0, 0.0], 2.0)

    def _close_gripper(self):
        """Close gripper."""
        rospy.loginfo("Closing gripper")
        return self._execute_finger_trajectory([1.2, 1.2, 1.2], 2.0)

    def _move_to_joints(self, joints_deg, duration=3.0):
        """Move to specific joint positions (in degrees)."""
        joints_rad = [np.radians(d) for d in joints_deg]
        rospy.loginfo(f"Moving to joints: {joints_deg} degrees")
        return self._execute_arm_trajectory([joints_rad], [duration])

    def _point(self, direction):
        """Point in a direction."""
        rospy.loginfo(f"Pointing {direction}")

        # Define pointing positions based on direction
        positions = {
            'forward': [np.radians(0), np.radians(180), np.radians(180), 0, np.radians(90), 0],
            'left': [np.radians(90), np.radians(180), np.radians(180), 0, np.radians(90), 0],
            'right': [np.radians(-90), np.radians(180), np.radians(180), 0, np.radians(90), 0],
            'up': [np.radians(0), np.radians(90), np.radians(180), 0, 0, 0],
            'down': [np.radians(0), np.radians(270), np.radians(180), 0, 0, 0],
        }

        target = positions.get(direction.lower(), positions['forward'])
        return self._execute_arm_trajectory([target], [3.0])

    def _move_to_cartesian(self, x, y, z):
        """Move end effector to a Cartesian position using MoveIt."""
        if not self.cartesian_path_service:
            rospy.logwarn("Cartesian path service not available, using joint interpolation")
            # Fallback to approximate joint positions
            # This is a simplified approximation - real IK would be needed
            return self._approximate_cartesian_motion(x, y, z)

        rospy.loginfo(f"Moving to Cartesian position: ({x:.3f}, {y:.3f}, {z:.3f})")

        try:
            req = GetCartesianPathRequest()
            req.header.frame_id = 'world'
            req.header.stamp = rospy.Time.now()
            req.group_name = self.planning_group
            req.link_name = self.end_effector_link

            # Create target waypoint
            target_pose = Pose()
            target_pose.position.x = x
            target_pose.position.y = y
            target_pose.position.z = z
            # Keep current orientation (simplified)
            target_pose.orientation.w = 1.0

            req.waypoints = [target_pose]
            req.max_step = 0.01
            req.jump_threshold = 0.0
            req.avoid_collisions = True

            # Get current robot state
            if self.joint_state_received:
                req.start_state.joint_state.name = self.arm_joint_names
                req.start_state.joint_state.position = self.current_joint_positions[:6]

            response = self.cartesian_path_service(req)

            if response.fraction < 0.9:
                rospy.logwarn(f"Could only plan {response.fraction*100:.1f}% of path")
                return False

            # Execute the trajectory
            trajectory = response.solution.joint_trajectory
            if len(trajectory.points) > 0:
                waypoints = [list(p.positions) for p in trajectory.points]
                durations = [p.time_from_start.to_sec() for p in trajectory.points]
                return self._execute_arm_trajectory(waypoints, durations)

            return False

        except Exception as e:
            rospy.logerr(f"Cartesian motion error: {e}")
            return False

    def _approximate_cartesian_motion(self, x, y, z):
        """Approximate Cartesian motion using predefined positions."""
        # This is a fallback when MoveIt cartesian path is not available
        # Map target position to approximate joint angles

        # Simple heuristic based on workspace zones
        base_angle = np.arctan2(y, x)  # Approximate base rotation

        # Very simplified approximation
        joints = [
            base_angle,
            np.radians(180),  # shoulder
            np.radians(180),  # elbow
            np.radians(0),    # wrist 1
            np.radians(90),   # wrist 2
            np.radians(0),    # wrist 3
        ]

        rospy.loginfo("Using approximate joint motion (MoveIt not available)")
        return self._execute_arm_trajectory([joints], [3.0])

    def _move_relative(self, dx, dy, dz):
        """Move end effector relative to current position."""
        # Get current end effector position from TF
        try:
            transform = self.tf_buffer.lookup_transform(
                'world', self.end_effector_link, rospy.Time(0), rospy.Duration(1.0)
            )
            current_x = transform.transform.translation.x
            current_y = transform.transform.translation.y
            current_z = transform.transform.translation.z

            return self._move_to_cartesian(current_x + dx, current_y + dy, current_z + dz)
        except Exception as e:
            rospy.logwarn(f"Could not get current position: {e}")
            # Fallback to simple relative motion
            return self._approximate_cartesian_motion(dx, dy, dz)

    def _pick_at_position(self, x, y, z, approach_height=0.1):
        """
        Composite pick action:
        1. Move above the object
        2. Open gripper
        3. Descend to object
        4. Close gripper
        5. Lift up
        """
        rospy.loginfo(f"Executing pick at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Step 1: Move above object
        rospy.loginfo("Step 1: Moving above object")
        if not self._move_to_cartesian(x, y, z + approach_height):
            rospy.logwarn("Failed to move above object")
            return False
        rospy.sleep(0.5)

        # Step 2: Open gripper
        rospy.loginfo("Step 2: Opening gripper")
        if not self._open_gripper():
            rospy.logwarn("Failed to open gripper")
            return False
        rospy.sleep(0.5)

        # Step 3: Descend to object
        rospy.loginfo("Step 3: Descending to object")
        if not self._move_to_cartesian(x, y, z):
            rospy.logwarn("Failed to descend to object")
            return False
        rospy.sleep(0.5)

        # Step 4: Close gripper
        rospy.loginfo("Step 4: Closing gripper")
        if not self._close_gripper():
            rospy.logwarn("Failed to close gripper")
            return False
        rospy.sleep(0.5)

        # Step 5: Lift up
        rospy.loginfo("Step 5: Lifting up")
        if not self._move_to_cartesian(x, y, z + approach_height):
            rospy.logwarn("Failed to lift object")
            return False

        rospy.loginfo("Pick completed successfully!")
        return True

    def _place_at_position(self, x, y, z, retract_height=0.1):
        """
        Composite place action:
        1. Move above target position
        2. Descend to target
        3. Open gripper
        4. Retract upward
        """
        rospy.loginfo(f"Executing place at ({x:.3f}, {y:.3f}, {z:.3f})")

        # Step 1: Move above target
        rospy.loginfo("Step 1: Moving above target")
        if not self._move_to_cartesian(x, y, z + retract_height):
            rospy.logwarn("Failed to move above target")
            return False
        rospy.sleep(0.5)

        # Step 2: Descend to target
        rospy.loginfo("Step 2: Descending to target")
        if not self._move_to_cartesian(x, y, z):
            rospy.logwarn("Failed to descend to target")
            return False
        rospy.sleep(0.5)

        # Step 3: Open gripper
        rospy.loginfo("Step 3: Opening gripper")
        if not self._open_gripper():
            rospy.logwarn("Failed to open gripper")
            return False
        rospy.sleep(0.5)

        # Step 4: Retract upward
        rospy.loginfo("Step 4: Retracting")
        if not self._move_to_cartesian(x, y, z + retract_height):
            rospy.logwarn("Failed to retract")
            return False

        rospy.loginfo("Place completed successfully!")
        return True

    def publish_status(self, status):
        """Publish status."""
        self.status_pub.publish(String(data=status))

    def run(self):
        """Main run loop."""
        rospy.loginfo("LLM Controller running. Waiting for commands on /user_command")
        rospy.spin()


def main():
    """Main entry point."""
    try:
        controller = LLMController()
        controller.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"LLM Controller failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
