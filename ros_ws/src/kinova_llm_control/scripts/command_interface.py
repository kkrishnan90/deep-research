#!/usr/bin/env python3
"""
Command Interface for Kinova LLM Controller.

This node provides a simple command-line interface for sending
commands to the LLM controller.

Usage:
    rosrun kinova_llm_control command_interface.py

Author: Developer
License: MIT
"""

import rospy
from std_msgs.msg import String


class CommandInterface:
    """Simple command-line interface for sending commands."""

    def __init__(self):
        """Initialize the command interface."""
        rospy.init_node('command_interface', anonymous=True)

        # Publishers
        self.command_pub = rospy.Publisher(
            '/user_command',
            String,
            queue_size=10
        )

        # Subscribers
        self.status_sub = rospy.Subscriber(
            '/llm_status',
            String,
            self.status_callback,
            queue_size=10
        )
        self.response_sub = rospy.Subscriber(
            '/llm_response',
            String,
            self.response_callback,
            queue_size=10
        )

        self.current_status = "unknown"

        rospy.loginfo("Command Interface initialized")
        rospy.loginfo("Type your commands below. Press Ctrl+C to exit.")

    def status_callback(self, msg):
        """Handle status updates."""
        self.current_status = msg.data

    def response_callback(self, msg):
        """Handle response from LLM controller."""
        print("\n" + "=" * 60)
        print("LLM Response:")
        print("=" * 60)
        print(msg.data)
        print("=" * 60 + "\n")

    def send_command(self, command):
        """Send a command to the LLM controller."""
        if not command.strip():
            return

        rospy.loginfo(f"Sending command: {command}")
        self.command_pub.publish(String(data=command))

    def run(self):
        """Main run loop with command-line input."""
        # Wait for publisher to connect
        rospy.sleep(1.0)

        print("\n" + "=" * 60)
        print("Kinova Mico LLM Command Interface")
        print("=" * 60)
        print("Enter natural language commands to control the robot.")
        print("Examples:")
        print("  - 'Pick up the red cube'")
        print("  - 'Move to home position'")
        print("  - 'What objects do you see?'")
        print("  - 'Place the object on the right side'")
        print("=" * 60 + "\n")

        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            try:
                command = input(f"[{self.current_status}] > ")
                self.send_command(command)
            except EOFError:
                break
            except KeyboardInterrupt:
                break
            rate.sleep()


def main():
    """Main entry point."""
    try:
        interface = CommandInterface()
        interface.run()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
