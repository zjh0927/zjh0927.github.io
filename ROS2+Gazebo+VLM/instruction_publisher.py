"""
指令发布节点：向 /vla/instruction 话题发布自然语言控制指令。

用法:
    python instruction_publisher.py "走到厨房"
    python instruction_publisher.py "去客厅"
"""

import sys
import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class InstructionPublisher(Node):
    def __init__(self, instruction: str):
        super().__init__('instruction_publisher')
        self.publisher = self.create_publisher(String, '/vla/instruction', 10)
        self.instruction = instruction

    def publish_once(self):
        msg = String()
        msg.data = self.instruction
        self.publisher.publish(msg)
        self.get_logger().info(f'Published instruction: {self.instruction}')


def main():
    rclpy.init(args=sys.argv)

    if len(sys.argv) < 2:
        print("Usage: python instruction_publisher.py \"<instruction>\"")
        sys.exit(1)

    instruction = ' '.join(sys.argv[1:])
    node = InstructionPublisher(instruction)
    node.publish_once()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
