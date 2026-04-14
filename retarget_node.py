import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
import json
import numpy as np

def angle_between(a, b, c):
    """Calculate angle at point B between points A, B, C"""
    ba = np.array([a['x'] - b['x'], a['y'] - b['y'], a['z'] - b['z']])
    bc = np.array([c['x'] - b['x'], c['y'] - b['y'], c['z'] - b['z']])
    cosine = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))

class RetargetNode(Node):
    def __init__(self):
        super().__init__('retarget_node')
        self.subscription = self.create_subscription(
            String,
            '/pose_landmarks',
            self.listener_callback,
            10)
        self.publisher_ = self.create_publisher(JointState, '/joint_states', 10)
        self.get_logger().info('RetargetNode started.')

    def listener_callback(self, msg):
        coords = json.loads(msg.data)

        # Calculate joint angles
        left_elbow_angle  = angle_between(coords['LEFT_SHOULDER'],  coords['LEFT_ELBOW'],  coords['LEFT_WRIST'])
        right_elbow_angle = angle_between(coords['RIGHT_SHOULDER'], coords['RIGHT_ELBOW'], coords['RIGHT_WRIST'])
        left_knee_angle   = angle_between(coords['LEFT_HIP'],       coords['LEFT_KNEE'],   coords['LEFT_KNEE'])
        right_knee_angle  = angle_between(coords['RIGHT_HIP'],      coords['RIGHT_KNEE'],  coords['RIGHT_KNEE'])

        # Pack into JointState message
        joint_msg = JointState()
        joint_msg.header.stamp = self.get_clock().now().to_msg()
        joint_msg.name     = ['left_elbow', 'right_elbow', 'left_knee', 'right_knee']
        joint_msg.position = [left_elbow_angle, right_elbow_angle, left_knee_angle, right_knee_angle]

        self.publisher_.publish(joint_msg)
        self.get_logger().info(
            f'L_elbow: {left_elbow_angle:.2f} | R_elbow: {right_elbow_angle:.2f} | '
            f'L_knee: {left_knee_angle:.2f} | R_knee: {right_knee_angle:.2f} rad'
        )

def main(args=None):
    rclpy.init(args=args)
    node = RetargetNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
