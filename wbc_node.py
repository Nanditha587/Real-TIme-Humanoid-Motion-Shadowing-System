mport rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# Safety limit — max joint angle change per step (radians)
MAX_DELTA = 0.1

class WBCNode(Node):
    def __init__(self):
        super().__init__('wbc_node')

        # Subscribe to joint angles from retarget_node
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.joint_callback, 10)

        # Subscribe to CoM state from lipm_node
        self.com_sub = self.create_subscription(
            Float64MultiArray, '/com_state', self.com_callback, 10)

        # Publish final safe joint commands
        self.publisher_ = self.create_publisher(JointState, '/safe_joint_commands', 10)

        self.latest_joints = {}
        self.com_x = 0.0
        self.com_y = 0.0

        self.get_logger().info('WBCNode started.')

    def com_callback(self, msg):
        self.com_x = msg.data[0]
        self.com_y = msg.data[1]

    def joint_callback(self, msg):
        angles = dict(zip(msg.name, msg.position))

        # Balance correction — if CoM drifts right, bend left knee to rebalance
        balance_correction = np.clip(-self.com_x * 0.3, -0.3, 0.3)

        # Apply correction to knee joints
        left_knee  = angles.get('left_knee',  0.0) + balance_correction
        right_knee = angles.get('right_knee', 0.0) - balance_correction

        # Clamp all joints within safety limits
        safe_angles = {
            'left_elbow':  np.clip(angles.get('left_elbow',  0.0), 0.0, np.pi),
            'right_elbow': np.clip(angles.get('right_elbow', 0.0), 0.0, np.pi),
            'left_knee':   np.clip(left_knee,  0.0, np.pi),
            'right_knee':  np.clip(right_knee, 0.0, np.pi),
        }

        # Publish safe commands
        out = JointState()
        out.header.stamp = self.get_clock().now().to_msg()
        out.name     = list(safe_angles.keys())
        out.position = list(safe_angles.values())
        self.publisher_.publish(out)

        self.get_logger().info(
            f'L_elbow: {safe_angles["left_elbow"]:.2f} | '
            f'R_elbow: {safe_angles["right_elbow"]:.2f} | '
            f'L_knee: {safe_angles["left_knee"]:.2f} | '
            f'R_knee: {safe_angles["right_knee"]:.2f} | '
            f'balance: {balance_correction:.3f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = WBCNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
