import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# Robot constants
G = 9.81        # gravity (m/s²)
Z_HEIGHT = 0.8  # robot CoM height from ground (metres)

class LIPMNode(Node):
    def __init__(self):
        super().__init__('lipm_node')

        # Subscribe to joint angles from retarget_node
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10)




        # Publish CoM state [x, y, vx, vy]
        self.publisher_ = self.create_publisher(Float64MultiArray, '/com_state', 10)

        # CoM state: position and velocity
        self.com_x  = 0.0
        self.com_y  = 0.0
        self.com_vx = 0.0
        self.com_vy = 0.0

        # ZMP (feet center — with correction limits)
        self.zmp_x = 0.0
        self.zmp_y = 0.0
        self.zmp_limit = 0.1  # max ZMP shift in metres (foot size boundary)

        self.dt = 0.033  # 30fps timestep
        self.get_logger().info('LIPMNode started.')

    def joint_callback(self, msg):
        # Extract left and right elbow angles
        angles = dict(zip(msg.name, msg.position))
        left_elbow  = angles.get('left_elbow',  0.0)
        right_elbow = angles.get('right_elbow', 0.0)

        # Estimate CoM shift from arm asymmetry
        # If right arm raises more → CoM shifts right
        arm_asymmetry = right_elbow - left_elbow
      # Drive CoM directly from asymmetry instead of accumulating
        self.com_x = arm_asymmetry * 0.05 # scaling factor

       # ZMP correction — shift ZMP toward CoM to resist falling
        self.zmp_x = np.clip(self.com_x * 0.5, -self.zmp_limit, self.zmp_limit)
        self.zmp_y = np.clip(self.com_y * 0.5, -self.zmp_limit, self.zmp_limit)

        # 3D-LIPM equations
        accel_x = (G / Z_HEIGHT) * (self.com_x - self.zmp_x)
        accel_y = (G / Z_HEIGHT) * (self.com_y - self.zmp_y)

        # Integrate acceleration → velocity → position
        self.com_vx += accel_x * self.dt
        self.com_vy += accel_y * self.dt
        self.com_x  += self.com_vx * self.dt
        self.com_y  += self.com_vy * self.dt

        # Publish CoM state
        msg_out = Float64MultiArray()
        msg_out.data = [self.com_x, self.com_y, self.com_vx, self.com_vy]
        self.publisher_.publish(msg_out)

        self.get_logger().info(
            f'CoM x: {self.com_x:.4f}  y: {self.com_y:.4f} | '
            f'vel x: {self.com_vx:.4f}  y: {self.com_vy:.4f}'
        )

def main(args=None):
    rclpy.init(args=args)
    node = LIPMNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
