import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import mujoco
import mujoco.viewer
import numpy as np
import threading

SCENE_PATH = '/home/nanditha/mujoco_menagerie/unitree_h1/scene.xml'

# Target standing pose (radians) for each actuator
STAND_POSE = {
    0:  0.0,   # left_hip_yaw
    1:  0.0,   # left_hip_roll
    2:  0.4,   # left_hip_pitch
    3:  0.8,   # left_knee
    4: -0.4,   # left_ankle
    5:  0.0,   # right_hip_yaw
    6:  0.0,   # right_hip_roll
    7:  0.4,   # right_hip_pitch
    8:  0.8,   # right_knee
    9: -0.4,   # right_ankle
    10: 0.0,   # torso
    11: 0.0,   # left_shoulder_pitch
    12: 0.0,   # left_shoulder_roll
    13: 0.0,   # left_shoulder_yaw
    14: 0.0,   # left_elbow
    15: 0.0,   # right_shoulder_pitch
    16: 0.0,   # right_shoulder_roll
    17: 0.0,   # right_shoulder_yaw
    18: 0.0,   # right_elbow
}

# PD gains
KP = 50.0
KD = 5.0N

class MuJoCoNode(Node):
    def __init__(self):
        super().__init__('mujoco_node')

        self.model = mujoco.MjModel.from_xml_path(SCENE_PATH)
        self.data  = mujoco.MjData(self.model)
        self.lock  = threading.Lock()

        # Target angles start at stand pose
        self.target = dict(STAND_POSE)

        self.get_logger().info(f'Loaded H1 model. DOF: {self.model.nv}')

        self.subscription = self.create_subscription(
            JointState,
            '/safe_joint_commands',
            self.joint_callback,
            10)

        self.timer = self.create_timer(0.002, self.control_loop)  # 500Hz control

        self.viewer_thread = threading.Thread(target=self.run_viewer, daemon=True)
        self.viewer_thread.start()
        self.get_logger().info('MuJoCoNode started.')

    def joint_callback(self, msg):
        angles = dict(zip(msg.name, msg.position))
        with self.lock:
            if 'left_elbow'  in angles: self.target[14] = angles['left_elbow']
            if 'right_elbow' in angles: self.target[18] = angles['right_elbow']

    def control_loop(self):
        with self.lock:
            for idx, target_pos in self.target.items():
                actual_pos = self.data.qpos[idx + 7]  # +7 skips root joint
                actual_vel = self.data.qvel[idx + 6]
                self.data.ctrl[idx] = KP * (target_pos - actual_pos) - KD * actual_vel
            mujoco.mj_step(self.model, self.data)

    def run_viewer(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            while viewer.is_running():
                with self.lock:
                    viewer.sync()

def main(args=None):
    rclpy.init(args=args)
    node = MuJoCoNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
