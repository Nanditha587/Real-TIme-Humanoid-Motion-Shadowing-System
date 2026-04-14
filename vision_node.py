import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import cv2
import mediapipe as mp
import json

mp_pose = mp.solutions.pose #POSE ESTIMATION MODULE SO THAT WE DONT HAVE TO WRITE THIS ALL THE TIME

#ENUMERATION
JOINTS = {
    "LEFT_SHOULDER":  mp_pose.PoseLandmark.LEFT_SHOULDER,
    "RIGHT_SHOULDER": mp_pose.PoseLandmark.RIGHT_SHOULDER,
    "LEFT_ELBOW":     mp_pose.PoseLandmark.LEFT_ELBOW,
    "RIGHT_ELBOW":    mp_pose.PoseLandmark.RIGHT_ELBOW,
    "LEFT_WRIST":     mp_pose.PoseLandmark.LEFT_WRIST,
    "RIGHT_WRIST":    mp_pose.PoseLandmark.RIGHT_WRIST,
    "LEFT_HIP":       mp_pose.PoseLandmark.LEFT_HIP,
    "RIGHT_HIP":      mp_pose.PoseLandmark.RIGHT_HIP,
    "LEFT_KNEE":      mp_pose.PoseLandmark.LEFT_KNEE,
    "RIGHT_KNEE":     mp_pose.PoseLandmark.RIGHT_KNEE,
}

class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.publisher_ = self.create_publisher(String, '/pose_landmarks', 10)
        self.cap = cv2.VideoCapture(0)
        self.pose = mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6)
        self.timer = self.create_timer(0.033, self.timer_callback)
        self.get_logger().info('VisionNode started.')

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb)
        if results.pose_landmarks:
            coords = {}
            for name, idx in JOINTS.items():
                lm = results.pose_landmarks.landmark[idx.value]
                coords[name] = {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
            msg = String()
            msg.data = json.dumps(coords)
            self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
