import rclpy
from rclpy.node import Node
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo

class USBCameraNode(Node):
    def __init__(self):
        super().__init__('usb_camera_node')

        # Declare params
        self.declare_parameter("video_device", "/dev/video1")
        self.declare_parameter("width", 640)
        self.declare_parameter("height", 480)
        self.declare_parameter("fps", 60)

        device = self.get_parameter("video_device").value
        width = self.get_parameter("width").value
        height = self.get_parameter("height").value
        fps = self.get_parameter("fps").value

        # OpenCV video capture
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)

        if not self.cap.isOpened():
            self.get_logger().error(f"Cannot open camera device: {device}")

        # Publishers
        self.image_pub = self.create_publisher(Image, "/camera/image_raw", 10)
        self.info_pub  = self.create_publisher(CameraInfo, "/camera/camera_info", 10)

        self.bridge = CvBridge()

        # Timer — publish at FPS
        timer_period = 1.0 / fps
        self.timer = self.create_timer(timer_period, self.publish_frame)

        self.get_logger().info("USB camera node started.")

    def publish_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warning("Camera frame grab failed.")
            return

        # Convert to ROS2 Image
        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"

        # CameraInfo (minimal example)
        info = CameraInfo()
        info.header = msg.header
        info.width = frame.shape[1]
        info.height = frame.shape[0]

        # Publish both
        self.image_pub.publish(msg)
        self.info_pub.publish(info)

def main(args=None):
    rclpy.init(args=args)
    node = USBCameraNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
