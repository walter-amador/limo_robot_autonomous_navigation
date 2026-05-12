#!/usr/bin/env python3
"""
HSV Color Tuner for Orange Cone Detection (ROS2 Version)
Use this script to find the optimal HSV range for your specific orange cones.

Usage:
  - Run as ROS2 node: ros2 run <package> hsv_tuner
  - Or: python3 hsv_tuner.py

Controls:
  - Adjust trackbars to find the optimal HSV range
  - Press 's' to save current values
  - Press 'q' to quit
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def nothing(x):
    pass


class HSVTunerNode(Node):
    def __init__(self):
        super().__init__("hsv_tuner_node")
        
        self.bridge = CvBridge()
        self.frame = None
        
        # Initialize default HSV values
        self.h_min, self.h_max = 0, 25
        self.s_min, self.s_max = 100, 255
        self.v_min, self.v_max = 100, 255
        
        # Create windows
        cv2.namedWindow("HSV Tuner", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Mask", cv2.WINDOW_NORMAL)

        # Create trackbars for HSV range
        cv2.createTrackbar("H Min", "HSV Tuner", 0, 180, nothing)
        cv2.createTrackbar("H Max", "HSV Tuner", 25, 180, nothing)
        cv2.createTrackbar("S Min", "HSV Tuner", 100, 255, nothing)
        cv2.createTrackbar("S Max", "HSV Tuner", 255, 255, nothing)
        cv2.createTrackbar("V Min", "HSV Tuner", 100, 255, nothing)
        cv2.createTrackbar("V Max", "HSV Tuner", 255, 255, nothing)

        # Configure QoS profile to match camera publisher
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Subscriber for camera images
        self.image_sub = self.create_subscription(
            Image, "/camera/image_raw", self.image_callback, qos_profile
        )

        # Timer for processing (30 Hz)
        self.timer = self.create_timer(0.033, self.process_frame)

        self.get_logger().info("HSV Tuner Node started")
        self.get_logger().info("Subscribing to: /camera/image_raw")
        
        print("\n" + "=" * 50)
        print("HSV Color Tuner for Orange Cone Detection")
        print("=" * 50)
        print("\nTypical HSV ranges for orange cones:")
        print("  - Traffic cone orange: H=5-25, S=100-255, V=100-255")
        print("  - Bright orange:       H=10-20, S=150-255, V=150-255")
        print("  - Red-orange:          H=0-15, S=100-255, V=100-255")
        print("\nNote: If cones appear red-ish, try H Min=0")
        print("      If cones appear yellow-ish, try H Max=30")
        print("=" * 50 + "\n")

    def image_callback(self, msg):
        """Receive camera images"""
        try:
            self.frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self.frame = cv2.resize(self.frame, (640, 480))
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")

    def process_frame(self):
        """Process the current frame with HSV filtering"""
        if self.frame is None:
            return

        frame = self.frame.copy()

        # Get current trackbar values
        self.h_min = cv2.getTrackbarPos("H Min", "HSV Tuner")
        self.h_max = cv2.getTrackbarPos("H Max", "HSV Tuner")
        self.s_min = cv2.getTrackbarPos("S Min", "HSV Tuner")
        self.s_max = cv2.getTrackbarPos("S Max", "HSV Tuner")
        self.v_min = cv2.getTrackbarPos("V Min", "HSV Tuner")
        self.v_max = cv2.getTrackbarPos("V Max", "HSV Tuner")

        # Create HSV mask
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([self.h_min, self.s_min, self.v_min])
        upper = np.array([self.h_max, self.s_max, self.v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Apply morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_CLOSE, kernel)

        # Find contours and draw on frame
        result = frame.copy()
        contours, _ = cv2.findContours(
            mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_area = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Same threshold as in main code
                total_area += area
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(
                    result,
                    f"Area: {int(area)}",
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),
                    1,
                )

        # Display current HSV values on frame
        info_text = f"H: [{self.h_min}, {self.h_max}]  S: [{self.s_min}, {self.s_max}]  V: [{self.v_min}, {self.v_max}]"
        cv2.putText(
            result, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        cv2.putText(
            result,
            f"Detected area: {int(total_area)}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            result,
            "Press 's' to save, 'q' to quit",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Show windows
        cv2.imshow("Original", frame)
        cv2.imshow("HSV Tuner", result)
        cv2.imshow("Mask", mask_cleaned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.print_values("FINAL")
            rclpy.shutdown()
        elif key == ord("s"):
            self.print_values("SAVED")

    def print_values(self, label):
        """Print current HSV values"""
        print("\n" + "=" * 50)
        print(f"{label} HSV VALUES:")
        print("=" * 50)
        print(f"self.CONE_LOWER_HSV = np.array([{self.h_min}, {self.s_min}, {self.v_min}])")
        print(f"self.CONE_UPPER_HSV = np.array([{self.h_max}, {self.s_max}, {self.v_max}])")
        print("=" * 50 + "\n")

    def destroy_node(self):
        """Cleanup"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)

    try:
        node = HSVTunerNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            cv2.destroyAllWindows()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
