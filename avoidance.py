#!/usr/bin/env python3

"""
ROS2 Obstacle Detection Node
Tests orange cone obstacle detection using HSV color segmentation
Subscribes to /camera/image_raw ROS2 topic
"""

import cv2
import numpy as np
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ObstacleDetectionNode(Node):
    """
    ROS2 node for obstacle detection
    """

    def __init__(self):
        """
        Initialize the obstacle detection node.
        """
        super().__init__("obstacle_detection_node")

        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()
        # --- Configuration ---

        # Orange cone HSV range (orange color detection)
        self.CONE_LOWER_HSV = np.array([100, 100, 150])
        self.CONE_UPPER_HSV = np.array([180, 255, 255])

        # Obstacle detection parameters
        self.OBSTACLE_MIN_AREA = 25  # Minimum contour area to consider as obstacle
        self.OBSTACLE_DANGER_ZONE_Y = 0.70  # Bottom 30% of frame is danger zone
        self.OBSTACLE_WARNING_ZONE_Y = 0.50  # Bottom 50% is warning zone

        # ROI boundaries
        self.ROI_BLIND_SPOT_Y = 1.0  # Bottom edge (100% of frame height)
        self.ROI_TOP_Y = 0.48  # Start detection from 40% down
        self.ROI_BOTTOM_WIDTH = 0.88  # 100% of frame width at bottom
        self.ROI_TOP_WIDTH = 0.5  # 50% of frame width at top

        # Obstacle detection results
        self.obstacles = []

        # FPS tracking
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Display window option
        self.show_visualization = True

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

        self.get_logger().info("Obstacle Detection Node started")
        self.get_logger().info("Subscribing to: /camera/image_raw")
        self.get_logger().info("Press 'q' to quit")

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            processed_frame, obstacle_mask = self.process_frame(frame)

            if self.show_visualization:
                cv2.imshow("Obstacle Detection", processed_frame)
                cv2.imshow("Obstacle Mask", obstacle_mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.get_logger().info("Quit requested")
                    rclpy.shutdown()

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def create_obstacle_roi_mask(self, height, width):
        """
        Create ROI mask for obstacle detection.
        Focuses on the path ahead where robot will travel.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # ROI boundaries
        blind_spot_y = int(height * self.ROI_BLIND_SPOT_Y)  # Bottom edge
        top_y = int(height * self.ROI_TOP_Y)  # Start detection from top

        # Create a trapezoidal ROI that matches the robot's path
        bottom_width = int(width * self.ROI_BOTTOM_WIDTH)  # Width at bottom
        top_width = int(width * self.ROI_TOP_WIDTH)  # Width at top

        center_x = width // 2

        # Define trapezoid points
        pts = np.array(
            [
                [center_x - top_width // 2, top_y],  # Top-left
                [center_x + top_width // 2, top_y],  # Top-right
                [center_x + bottom_width // 2, blind_spot_y],  # Bottom-right
                [center_x - bottom_width // 2, blind_spot_y],  # Bottom-left
            ],
            np.int32,
        )

        cv2.fillPoly(mask, [pts], 255)
        return mask, pts

    def detect_obstacles(self, frame, height, width):
        """
        Detect orange cones in the frame using HSV color segmentation.
        Returns list of obstacles with their position, size, and zone.
        """
        # Create obstacle detection ROI
        roi_mask, roi_pts = self.create_obstacle_roi_mask(height, width)

        # Convert to HSV and detect orange
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        orange_mask = cv2.inRange(hsv, self.CONE_LOWER_HSV, self.CONE_UPPER_HSV)

        # Apply ROI mask
        orange_mask = cv2.bitwise_and(orange_mask, roi_mask)

        # Morphological operations to clean up noise
        kernel = np.ones((25, 25), np.uint8)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

        # Find contours
        contours, _ = cv2.findContours(
            orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        obstacles = []
        danger_zone_y = int(height * self.OBSTACLE_DANGER_ZONE_Y)
        warning_zone_y = int(height * self.OBSTACLE_WARNING_ZONE_Y)
        center_x = width // 2

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.OBSTACLE_MIN_AREA:
                continue

            # Get bounding box and centroid
            x, y, w, h = cv2.boundingRect(contour)

            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx = x + w // 2
                cy = y + h // 2

            # Determine zone based on y position (bottom of bounding box)
            bottom_y = y + h
            if bottom_y >= danger_zone_y:
                zone = "danger"
            elif bottom_y >= warning_zone_y:
                zone = "warning"
            else:
                zone = "safe"

            # Calculate horizontal offset from center
            offset_from_center = cx - center_x

            # Estimate relative size/distance
            relative_size = area / (height * width)

            obstacles.append(
                {
                    "contour": contour,
                    "bbox": (x, y, w, h),
                    "centroid": (cx, cy),
                    "area": area,
                    "zone": zone,
                    "offset": offset_from_center,
                    "relative_size": relative_size,
                    "bottom_y": bottom_y,
                }
            )

        # Sort by bottom_y (closest first)
        obstacles.sort(key=lambda o: o["bottom_y"], reverse=True)

        return obstacles, orange_mask, roi_pts

    def visualize_obstacles(self, frame, obstacles, roi_pts, height, width):
        """Visualize detected obstacles and zones."""
        # Draw obstacle detection ROI (trapezoid)
        cv2.polylines(frame, [roi_pts], True, (255, 128, 0), 2)

        # Draw zone lines
        danger_y = int(height * self.OBSTACLE_DANGER_ZONE_Y)
        warning_y = int(height * self.OBSTACLE_WARNING_ZONE_Y)

        cv2.line(frame, (0, danger_y), (width, danger_y), (0, 0, 255), 1)
        cv2.putText(frame, "DANGER ZONE", (width - 120, danger_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        cv2.line(frame, (0, warning_y), (width, warning_y), (0, 165, 255), 1)
        cv2.putText(frame, "WARNING ZONE", (width - 130, warning_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

        # Draw detected obstacles
        for obs in obstacles:
            x, y, w, h = obs["bbox"]
            cx, cy = obs["centroid"]
            zone = obs["zone"]

            # Color based on zone
            if zone == "danger":
                color = (0, 0, 255)  # Red
            elif zone == "warning":
                color = (0, 165, 255)  # Orange
            else:
                color = (0, 255, 255)  # Yellow

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (cx, cy), 5, color, -1)

            label = f"{zone.upper()} off:{obs['offset']:+d}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Show obstacle count
        if len(obstacles) > 0:
            cv2.putText(frame, f"Obstacles: {len(obstacles)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def process_frame(self, frame):
        """Process a single frame."""
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]

        # Obstacle Detection
        self.obstacles, obstacle_mask, obstacle_roi_pts = self.detect_obstacles(frame, height, width)

        # Visualizations
        self.visualize_obstacles(frame, self.obstacles, obstacle_roi_pts, height, width)

        # Update FPS
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 0.5:
            self.fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

        # Display FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (width - 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        return frame, obstacle_mask

    def destroy_node(self):
        """Cleanup resources"""
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = ObstacleDetectionNode()

        print("\n" + "=" * 60)
        print("ROS2 Obstacle Detection Node")
        print("=" * 60)
        print("Subscribing to: /camera/image_raw")
        print("Orange obstacle HSV range:", node.CONE_LOWER_HSV, "-", node.CONE_UPPER_HSV)
        print("\nPress 'q' in visualization window to quit")
        print("=" * 60 + "\n")

        rclpy.spin(node)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
