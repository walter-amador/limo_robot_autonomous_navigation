#!/usr/bin/env python3

"""
ROS2 Line Following Node for Agilex LIMO Robot
Subscribes to /camera/color/image_raw and follows green line using PID control
Detects road signs using YOLO and makes navigation decisions
Based on ros_vid_sim.py logic
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np
import json
import time
from ultralytics import YOLO, FastSAM


class LineFollowerNode(Node):
    """
    Line following robot with road sign detection and FSM-based navigation
    """

    def __init__(self):
        super().__init__("line_follower_node")

        # CV Bridge for converting ROS images to OpenCV
        self.bridge = CvBridge()

        # --- Configuration ---
        self.MODEL_PATH = "model/road_signs_yolov8n.pt"
        self.OBSTACLE_MODEL_PATH = "model/FastSAM-s.pt"
        self.LOWER_HSV = np.array([30, 90, 25])
        self.UPPER_HSV = np.array([85, 250, 255])

        # --- Obstacle Avoidance Configuration ---
        self.OBSTACLE_CONFIDENCE = 0.90
        # Robot physical parameters
        self.ROBOT_WIDTH = 0.22  # 22cm width
        self.ROBOT_LENGTH = 0.32  # 32cm length
        self.CAMERA_BLIND_SPOT = 0.10  # 10cm blind spot in front

        # Proximity thresholds (normalized Y position: 0=top/far, 1=bottom/close)
        # Adjusted for 10cm blind spot - must avoid BEFORE obstacle reaches bottom of frame
        self.OBSTACLE_DANGER_ZONE = (
            0.65  # Y > this = very close, MUST steer NOW (was 0.75)
        )
        self.OBSTACLE_WARNING_ZONE = 0.45  # Y > this = start steering (was 0.55)
        self.OBSTACLE_CLEARED_ZONE = 0.85  # Y > this = obstacle passed behind robot
        # Driving corridor (center portion of frame where robot travels)
        # Narrowed to match robot width better
        self.CORRIDOR_LEFT = 0.30  # Left boundary of driving path (30% from left)
        self.CORRIDOR_RIGHT = 0.70  # Right boundary of driving path (70% from left)
        # Minimum obstacle area to consider (filters noise)
        self.MIN_OBSTACLE_AREA = 300  # Lowered to detect smaller obstacles
        # Avoidance steering gain - INCREASED for actual steering
        self.AVOIDANCE_GAIN = 2.5  # Was 1.5, increased for more aggressive avoidance
        # Active avoidance steering (direct angular velocity)
        self.AVOIDANCE_ANGULAR_SPEED = 0.5  # rad/s - steering speed during avoidance
        self.AVOIDANCE_LINEAR_SPEED = 0.08  # m/s - slow forward during avoidance
        # Green color range for filtering (same as lane detection)
        self.OBSTACLE_FILTER_GREEN_RATIO = (
            0.3  # If >30% of obstacle is green, it's likely the line
        )
        # Avoidance timing
        self.AVOIDANCE_TIMEOUT = 8.0  # Max seconds to stay in avoidance mode
        self.OBSTACLE_CLEAR_TIME = (
            2.0  # Seconds without seeing obstacle before considering it cleared
        )
        self.MIN_AVOIDANCE_DURATION = (
            2.0  # Minimum time to stay in avoidance mode before clearing
        )

        # Timing constants
        self.STOP_COOLDOWN = 5.0
        self.WAIT_DURATION = 3.0
        self.LOST_LINE_TIMEOUT = 0.5
        self.SEARCH_TIMEOUT = 2.0
        self.JUNCTION_CONFIRM_TIME = 0.7  # Wait before confirming junction
        self.TURN_180_DURATION = 3.0

        # Robot control parameters
        self.LINEAR_SPEED = 0.15  # m/s - forward speed
        self.LINEAR_SPEED_SLOW = 0.05  # m/s - slow speed for junctions
        self.ANGULAR_SPEED = 0.3  # rad/s - base turning speed
        self.KP = 0.003  # Proportional gain for steering
        self.KD = 0.001  # Derivative gain for damping oscillations

        # PID state variables
        self.previous_error = 0.0
        self.last_error_time = None

        # Display window option
        self.show_visualization = True

        # Load YOLO model
        self.get_logger().info(f"Loading YOLO model from {self.MODEL_PATH}...")
        self.model = YOLO(self.MODEL_PATH)
        self.class_names = self.model.names
        self.get_logger().info("YOLO model loaded successfully")

        # Load FastSAM model for obstacle detection
        self.get_logger().info(
            f"Loading FastSAM model from {self.OBSTACLE_MODEL_PATH}..."
        )
        self.obstacle_model = FastSAM(self.OBSTACLE_MODEL_PATH)
        self.get_logger().info("FastSAM model loaded successfully")

        # State dictionary for FSM
        self.state = {
            "fsm": "FOLLOW_LANE",  # FOLLOW_LANE, STOP_WAIT, TURN_180, AVOID_OBSTACLE
            "roi_mode": "custom_rect",
            "stop_start_time": 0,
            "last_stop_time": 0,
            "pending_roi_mode": None,
            "lost_line_time": None,
            "search_direction": None,
            "junction_detected_time": None,
            "turn_start_time": 0,
            "search_start_time": 0,
            "junction_deciding": False,  # True while scanning/deciding at junction
            "sign_directed": False,  # True when a sign has set the direction (priority over junction)
            "sign_directed_time": None,  # Time when sign direction was set (for timeout)
            # Obstacle avoidance state - path planning approach
            "avoiding_obstacle": False,
            "avoidance_side": None,  # "left" or "right" - which side to pass obstacle
            "avoidance_start_time": None,
            "obstacle_last_seen": None,  # Track when we last saw the obstacle
            "avoidance_phase": None,  # "approach", "passing", "clearing"
            "original_roi_mode": None,  # Store original ROI to restore after avoidance
        }

        # Current error for control
        self.current_error = None

        # Obstacle detection state (cached between frames for performance)
        self.obstacle_mask = np.zeros((480, 640), dtype=np.uint8)
        self.obstacle_info = []
        self.closest_obstacle = None
        self.avoidance_action = None
        self.obstacle_frame_count = 0  # For running detection every Nth frame

        # FPS tracking
        self.last_fps_update = time.time()
        self.frame_count = 0
        self.fps = 0.0

        # Camera timeout tracking
        self.last_camera_time = None
        self.camera_timeout = 1.0
        self.camera_active = False

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

        # Publisher for status messages
        self.status_pub = self.create_publisher(String, "/line_follower/status", 10)

        # Publisher for processed image
        self.processed_image_pub = self.create_publisher(
            Image, "/line_follower/debug_image", 10
        )

        # Publisher for velocity commands
        self.velocity_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Timer for control loop (20 Hz)
        self.control_timer = self.create_timer(0.05, self.control_callback)

        self.get_logger().info("Line Follower Node started")
        self.get_logger().info("Subscribing to: /camera/color/image_raw")
        self.get_logger().info("Publishing velocity to: /cmd_vel")

    def detect_and_annotate_signs(self, frame):
        """Detect signs using YOLO and annotate the frame."""
        detections = []
        results = self.model(frame, conf=0.5, iou=0.45, verbose=False)

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = self.class_names[class_id]

                detections.append({"class_name": class_name, "box": (x1, y1, x2, y2)})

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name}: {confidence:.2f}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                )

        return frame, detections

    def detect_obstacles(self, frame):
        """
        Detect obstacles using FastSAM segmentation.

        Args:
            frame: Input image frame

        Returns:
            obstacle_mask: Binary mask of detected obstacles
            obstacle_info: List of obstacle bounding boxes and areas
        """
        height, width = frame.shape[:2]
        obstacle_mask = np.zeros((height, width), dtype=np.uint8)
        obstacle_info = []

        # Run FastSAM inference on a smaller frame for performance
        small_frame = cv2.resize(frame, (320, 240))
        results = self.obstacle_model(
            small_frame,
            conf=self.OBSTACLE_CONFIDENCE,
            iou=0.9,
            retina_masks=False,
            verbose=False,
        )

        if results and len(results) > 0:
            result = results[0]

            if result.masks is not None:
                masks = result.masks.data.cpu().numpy()

                for i, mask in enumerate(masks):
                    # Resize mask to frame size if needed
                    if mask.shape != (height, width):
                        mask_resized = cv2.resize(
                            mask.astype(np.float32), (width, height)
                        )
                    else:
                        mask_resized = mask

                    # Convert to binary mask
                    binary_mask = (mask_resized > 0.5).astype(np.uint8) * 255

                    # Add to combined obstacle mask
                    obstacle_mask = cv2.bitwise_or(obstacle_mask, binary_mask)

                    # Get bounding box and area for obstacle info
                    contours, _ = cv2.findContours(
                        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > 100:  # Filter small noise
                            x, y, w, h = cv2.boundingRect(contour)
                            obstacle_info.append(
                                {
                                    "box": (x, y, x + w, y + h),
                                    "area": area,
                                    "centroid": (x + w // 2, y + h // 2),
                                }
                            )

        return obstacle_mask, obstacle_info

    def filter_false_obstacles(self, frame, obstacle_info, sign_detections):
        """
        Filter out false positives: green line segments and detected road signs.

        Args:
            frame: Original BGR frame
            obstacle_info: List of detected obstacles
            sign_detections: List of detected road signs from YOLO

        Returns:
            filtered_obstacles: List with green line and signs removed
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        filtered = []

        for obs in obstacle_info:
            x1, y1, x2, y2 = obs["box"]

            # Ensure bounds are within frame
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            if x2 <= x1 or y2 <= y1:
                continue

            # Check 1: Is this obstacle mostly green (likely the lane line)?
            roi_hsv = hsv[y1:y2, x1:x2]
            green_mask = cv2.inRange(roi_hsv, self.LOWER_HSV, self.UPPER_HSV)
            green_pixels = cv2.countNonZero(green_mask)
            total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
            green_ratio = green_pixels / max(total_pixels, 1)

            if green_ratio > self.OBSTACLE_FILTER_GREEN_RATIO:
                # This is likely the green line, skip it
                continue

            # Check 2: Does this obstacle overlap with a detected road sign?
            is_sign = False
            for sign in sign_detections:
                sx1, sy1, sx2, sy2 = sign["box"]
                # Check for overlap (intersection)
                overlap_x = max(0, min(x2, sx2) - max(x1, sx1))
                overlap_y = max(0, min(y2, sy2) - max(y1, sy1))
                overlap_area = overlap_x * overlap_y
                obs_area = (x2 - x1) * (y2 - y1)

                if overlap_area > 0.3 * obs_area:  # >30% overlap with sign
                    is_sign = True
                    break

            if is_sign:
                continue

            # This is a real obstacle
            filtered.append(obs)

        return filtered

    def annotate_obstacles(self, frame, obstacle_mask, obstacle_info):
        """
        Annotate frame with obstacle detection visualization.

        Args:
            frame: Input frame to annotate
            obstacle_mask: Binary mask of obstacles
            obstacle_info: List of obstacle information dicts

        Returns:
            annotated_frame: Frame with obstacle visualizations
        """
        annotated = frame.copy()

        # Create colored overlay for obstacles
        overlay = annotated.copy()
        overlay[obstacle_mask > 0] = [0, 0, 255]  # Red for obstacles
        annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        # Draw bounding boxes and centroids
        for obs in obstacle_info:
            x1, y1, x2, y2 = obs["box"]
            centroid = obs["centroid"]
            area = obs["area"]

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)

            # Draw centroid
            cv2.circle(annotated, centroid, 5, (255, 0, 255), -1)

            # Label with area
            label = f"Obs: {area:.0f}px"
            cv2.putText(
                annotated,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 165, 255),
                1,
            )

        # Add obstacle count
        cv2.putText(
            annotated,
            f"Obstacles: {len(obstacle_info)}",
            (10, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

        return annotated

    def check_obstacle_proximity(self, obstacle_info, frame_height, frame_width):
        """
        Analyze obstacles and find the closest one in the driving corridor.

        Args:
            obstacle_info: List of detected obstacles
            frame_height: Frame height for normalization
            frame_width: Frame width for corridor calculation

        Returns:
            closest_obstacle: Dict with obstacle info and proximity, or None
            all_in_path: List of all obstacles in the driving corridor
        """
        corridor_left = int(frame_width * self.CORRIDOR_LEFT)
        corridor_right = int(frame_width * self.CORRIDOR_RIGHT)

        obstacles_in_path = []

        for obs in obstacle_info:
            if obs["area"] < self.MIN_OBSTACLE_AREA:
                continue

            cx, cy = obs["centroid"]
            x1, y1, x2, y2 = obs["box"]

            # Check if obstacle overlaps with driving corridor
            obs_in_corridor = not (x2 < corridor_left or x1 > corridor_right)

            if obs_in_corridor:
                # Calculate normalized proximity (0=far/top, 1=close/bottom)
                proximity = y2 / frame_height

                # Determine zone
                if proximity > self.OBSTACLE_DANGER_ZONE:
                    zone = "DANGER"
                elif proximity > self.OBSTACLE_WARNING_ZONE:
                    zone = "WARNING"
                else:
                    zone = "FAR"

                obstacles_in_path.append(
                    {
                        **obs,
                        "proximity": proximity,
                        "zone": zone,
                        "relative_x": (cx - frame_width / 2)
                        / (frame_width / 2),  # -1 to 1
                    }
                )

        # Sort by proximity (closest first)
        obstacles_in_path.sort(key=lambda x: x["proximity"], reverse=True)

        closest = obstacles_in_path[0] if obstacles_in_path else None
        return closest, obstacles_in_path

    def calculate_avoidance_steering(
        self, closest_obstacle, frame_width, lane_error=None
    ):
        """
        Calculate steering correction to avoid obstacle while trying to follow the line.

        Args:
            closest_obstacle: The closest obstacle in path (from check_obstacle_proximity)
            frame_width: Frame width
            lane_error: Current lane following error (to blend with avoidance)

        Returns:
            avoidance_error: Steering error to avoid obstacle (negative=left, positive=right)
            action: String describing the avoidance action
        """
        if closest_obstacle is None:
            return 0, None

        cx, cy = closest_obstacle["centroid"]
        proximity = closest_obstacle["proximity"]
        zone = closest_obstacle["zone"]
        relative_x = closest_obstacle["relative_x"]  # -1 (left) to 1 (right)

        # Base avoidance direction (opposite to obstacle position)
        avoidance_direction = -1 if relative_x > 0 else 1  # Steer away from obstacle

        # Scale avoidance based on proximity (closer = stronger avoidance)
        if zone == "DANGER":
            # Immediate avoidance - strong steering
            avoidance_magnitude = frame_width * 0.3 * self.AVOIDANCE_GAIN
            action = "AVOID_IMMEDIATE"
        elif zone == "WARNING":
            # Gradual avoidance - moderate steering
            avoidance_magnitude = frame_width * 0.15 * self.AVOIDANCE_GAIN
            action = "AVOID_GRADUAL"
        else:
            # Far obstacle - minimal adjustment
            avoidance_magnitude = frame_width * 0.05
            action = "AVOID_PREPARE"

        avoidance_error = int(avoidance_direction * avoidance_magnitude)

        # Blend with lane following if available (prioritize avoidance when close)
        if lane_error is not None and zone != "DANGER":
            blend_factor = proximity  # 0 to 1
            avoidance_error = int(
                avoidance_error * blend_factor + lane_error * (1 - blend_factor)
            )

        return avoidance_error, action

    def visualize_obstacle_avoidance(
        self, frame, closest_obstacle, avoidance_action, height, width
    ):
        """
        Visualize obstacle avoidance status on frame.

        Args:
            frame: Frame to annotate
            closest_obstacle: Closest obstacle info
            avoidance_action: Current avoidance action string
            height, width: Frame dimensions
        """
        # Draw driving corridor
        corridor_left = int(width * self.CORRIDOR_LEFT)
        corridor_right = int(width * self.CORRIDOR_RIGHT)
        cv2.line(frame, (corridor_left, 0), (corridor_left, height), (255, 0, 255), 1)
        cv2.line(frame, (corridor_right, 0), (corridor_right, height), (255, 0, 255), 1)

        # Draw proximity zones
        danger_y = int(height * self.OBSTACLE_DANGER_ZONE)
        warning_y = int(height * self.OBSTACLE_WARNING_ZONE)
        cv2.line(
            frame, (corridor_left, danger_y), (corridor_right, danger_y), (0, 0, 255), 2
        )
        cv2.line(
            frame,
            (corridor_left, warning_y),
            (corridor_right, warning_y),
            (0, 165, 255),
            2,
        )

        # Label zones
        cv2.putText(
            frame,
            "DANGER",
            (corridor_right + 5, danger_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )
        cv2.putText(
            frame,
            "WARNING",
            (corridor_right + 5, warning_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

        if closest_obstacle:
            cx, cy = closest_obstacle["centroid"]
            zone = closest_obstacle["zone"]
            proximity = closest_obstacle["proximity"]
            relative_x = closest_obstacle.get("relative_x", 0)

            # Highlight the threatening obstacle
            x1, y1, x2, y2 = closest_obstacle["box"]
            color = (
                (0, 0, 255)
                if zone == "DANGER"
                else (0, 165, 255) if zone == "WARNING" else (0, 255, 255)
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

            # Show zone label on obstacle
            cv2.putText(
                frame,
                f"ZONE: {zone}",
                (x1, y1 - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

            # Show proximity percentage and relative position
            cv2.putText(
                frame,
                f"Proximity: {proximity*100:.0f}%",
                (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            cv2.putText(
                frame,
                f"RelX: {relative_x:.2f}",
                (10, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

            if avoidance_action:
                cv2.putText(
                    frame,
                    f"Action: {avoidance_action}",
                    (10, 175),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    def create_roi_mask(self, height, width, mode):
        """Create a binary mask for the Region of Interest."""
        mask = np.zeros((height, width), dtype=np.uint8)

        if mode == "bottom_left":
            h = int(height * 0.5)
            w = int(width * 0.5)
            mask[height - h :, :w] = 255
        elif mode == "bottom_right":
            h = int(height * 0.5)
            w = int(width * 0.5)
            mask[height - h :, width - w :] = 255
        elif mode == "custom_rect":
            # Start ROI higher to detect junctions earlier
            rx, ry = int(0.25 * width), int(0.35 * height)
            rw, rh = int(0.5 * width), int(0.65 * height)
            mask[ry : ry + rh, rx : rx + rw] = 255
        else:
            h = int(height * 0.5)
            mask[height - h :, :] = 255

        return mask

    def process_lane_detection(self, frame, roi_mask):
        """Process frame to detect lane line and calculate error."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.LOWER_HSV, self.UPPER_HSV)

        combined_mask = cv2.bitwise_and(color_mask, roi_mask)

        kernel = np.ones((5, 5), np.uint8)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        largest_contour = None
        centroid = None
        error = None

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroid = (cx, cy)
                center_x = frame.shape[1] // 2
                error = cx - center_x

        return combined_mask, largest_contour, centroid, error

    def detect_junction(self, contour, frame_height, threshold=0.3):
        """Detect if approaching a junction (line doesn't reach bottom of ROI)."""
        if contour is None:
            return False, None

        lowest_y = contour[:, :, 1].max()
        bottom_threshold = int(frame_height * threshold)
        line_reaches_bottom = lowest_y >= bottom_threshold

        print("+++line_reaches_bottom", line_reaches_bottom)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        is_wide = aspect_ratio > 1.5
        print("+++is_wide", is_wide)

        junction_detected = not line_reaches_bottom or is_wide
        return junction_detected, lowest_y

    def scan_for_best_direction(self, frame, height, width):
        """Scan left and right ROIs to find the best direction."""
        left_mask = self.create_roi_mask(height, width, "bottom_left")
        right_mask = self.create_roi_mask(height, width, "bottom_right")

        _, left_contour, _, _ = self.process_lane_detection(frame, left_mask)
        _, right_contour, _, _ = self.process_lane_detection(frame, right_mask)

        left_area = cv2.contourArea(left_contour) if left_contour is not None else 0
        right_area = cv2.contourArea(right_contour) if right_contour is not None else 0

        MIN_AREA = 500

        left_valid = left_area > MIN_AREA
        right_valid = right_area > MIN_AREA

        if left_valid and right_valid:
            if left_area > right_area * 1.2:
                return "left", left_area, right_area
            elif right_area > left_area * 1.2:
                return "right", left_area, right_area
            else:
                return "right", left_area, right_area
        elif left_valid:
            return "left", left_area, right_area
        elif right_valid:
            return "right", left_area, right_area
        else:
            return None, left_area, right_area

    def pid_follow_line(self, error):
        """Calculate angular velocity based on error (PD-Controller)."""
        if error is None:
            # Reset derivative state when line is lost
            self.previous_error = 0.0
            self.last_error_time = None
            return 0.0

        current_time = time.time()

        # Proportional term
        p_term = -self.KP * error

        # Derivative term
        d_term = 0.0
        if self.last_error_time is not None:
            dt = current_time - self.last_error_time
            if dt > 0:
                error_derivative = (error - self.previous_error) / dt
                d_term = -self.KD * error_derivative

        # Update state for next iteration
        self.previous_error = error
        self.last_error_time = current_time

        return p_term + d_term

    def image_callback(self, msg):
        """Process incoming camera images"""
        try:
            self.last_camera_time = time.time()
            self.camera_active = True

            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            frame = cv2.resize(frame, (640, 480))
            height, width = frame.shape[:2]

            # 1. Detect Signs
            frame, detections = self.detect_and_annotate_signs(frame)

            # 1.5. Detect Obstacles using FastSAM (only every 3rd frame for performance)
            self.obstacle_frame_count += 1
            if self.obstacle_frame_count % 3 == 0:
                self.obstacle_mask, self.obstacle_info = self.detect_obstacles(frame)
                # Filter out green line and road signs from obstacles
                self.obstacle_info = self.filter_false_obstacles(
                    frame, self.obstacle_info, detections
                )

            # Always annotate with the latest obstacle data (cached from last detection)
            frame = self.annotate_obstacles(
                frame, self.obstacle_mask, self.obstacle_info
            )

            # 1.6. Check obstacle proximity and calculate avoidance
            self.closest_obstacle, obstacles_in_path = self.check_obstacle_proximity(
                self.obstacle_info, height, width
            )
            self.avoidance_error, self.avoidance_action = (
                self.calculate_avoidance_steering(
                    self.closest_obstacle,
                    width,
                    lane_error=None,  # Will blend with lane error later
                )
            )

            # 2. Draw sign detection ROI lines
            sign_roi_top = int(height * 0.65)
            sign_roi_bottom = int(height * 0.7)
            cv2.line(frame, (0, sign_roi_top), (width, sign_roi_top), (0, 0, 255), 2)
            cv2.line(
                frame, (0, sign_roi_bottom), (width, sign_roi_bottom), (0, 0, 255), 2
            )

            # 3. FSM Logic Block
            self.process_fsm(
                frame, detections, sign_roi_top, sign_roi_bottom, height, width
            )

            # 4. Lane Detection
            roi_mask = self.create_roi_mask(height, width, self.state["roi_mode"])
            processed_mask, contour, centroid, error = self.process_lane_detection(
                frame, roi_mask
            )
            self.current_error = error

            # 5. Junction detection & Auto-switch ROI
            self.process_junction_logic(frame, contour, error, height, width)

            # 6. Visualization
            self.visualize(frame, roi_mask, contour, centroid, error, height, width)

            # Visualize obstacle avoidance zones and closest obstacle
            self.visualize_obstacle_avoidance(
                frame, self.closest_obstacle, self.avoidance_action, height, width
            )

            # Update FPS
            self.frame_count += 1
            current_time = time.time()
            if current_time - self.last_fps_update >= 0.5:
                self.fps = self.frame_count / (current_time - self.last_fps_update)
                self.frame_count = 0
                self.last_fps_update = current_time

            # Display FPS
            cv2.putText(
                frame,
                f"FPS: {self.fps:.1f}",
                (width - 100, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                frame,
                f"FSM: {self.state['fsm']}",
                (10, height - 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"ROI: {self.state['roi_mode']}",
                (10, height - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

            if self.show_visualization:
                cv2.imshow("Line Follower", frame)
                cv2.imshow("Mask", processed_mask)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    self.get_logger().info("Quit requested")
                    rclpy.shutdown()

            # Publish processed image
            try:
                processed_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                processed_msg.header = msg.header
                self.processed_image_pub.publish(processed_msg)
            except Exception as e:
                self.get_logger().error(f"Error publishing image: {str(e)}")

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def process_fsm(
        self, frame, detections, sign_roi_top, sign_roi_bottom, height, width
    ):
        """Process FSM state transitions based on sign detections."""
        if self.state["fsm"] == "FOLLOW_LANE":
            stop_detected = False
            direction_detected = None

            for det in detections:
                _, y1, _, _ = det["box"]
                if sign_roi_top < y1 < sign_roi_bottom:
                    if det["class_name"] == "stop":
                        if (
                            time.time() - self.state["last_stop_time"]
                            > self.STOP_COOLDOWN
                        ):
                            stop_detected = True
                    elif det["class_name"] in ["left", "right", "forward"]:
                        direction_detected = det["class_name"]

            if stop_detected:
                self.state["fsm"] = "STOP_WAIT"
                self.state["stop_start_time"] = time.time()
                # Signs have priority - disable junction auto-detection
                self.state["junction_deciding"] = False
                self.state["junction_detected_time"] = None
                if direction_detected == "left":
                    self.state["pending_roi_mode"] = "bottom_left"
                    self.state["sign_directed"] = True
                    self.state["sign_directed_time"] = time.time()
                elif direction_detected == "right":
                    self.state["pending_roi_mode"] = "bottom_right"
                    self.state["sign_directed"] = True
                    self.state["sign_directed_time"] = time.time()
                elif direction_detected == "forward":
                    self.state["pending_roi_mode"] = "custom_rect"
                    self.state["sign_directed"] = True
                    self.state["sign_directed_time"] = time.time()
                else:
                    self.state["pending_roi_mode"] = None
                self.get_logger().warn(
                    f"STOP DETECTED. Pending: {self.state['pending_roi_mode']}"
                )

            elif direction_detected:
                # Signs have priority - disable junction auto-detection
                self.state["sign_directed"] = True
                self.state["sign_directed_time"] = time.time()
                self.state["junction_deciding"] = False
                self.state["junction_detected_time"] = None
                if direction_detected == "left":
                    self.state["roi_mode"] = "bottom_left"
                elif direction_detected == "right":
                    self.state["roi_mode"] = "bottom_right"
                elif direction_detected == "forward":
                    self.state["roi_mode"] = "custom_rect"
                self.get_logger().info(
                    f"Sign: {direction_detected} -> ROI: {self.state['roi_mode']} (sign priority)"
                )

            # Check for obstacle in WARNING or DANGER zone - start avoidance immediately
            # With 10cm blind spot, we must react early
            elif self.closest_obstacle and self.closest_obstacle["zone"] in [
                "WARNING",
                "DANGER",
            ]:
                # Log detailed obstacle info for debugging
                obs_zone = self.closest_obstacle["zone"]
                obs_proximity = self.closest_obstacle["proximity"]
                obs_rel_x = self.closest_obstacle["relative_x"]
                self.get_logger().warn(
                    f"OBSTACLE IN {obs_zone} ZONE! Proximity: {obs_proximity:.2f}, RelX: {obs_rel_x:.2f}"
                )

                self.state["fsm"] = "AVOID_OBSTACLE"
                self.state["avoiding_obstacle"] = True
                self.state["avoidance_start_time"] = time.time()
                self.state["obstacle_last_seen"] = time.time()
                self.state["original_roi_mode"] = self.state["roi_mode"]
                self.state["avoidance_phase"] = "approach"

                # Determine which side to pass: go to the side with more clearance
                # If obstacle is on the right (relative_x > 0), pass on left
                # If obstacle is on the left (relative_x < 0), pass on right
                if obs_rel_x > 0:
                    self.state["avoidance_side"] = "left"
                    self.state["roi_mode"] = "bottom_left"
                else:
                    self.state["avoidance_side"] = "right"
                    self.state["roi_mode"] = "bottom_right"
                self.get_logger().warn(
                    f"AVOIDING OBSTACLE! Steering {self.state['avoidance_side'].upper()}, ROI: {self.state['roi_mode']}"
                )

        elif self.state["fsm"] == "STOP_WAIT":
            elapsed = time.time() - self.state["stop_start_time"]
            remaining = int(self.WAIT_DURATION - elapsed) + 1
            cv2.putText(
                frame,
                f"STOP: {remaining}s",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4,
            )

            if elapsed >= self.WAIT_DURATION:
                self.state["fsm"] = "FOLLOW_LANE"
                self.state["last_stop_time"] = time.time()
                if self.state["pending_roi_mode"]:
                    self.state["roi_mode"] = self.state["pending_roi_mode"]
                    self.get_logger().info(f"Wait done. ROI: {self.state['roi_mode']}")
                self.state["pending_roi_mode"] = None

        elif self.state["fsm"] == "TURN_180":
            elapsed = time.time() - self.state["turn_start_time"]
            remaining = int(self.TURN_180_DURATION - elapsed) + 1
            cv2.putText(
                frame,
                f"TURN 180: {remaining}s",
                (width // 2 - 150, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 165, 0),
                4,
            )

            if elapsed >= self.TURN_180_DURATION:
                self.state["fsm"] = "FOLLOW_LANE"
                self.state["roi_mode"] = "custom_rect"
                self.state["lost_line_time"] = None
                self.state["search_direction"] = None
                self.state["sign_directed"] = False  # Reset sign priority
                self.state["sign_directed_time"] = None
                # Reset PID state after turn
                self.previous_error = 0.0
                self.last_error_time = None
                # Reset any deciding/avoiding flags
                self.state["junction_deciding"] = False
                self.state["avoiding_obstacle"] = False
                self.get_logger().info("Turn complete. Resuming lane follow.")

        elif self.state["fsm"] == "AVOID_OBSTACLE":
            # Path planning obstacle avoidance:
            # Phase 1 (approach): Shift to side ROI, steer away from center
            # Phase 2 (passing): Stay on offset path while obstacle is beside us
            # Phase 3 (clearing): Obstacle behind us, gradually return to center

            elapsed = time.time() - self.state["avoidance_start_time"]

            # Update obstacle tracking - keep updating as long as obstacle is visible and not FAR
            if self.closest_obstacle and self.closest_obstacle["zone"] in [
                "WARNING",
                "DANGER",
            ]:
                self.state["obstacle_last_seen"] = time.time()

            # Calculate time since we last saw the obstacle in a threatening zone
            if self.state["obstacle_last_seen"] is not None:
                time_since_obstacle = time.time() - self.state["obstacle_last_seen"]
            else:
                time_since_obstacle = 0  # Just started, haven't lost sight yet

            # Display current avoidance status
            phase_text = (
                self.state["avoidance_phase"].upper()
                if self.state["avoidance_phase"]
                else "AVOIDING"
            )
            side_text = (
                self.state["avoidance_side"].upper()
                if self.state["avoidance_side"]
                else ""
            )
            cv2.putText(
                frame,
                f"AVOIDING: {phase_text} ({side_text})",
                (width // 2 - 150, height // 2 - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2,
            )
            # Show elapsed time for debugging
            cv2.putText(
                frame,
                f"Avoid time: {elapsed:.1f}s",
                (width // 2 - 80, height // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

            # Draw steering direction arrow
            arrow_center = (width // 2, height - 50)
            if side_text == "LEFT":
                arrow_end = (width // 2 - 80, height - 50)
                cv2.arrowedLine(
                    frame, arrow_center, arrow_end, (0, 255, 0), 4, tipLength=0.3
                )
                cv2.putText(
                    frame,
                    "STEERING LEFT",
                    (width // 2 - 100, height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            elif side_text == "RIGHT":
                arrow_end = (width // 2 + 80, height - 50)
                cv2.arrowedLine(
                    frame, arrow_center, arrow_end, (0, 255, 0), 4, tipLength=0.3
                )
                cv2.putText(
                    frame,
                    "STEERING RIGHT",
                    (width // 2 - 100, height - 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            # Phase transitions based on obstacle position
            if self.closest_obstacle:
                proximity = self.closest_obstacle["proximity"]

                if proximity >= self.OBSTACLE_DANGER_ZONE:
                    # Very close - we're passing the obstacle
                    self.state["avoidance_phase"] = "passing"
                elif proximity >= self.OBSTACLE_WARNING_ZONE:
                    # Approaching - continue avoidance maneuver
                    self.state["avoidance_phase"] = "approach"
                else:
                    # Obstacle now far - we may have passed it
                    self.state["avoidance_phase"] = "clearing"
            else:
                # No obstacle detected - likely passed it
                self.state["avoidance_phase"] = "clearing"

            # Check completion conditions - must meet minimum duration first
            obstacle_cleared = False

            if elapsed >= self.MIN_AVOIDANCE_DURATION:
                # Condition 1: Haven't seen obstacle in WARNING/DANGER zone for a while
                if time_since_obstacle > self.OBSTACLE_CLEAR_TIME:
                    obstacle_cleared = True
                    self.get_logger().info(
                        f"Obstacle not seen for {self.OBSTACLE_CLEAR_TIME}s - cleared!"
                    )

                # Condition 2: Obstacle is now in FAR zone (we passed it)
                elif self.closest_obstacle and self.closest_obstacle["zone"] == "FAR":
                    obstacle_cleared = True
                    self.get_logger().info("Obstacle now in FAR zone - cleared!")

                # Condition 3: No obstacle detected at all
                elif self.closest_obstacle is None and time_since_obstacle > 0.5:
                    obstacle_cleared = True
                    self.get_logger().info("No obstacle detected - cleared!")

            # Condition 4: Timeout (always applies)
            if elapsed > self.AVOIDANCE_TIMEOUT:
                obstacle_cleared = True
                self.get_logger().warn("Avoidance timeout - returning to lane.")

            if obstacle_cleared:
                # Return to normal lane following
                self.state["fsm"] = "FOLLOW_LANE"
                self.state["avoiding_obstacle"] = False
                self.state["avoidance_side"] = None
                self.state["avoidance_phase"] = None
                self.state["roi_mode"] = "custom_rect"  # Return to center following
                self.state["obstacle_last_seen"] = None
                self.state["original_roi_mode"] = None
                self.get_logger().info("Obstacle cleared. Resuming center lane follow.")

    def process_junction_logic(self, frame, contour, error, height, width):
        """Process junction detection and search logic."""
        junction_detected, _ = self.detect_junction(contour, height)

        if error is not None:
            self.state["lost_line_time"] = None

            # Only do automatic junction detection if signs haven't already set direction
            if (
                junction_detected
                and self.state["roi_mode"] == "custom_rect"
                and self.state["fsm"] == "FOLLOW_LANE"
                and not self.state["sign_directed"]  # Signs have priority
            ):

                if self.state["junction_detected_time"] is None:
                    self.state["junction_detected_time"] = time.time()
                    self.state["junction_deciding"] = True  # Start slowing down

                junction_duration = time.time() - self.state["junction_detected_time"]
                cv2.putText(
                    frame,
                    "JUNCTION - SLOWING",
                    (width // 2 - 140, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

                if (
                    junction_duration > self.JUNCTION_CONFIRM_TIME
                    and self.state["search_direction"] is None
                ):
                    best_dir, left_area, right_area = self.scan_for_best_direction(
                        frame, height, width
                    )
                    self.get_logger().info(
                        f"Junction scan - L:{left_area}, R:{right_area}, Best:{best_dir}"
                    )

                    if best_dir == "left":
                        self.state["search_direction"] = "left"
                        self.state["roi_mode"] = "bottom_left"
                    elif best_dir == "right":
                        self.state["search_direction"] = "right"
                        self.state["roi_mode"] = "bottom_right"
                    else:
                        self.state["search_direction"] = "right"
                        self.state["roi_mode"] = "bottom_right"
                        self.state["search_start_time"] = time.time()
            else:
                # No junction detected or not in custom_rect mode
                if self.state["search_direction"] is None:
                    self.state["junction_detected_time"] = None
                    # Reset junction_deciding if no junction and no active search
                    if not junction_detected:
                        self.state["junction_deciding"] = False

                # Check if we've completed the turn and can return to normal following
                if self.state["roi_mode"] == "bottom_left":
                    if -50 <= error <= 10:  # Wider tolerance for resuming
                        self.state["roi_mode"] = "custom_rect"
                        self.state["search_direction"] = None
                        self.state["junction_detected_time"] = None
                        self.state["junction_deciding"] = False  # Resume normal speed
                        self.state["sign_directed"] = False  # Reset sign priority
                        self.state["sign_directed_time"] = None
                        self.get_logger().info(
                            "Left turn complete, resuming normal speed"
                        )
                elif self.state["roi_mode"] == "bottom_right":
                    if -10 <= error <= 50:  # Wider tolerance for resuming
                        self.state["roi_mode"] = "custom_rect"
                        self.state["search_direction"] = None
                        self.state["junction_detected_time"] = None
                        self.state["junction_deciding"] = False  # Resume normal speed
                        self.state["sign_directed"] = False  # Reset sign priority
                        self.state["sign_directed_time"] = None
                        self.get_logger().info(
                            "Right turn complete, resuming normal speed"
                        )
        else:
            # Line lost - handle sign-directed turns or search logic
            if self.state["sign_directed"] and self.state["fsm"] == "FOLLOW_LANE":
                # Sign directed a turn but line is lost - keep turning
                # But add a timeout to prevent infinite turning
                if self.state["sign_directed_time"] is not None:
                    sign_turn_duration = time.time() - self.state["sign_directed_time"]
                    # If we've been turning for too long (e.g., 5 seconds), give up and search
                    if sign_turn_duration > 5.0:
                        self.get_logger().warn(
                            "Sign-directed turn timeout, switching to search mode"
                        )
                        self.state["sign_directed"] = False
                        self.state["sign_directed_time"] = None
                        self.state["search_direction"] = (
                            "left"
                            if self.state["roi_mode"] == "bottom_left"
                            else "right"
                        )
                        self.state["search_start_time"] = time.time()
                # Display that we're still turning based on sign
                cv2.putText(
                    frame,
                    "SIGN TURN - SEARCHING",
                    (width // 2 - 140, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
            elif (
                self.state["roi_mode"] == "custom_rect"
                and self.state["fsm"] == "FOLLOW_LANE"
            ):
                if self.state["lost_line_time"] is None:
                    self.state["lost_line_time"] = time.time()

                lost_duration = time.time() - self.state["lost_line_time"]

                if (
                    lost_duration > self.LOST_LINE_TIMEOUT
                    and self.state["search_direction"] is None
                ):
                    self.state["search_direction"] = "right"
                    self.state["roi_mode"] = "bottom_right"
                    self.state["search_start_time"] = time.time()

            elif (
                self.state["roi_mode"] == "bottom_right"
                and self.state["search_direction"] == "right"
            ):
                search_elapsed = time.time() - self.state.get(
                    "search_start_time", time.time()
                )
                if search_elapsed > self.SEARCH_TIMEOUT:
                    self.state["search_direction"] = "left"
                    self.state["roi_mode"] = "bottom_left"
                    self.state["search_start_time"] = time.time()

            elif (
                self.state["roi_mode"] == "bottom_left"
                and self.state["search_direction"] == "left"
            ):
                search_elapsed = time.time() - self.state.get(
                    "search_start_time", time.time()
                )
                if search_elapsed > self.SEARCH_TIMEOUT:
                    self.state["fsm"] = "TURN_180"
                    self.state["turn_start_time"] = time.time()

    def visualize(self, frame, roi_mask, contour, centroid, error, height, width):
        """Add visualization overlays to frame."""
        roi_contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, roi_contours, -1, (255, 255, 0), 2)

        # Draw junction detection threshold line (cyan dashed effect)
        junction_threshold = 0.4  # Same as detect_junction threshold
        junction_y = int(height * junction_threshold)
        # Draw the line where junction is detected (if line ends ABOVE this = junction)
        cv2.line(frame, (0, junction_y), (width, junction_y), (255, 255, 0), 2)
        cv2.putText(
            frame,
            "JUNCTION THRESHOLD",
            (width // 2 - 100, junction_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 0),
            1,
        )
        cv2.putText(
            frame,
            "(line must reach below this)",
            (width // 2 - 110, junction_y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 0),
            1,
        )

        # Draw where the line currently ends (if contour exists)
        if contour is not None:
            lowest_y = contour[:, :, 1].max()
            cv2.line(frame, (0, lowest_y), (width, lowest_y), (0, 165, 255), 2)
            cv2.putText(
                frame,
                f"LINE END: y={lowest_y}",
                (10, lowest_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 165, 255),
                1,
            )

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

            # Control visualization
            if error is not None:
                center_x = width // 2
                steering = int(error * 1.0)
                arrow_start = (center_x, height)
                arrow_end = (center_x + steering, height - 100)
                cv2.arrowedLine(
                    frame, arrow_start, arrow_end, (255, 0, 255), 5, tipLength=0.3
                )

                if abs(error) < 20:
                    text, color = "FORWARD", (0, 255, 0)
                elif error > 0:
                    text, color = f"RIGHT ({abs(error)})", (0, 165, 255)
                else:
                    text, color = f"LEFT ({abs(error)})", (0, 165, 255)

                cv2.putText(
                    frame,
                    f"Action: {text}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                )
                cv2.putText(
                    frame,
                    f"Error: {error}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

    def control_callback(self):
        """Control loop - sends velocity commands to robot."""
        twist = Twist()

        # Check camera timeout
        current_time = time.time()
        if self.last_camera_time is None:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)
            return

        if current_time - self.last_camera_time > self.camera_timeout:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.velocity_pub.publish(twist)
            if self.camera_active:
                self.get_logger().error("CAMERA TIMEOUT - STOPPING ROBOT")
                self.camera_active = False
            return

        # FSM-based control
        if self.state["fsm"] == "STOP_WAIT":
            # Robot stopped at stop sign
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        elif self.state["fsm"] == "TURN_180":
            # Turning 180 degrees
            twist.linear.x = 0.0
            twist.angular.z = self.ANGULAR_SPEED  # Turn left

        elif self.state["fsm"] == "AVOID_OBSTACLE":
            # Active obstacle avoidance - steer away from the obstacle
            # The key is to actively steer in the avoidance direction, not just follow the shifted ROI

            avoidance_side = self.state["avoidance_side"]
            avoidance_phase = self.state["avoidance_phase"]

            # Determine steering based on obstacle position and avoidance phase
            if avoidance_phase == "passing" or avoidance_phase == "approach":
                # Actively steer away from obstacle
                twist.linear.x = self.AVOIDANCE_LINEAR_SPEED

                # Steer based on which side we're avoiding to
                if avoidance_side == "left":
                    twist.angular.z = (
                        self.AVOIDANCE_ANGULAR_SPEED
                    )  # Turn left (positive)
                elif avoidance_side == "right":
                    twist.angular.z = (
                        -self.AVOIDANCE_ANGULAR_SPEED
                    )  # Turn right (negative)
                else:
                    twist.angular.z = 0.0

                # If we still see the obstacle in danger zone, steer more aggressively
                if self.closest_obstacle and self.closest_obstacle["zone"] == "DANGER":
                    twist.angular.z *= 1.5  # More aggressive steering
                    twist.linear.x = self.AVOIDANCE_LINEAR_SPEED * 0.5  # Slow down more

            elif avoidance_phase == "clearing":
                # Obstacle is behind/far, gently return toward center while moving forward
                twist.linear.x = self.LINEAR_SPEED_SLOW

                # Use lane error if available to return to line
                if self.current_error is not None:
                    twist.angular.z = self.pid_follow_line(self.current_error)
                    # Limit angular velocity during clearing
                    max_angular = 0.4
                    twist.angular.z = max(
                        -max_angular, min(max_angular, twist.angular.z)
                    )
                else:
                    # Slowly straighten out
                    if avoidance_side == "left":
                        twist.angular.z = -0.1  # Gently turn right to straighten
                    elif avoidance_side == "right":
                        twist.angular.z = 0.1  # Gently turn left to straighten
                    else:
                        twist.angular.z = 0.0
            else:
                # Default: slow forward with lane following
                twist.linear.x = self.AVOIDANCE_LINEAR_SPEED
                if self.current_error is not None:
                    twist.angular.z = self.pid_follow_line(self.current_error)
                else:
                    twist.angular.z = 0.0

            # Limit angular velocity
            max_angular = 0.8
            twist.angular.z = max(-max_angular, min(max_angular, twist.angular.z))

        elif self.state["fsm"] == "FOLLOW_LANE":
            if self.current_error is not None:
                # Follow line using PD-controller
                # Slow down if deciding at junction OR if sign directed a turn
                if self.state["junction_deciding"] or self.state["sign_directed"]:
                    twist.linear.x = self.LINEAR_SPEED_SLOW
                else:
                    twist.linear.x = self.LINEAR_SPEED

                twist.angular.z = self.pid_follow_line(self.current_error)

                # Limit angular velocity
                max_angular = 0.8
                twist.angular.z = max(-max_angular, min(max_angular, twist.angular.z))
            else:
                # Line lost - continue turning based on sign direction or search
                if self.state["sign_directed"]:
                    # Sign directed a turn - keep turning in that direction with slow forward motion
                    twist.linear.x = self.LINEAR_SPEED_SLOW * 0.5  # Very slow forward
                    if self.state["roi_mode"] == "bottom_left":
                        twist.angular.z = self.ANGULAR_SPEED * 0.7  # Turn left
                    elif self.state["roi_mode"] == "bottom_right":
                        twist.angular.z = -self.ANGULAR_SPEED * 0.7  # Turn right
                    else:
                        twist.linear.x = 0.0
                        twist.angular.z = 0.0
                elif self.state["search_direction"] == "left":
                    twist.linear.x = 0.0
                    twist.angular.z = self.ANGULAR_SPEED * 0.5
                elif self.state["search_direction"] == "right":
                    twist.linear.x = 0.0
                    twist.angular.z = -self.ANGULAR_SPEED * 0.5
                else:
                    # No direction set yet, stop
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

        self.velocity_pub.publish(twist)

        # Publish status
        status_msg = String()
        status_data = {
            "fsm": self.state["fsm"],
            "roi_mode": self.state["roi_mode"],
            "error": self.current_error,
            "linear": twist.linear.x,
            "angular": twist.angular.z,
        }
        status_msg.data = json.dumps(status_data)
        self.status_pub.publish(status_msg)

    def destroy_node(self):
        """Cleanup resources"""
        # Stop the robot
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.velocity_pub.publish(twist)

        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    """Main entry point"""
    rclpy.init(args=args)

    try:
        node = LineFollowerNode()

        print("\n" + "=" * 60)
        print("LIMO Line Follower with Road Sign Detection")
        print("=" * 60)
        print("Subscribing to: /camera/color/image_raw")
        print("Publishing velocity to: /cmd_vel")
        print("Publishing status to: /line_follower/status")
        print("Publishing debug image to: /line_follower/debug_image")
        print("\nBehavior:")
        print("  - Follows green line using PID control")
        print("  - Detects road signs (left, right, forward, stop)")
        print("  - Handles junctions automatically")
        print("  - Detects and avoids obstacles using FastSAM")
        print("  - Turns 180° if line is lost")
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
