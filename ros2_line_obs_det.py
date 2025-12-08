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
from ultralytics import YOLO


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

        # Green line HSV range
        self.LOWER_HSV = np.array([30, 90, 25])
        self.UPPER_HSV = np.array([85, 250, 255])

        # Orange cone HSV range (orange color detection)
        self.CONE_LOWER_HSV = np.array([100, 100, 150])
        self.CONE_UPPER_HSV = np.array([180, 255, 255])

        # Robot physical parameters (in meters)
        self.ROBOT_WIDTH = 0.26  # 22cm
        self.ROBOT_LENGTH = 0.32  # 32cm
        self.BLIND_SPOT = 0.10  # 10cm blind spot in front due to camera tilt

        # Obstacle avoidance parameters
        self.OBSTACLE_MIN_AREA = 25  # Minimum contour area to consider as obstacle
        self.OBSTACLE_DANGER_ZONE_Y = (
            0.70  # Bottom 15% of frame is danger zone (very close)
        )
        self.OBSTACLE_WARNING_ZONE_Y = 0.50  # Bottom 35% is warning zone
        self.OBSTACLE_AVOIDANCE_GAIN = 0.25  # How aggressively to steer away from obstacles (increased for stronger response)
        self.OBSTACLE_SLOWDOWN_FACTOR = 0.8  # Speed reduction when obstacle detected
        self.OBSTACLE_MIN_AVOIDANCE = (
            0.15  # Minimum avoidance steering when obstacle in path
        )
        self.OBSTACLE_AVOIDANCE_PERSIST_TIME = (
            2  # Keep avoiding for this long after obstacle leaves view
        )

        # Timing constants
        self.STOP_COOLDOWN = 5.0
        self.WAIT_DURATION = 3.0
        self.LOST_LINE_TIMEOUT = 0.5
        self.SEARCH_TIMEOUT = 2.0
        self.JUNCTION_CONFIRM_TIME = 0.5  # Wait before confirming junction
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

        # State dictionary for FSM
        self.state = {
            "fsm": "FOLLOW_LANE",  # FOLLOW_LANE, STOP_WAIT, TURN_180
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
            # Obstacle avoidance state
            "obstacle_detected": False,
            "obstacle_avoidance_steering": 0.0,  # Direct steering component for avoidance
            "obstacle_in_danger_zone": False,  # True if obstacle is very close
            # Path-aware obstacle avoidance
            "obstacle_relative_to_line": None,  # 'left' or 'right' of the line
            "avoidance_offset": 0,  # Pixel offset to apply to line following target
            # Avoidance persistence (keep avoiding after obstacle leaves view)
            "last_avoidance_offset": 0,  # Last non-zero avoidance offset
            "last_obstacle_time": None,  # Time when obstacle was last seen
            "avoidance_active": False,  # True while actively avoiding (including persistence)
            "avoidance_state_changed": False,  # True when avoidance state just changed
        }

        # Line trajectory points (updated each frame)
        self.line_points = []  # List of (x, y) points along the detected line
        self.extrapolated_line_points = (
            []
        )  # Line points including extrapolation through obstacles
        self.last_good_line_points = (
            []
        )  # Remember last good trajectory for occlusion recovery
        self.line_occluded = False  # True if obstacle is covering part of the line

        # Obstacle detection results (updated each frame)
        self.obstacles = []  # List of detected obstacle info

        # Current error for control
        self.current_error = None

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
        results = self.model(frame, conf=0.7, iou=0.45, verbose=False)

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

    def extract_line_trajectory(self, contour, height):
        """
        Extract points along the detected line to understand the path trajectory.
        Returns a list of (x, y) points from bottom to top of the line.
        """
        if contour is None:
            return []

        # Get all points from the contour
        points = contour.reshape(-1, 2)

        # Group points by y-coordinate bands and find the center x for each band
        # This gives us the line's trajectory from bottom to top
        num_bands = 10
        min_y = points[:, 1].min()
        max_y = points[:, 1].max()

        if max_y - min_y < 20:  # Line too short
            return []

        band_height = (max_y - min_y) / num_bands
        trajectory = []

        for i in range(num_bands):
            band_bottom = max_y - i * band_height
            band_top = band_bottom - band_height

            # Find points in this band
            band_mask = (points[:, 1] >= band_top) & (points[:, 1] < band_bottom)
            band_points = points[band_mask]

            if len(band_points) > 0:
                # Use the mean x position for this band
                mean_x = int(np.mean(band_points[:, 0]))
                mean_y = int((band_top + band_bottom) / 2)
                trajectory.append((mean_x, mean_y))

        return trajectory

    def get_line_x_at_y(self, trajectory, target_y):
        """
        Interpolate to find the x position of the line at a given y coordinate.
        """
        if len(trajectory) < 2:
            return None

        # Find the two trajectory points that bracket target_y
        trajectory_sorted = sorted(trajectory, key=lambda p: p[1])  # Sort by y

        for i in range(len(trajectory_sorted) - 1):
            y1 = trajectory_sorted[i][1]
            y2 = trajectory_sorted[i + 1][1]

            if y1 <= target_y <= y2 or y2 <= target_y <= y1:
                # Interpolate x
                x1 = trajectory_sorted[i][0]
                x2 = trajectory_sorted[i + 1][0]

                if y2 == y1:
                    return (x1 + x2) // 2

                t = (target_y - y1) / (y2 - y1)
                return int(x1 + t * (x2 - x1))

        # If target_y is outside the trajectory, extrapolate from nearest end
        if target_y < trajectory_sorted[0][1]:
            return trajectory_sorted[0][0]
        else:
            return trajectory_sorted[-1][0]

    def check_line_occlusion(self, obstacles, line_mask, height, width):
        """
        Check if any obstacle is occluding (covering) the line.
        Returns: (is_occluded, occluding_obstacles)
        """
        occluding_obstacles = []

        for obstacle in obstacles:
            x, y, w, h = obstacle["bbox"]

            # Check if there's line mask pixels near the obstacle
            # Expand the bounding box slightly to check around it
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(width, x + w + margin)
            y2 = min(height, y + h + margin)

            # Check the region around the obstacle for line pixels
            region = line_mask[y1:y2, x1:x2]
            line_pixels_near = np.sum(region > 0)

            # Check inside the obstacle bounding box
            obs_region = line_mask[y : y + h, x : x + w]
            line_pixels_inside = np.sum(obs_region > 0)

            # If there are line pixels near but few inside, obstacle might be occluding
            # Or if the obstacle is positioned where we expect the line to be
            if line_pixels_near > 100 and line_pixels_inside < line_pixels_near * 0.3:
                occluding_obstacles.append(obstacle)

            # Also check if obstacle is on the expected line path
            obs_cx = obstacle["centroid"][0]
            obs_cy = obstacle["centroid"][1]

            # Use last known good trajectory or current trajectory
            trajectory = (
                self.last_good_line_points
                if len(self.last_good_line_points) > 2
                else self.line_points
            )
            expected_line_x = self.get_line_x_at_y(trajectory, obs_cy)

            if expected_line_x is not None:
                # If obstacle center is close to where line should be
                if abs(obs_cx - expected_line_x) < w:
                    if obstacle not in occluding_obstacles:
                        occluding_obstacles.append(obstacle)

        return len(occluding_obstacles) > 0, occluding_obstacles

    def extrapolate_line_through_obstacle(self, line_points, obstacles, height, width):
        """
        If an obstacle is covering part of the line, extrapolate the line through it
        using the visible portions above and below the obstacle.

        Returns: extrapolated trajectory points
        """
        if len(line_points) < 3:
            # Not enough points to extrapolate, use last known good trajectory
            if len(self.last_good_line_points) > 3:
                return self.last_good_line_points
            return line_points

        extrapolated = list(line_points)

        for obstacle in obstacles:
            x, y, w, h = obstacle["bbox"]
            obs_top = y
            obs_bottom = y + h
            obs_cx = obstacle["centroid"][0]

            # Find line points above and below the obstacle
            points_above = [(px, py) for px, py in extrapolated if py < obs_top - 10]
            points_below = [(px, py) for px, py in extrapolated if py > obs_bottom + 10]

            # If we have points on both sides, we can extrapolate through
            if len(points_above) >= 2 and len(points_below) >= 2:
                # Fit a line to the points above
                above_sorted = sorted(points_above, key=lambda p: p[1], reverse=True)[
                    :3
                ]

                # Fit a line to the points below
                below_sorted = sorted(points_below, key=lambda p: p[1])[:3]

                # Simple linear interpolation between the closest points
                if above_sorted and below_sorted:
                    closest_above = above_sorted[0]
                    closest_below = below_sorted[0]

                    # Generate interpolated points through the obstacle region
                    num_interp = max(3, (obs_bottom - obs_top) // 30)
                    for i in range(1, num_interp + 1):
                        t = i / (num_interp + 1)
                        interp_y = int(
                            closest_above[1] + t * (closest_below[1] - closest_above[1])
                        )
                        interp_x = int(
                            closest_above[0] + t * (closest_below[0] - closest_above[0])
                        )
                        extrapolated.append((interp_x, interp_y))

            elif len(points_above) >= 2:
                # Only have points above - extrapolate downward
                # Use the trend from the visible portion
                above_sorted = sorted(points_above, key=lambda p: p[1], reverse=True)
                if len(above_sorted) >= 2:
                    # Calculate slope from last two visible points
                    p1 = above_sorted[0]
                    p2 = above_sorted[1]
                    if p1[1] != p2[1]:
                        slope = (p1[0] - p2[0]) / (p1[1] - p2[1])
                        # Extrapolate through obstacle
                        for dy in range(30, obs_bottom - p1[1] + 30, 30):
                            ext_y = p1[1] + dy
                            ext_x = int(p1[0] + slope * dy)
                            ext_x = max(0, min(width - 1, ext_x))
                            extrapolated.append((ext_x, ext_y))

            elif len(points_below) >= 2:
                # Only have points below - extrapolate upward
                below_sorted = sorted(points_below, key=lambda p: p[1])
                if len(below_sorted) >= 2:
                    p1 = below_sorted[0]
                    p2 = below_sorted[1]
                    if p1[1] != p2[1]:
                        slope = (p2[0] - p1[0]) / (p2[1] - p1[1])
                        for dy in range(-30, p1[1] - obs_top - 30, -30):
                            ext_y = p1[1] + dy
                            ext_x = int(p1[0] + slope * dy)
                            ext_x = max(0, min(width - 1, ext_x))
                            extrapolated.append((ext_x, ext_y))

        # Sort by y coordinate
        extrapolated = sorted(set(extrapolated), key=lambda p: p[1])
        return extrapolated

    def create_obstacle_roi_mask(self, height, width):
        """
        Create ROI mask for obstacle detection.
        Accounts for 10cm blind spot (camera tilt) - excludes bottom portion of frame.
        Focuses on the path ahead where robot will travel.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # The blind spot maps to roughly bottom 10-15% of frame
        # We want to detect obstacles from just above blind spot to middle of frame
        blind_spot_y = int(height * 1)  # Bottom 10% is blind spot
        top_y = int(height * 0.40)  # Start detection from 30% down

        # Create a trapezoidal ROI that matches the robot's path
        # Wider at bottom (closer), narrower at top (further away)
        bottom_width = int(width * 1)  # 90% of frame width at bottom
        top_width = int(width * 0.5)  # 50% of frame width at top

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

    def obstacle_overlaps_yolo_sign(
        self, obstacle_bbox, yolo_detections, overlap_threshold=0.3
    ):
        """
        Check if an obstacle bounding box overlaps significantly with any YOLO-detected sign.

        Args:
            obstacle_bbox: (x, y, w, h) of the obstacle
            yolo_detections: List of YOLO detections with 'box' key containing (x1, y1, x2, y2)
            overlap_threshold: Minimum IoU or overlap ratio to consider as a sign (default 0.3)

        Returns:
            True if the obstacle overlaps with a sign, False otherwise
        """
        if not yolo_detections:
            return False

        ox, oy, ow, oh = obstacle_bbox
        obstacle_area = ow * oh

        for detection in yolo_detections:
            sx1, sy1, sx2, sy2 = detection["box"]

            # Calculate intersection
            inter_x1 = max(ox, sx1)
            inter_y1 = max(oy, sy1)
            inter_x2 = min(ox + ow, sx2)
            inter_y2 = min(oy + oh, sy2)

            if inter_x1 < inter_x2 and inter_y1 < inter_y2:
                inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

                # Check if significant portion of obstacle overlaps with sign
                # Using overlap ratio relative to obstacle area
                overlap_ratio = inter_area / obstacle_area if obstacle_area > 0 else 0

                if overlap_ratio >= overlap_threshold:
                    return True

        return False

    def detect_obstacles(self, frame, height, width, yolo_detections=None):
        """
        Detect orange cones in the frame using HSV color segmentation.
        Returns list of obstacles with their position, size, and zone (danger/warning/safe).

        Args:
            yolo_detections: List of YOLO detections to filter out (avoid detecting signs as obstacles)
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
        # orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)
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

            # Skip if this obstacle overlaps with a YOLO-detected sign
            if self.obstacle_overlaps_yolo_sign((x, y, w, h), yolo_detections):
                continue

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

            # Calculate horizontal offset from center (for steering)
            offset_from_center = cx - center_x

            # Estimate relative size/distance (larger area = closer)
            # Normalize by frame area for consistency
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

    def calculate_obstacle_avoidance_path_aware(self, obstacles, width, height):
        """
        PATH-AWARE obstacle avoidance.
        Instead of steering based on obstacle position relative to robot center,
        we determine which side of the LINE the obstacle is on, and pass on the opposite side.

        KEY INSIGHT: We need to consider the ENTIRE obstacle (including its width),
        not just its centroid, and add enough offset to clear both the obstacle
        AND the robot's width with a safety margin.

        Returns: (avoidance_offset, should_slow_down, is_danger, obstacle_side)
        - avoidance_offset: pixel offset to add to the line following target
        - obstacle_side: 'left' or 'right' of the line (or None)
        """
        if not obstacles:
            return 0, False, False, None

        should_slow_down = False
        is_danger = False
        obstacle_side = None
        avoidance_offset = 0

        # Robot width in pixels (approximate) - 26cm robot on ~1m visible width
        # At 640px width viewing ~1m, robot is roughly 26% of visible width at close range
        # But at obstacle detection distance, it's less. Use conservative estimate.
        robot_half_width_px = int(width * 0.15)  # ~15% of frame width for safety

        # Safety margin in pixels
        safety_margin_px = int(width * 0.08)  # Extra 8% margin

        for obstacle in obstacles:
            obs_cx, obs_cy = obstacle["centroid"]
            obs_x, obs_y, obs_w, obs_h = obstacle["bbox"]
            zone = obstacle["zone"]

            # Determine zone urgency
            if zone == "danger":
                is_danger = True
                should_slow_down = True
                urgency = 1.0
            elif zone == "warning":
                should_slow_down = True
                urgency = 0.8
            else:
                urgency = 0.5

            # Find where the line is at the obstacle's y-coordinate
            line_x_at_obstacle = self.get_line_x_at_y(self.line_points, obs_cy)

            if line_x_at_obstacle is not None:
                # Calculate obstacle EDGES relative to the line (not just centroid)
                obs_left_edge = obs_x
                obs_right_edge = obs_x + obs_w

                # Distance from line to each edge of the obstacle
                left_edge_to_line = (
                    line_x_at_obstacle - obs_left_edge
                )  # positive if obstacle left of line
                right_edge_to_line = (
                    line_x_at_obstacle - obs_right_edge
                )  # negative if obstacle right of line

                # Check if obstacle overlaps with the robot's path
                # Robot path is: line_x ± robot_half_width
                robot_left_path = line_x_at_obstacle - robot_half_width_px
                robot_right_path = line_x_at_obstacle + robot_half_width_px

                # Does obstacle overlap with robot's path?
                obstacle_blocks_path = not (
                    obs_right_edge < robot_left_path or obs_left_edge > robot_right_path
                )

                if obstacle_blocks_path:
                    # Calculate how much we need to shift to clear the obstacle
                    # Option 1: Go LEFT of obstacle (shift target left = negative offset)
                    # Need to clear: obstacle's left edge + robot half width + safety
                    clearance_if_go_left = (
                        obs_left_edge
                        - robot_half_width_px
                        - safety_margin_px
                        - line_x_at_obstacle
                    )

                    # Option 2: Go RIGHT of obstacle (shift target right = positive offset)
                    # Need to clear: obstacle's right edge + robot half width + safety
                    clearance_if_go_right = (
                        obs_right_edge
                        + robot_half_width_px
                        + safety_margin_px
                        - line_x_at_obstacle
                    )

                    # Choose the direction with smaller deviation from line
                    if abs(clearance_if_go_left) < abs(clearance_if_go_right):
                        # Go left is shorter
                        obstacle_side = "right"  # obstacle is to our right
                        offset_contribution = clearance_if_go_left * urgency
                    else:
                        # Go right is shorter
                        obstacle_side = "left"  # obstacle is to our left
                        offset_contribution = clearance_if_go_right * urgency

                    # Use the largest needed offset (most critical obstacle)
                    if abs(offset_contribution) > abs(avoidance_offset):
                        avoidance_offset = int(offset_contribution)

            else:
                # No line trajectory available - fall back to robot-center based avoidance
                center_x = width // 2
                obs_left_edge = obs_x
                obs_right_edge = obs_x + obs_w

                # Check if obstacle blocks center path
                if (
                    obs_left_edge < center_x + robot_half_width_px
                    and obs_right_edge > center_x - robot_half_width_px
                ):
                    # Calculate clearance needed
                    clearance_left = (
                        obs_left_edge
                        - robot_half_width_px
                        - safety_margin_px
                        - center_x
                    )
                    clearance_right = (
                        obs_right_edge
                        + robot_half_width_px
                        + safety_margin_px
                        - center_x
                    )

                    if abs(clearance_left) < abs(clearance_right):
                        obstacle_side = "right"
                        avoidance_offset = int(clearance_left * urgency)
                    else:
                        obstacle_side = "left"
                        avoidance_offset = int(clearance_right * urgency)

        # Clamp the offset to reasonable bounds
        max_offset = int(
            width * 0.35
        )  # Max 35% of frame width (increased for better clearance)
        avoidance_offset = max(-max_offset, min(max_offset, avoidance_offset))

        return avoidance_offset, should_slow_down, is_danger, obstacle_side

    def calculate_obstacle_avoidance(self, obstacles, width):
        """
        Legacy method - now calls path-aware version.
        Kept for compatibility with existing code structure.
        Returns: (avoidance_steering, should_slow_down, is_danger)
        """
        # This is now handled by calculate_obstacle_avoidance_path_aware
        # Return zeros here as the actual avoidance is done via offset
        if not obstacles:
            return 0.0, False, False

        is_danger = any(obs["zone"] == "danger" for obs in obstacles)
        should_slow = any(obs["zone"] in ["danger", "warning"] for obs in obstacles)

        return 0.0, should_slow, is_danger

    def detect_junction(self, contour, frame_height, threshold=0.4):
        """Detect if approaching a junction (line doesn't reach bottom of ROI)."""
        if contour is None:
            return False, None

        lowest_y = contour[:, :, 1].max()
        bottom_threshold = int(frame_height * threshold)
        line_reaches_bottom = lowest_y >= bottom_threshold

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        is_wide = aspect_ratio > 1.5

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

    def pid_follow_line(self, error, reset_derivative=False):
        """Calculate angular velocity based on error (PD-Controller).

        Args:
            error: The line following error (pixels from center)
            reset_derivative: If True, reset derivative state (use when avoidance state changes)
        """
        if error is None:
            # Reset derivative state when line is lost
            self.previous_error = 0.0
            self.last_error_time = None
            return 0.0

        current_time = time.time()

        # Reset derivative if requested (prevents fighting when avoidance kicks in)
        if reset_derivative:
            self.previous_error = error  # Set to current to avoid derivative spike
            self.last_error_time = current_time
            # Return just P term on reset frame
            return -self.KP * error

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

            # 4.1 Extract line trajectory for path-aware obstacle avoidance
            self.line_points = self.extract_line_trajectory(contour, height)

            # Save good line trajectory for occlusion recovery
            if len(self.line_points) > 5:
                self.last_good_line_points = list(self.line_points)

            # 4.5. Obstacle Detection and Avoidance (PATH-AWARE)
            # Pass YOLO detections to filter out signs from obstacle detection
            self.obstacles, obstacle_mask, obstacle_roi_pts = self.detect_obstacles(
                frame, height, width, yolo_detections=detections
            )

            # 4.6 Check if obstacle is occluding the line
            self.line_occluded, occluding_obstacles = self.check_line_occlusion(
                self.obstacles, processed_mask, height, width
            )

            # 4.7 If line is occluded, extrapolate through the obstacle
            if self.line_occluded and len(occluding_obstacles) > 0:
                self.extrapolated_line_points = self.extrapolate_line_through_obstacle(
                    self.line_points, occluding_obstacles, height, width
                )
                # Use extrapolated points for path-aware avoidance
                working_line_points = self.extrapolated_line_points
                self.get_logger().info(
                    f"Line occluded by {len(occluding_obstacles)} obstacle(s), using extrapolation"
                )
            else:
                self.extrapolated_line_points = []
                working_line_points = self.line_points

            # Store for use in avoidance calculation
            original_line_points = self.line_points
            self.line_points = working_line_points

            # Calculate path-aware avoidance offset
            avoidance_offset, should_slow, is_danger, obstacle_side = (
                self.calculate_obstacle_avoidance_path_aware(
                    self.obstacles, width, height
                )
            )

            # Restore original line points for visualization
            self.line_points = original_line_points

            # --- Avoidance Persistence Logic ---
            # Keep avoiding for a period after obstacle leaves the field of view
            # to ensure the robot fully passes the obstacle before returning to line.
            #
            # KEY CHANGE: Don't fade during the first part of persistence!
            # The robot needs to maintain full offset while passing the obstacle,
            # then fade smoothly back to the line.
            current_time = time.time()

            # Define persistence phases:
            # Phase 1 (0 to 60% of persist time): Maintain FULL offset - robot is passing obstacle
            # Phase 2 (60% to 100%): Fade back to line - robot has cleared obstacle
            MAINTAIN_PHASE_RATIO = 0.6  # First 60% at full offset

            if avoidance_offset != 0:
                # Obstacle currently visible - save the avoidance state
                self.state["last_avoidance_offset"] = avoidance_offset
                self.state["last_obstacle_time"] = current_time
                self.state["avoidance_active"] = True
                effective_avoidance_offset = avoidance_offset
            elif self.state["last_obstacle_time"] is not None:
                # No obstacle visible - check if we should persist the avoidance
                time_since_obstacle = current_time - self.state["last_obstacle_time"]
                persist_time = self.OBSTACLE_AVOIDANCE_PERSIST_TIME
                maintain_duration = persist_time * MAINTAIN_PHASE_RATIO
                fade_duration = persist_time * (1 - MAINTAIN_PHASE_RATIO)

                if time_since_obstacle < maintain_duration:
                    # Phase 1: Maintain full offset - robot is still passing the obstacle
                    effective_avoidance_offset = self.state["last_avoidance_offset"]
                    self.state["avoidance_active"] = True
                    self.get_logger().debug(
                        f"Avoidance persistence (MAINTAIN): {time_since_obstacle:.1f}s, offset={effective_avoidance_offset}"
                    )
                elif time_since_obstacle < persist_time:
                    # Phase 2: Fade back to line - robot has cleared obstacle
                    fade_progress = (
                        time_since_obstacle - maintain_duration
                    ) / fade_duration
                    fade_factor = 1.0 - fade_progress
                    effective_avoidance_offset = int(
                        self.state["last_avoidance_offset"] * fade_factor
                    )
                    self.state["avoidance_active"] = True
                    self.get_logger().debug(
                        f"Avoidance persistence (FADE): {time_since_obstacle:.1f}s, offset={effective_avoidance_offset}"
                    )
                else:
                    # Persistence window expired - return to normal line following
                    effective_avoidance_offset = 0
                    self.state["avoidance_active"] = False
                    self.state["last_avoidance_offset"] = 0
                    self.state["last_obstacle_time"] = None
            else:
                effective_avoidance_offset = 0
                self.state["avoidance_active"] = False

            # === SEPARATE AVOIDANCE STEERING (NOT added to PID error) ===
            # Instead of modifying the error (which causes PID derivative fighting),
            # we calculate a separate avoidance angular velocity that will be
            # added to the PID output in control_callback.
            #
            # This approach:
            # 1. PID works on pure line-following error (smooth, no jumps)
            # 2. Avoidance steering is a separate, direct angular velocity command
            # 3. Combines naturally without derivative spikes

            # Convert pixel offset to angular velocity
            # Positive offset = obstacle on left = need to turn RIGHT = negative angular velocity
            # (In ROS: positive angular.z = turn left, negative = turn right)
            AVOIDANCE_ANGULAR_GAIN = 0.008  # rad/s per pixel offset (tune as needed)

            if effective_avoidance_offset != 0:
                # Direct steering command based on offset
                avoidance_angular_velocity = (
                    -effective_avoidance_offset * AVOIDANCE_ANGULAR_GAIN
                )
                self.get_logger().debug(
                    f"Path-aware avoidance: offset={effective_avoidance_offset}, "
                    f"angular={avoidance_angular_velocity:.3f}, side={obstacle_side}"
                )
            else:
                avoidance_angular_velocity = 0.0

            # Handle line occlusion case - error from extrapolated line
            if (
                error is None
                and self.line_occluded
                and len(self.extrapolated_line_points) > 0
            ):
                # Line is fully occluded but we have extrapolation - use extrapolated centroid
                # Calculate error from extrapolated line (NO offset added here)
                bottom_points = sorted(
                    self.extrapolated_line_points, key=lambda p: p[1], reverse=True
                )[:3]
                if bottom_points:
                    avg_x = sum(p[0] for p in bottom_points) // len(bottom_points)
                    error = avg_x - (width // 2)  # Pure line error, no offset
                    self.get_logger().info(
                        f"Using extrapolated line for navigation, error={error}"
                    )

            # Store pure line-following error (NOT modified by avoidance)
            self.current_error = error

            self.state["obstacle_detected"] = (
                len(self.obstacles) > 0 or self.state["avoidance_active"]
            )
            # Store the calculated avoidance angular velocity for use in control_callback
            self.state["obstacle_avoidance_steering"] = avoidance_angular_velocity
            self.state["obstacle_in_danger_zone"] = is_danger
            self.state["obstacle_relative_to_line"] = obstacle_side
            self.state["avoidance_offset"] = effective_avoidance_offset

            # Visualize obstacle detection
            self.visualize_obstacles(
                frame, self.obstacles, obstacle_roi_pts, height, width
            )

            # 5. Junction detection & Auto-switch ROI
            self.process_junction_logic(frame, contour, error, height, width)

            # 6. Visualization
            self.visualize(frame, roi_mask, contour, centroid, error, height, width)

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
                # cv2.imshow("Mask", processed_mask)
                cv2.imshow("Obstacle Mask", obstacle_mask)
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
                self.get_logger().info("Turn complete. Resuming lane follow.")

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

    def visualize_obstacles(self, frame, obstacles, roi_pts, height, width):
        """Visualize detected obstacles and avoidance zones."""
        # Draw obstacle detection ROI (trapezoid)
        cv2.polylines(frame, [roi_pts], True, (255, 128, 0), 2)

        # Draw zone lines
        danger_y = int(height * self.OBSTACLE_DANGER_ZONE_Y)
        warning_y = int(height * self.OBSTACLE_WARNING_ZONE_Y)

        cv2.line(frame, (0, danger_y), (width, danger_y), (0, 0, 255), 1)
        cv2.putText(
            frame,
            "DANGER ZONE",
            (width - 120, danger_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
        )

        cv2.line(frame, (0, warning_y), (width, warning_y), (0, 165, 255), 1)
        cv2.putText(
            frame,
            "WARNING ZONE",
            (width - 130, warning_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 165, 255),
            1,
        )

        # Draw line trajectory points (for debugging path-aware avoidance)
        if len(self.line_points) > 1:
            for i, pt in enumerate(self.line_points):
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame, self.line_points[i - 1], pt, (0, 255, 0), 1)

        # Draw extrapolated line points (in cyan) if line is occluded
        if len(self.extrapolated_line_points) > 1:
            for i, pt in enumerate(self.extrapolated_line_points):
                # Check if this is an extrapolated point (not in original)
                is_extrapolated = pt not in self.line_points
                color = (
                    (255, 255, 0) if is_extrapolated else (0, 255, 0)
                )  # Cyan for extrapolated
                cv2.circle(frame, pt, 4, color, -1 if is_extrapolated else 1)
                if i > 0:
                    prev_pt = self.extrapolated_line_points[i - 1]
                    cv2.line(frame, prev_pt, pt, (255, 255, 0), 1)

        # Show occlusion status
        if self.line_occluded:
            cv2.putText(
                frame,
                "LINE OCCLUDED - EXTRAPOLATING",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )

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

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Draw centroid
            cv2.circle(frame, (cx, cy), 5, color, -1)

            # Draw line from obstacle to where line is at that y-coordinate
            line_x = self.get_line_x_at_y(self.line_points, cy)
            if line_x is not None:
                # Draw reference line showing obstacle-to-line relationship
                cv2.line(frame, (cx, cy), (line_x, cy), (255, 0, 255), 2)
                # Mark the line position
                cv2.circle(frame, (line_x, cy), 6, (255, 0, 255), 2)

                # Label with relative position to line
                rel_pos = "L of line" if cx < line_x else "R of line"
                label = f"{zone.upper()} {rel_pos}"
            else:
                label = f"{zone.upper()} off:{obs['offset']:+d}"

            cv2.putText(
                frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        # Show avoidance status (path-aware)
        if self.state["obstacle_detected"] or self.state["avoidance_active"]:
            obstacle_side = self.state.get("obstacle_relative_to_line", None)
            avoidance_offset = self.state.get("avoidance_offset", 0)

            # Check if we're in persistence mode (obstacle not currently visible but still avoiding)
            is_persisting = self.state["avoidance_active"] and len(obstacles) == 0

            if self.state["obstacle_in_danger_zone"]:
                if obstacle_side:
                    status_text = f"DANGER! Obs {obstacle_side.upper()} of line, pass {'RIGHT' if obstacle_side == 'left' else 'LEFT'}"
                else:
                    status_text = "OBSTACLE DANGER!"
                status_color = (0, 0, 255)
            elif is_persisting:
                # Show persistence status
                status_text = f"AVOIDING (persist), offset: {avoidance_offset:+d}px"
                status_color = (0, 200, 255)  # Light orange
            else:
                if obstacle_side:
                    status_text = (
                        f"Obs {obstacle_side} of line, offset: {avoidance_offset:+d}px"
                    )
                else:
                    status_text = f"Obstacle detected, offset: {avoidance_offset:+d}px"
                status_color = (0, 165, 255)

            cv2.putText(
                frame,
                status_text,
                (10, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                status_color,
                2,
            )

            # Draw avoidance offset arrow (shows which direction robot will shift)
            if avoidance_offset != 0:
                center_x = width // 2
                arrow_start = (center_x, height - 50)
                arrow_end = (center_x + avoidance_offset, height - 100)
                # Green arrow for path-aware avoidance direction
                cv2.arrowedLine(
                    frame, arrow_start, arrow_end, (0, 255, 128), 3, tipLength=0.3
                )
                cv2.putText(
                    frame,
                    f"Shift: {avoidance_offset:+d}px",
                    (arrow_end[0] - 40, arrow_end[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 128),
                    1,
                )

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

        elif self.state["fsm"] == "FOLLOW_LANE":
            if self.current_error is not None:
                # Follow line using PD-controller
                # Slow down if deciding at junction OR if sign directed a turn OR obstacle detected
                if self.state["junction_deciding"] or self.state["sign_directed"]:
                    twist.linear.x = self.LINEAR_SPEED_SLOW
                elif self.state["obstacle_in_danger_zone"]:
                    # Very slow when obstacle in danger zone
                    twist.linear.x = (
                        self.LINEAR_SPEED_SLOW * self.OBSTACLE_SLOWDOWN_FACTOR
                    )
                elif self.state["obstacle_detected"]:
                    # Slow down when obstacle detected
                    twist.linear.x = self.LINEAR_SPEED * self.OBSTACLE_SLOWDOWN_FACTOR
                else:
                    twist.linear.x = self.LINEAR_SPEED

                # === COMBINED STEERING: PID Line Following + Separate Avoidance ===
                #
                # 1. PID works on pure line error (smooth derivative)
                # 2. Avoidance steering is added as a separate component
                # 3. This prevents PID from "fighting" the avoidance offset

                # Get avoidance steering (calculated in image_callback)
                avoidance_steering = self.state.get("obstacle_avoidance_steering", 0.0)

                # Check if avoidance state changed (for PID derivative reset)
                was_avoiding = getattr(self, "_was_avoiding", False)
                is_avoiding = self.state.get("avoidance_active", False)
                avoidance_state_changed = was_avoiding != is_avoiding
                self._was_avoiding = is_avoiding

                # Calculate line-following steering via PID
                # Reset derivative if avoidance state just changed to prevent spikes
                line_steering = self.pid_follow_line(
                    self.current_error, reset_derivative=avoidance_state_changed
                )

                # Combine: line following + obstacle avoidance
                # Both components work together smoothly
                total_steering = line_steering + avoidance_steering

                if avoidance_steering != 0:
                    self.get_logger().debug(
                        f"Steering: line={line_steering:.3f}, avoidance={avoidance_steering:.3f}, "
                        f"total={total_steering:.3f}"
                    )

                twist.angular.z = total_steering

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
            "obstacle_detected": self.state["obstacle_detected"],
            "obstacle_count": len(self.obstacles),
            "obstacle_danger": self.state["obstacle_in_danger_zone"],
            "obstacle_avoidance": self.state["obstacle_avoidance_steering"],
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
        print("LIMO Line Follower with Road Sign & Obstacle Detection")
        print("=" * 60)
        print("Subscribing to: /camera/color/image_raw")
        print("Publishing velocity to: /cmd_vel")
        print("Publishing status to: /line_follower/status")
        print("Publishing debug image to: /line_follower/debug_image")
        print("\nBehavior:")
        print("  - Follows green line using PID control")
        print("  - Detects road signs (left, right, forward, stop)")
        print("  - Handles junctions automatically")
        print("  - Detects orange cones and avoids obstacles")
        print("  - Turns 180° if line is lost")
        print(f"\nRobot dimensions: {0.22*100:.0f}cm x {0.32*100:.0f}cm")
        print(f"Camera blind spot: {0.10*100:.0f}cm")
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
