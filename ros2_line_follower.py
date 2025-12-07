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
        self.LOWER_HSV = np.array([30, 90, 25])
        self.UPPER_HSV = np.array([85, 250, 255])

        # Orange cone HSV range (orange color detection)
        self.CONE_LOWER_HSV = np.array([120, 100, 150])
        self.CONE_UPPER_HSV = np.array([180, 255, 255])

        # Obstacle detection parameters
        self.OBSTACLE_MIN_AREA = 250  # Minimum contour area to consider as obstacle
        self.OBSTACLE_DANGER_ZONE_Y = 0.70  # Bottom 30% of frame is danger zone
        self.OBSTACLE_WARNING_ZONE_Y = 0.50  # Bottom 50% is warning zone

        # ROI boundaries for Avoidance
        self.ROI_BLIND_SPOT_Y = 1.0  # Bottom edge (100% of frame height)
        self.ROI_TOP_Y = 0.45  # Start detection from 40% down
        self.ROI_BOTTOM_WIDTH = 0.88  # 100% of frame width at bottom
        self.ROI_TOP_WIDTH = 0.5  # 50% of frame width at top

        # Obstacle avoidance PD controller parameters
        self.KP_AVOIDANCE = 0.005  # Proportional gain for avoidance
        self.KD_AVOIDANCE = 0.002  # Derivative gain for damping
        self.LINEAR_SPEED_AVOIDANCE = (
            0.08  # Normal forward speed during avoidance (m/s)
        )
        self.LINEAR_SPEED_SLOW_AVOIDANCE = 0.07  # Slow speed during avoidance (m/s)
        self.ANGULAR_SPEED_MAX_AVOIDANCE = 0.3  # Max angular velocity (rad/s)

        # PD state variables for avoidance
        self.previous_avoidance_error = 0.0
        self.last_avoidance_error_time = None

        # Avoidance state
        self.avoidance_active = False
        self.avoidance_phase = 1  # Phase 1: Turn, Phase 2: Move forward
        self.avoidance_direction = None  # 'left' or 'right' (direction to turn)
        self.obstacle_side = None  # 'left' or 'right' (side where obstacle is)
        self.phase_start_time = None  # Track when current phase started
        self.phase1_alignment_achieved = (
            False  # Track if Phase 1 corner alignment is done
        )
        self.phase1_restart_count = (
            0  # Count how many times Phase 1 restarts due to obstacle
        )
        self.PHASE1_MAX_RESTARTS = (
            10  # After this many restarts, force turn until obstacle clears
        )
        self.PHASE_TIMEOUT_TURN = 5.0  # Maximum seconds for turning phases (1 and 3)
        self.PHASE_TIMEOUT_FORWARD = 3.0  # Maximum seconds for forward phase (2)

        # Robot and obstacle dimensions for Phase 4 clearance calculation (in meters)
        self.ROBOT_WIDTH = 0.22  # 22 cm
        self.ROBOT_LENGTH = 0.32  # 32 cm
        self.CONE_WIDTH = 0.14  # 14 cm
        self.CONE_LENGTH = 0.14  # 14 cm
        self.CLEARANCE_SAFETY_MARGIN = 0.1  # 10 cm extra safety margin

        # Phase 4 forward clearance distance calculation:
        # Need to clear half robot width + half cone width + safety margin
        self.PHASE4_CLEARANCE_DISTANCE = (
            (self.ROBOT_WIDTH / 2)
            + (self.CONE_WIDTH / 2)
            + self.CLEARANCE_SAFETY_MARGIN
        )  # ~0.23 meters

        # Obstacle detection results
        self.obstacles = []
        self.lane_contour = None  # Store lane contour for edge detection
        self.lane_centroid = None  # Store lane centroid for visualization

        # Obstacle detection cooldown after stop sign
        self.OBSTACLE_DETECTION_COOLDOWN = (
            3.0  # Seconds to ignore obstacles after stop sign
        )
        self.last_stop_sign_detection_time = (
            None  # Track when stop sign was last detected
        )

        # Timing constants
        self.STOP_COOLDOWN = 5.0
        self.WAIT_DURATION = 3.0
        self.LOST_LINE_TIMEOUT = 0.5
        self.SEARCH_TIMEOUT = 2.0
        self.JUNCTION_CONFIRM_TIME = 0.7  # Wait before confirming junction
        self.TURN_180_DURATION = 3.0

        # Robot control parameters
        self.LINEAR_SPEED = 0.07  # m/s - forward speed
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
        }

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

    def is_obstacle_inside_sign(self, obstacle_bbox, sign_detections):
        """
        Check if an obstacle bounding box is inside any sign bounding box.
        """
        ox, oy, ow, oh = obstacle_bbox

        for sign in sign_detections:
            sx1, sy1, sx2, sy2 = sign["box"]

            # Check if obstacle center is inside sign box
            ocx = ox + ow / 2
            ocy = oy + oh / 2

            if sx1 < ocx < sx2 and sy1 < ocy < sy2:
                return True

            # Check for significant overlap (IoU-like)
            # Intersection
            ix1 = max(ox, sx1)
            iy1 = max(oy, sy1)
            ix2 = min(ox + ow, sx2)
            iy2 = min(oy + oh, sy2)

            if ix1 < ix2 and iy1 < iy2:
                intersection_area = (ix2 - ix1) * (iy2 - iy1)
                obstacle_area = ow * oh
                # If more than 50% of obstacle is inside sign, ignore it
                if intersection_area > 0.5 * obstacle_area:
                    return True

        return False

    def detect_obstacles(self, frame, height, width, sign_detections):
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

            # Check if this obstacle is actually a sign
            if self.is_obstacle_inside_sign((x, y, w, h), sign_detections):
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

    def get_lane_edge_x(self, side, height, width):
        """
        Get the x-coordinate of the lane edge at a specific y level.
        Returns the left or right edge of the detected green line.
        """
        if self.lane_contour is None:
            return None

        # Get the y-level at the top of the obstacle ROI
        top_y = int(height * self.ROI_TOP_Y)

        # Find lane contour points near the top_y level
        tolerance = 30  # pixels tolerance for y-level matching
        edge_points = []

        for point in self.lane_contour:
            px, py = point[0]
            if abs(py - top_y) < tolerance:
                edge_points.append(px)

        if not edge_points:
            # If no points at that level, use the overall bounding box
            x, y, w, h = cv2.boundingRect(self.lane_contour)
            if side == "left":
                return x
            else:
                return x + w

        if side == "left":
            return min(edge_points)
        else:
            return max(edge_points)

    def get_obstacle_roi_corner_x(self, side, height, width):
        """
        Get the x-coordinate of the obstacle ROI corner at top.
        """
        top_y = int(height * self.ROI_TOP_Y)
        top_width = int(width * self.ROI_TOP_WIDTH)
        center_x = width // 2

        if side == "left":
            return center_x - top_width // 2
        else:
            return center_x + top_width // 2

    def get_obstacle_roi_bottom_corner_x(self, side, height, width):
        """
        Get the x-coordinate of the obstacle ROI corner at bottom.
        """
        bottom_width = int(width * self.ROI_BOTTOM_WIDTH)
        center_x = width // 2

        if side == "left":
            return center_x - bottom_width // 2
        else:
            return center_x + bottom_width // 2

    def get_lane_roi_bottom_corner_x(self, side, height, width):
        """
        Get the x-coordinate of the lane ROI corner at bottom.
        Lane ROI has full width at bottom.
        """
        bottom_width = width  # Lane ROI is full width at bottom
        center_x = width // 2

        if side == "left":
            return center_x - bottom_width // 2
        else:
            return center_x + bottom_width // 2

    def get_lane_edge_x_at_bottom(self, side, height, width):
        """
        Get the x-coordinate of the lane edge at the bottom of the ROI.
        """
        if self.lane_contour is None:
            return None

        # Get the y-level at the bottom of the obstacle ROI
        bottom_y = int(height * self.ROI_BLIND_SPOT_Y)

        # Find lane contour points near the bottom_y level
        tolerance = 30  # pixels tolerance for y-level matching
        edge_points = []

        for point in self.lane_contour:
            px, py = point[0]
            if abs(py - bottom_y) < tolerance:
                edge_points.append(px)

        if not edge_points:
            # If no points at that level, use the overall bounding box
            x, y, w, h = cv2.boundingRect(self.lane_contour)
            if side == "left":
                return x
            else:
                return x + w

        if side == "left":
            return min(edge_points)
        else:
            return max(edge_points)

    def calculate_avoidance_error(self, height, width):
        """
        Calculate the error for obstacle avoidance.

        Phase 1 (Turn):
        - Obstacle on left -> turn right -> target: left ROI top corner reaches right lane edge
        - Obstacle on right -> turn left -> target: right ROI top corner reaches left lane edge

        Phase 2 (Move forward):
        - Obstacle on left -> target: left ROI bottom corner reaches right lane edge
        - Obstacle on right -> target: right ROI bottom corner reaches left lane edge

        Phase 3 (Align back to opposite edge):
        - Obstacle was on left -> turn left -> target: left ROI top corner aligns with RIGHT lane edge
        - Obstacle was on right -> turn right -> target: right ROI top corner aligns with LEFT lane edge

        Phase 4 (Timed forward clearance):
        - Move forward for calculated time to clear the cone completely
        - No error calculation needed (timed movement)

        Phase 5 (Center on lane):
        - Obstacle was on left -> turn right -> target: left ROI top corner aligns with LEFT lane edge
        - Obstacle was on right -> turn left -> target: right ROI top corner aligns with RIGHT lane edge

        Returns: error (positive = turn right, negative = turn left)
        """
        if not self.avoidance_active or self.obstacle_side is None:
            return None

        if self.avoidance_phase == 1:
            # Phase 1: Turn until top corner meets lane edge
            if self.obstacle_side == "left":
                # Obstacle on left -> turn right
                # Target: left ROI top corner reaches right lane edge
                roi_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x("right", height, width)

                if lane_edge_x is None:
                    return 100  # Default turn right if no lane detected

                # Error is distance between corner and edge
                # We want corner to be AT the edge
                error = lane_edge_x - roi_corner_x

            else:
                # Obstacle on right -> turn left
                # Target: right ROI top corner reaches left lane edge
                roi_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x("left", height, width)

                if lane_edge_x is None:
                    return -100  # Default turn left if no lane detected

                error = lane_edge_x - roi_corner_x

        elif self.avoidance_phase == 2:
            # Phase 2: Move forward until lane ROI bottom corner aligns with lane edge
            if self.obstacle_side == "left":
                # Target: left ROI bottom corner reaches right lane edge
                roi_corner_x = self.get_lane_roi_bottom_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x_at_bottom("right", height, width)

                if lane_edge_x is None:
                    return 100  # Default if no lane detected

                error = lane_edge_x - roi_corner_x

            else:
                # Target: right ROI bottom corner reaches left lane edge
                roi_corner_x = self.get_lane_roi_bottom_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x_at_bottom("left", height, width)

                if lane_edge_x is None:
                    return -100  # Default if no lane detected

                error = lane_edge_x - roi_corner_x

        elif self.avoidance_phase == 3:
            # Phase 3: Align back - turn to align with OPPOSITE side lane edge
            if self.obstacle_side == "left":
                # Obstacle was on left -> turn left to realign
                # Target: left ROI top corner aligns with RIGHT lane edge (opposite side)
                roi_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x("right", height, width)

                if lane_edge_x is None:
                    return -100  # Default turn left if no lane detected

                error = lane_edge_x - roi_corner_x

            else:
                # Obstacle was on right -> turn right to realign
                # Target: right ROI top corner aligns with LEFT lane edge (opposite side)
                roi_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x("left", height, width)

                if lane_edge_x is None:
                    return 100  # Default turn right if no lane detected

                error = lane_edge_x - roi_corner_x

        elif self.avoidance_phase == 4:
            # Phase 4: Timed forward movement - no error calculation needed
            return None

        elif self.avoidance_phase == 5:
            # Phase 5: Center on lane - align same-side corner with same-side edge
            if self.obstacle_side == "left":
                # Obstacle was on left -> robot is displaced to the RIGHT of lane
                # Need to turn LEFT to center back on lane
                # Target: left ROI top corner aligns with LEFT lane edge (same side)
                roi_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x("left", height, width)

                if lane_edge_x is None:
                    return (
                        -100
                    )  # Default turn LEFT if no lane detected (lane is to our left)

                error = lane_edge_x - roi_corner_x

            else:
                # Obstacle was on right -> robot is displaced to the LEFT of lane
                # Need to turn RIGHT to center back on lane
                # Target: right ROI top corner aligns with RIGHT lane edge (same side)
                roi_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x("right", height, width)

                if lane_edge_x is None:
                    return 100  # Default turn RIGHT if no lane detected (lane is to our right)

                error = lane_edge_x - roi_corner_x

        else:
            return None

        return error

    def pd_avoidance(self, error):
        """
        Calculate angular velocity using PD controller for avoidance.

        Note: In ROS2, positive angular.z = turn LEFT, negative angular.z = turn RIGHT
        Our error convention: positive error = need to turn right, negative = turn left
        So we negate the output to match ROS2 convention.
        """
        if error is None:
            self.previous_avoidance_error = 0.0
            self.last_avoidance_error_time = None
            return 0.0

        current_time = time.time()

        # Proportional term
        p_term = self.KP_AVOIDANCE * error

        # Derivative term
        d_term = 0.0
        if self.last_avoidance_error_time is not None:
            dt = current_time - self.last_avoidance_error_time
            if dt > 0:
                error_derivative = (error - self.previous_avoidance_error) / dt
                d_term = self.KD_AVOIDANCE * error_derivative

        # Update state
        self.previous_avoidance_error = error
        self.last_avoidance_error_time = current_time

        # Negate to match ROS2 convention: positive error -> turn right (negative angular.z)
        return -(p_term + d_term)

    def update_avoidance_state(self, height, width):
        """
        Update the avoidance state based on detected obstacles.
        """
        center_x = width // 2

        # Check for obstacles in warning or danger zone
        active_obstacle = None
        for obs in self.obstacles:
            if obs["zone"] in ["warning", "danger"]:
                active_obstacle = obs
                break

        if active_obstacle is not None:
            # Determine which side the obstacle is on
            if active_obstacle["offset"] < 0:
                self.obstacle_side = "left"
            else:
                self.obstacle_side = "right"

            if not self.avoidance_active:
                self.avoidance_active = True
                self.avoidance_phase = 1
                self.phase_start_time = time.time()
                self.phase1_alignment_achieved = False
                self.phase1_restart_count = 0
                self.get_logger().info(
                    f"AVOIDANCE START: Obstacle on {self.obstacle_side}"
                )

                # Set initial direction
                if self.obstacle_side == "left":
                    self.avoidance_direction = "right"
                else:
                    self.avoidance_direction = "left"
            elif self.avoidance_phase == 1:
                # If we are in Phase 1 and see an obstacle, we might need to keep turning
                # Reset phase 1 start time to prevent timeout if we are still seeing obstacle
                # But also track restarts to avoid infinite loops
                self.phase1_restart_count += 1
                if self.phase1_restart_count < self.PHASE1_MAX_RESTARTS:
                    self.phase_start_time = time.time()

        else:
            # No active obstacle - check phase transitions and completion
            if self.avoidance_active:
                phase_elapsed = time.time() - self.phase_start_time

                if self.avoidance_phase == 1:
                    # Check if we have aligned (error close to 0)
                    error = self.calculate_avoidance_error(height, width)
                    if error is not None and abs(error) < 20:  # 20px tolerance
                        self.phase1_alignment_achieved = True

                    # Transition to Phase 2 if aligned OR timeout
                    if (
                        self.phase1_alignment_achieved
                        or phase_elapsed > self.PHASE_TIMEOUT_TURN
                    ):
                        self.avoidance_phase = 2
                        self.phase_start_time = time.time()
                        self.get_logger().info("AVOIDANCE PHASE 2: Move Forward")

                elif self.avoidance_phase == 2:
                    # Check alignment at bottom
                    error = self.calculate_avoidance_error(height, width)
                    is_aligned = error is not None and abs(error) < 20

                    if is_aligned or phase_elapsed > self.PHASE_TIMEOUT_FORWARD:
                        self.avoidance_phase = 3
                        self.phase_start_time = time.time()
                        self.get_logger().info("AVOIDANCE PHASE 3: Align Back")

                elif self.avoidance_phase == 3:
                    # Check alignment
                    error = self.calculate_avoidance_error(height, width)
                    is_aligned = error is not None and abs(error) < 20

                    if is_aligned or phase_elapsed > self.PHASE_TIMEOUT_TURN:
                        self.avoidance_phase = 4
                        self.phase_start_time = time.time()
                        # Calculate Phase 4 duration based on clearance distance and speed
                        self.phase4_duration = (
                            self.PHASE4_CLEARANCE_DISTANCE / self.LINEAR_SPEED_AVOIDANCE
                        )
                        self.get_logger().info(
                            f"AVOIDANCE PHASE 4: Forward Clearance ({self.PHASE4_CLEARANCE_DISTANCE:.2f}m in {self.phase4_duration:.2f}s)"
                        )

                elif self.avoidance_phase == 4:
                    # Phase 4: Timed forward movement to clear the cone
                    if phase_elapsed >= self.phase4_duration:
                        self.avoidance_phase = 5
                        self.phase_start_time = time.time()
                        self.get_logger().info("AVOIDANCE PHASE 5: Center on Lane")

                elif self.avoidance_phase == 5:
                    # Phase 5: Center on lane
                    error = self.calculate_avoidance_error(height, width)
                    is_aligned = error is not None and abs(error) < 20

                    if is_aligned or phase_elapsed > self.PHASE_TIMEOUT_TURN:
                        self.avoidance_active = False
                        self.get_logger().info("AVOIDANCE COMPLETE")

        # Also check Phase 3 timeout outside the obstacle check (fallback if stuck)
        if self.avoidance_active and self.avoidance_phase == 3:
            phase3_elapsed = (
                time.time() - self.phase_start_time if self.phase_start_time else 0
            )
            if phase3_elapsed >= self.PHASE_TIMEOUT_TURN:
                self.avoidance_phase = 4
                self.phase_start_time = time.time()
                self.phase4_duration = (
                    self.PHASE4_CLEARANCE_DISTANCE / self.LINEAR_SPEED_AVOIDANCE
                )
                self.get_logger().info(
                    f"AVOIDANCE PHASE 4: Forward Clearance (Timeout) ({self.PHASE4_CLEARANCE_DISTANCE:.2f}m in {self.phase4_duration:.2f}s)"
                )

        # Also check Phase 4 completion outside obstacle check
        if self.avoidance_active and self.avoidance_phase == 4:
            phase4_elapsed = (
                time.time() - self.phase_start_time if self.phase_start_time else 0
            )
            if (
                hasattr(self, "phase4_duration")
                and phase4_elapsed >= self.phase4_duration
            ):
                self.avoidance_phase = 5
                self.phase_start_time = time.time()
                self.get_logger().info("AVOIDANCE PHASE 5: Center on Lane")

        # Also check Phase 5 timeout outside obstacle check
        if self.avoidance_active and self.avoidance_phase == 5:
            phase5_elapsed = (
                time.time() - self.phase_start_time if self.phase_start_time else 0
            )
            if phase5_elapsed >= self.PHASE_TIMEOUT_TURN:
                self.avoidance_active = False
                self.get_logger().info("AVOIDANCE COMPLETE (Phase 5 Timeout)")

    def visualize_avoidance(self, frame, obstacle_roi_pts, height, width):
        """
        Visualize avoidance state and targets for all three phases.
        """
        if not self.avoidance_active:
            return

        top_y = int(height * self.ROI_TOP_Y)
        bottom_y = (
            int(height * self.ROI_BLIND_SPOT_Y) - 1
        )  # Slight offset to stay in frame

        status = ""
        error = self.calculate_avoidance_error(height, width)

        if self.avoidance_phase == 1:
            # Phase 1: Show top corner and lane edge target
            if self.obstacle_side == "left":
                top_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x("right", height, width)
            else:
                top_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x("left", height, width)

            # Draw top corner and target (cyan to magenta)
            cv2.circle(frame, (top_corner_x, top_y), 10, (255, 255, 0), -1)
            if lane_edge_x is not None:
                cv2.circle(frame, (lane_edge_x, top_y), 10, (255, 0, 255), -1)
                cv2.line(
                    frame,
                    (top_corner_x, top_y),
                    (lane_edge_x, top_y),
                    (255, 255, 255),
                    2,
                )

            status = f"PHASE 1: Turn {self.avoidance_direction.upper()}"

        elif self.avoidance_phase == 2:
            # Phase 2: Show lane ROI bottom corner and lane edge target
            if self.obstacle_side == "left":
                bottom_corner_x = self.get_lane_roi_bottom_corner_x(
                    "left", height, width
                )
                lane_edge_x = self.get_lane_edge_x_at_bottom("right", height, width)
            else:
                bottom_corner_x = self.get_lane_roi_bottom_corner_x(
                    "right", height, width
                )
                lane_edge_x = self.get_lane_edge_x_at_bottom("left", height, width)

            # Draw bottom corner and target
            cv2.circle(frame, (bottom_corner_x, bottom_y), 10, (255, 255, 0), -1)
            if lane_edge_x is not None:
                cv2.circle(frame, (lane_edge_x, bottom_y), 10, (255, 0, 255), -1)
                cv2.line(
                    frame,
                    (bottom_corner_x, bottom_y),
                    (lane_edge_x, bottom_y),
                    (255, 255, 255),
                    2,
                )

            status = f"PHASE 2: Move FORWARD"

        elif self.avoidance_phase == 3:
            # Phase 3: Show top corner aligning with OPPOSITE side lane edge
            turn_dir = "LEFT" if self.obstacle_side == "left" else "RIGHT"

            if self.obstacle_side == "left":
                top_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                top_lane_edge_x = self.get_lane_edge_x(
                    "right", height, width
                )  # Opposite side
            else:
                top_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                top_lane_edge_x = self.get_lane_edge_x(
                    "left", height, width
                )  # Opposite side

            # Draw top corner and target (cyan to magenta)
            cv2.circle(frame, (top_corner_x, top_y), 10, (255, 255, 0), -1)
            if top_lane_edge_x is not None:
                cv2.circle(frame, (top_lane_edge_x, top_y), 10, (255, 0, 255), -1)
                cv2.line(
                    frame,
                    (top_corner_x, top_y),
                    (top_lane_edge_x, top_y),
                    (255, 255, 255),
                    2,
                )

            status = f"PHASE 3: Turn {turn_dir} (align)"

        elif self.avoidance_phase == 4:
            # Phase 4: Timed forward movement to clear the cone
            phase4_elapsed = (
                time.time() - self.phase_start_time if self.phase_start_time else 0
            )
            phase4_duration = getattr(
                self,
                "phase4_duration",
                self.PHASE4_CLEARANCE_DISTANCE / self.LINEAR_SPEED_AVOIDANCE,
            )
            progress = (
                min(1.0, phase4_elapsed / phase4_duration)
                if phase4_duration > 0
                else 1.0
            )

            # Draw progress bar for Phase 4
            bar_x = 10
            bar_y = 150
            bar_width = 200
            bar_height = 20

            # Background bar
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1,
            )
            # Progress bar
            progress_width = int(bar_width * progress)
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + progress_width, bar_y + bar_height),
                (0, 255, 0),
                -1,
            )
            # Border
            cv2.rectangle(
                frame,
                (bar_x, bar_y),
                (bar_x + bar_width, bar_y + bar_height),
                (255, 255, 255),
                2,
            )

            status = f"PHASE 4: Forward Clearance ({progress*100:.0f}%)"

        else:  # Phase 5
            # Phase 5: Center on lane - show same-side corner aligning with same-side edge
            turn_dir = "RIGHT" if self.obstacle_side == "left" else "LEFT"

            if self.obstacle_side == "left":
                top_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                top_lane_edge_x = self.get_lane_edge_x(
                    "left", height, width
                )  # Same side
            else:
                top_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                top_lane_edge_x = self.get_lane_edge_x(
                    "right", height, width
                )  # Same side

            # Draw top corner and target (cyan to magenta)
            cv2.circle(frame, (top_corner_x, top_y), 10, (255, 255, 0), -1)
            if top_lane_edge_x is not None:
                cv2.circle(frame, (top_lane_edge_x, top_y), 10, (255, 0, 255), -1)
                cv2.line(
                    frame,
                    (top_corner_x, top_y),
                    (top_lane_edge_x, top_y),
                    (255, 255, 255),
                    2,
                )

            status = f"PHASE 5: Turn {turn_dir} (center)"

        # Display avoidance status
        cv2.putText(
            frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        if error is not None:
            cv2.putText(
                frame,
                f"Avoid Err: {error}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )

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

    def create_avoidance_lane_roi_mask(self, height, width):
        """
        Create ROI mask for lane detection during avoidance.
        Full width trapezoid aligned with obstacle ROI vertical boundaries.
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        # Use same vertical boundaries as obstacle ROI
        blind_spot_y = int(height * self.ROI_BLIND_SPOT_Y)  # Bottom edge
        top_y = int(height * self.ROI_TOP_Y)  # Start detection from top

        # Full width trapezoid
        bottom_width = width  # 100% of frame width at bottom
        top_width = int(width * 0.6)  # 60% of frame width at top

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

            # Initialize processed_mask for visualization
            processed_mask = np.zeros((height, width), dtype=np.uint8)

            # 1. Obstacle Detection (always run to check for obstacles)
            # We need sign detections to filter out false positives, but only if not in avoidance
            if self.avoidance_active:
                # During avoidance, don't detect signs - pass empty list
                detections = []
            else:
                # Normal mode - detect signs
                frame, detections = self.detect_and_annotate_signs(frame)

                # Track stop sign detection time for obstacle cooldown
                for det in detections:
                    if det["class_name"] == "stop":
                        self.last_stop_sign_detection_time = time.time()
                        break

            # 2. Obstacle Detection (with cooldown after stop sign)
            # Check if we're in cooldown period after stop sign detection
            obstacle_cooldown_active = False
            if self.last_stop_sign_detection_time is not None:
                time_since_stop = time.time() - self.last_stop_sign_detection_time
                if time_since_stop < self.OBSTACLE_DETECTION_COOLDOWN:
                    obstacle_cooldown_active = True

            if obstacle_cooldown_active:
                # Don't detect obstacles during cooldown - clear obstacle list
                self.obstacles = []
                obstacle_mask = np.zeros((height, width), dtype=np.uint8)
                obstacle_roi_pts, _ = self.create_obstacle_roi_mask(height, width)
                obstacle_roi_pts = (
                    obstacle_roi_pts
                    if isinstance(obstacle_roi_pts, np.ndarray)
                    else np.array([])
                )
                # Get the ROI pts for visualization
                _, obstacle_roi_pts = self.create_obstacle_roi_mask(height, width)
            else:
                self.obstacles, obstacle_mask, obstacle_roi_pts = self.detect_obstacles(
                    frame, height, width, detections
                )

            # 3. Lane detection for avoidance (always run to get lane contour for avoidance calculations)
            avoidance_lane_roi_mask, _ = self.create_avoidance_lane_roi_mask(
                height, width
            )
            _, self.lane_contour, self.lane_centroid, _ = self.process_lane_detection(
                frame, avoidance_lane_roi_mask
            )

            # 4. Update avoidance state machine
            self.update_avoidance_state(height, width)
            self.visualize_avoidance(frame, obstacle_roi_pts, height, width)

            if self.avoidance_active:
                # === AVOIDANCE MODE ===
                # Sign detection and line following are INACTIVE during avoidance

                # Calculate error based on current phase
                avoidance_error = self.calculate_avoidance_error(height, width)

                # Use separate PID for avoidance
                avoidance_steering = self.pd_avoidance(avoidance_error)

                # Override current error/steering for control callback
                self.current_error = None  # Disable normal line following error
                self.avoidance_steering_cmd = avoidance_steering

                # Visualize avoidance (includes both ROIs like avoidance.py)
                # self.visualize_avoidance(frame, obstacle_roi_pts, height, width)

                # Draw obstacle ROI (orange)
                cv2.polylines(frame, [obstacle_roi_pts], True, (255, 128, 0), 2)

                # Draw lane ROI (green) - same as avoidance.py
                _, lane_roi_pts = self.create_avoidance_lane_roi_mask(height, width)
                cv2.polylines(frame, [lane_roi_pts], True, (0, 255, 0), 2)

                # Draw lane contour if available
                if self.lane_contour is not None:
                    cv2.drawContours(frame, [self.lane_contour], -1, (0, 255, 0), 2)
                    if self.lane_centroid is not None:
                        cv2.circle(frame, self.lane_centroid, 8, (0, 255, 0), -1)

                # Draw zone lines (like avoidance.py)
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

                # Draw detected obstacles with bounding boxes
                for obs in self.obstacles:
                    x, y, w, h = obs["bbox"]
                    cx, cy = obs["centroid"]
                    zone = obs["zone"]
                    if zone == "danger":
                        color = (0, 0, 255)  # Red
                    elif zone == "warning":
                        color = (0, 165, 255)  # Orange
                    else:
                        color = (0, 255, 255)  # Yellow
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    cv2.circle(frame, (cx, cy), 5, color, -1)
                    label = f"{zone.upper()} off:{obs['offset']:+d}"
                    cv2.putText(
                        frame,
                        label,
                        (x, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

                # Use obstacle mask for visualization
                processed_mask = obstacle_mask

            else:
                # === NORMAL MODE ===
                self.avoidance_steering_cmd = None  # Reset avoidance command

                # Draw sign detection ROI lines (only in normal mode)
                sign_roi_top = int(height * 0.65)
                sign_roi_bottom = int(height * 0.7)
                cv2.line(
                    frame, (0, sign_roi_top), (width, sign_roi_top), (0, 0, 255), 2
                )
                cv2.line(
                    frame,
                    (0, sign_roi_bottom),
                    (width, sign_roi_bottom),
                    (0, 0, 255),
                    2,
                )

                # 5. FSM Logic Block (Signs)
                self.process_fsm(
                    frame, detections, sign_roi_top, sign_roi_bottom, height, width
                )

                # 6. Lane Detection (Normal)
                roi_mask = self.create_roi_mask(height, width, self.state["roi_mode"])
                processed_mask, contour, centroid, error = self.process_lane_detection(
                    frame, roi_mask
                )
                self.current_error = error

                # 7. Junction detection & Auto-switch ROI
                self.process_junction_logic(frame, contour, error, height, width)

                # 8. Visualization
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
                # cv2.imshow("Mask", processed_mask)
                # Show obstacle detection mask (orange cone HSV detection)
                # cv2.imshow("Obstacle Mask", obstacle_mask)
                cv2.imshow("Line Follower", frame)
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
        if self.avoidance_active:
            # === AVOIDANCE PRIORITY ===
            # Override all other behaviors when avoiding an obstacle
            # Phase-specific control matching avoidance.py logic

            if self.avoidance_phase == 1:
                # Phase 1: Turn to align corner with lane edge (no forward movement)
                twist.linear.x = 0.0

                if self.phase1_alignment_achieved:
                    # Alignment done, but obstacle still visible - turn at constant speed
                    if self.avoidance_direction == "right":
                        twist.angular.z = (
                            -self.ANGULAR_SPEED_MAX_AVOIDANCE
                        )  # Turn right (negative)
                    else:
                        twist.angular.z = (
                            self.ANGULAR_SPEED_MAX_AVOIDANCE
                        )  # Turn left (positive)
                else:
                    # Use PD controller to achieve corner alignment
                    if self.avoidance_steering_cmd is not None:
                        twist.angular.z = max(
                            -self.ANGULAR_SPEED_MAX_AVOIDANCE,
                            min(
                                self.ANGULAR_SPEED_MAX_AVOIDANCE,
                                self.avoidance_steering_cmd,
                            ),
                        )
                    else:
                        twist.angular.z = 0.0

            elif self.avoidance_phase == 2:
                # Phase 2: Move forward only (no steering)
                twist.linear.x = self.LINEAR_SPEED_AVOIDANCE  # Normal forward speed
                twist.angular.z = 0.0  # No turning, only forward

            elif self.avoidance_phase == 3:
                # Phase 3: Turn back to align with lane (no forward movement)
                twist.linear.x = 0.0
                if self.avoidance_steering_cmd is not None:
                    twist.angular.z = max(
                        -self.ANGULAR_SPEED_MAX_AVOIDANCE,
                        min(
                            self.ANGULAR_SPEED_MAX_AVOIDANCE,
                            self.avoidance_steering_cmd,
                        ),
                    )
                else:
                    twist.angular.z = 0.0

            elif self.avoidance_phase == 4:
                # Phase 4: Move forward to clear the cone (no steering)
                twist.linear.x = self.LINEAR_SPEED_AVOIDANCE
                twist.angular.z = 0.0

            else:  # Phase 5
                # Phase 5: Turn to center on lane (no forward movement)
                twist.linear.x = 0.0
                if self.avoidance_steering_cmd is not None:
                    twist.angular.z = max(
                        -self.ANGULAR_SPEED_MAX_AVOIDANCE,
                        min(
                            self.ANGULAR_SPEED_MAX_AVOIDANCE,
                            self.avoidance_steering_cmd,
                        ),
                    )
                else:
                    twist.angular.z = 0.0

        elif self.state["fsm"] == "STOP_WAIT":
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
