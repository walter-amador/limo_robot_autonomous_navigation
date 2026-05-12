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
from geometry_msgs.msg import Twist
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

        # Green line HSV range (lane detection)
        self.LANE_LOWER_HSV = np.array([30, 90, 25])
        self.LANE_UPPER_HSV = np.array([85, 250, 255])

        # Obstacle detection parameters
        self.OBSTACLE_MIN_AREA = 250  # Minimum contour area to consider as obstacle
        self.OBSTACLE_DANGER_ZONE_Y = 0.70  # Bottom 30% of frame is danger zone
        self.OBSTACLE_WARNING_ZONE_Y = 0.50  # Bottom 50% is warning zone

        # ROI boundaries
        self.ROI_BLIND_SPOT_Y = 1.0  # Bottom edge (100% of frame height)
        self.ROI_TOP_Y = 0.45  # Start detection from 40% down
        self.ROI_BOTTOM_WIDTH = 0.88  # 100% of frame width at bottom
        self.ROI_TOP_WIDTH = 0.5  # 50% of frame width at top

        # Obstacle detection results
        self.obstacles = []

        # Lane detection results
        self.lane_centroid = None
        self.lane_error = None
        self.lane_contour = None  # Store lane contour for edge detection

        # Obstacle avoidance PD controller parameters
        self.KP_AVOIDANCE = 0.005  # Proportional gain for avoidance
        self.KD_AVOIDANCE = 0.002  # Derivative gain for damping
        self.LINEAR_SPEED = 0.08  # Normal forward speed (m/s)
        self.LINEAR_SPEED_SLOW = 0.05  # Slow speed during avoidance (m/s)
        self.ANGULAR_SPEED_MAX = 0.2  # Max angular velocity (rad/s)

        # PD state variables
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
        self.PHASE_TIMEOUT_TURN = 3.0  # Maximum seconds for turning phases (1 and 3)
        self.PHASE_TIMEOUT_FORWARD = 2.0  # Maximum seconds for forward phase (2)

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

        # Publisher for velocity commands
        self.velocity_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Timer for control loop (20 Hz)
        self.control_timer = self.create_timer(0.05, self.control_callback)

        self.get_logger().info("Obstacle Detection Node started")
        self.get_logger().info("Subscribing to: /camera/image_raw")
        self.get_logger().info("Publishing velocity to: /cmd_vel")
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

    def create_lane_roi_mask(self, height, width):
        """
        Create ROI mask for lane detection.
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

    def process_lane_detection(self, frame, roi_mask):
        """Process frame to detect lane line and calculate error."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, self.LANE_LOWER_HSV, self.LANE_UPPER_HSV)

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
            cv2.putText(
                frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1
            )

        # Show obstacle count
        if len(obstacles) > 0:
            cv2.putText(
                frame,
                f"Obstacles: {len(obstacles)}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

    def visualize_lane(
        self, frame, lane_roi_pts, lane_contour, centroid, error, height, width
    ):
        """Visualize lane detection."""
        # Draw lane detection ROI (trapezoid) in green
        cv2.polylines(frame, [lane_roi_pts], True, (0, 255, 0), 2)

        # Draw lane contour and centroid
        if lane_contour is not None:
            cv2.drawContours(frame, [lane_contour], -1, (0, 255, 0), 2)
            if centroid is not None:
                cv2.circle(frame, centroid, 8, (0, 255, 0), -1)

        # Draw center line
        center_x = width // 2
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)

        # Display error
        if error is not None:
            cv2.putText(
                frame,
                f"Lane Error: {error:+d}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )

    def process_frame(self, frame):
        """Process a single frame."""
        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]

        # Lane Detection
        lane_roi_mask, lane_roi_pts = self.create_lane_roi_mask(height, width)
        lane_mask, self.lane_contour, self.lane_centroid, self.lane_error = (
            self.process_lane_detection(frame, lane_roi_mask)
        )

        # Obstacle Detection
        self.obstacles, obstacle_mask, obstacle_roi_pts = self.detect_obstacles(
            frame, height, width
        )

        # Visualizations
        self.visualize_lane(
            frame,
            lane_roi_pts,
            self.lane_contour,
            self.lane_centroid,
            self.lane_error,
            height,
            width,
        )
        self.visualize_obstacles(frame, self.obstacles, obstacle_roi_pts, height, width)
        self.visualize_avoidance(frame, obstacle_roi_pts, height, width)

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

        # Combine masks for display
        combined_display_mask = cv2.bitwise_or(obstacle_mask, lane_mask)

        return frame, combined_display_mask

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

        Phase 3 (Align back):
        - Obstacle was on left -> turn left -> target: left ROI top corner aligns with left lane edge
        - Obstacle was on right -> turn right -> target: right ROI top corner aligns with right lane edge

        Returns: error (positive = turn right, negative = turn left)
        """
        if not self.avoidance_active or self.obstacle_side is None:
            return None

        if self.avoidance_phase == 1:
            # Phase 1: Turn until top corner meets lane edge
            if self.obstacle_side == "left":
                roi_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x("right", height, width)

                if lane_edge_x is None:
                    return 100  # Default turn right if no lane detected

                error = lane_edge_x - roi_corner_x

            else:  # obstacle on right
                roi_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x("left", height, width)

                if lane_edge_x is None:
                    return -100  # Default turn left if no lane detected

                error = lane_edge_x - roi_corner_x

        elif self.avoidance_phase == 2:
            # Phase 2: Move forward until lane ROI bottom corner aligns with lane edge
            if self.obstacle_side == "left":
                roi_corner_x = self.get_lane_roi_bottom_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x_at_bottom("right", height, width)

                if lane_edge_x is None:
                    return 100  # Default if no lane detected

                error = lane_edge_x - roi_corner_x

            else:  # obstacle on right
                roi_corner_x = self.get_lane_roi_bottom_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x_at_bottom("left", height, width)

                if lane_edge_x is None:
                    return -100  # Default if no lane detected

                error = lane_edge_x - roi_corner_x

        else:  # Phase 3: Align back - turn opposite direction to realign with lane
            if self.obstacle_side == "left":
                # Obstacle was on left, now align LEFT top corner with LEFT lane edge
                top_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                top_lane_edge_x = self.get_lane_edge_x("left", height, width)

                if top_lane_edge_x is None:
                    return -100  # Default turn left if no lane detected

                # Error based on top corner only
                error = top_lane_edge_x - top_corner_x

            else:  # obstacle was on right
                # Obstacle was on right, now align RIGHT top corner with RIGHT lane edge
                top_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                top_lane_edge_x = self.get_lane_edge_x("right", height, width)

                if top_lane_edge_x is None:
                    return 100  # Default turn right if no lane detected

                # Error based on top corner only
                error = top_lane_edge_x - top_corner_x

        return error

    def pd_avoidance(self, error):
        """
        Calculate angular velocity using PD controller for avoidance.
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

        return p_term + d_term

    def update_avoidance_state(self, height, width):
        """
        Update the avoidance state based on detected obstacles.
        Handles three phases:
        - Phase 1: Turn until top corner meets lane edge
        - Phase 2: Move forward until bottom corner aligns with lane edge
        - Phase 3: Turn back to align both corners with opposite lane edge
        """
        center_x = width // 2

        # Check for obstacles in warning or danger zone
        active_obstacle = None
        for obs in self.obstacles:
            if obs["zone"] in ["warning", "danger"]:
                active_obstacle = obs
                break  # Take the closest one (already sorted)

        if active_obstacle is not None:
            # Determine which side the obstacle is on
            if active_obstacle["offset"] < 0:
                self.obstacle_side = "left"
                self.avoidance_direction = "right"
            else:
                self.obstacle_side = "right"
                self.avoidance_direction = "left"

            if not self.avoidance_active:
                # Start new avoidance maneuver
                self.avoidance_active = True
                self.avoidance_phase = 1
                self.phase1_alignment_achieved = False  # Reset alignment flag
                self.phase_start_time = time.time()  # Start Phase 1 timer

                if self.phase1_restart_count > self.PHASE1_MAX_RESTARTS:
                    # Already in forced turn mode from previous alignment loops
                    self.phase1_alignment_achieved = (
                        True  # Skip alignment, go straight to turn mode
                    )
                    self.get_logger().info(
                        f"Avoidance started: FORCED TURN MODE - turning {self.avoidance_direction} until obstacle clears"
                    )
                else:
                    self.get_logger().info(
                        f"Avoidance started: obstacle on {self.obstacle_side}, turning {self.avoidance_direction}"
                    )
            elif self.avoidance_phase == 1:
                # In Phase 1 with obstacle still visible
                # Check if corner alignment is achieved (only if not already in forced turn mode)
                if not self.phase1_alignment_achieved:
                    error = self.calculate_avoidance_error(height, width)
                    if error is not None and abs(error) < 20:
                        # Alignment achieved but obstacle still visible
                        self.phase1_restart_count += 1  # Count this as a "loop"

                        if self.phase1_restart_count > self.PHASE1_MAX_RESTARTS:
                            # Too many alignment completions with obstacle still there
                            self.phase1_alignment_achieved = (
                                True  # Switch to forced turn mode
                            )
                            self.get_logger().info(
                                f"Phase 1: Alignment loop #{self.phase1_restart_count} - FORCED TURN MODE - turning {self.avoidance_direction} until obstacle clears"
                            )
                        else:
                            self.get_logger().info(
                                f"Phase 1: Alignment achieved but obstacle still visible (loop #{self.phase1_restart_count}/{self.PHASE1_MAX_RESTARTS})"
                            )
                # Keep turning until obstacle clears (handled by else branch below)
        else:
            # No active obstacle - check phase transitions and completion
            if self.avoidance_active:
                error = self.calculate_avoidance_error(height, width)

                if self.avoidance_phase == 1:
                    # Phase 1: Check completion condition
                    if self.phase1_alignment_achieved:
                        # In forced turn mode - obstacle cleared, go to Phase 2 (ignore alignment)
                        self.avoidance_phase = 2
                        self.phase1_alignment_achieved = (
                            False  # Reset for next avoidance
                        )
                        self.phase1_restart_count = (
                            0  # Reset restart counter on successful transition
                        )
                        self.phase_start_time = time.time()  # Start Phase 2 timer
                        self.previous_avoidance_error = 0.0
                        self.last_avoidance_error_time = None
                        self.get_logger().info(
                            "Phase 1 complete: Forced turn mode - obstacle cleared, moving forward"
                        )
                    else:
                        # Normal mode - must achieve alignment before transitioning
                        alignment_complete = error is not None and abs(error) < 20

                        if alignment_complete:
                            # Alignment achieved and obstacle cleared - transition to Phase 2
                            self.avoidance_phase = 2
                            self.phase1_restart_count = (
                                0  # Reset restart counter on successful transition
                            )
                            self.phase_start_time = time.time()  # Start Phase 2 timer
                            self.previous_avoidance_error = 0.0
                            self.last_avoidance_error_time = None
                            self.get_logger().info(
                                "Phase 1 complete: Alignment done and obstacle cleared, moving forward"
                            )
                        # else: Keep turning until alignment is achieved (PD controller will handle it)

                elif self.avoidance_phase == 2:
                    # Check if Phase 2 is complete (bottom corner near lane edge OR timeout)
                    phase2_elapsed = (
                        time.time() - self.phase_start_time
                        if self.phase_start_time
                        else 0
                    )
                    phase2_complete = (error is not None and abs(error) < 20) or (
                        phase2_elapsed >= self.PHASE_TIMEOUT_FORWARD
                    )

                    if phase2_complete:
                        self.avoidance_phase = 3
                        self.phase_start_time = time.time()  # Start Phase 3 timer
                        self.previous_avoidance_error = 0.0
                        self.last_avoidance_error_time = None
                        if phase2_elapsed >= self.PHASE_TIMEOUT_FORWARD:
                            self.get_logger().info(
                                f"Phase 2 timeout ({self.PHASE_TIMEOUT_FORWARD}s): Moving to Phase 3"
                            )
                        else:
                            self.get_logger().info(
                                "Phase 2 complete: Moving forward done, aligning back"
                            )

                elif self.avoidance_phase == 3:
                    # Check if Phase 3 is complete (top corner aligned with lane edge OR timeout)
                    phase3_elapsed = (
                        time.time() - self.phase_start_time
                        if self.phase_start_time
                        else 0
                    )
                    # Check error tolerance first
                    error_satisfied = error is not None and abs(error) < 50
                    # Check timeout as fallback
                    timeout_triggered = phase3_elapsed >= self.PHASE_TIMEOUT_TURN

                    phase3_complete = error_satisfied or timeout_triggered

                    if phase3_complete:
                        self.avoidance_active = False
                        self.avoidance_phase = 1
                        self.avoidance_direction = None
                        self.obstacle_side = None
                        self.phase_start_time = None  # Reset timer
                        self.previous_avoidance_error = 0.0
                        self.last_avoidance_error_time = None
                        if timeout_triggered and not error_satisfied:
                            self.get_logger().info(
                                f"Phase 3 timeout ({self.PHASE_TIMEOUT_TURN}s): Resuming lane following"
                            )
                        else:
                            self.get_logger().info(
                                "Avoidance complete: Resuming lane following"
                            )

        # Also check Phase 3 timeout outside the obstacle check (fallback if stuck)
        if self.avoidance_active and self.avoidance_phase == 3:
            phase3_elapsed = (
                time.time() - self.phase_start_time if self.phase_start_time else 0
            )
            if phase3_elapsed >= self.PHASE_TIMEOUT_TURN:
                self.avoidance_active = False
                self.avoidance_phase = 1
                self.avoidance_direction = None
                self.obstacle_side = None
                self.phase_start_time = None
                self.previous_avoidance_error = 0.0
                self.last_avoidance_error_time = None
                self.get_logger().info(
                    f"Phase 3 timeout ({self.PHASE_TIMEOUT_TURN}s): Forced exit, resuming lane following"
                )

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

        if self.avoidance_phase == 1:
            # Phase 1: Show top corner and lane edge target
            if self.obstacle_side == "left":
                corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x("right", height, width)
                # Draw corner point (cyan)
                cv2.circle(frame, (corner_x, top_y), 10, (255, 255, 0), -1)
                # Draw target lane edge (magenta)
                if lane_edge_x is not None:
                    cv2.circle(frame, (lane_edge_x, top_y), 10, (255, 0, 255), -1)
                    cv2.line(
                        frame, (corner_x, top_y), (lane_edge_x, top_y), (255, 0, 255), 2
                    )
            else:
                corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x("left", height, width)
                cv2.circle(frame, (corner_x, top_y), 10, (255, 255, 0), -1)
                if lane_edge_x is not None:
                    cv2.circle(frame, (lane_edge_x, top_y), 10, (255, 0, 255), -1)
                    cv2.line(
                        frame, (corner_x, top_y), (lane_edge_x, top_y), (255, 0, 255), 2
                    )

            status = f"PHASE 1: Turn {self.avoidance_direction.upper()}"

        elif self.avoidance_phase == 2:
            # Phase 2: Show lane ROI bottom corner and lane edge target
            if self.obstacle_side == "left":
                corner_x = self.get_lane_roi_bottom_corner_x("left", height, width)
                lane_edge_x = self.get_lane_edge_x_at_bottom("right", height, width)
                # Draw corner point (cyan)
                cv2.circle(frame, (corner_x, bottom_y), 10, (255, 255, 0), -1)
                # Draw target lane edge (magenta)
                if lane_edge_x is not None:
                    cv2.circle(frame, (lane_edge_x, bottom_y), 10, (255, 0, 255), -1)
                    cv2.line(
                        frame,
                        (corner_x, bottom_y),
                        (lane_edge_x, bottom_y),
                        (255, 0, 255),
                        2,
                    )
            else:
                corner_x = self.get_lane_roi_bottom_corner_x("right", height, width)
                lane_edge_x = self.get_lane_edge_x_at_bottom("left", height, width)
                cv2.circle(frame, (corner_x, bottom_y), 10, (255, 255, 0), -1)
                if lane_edge_x is not None:
                    cv2.circle(frame, (lane_edge_x, bottom_y), 10, (255, 0, 255), -1)
                    cv2.line(
                        frame,
                        (corner_x, bottom_y),
                        (lane_edge_x, bottom_y),
                        (255, 0, 255),
                        2,
                    )

            status = f"PHASE 2: Move FORWARD"

        else:  # Phase 3
            # Phase 3: Show top corner aligning with same-side lane edge
            turn_dir = "LEFT" if self.obstacle_side == "left" else "RIGHT"

            if self.obstacle_side == "left":
                # Align LEFT top corner with LEFT lane edge
                top_corner_x = self.get_obstacle_roi_corner_x("left", height, width)
                top_lane_edge_x = self.get_lane_edge_x("left", height, width)
            else:
                # Align RIGHT top corner with RIGHT lane edge
                top_corner_x = self.get_obstacle_roi_corner_x("right", height, width)
                top_lane_edge_x = self.get_lane_edge_x("right", height, width)

            # Draw top corner and target (cyan to magenta)
            cv2.circle(frame, (top_corner_x, top_y), 10, (255, 255, 0), -1)
            if top_lane_edge_x is not None:
                cv2.circle(frame, (top_lane_edge_x, top_y), 10, (255, 0, 255), -1)
                cv2.line(
                    frame,
                    (top_corner_x, top_y),
                    (top_lane_edge_x, top_y),
                    (255, 0, 255),
                    2,
                )

            status = f"PHASE 3: Turn {turn_dir} (align)"

        # Display avoidance status
        cv2.putText(
            frame, status, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        error = self.calculate_avoidance_error(height, width)
        if error is not None:
            cv2.putText(
                frame,
                f"Avoid Error: {error:+.0f}",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2,
            )

    def control_callback(self):
        """
        Control loop - sends velocity commands to robot.
        Phase 1: Turn in place (or slow forward) until top corner aligns
        Phase 2: Move forward with slight correction until bottom corner aligns
        Phase 3: Turn back to align both corners with opposite lane edge
        """
        twist = Twist()

        # Get frame dimensions (assuming 640x480)
        height, width = 480, 640

        # Update avoidance state
        self.update_avoidance_state(height, width)

        if self.avoidance_active:
            error = self.calculate_avoidance_error(height, width)

            if self.avoidance_phase == 1:
                # Phase 1: Turn to align corner with lane edge
                # If alignment not yet achieved, use PD controller
                # If alignment achieved but obstacle still visible, turn at constant speed
                twist.linear.x = 0.0  # No forward movement, only turn

                if self.phase1_alignment_achieved:
                    # Alignment done, but obstacle still visible - turn at constant speed
                    if self.avoidance_direction == "right":
                        twist.angular.z = (
                            -self.ANGULAR_SPEED_MAX
                        )  # Turn right (negative)
                    else:
                        twist.angular.z = self.ANGULAR_SPEED_MAX  # Turn left (positive)
                else:
                    # Use PD controller to achieve corner alignment
                    angular_z = self.pd_avoidance(error)
                    angular_z = max(
                        -self.ANGULAR_SPEED_MAX, min(self.ANGULAR_SPEED_MAX, angular_z)
                    )
                    twist.angular.z = -angular_z  # Inverted for correct robot direction

            elif self.avoidance_phase == 2:
                # Phase 2: Move forward only (no steering)
                twist.linear.x = self.LINEAR_SPEED  # Normal forward speed
                twist.angular.z = 0.0  # No turning, only forward

            else:  # Phase 3
                # Phase 3: Turn back to align with lane (no forward movement)
                angular_z = self.pd_avoidance(error)
                angular_z = max(
                    -self.ANGULAR_SPEED_MAX, min(self.ANGULAR_SPEED_MAX, angular_z)
                )

                twist.linear.x = 0.0  # No forward movement, only turn
                twist.angular.z = -angular_z  # Inverted for correct robot direction

        else:
            # Normal lane following mode (simple proportional)
            if self.lane_error is not None:
                twist.linear.x = self.LINEAR_SPEED
                # Simple proportional lane following
                twist.angular.z = (
                    -0.003 * self.lane_error
                )  # Negative sign for correct direction
                twist.angular.z = max(-0.3, min(0.3, twist.angular.z))
            else:
                # No lane detected, stop
                twist.linear.x = 0.0
                twist.angular.z = 0.0

        self.velocity_pub.publish(twist)

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
    """Main entry point."""
    rclpy.init(args=args)

    try:
        node = ObstacleDetectionNode()

        print("\n" + "=" * 60)
        print("ROS2 Obstacle Detection Node")
        print("=" * 60)
        print("Subscribing to: /camera/image_raw")
        print(
            "Orange obstacle HSV range:", node.CONE_LOWER_HSV, "-", node.CONE_UPPER_HSV
        )
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
