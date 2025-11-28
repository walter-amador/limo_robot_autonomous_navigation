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

        print('+++line_reaches_bottom', line_reaches_bottom)

        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / max(h, 1)
        is_wide = aspect_ratio > 1.5
        print('+++is_wide', is_wide)

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
                        self.get_logger().info("Left turn complete, resuming normal speed")
                elif self.state["roi_mode"] == "bottom_right":
                    if -10 <= error <= 50:  # Wider tolerance for resuming
                        self.state["roi_mode"] = "custom_rect"
                        self.state["search_direction"] = None
                        self.state["junction_detected_time"] = None
                        self.state["junction_deciding"] = False  # Resume normal speed
                        self.state["sign_directed"] = False  # Reset sign priority
                        self.state["sign_directed_time"] = None
                        self.get_logger().info("Right turn complete, resuming normal speed")
        else:
            # Line lost - handle sign-directed turns or search logic
            if self.state["sign_directed"] and self.state["fsm"] == "FOLLOW_LANE":
                # Sign directed a turn but line is lost - keep turning
                # But add a timeout to prevent infinite turning
                if self.state["sign_directed_time"] is not None:
                    sign_turn_duration = time.time() - self.state["sign_directed_time"]
                    # If we've been turning for too long (e.g., 5 seconds), give up and search
                    if sign_turn_duration > 5.0:
                        self.get_logger().warn("Sign-directed turn timeout, switching to search mode")
                        self.state["sign_directed"] = False
                        self.state["sign_directed_time"] = None
                        self.state["search_direction"] = "left" if self.state["roi_mode"] == "bottom_left" else "right"
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
