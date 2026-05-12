import cv2
import numpy as np
import time
from ultralytics import YOLO, FastSAM

# --- Configuration ---
MODEL_PATH = "model/road_signs_yolov8n.pt"
OBSTACLE_MODEL_ID = "model/FastSAM-s.pt"
OBSTACLE_CONFIDENCE = 0.90
LOWER_HSV = np.array([30, 90, 25])
UPPER_HSV = np.array([85, 250, 255])
STOP_COOLDOWN = 5.0
WAIT_DURATION = 3.0

# --- Obstacle Avoidance Configuration ---
# Proximity thresholds (normalized Y position: 0=top/far, 1=bottom/close)
OBSTACLE_DANGER_ZONE = 0.75  # Y > this = very close, must avoid immediately
OBSTACLE_WARNING_ZONE = 0.55  # Y > this = approaching, start planning avoidance
OBSTACLE_CLEARED_ZONE = 0.85  # Y > this = obstacle passed behind robot
# Driving corridor (center portion of frame where robot travels)
CORRIDOR_LEFT = 0.25  # Left boundary of driving path (25% from left)
CORRIDOR_RIGHT = 0.75  # Right boundary of driving path (75% from left)
# Minimum obstacle area to consider (filters noise)
MIN_OBSTACLE_AREA = 500
# Avoidance steering gain
AVOIDANCE_GAIN = 1.5
# Green color range for filtering (same as lane detection)
OBSTACLE_FILTER_GREEN_RATIO = 0.3  # If >30% of obstacle is green, it's likely the line


def init_camera(source=2):
    """Initialize video capture."""
    cap = cv2.VideoCapture(source)
    return cap


def load_yolo_model(path):
    """Load YOLO model and class names."""
    model = YOLO(path)
    return model, model.names


def load_obstacle_model(path):
    """Load FastSAM model for obstacle detection."""
    model = FastSAM(path)
    return model


def filter_false_obstacles(frame, obstacle_info, sign_detections):
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
        green_mask = cv2.inRange(roi_hsv, LOWER_HSV, UPPER_HSV)
        green_pixels = cv2.countNonZero(green_mask)
        total_pixels = roi_hsv.shape[0] * roi_hsv.shape[1]
        green_ratio = green_pixels / max(total_pixels, 1)

        if green_ratio > OBSTACLE_FILTER_GREEN_RATIO:
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


def detect_obstacles(frame, model, confidence=OBSTACLE_CONFIDENCE):
    """
    Detect obstacles using FastSAM segmentation.

    Args:
        frame: Input image frame
        model: FastSAM model
        confidence: Confidence threshold for detection (tunable)

    Returns:
        annotated_frame: Frame with obstacle annotations
        obstacle_mask: Binary mask of detected obstacles
        obstacle_info: List of obstacle bounding boxes and areas
    """
    height, width = frame.shape[:2]
    obstacle_mask = np.zeros((height, width), dtype=np.uint8)
    obstacle_info = []

    # Run FastSAM inference
    small_frame = cv2.resize(frame, (320, 240))
    results = model(
        small_frame, conf=confidence, iou=0.9, retina_masks=False, verbose=False
    )

    if results and len(results) > 0:
        result = results[0]

        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()

            for i, mask in enumerate(masks):
                # Resize mask to frame size if needed
                if mask.shape != (height, width):
                    mask_resized = cv2.resize(mask.astype(np.float32), (width, height))
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


def annotate_obstacles(frame, obstacle_mask, obstacle_info):
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


def check_obstacle_proximity(obstacle_info, frame_height, frame_width):
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
    corridor_left = int(frame_width * CORRIDOR_LEFT)
    corridor_right = int(frame_width * CORRIDOR_RIGHT)

    obstacles_in_path = []

    for obs in obstacle_info:
        if obs["area"] < MIN_OBSTACLE_AREA:
            continue

        cx, cy = obs["centroid"]
        x1, y1, x2, y2 = obs["box"]

        # Check if obstacle overlaps with driving corridor
        # (obstacle box intersects with corridor)
        obs_in_corridor = not (x2 < corridor_left or x1 > corridor_right)

        if obs_in_corridor:
            # Calculate normalized proximity (0=far/top, 1=close/bottom)
            # Use bottom of bounding box for proximity (closest point to robot)
            proximity = y2 / frame_height

            # Determine zone
            if proximity > OBSTACLE_DANGER_ZONE:
                zone = "DANGER"
            elif proximity > OBSTACLE_WARNING_ZONE:
                zone = "WARNING"
            else:
                zone = "FAR"

            obstacles_in_path.append(
                {
                    **obs,
                    "proximity": proximity,
                    "zone": zone,
                    "relative_x": (cx - frame_width / 2) / (frame_width / 2),  # -1 to 1
                }
            )

    # Sort by proximity (closest first)
    obstacles_in_path.sort(key=lambda x: x["proximity"], reverse=True)

    closest = obstacles_in_path[0] if obstacles_in_path else None
    return closest, obstacles_in_path


def calculate_avoidance_steering(closest_obstacle, frame_width, lane_error=None):
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

    # Determine which side has more clearance
    # If obstacle is on the left (relative_x < 0), steer right (positive error)
    # If obstacle is on the right (relative_x > 0), steer left (negative error)

    # Base avoidance direction (opposite to obstacle position)
    avoidance_direction = -1 if relative_x > 0 else 1  # Steer away from obstacle

    # Scale avoidance based on proximity (closer = stronger avoidance)
    if zone == "DANGER":
        # Immediate avoidance - strong steering
        avoidance_magnitude = frame_width * 0.3 * AVOIDANCE_GAIN
        action = "AVOID_IMMEDIATE"
    elif zone == "WARNING":
        # Gradual avoidance - moderate steering
        avoidance_magnitude = frame_width * 0.15 * AVOIDANCE_GAIN
        action = "AVOID_GRADUAL"
    else:
        # Far obstacle - minimal adjustment
        avoidance_magnitude = frame_width * 0.05
        action = "AVOID_PREPARE"

    avoidance_error = int(avoidance_direction * avoidance_magnitude)

    # Blend with lane following if available (prioritize avoidance when close)
    if lane_error is not None and zone != "DANGER":
        # Blend: more weight to avoidance as obstacle gets closer
        blend_factor = proximity  # 0 to 1
        avoidance_error = int(
            avoidance_error * blend_factor + lane_error * (1 - blend_factor)
        )

    return avoidance_error, action


def visualize_obstacle_avoidance(
    frame, closest_obstacle, avoidance_action, height, width
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
    corridor_left = int(width * CORRIDOR_LEFT)
    corridor_right = int(width * CORRIDOR_RIGHT)
    cv2.line(frame, (corridor_left, 0), (corridor_left, height), (255, 0, 255), 1)
    cv2.line(frame, (corridor_right, 0), (corridor_right, height), (255, 0, 255), 1)

    # Draw proximity zones
    danger_y = int(height * OBSTACLE_DANGER_ZONE)
    warning_y = int(height * OBSTACLE_WARNING_ZONE)
    cv2.line(
        frame, (corridor_left, danger_y), (corridor_right, danger_y), (0, 0, 255), 2
    )
    cv2.line(
        frame, (corridor_left, warning_y), (corridor_right, warning_y), (0, 165, 255), 2
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

        # Highlight the threatening obstacle
        x1, y1, x2, y2 = closest_obstacle["box"]
        color = (
            (0, 0, 255)
            if zone == "DANGER"
            else (0, 165, 255) if zone == "WARNING" else (0, 255, 255)
        )
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        # Show proximity percentage
        cv2.putText(
            frame,
            f"Proximity: {proximity*100:.0f}%",
            (10, 150),
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


def detect_and_annotate_signs(frame, model, class_names):
    """Detect signs using YOLO and annotate the frame."""
    detections = []
    results = model(frame, conf=0.5, iou=0.45, verbose=False)

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

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
                frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )

    return frame, detections


def create_roi_mask(height, width, mode):
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
        # Normalized coords: x=0.3, y=0.5, w=0.4, h=0.5
        rx, ry = int(0.3 * width), int(0.5 * height)
        rw, rh = int(0.4 * width), int(0.5 * height)
        mask[ry : ry + rh, rx : rx + rw] = 255
    else:  # bottom / fallback
        h = int(height * 0.5)
        mask[height - h :, :] = 255

    return mask


def process_lane_detection(frame, roi_mask):
    """Process frame to detect lane line and calculate error."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv, LOWER_HSV, UPPER_HSV)

    # Combine masks
    combined_mask = cv2.bitwise_and(color_mask, roi_mask)

    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)

    # Contours
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


def detect_junction(contour, frame_height, threshold=0.7):
    """Detect if approaching a junction (line doesn't reach bottom of frame)."""
    if contour is None:
        return False, None

    # Get the lowest point of the contour
    lowest_y = contour[:, :, 1].max()
    bottom_threshold = int(frame_height * threshold)

    # Check if line reaches near the bottom
    line_reaches_bottom = lowest_y >= bottom_threshold

    # Also check for multiple branches by looking at contour width at different heights
    # Get bounding rect to estimate line width
    x, y, w, h = cv2.boundingRect(contour)

    # If line is very wide relative to its height, might be a junction
    aspect_ratio = w / max(h, 1)
    is_wide = aspect_ratio > 2.0  # Line spreading out

    junction_detected = not line_reaches_bottom or is_wide

    return junction_detected, lowest_y


def scan_for_best_direction(frame, height, width):
    """Scan left and right ROIs to find the best direction with more line."""
    # Create masks for both directions
    left_mask = create_roi_mask(height, width, "bottom_left")
    right_mask = create_roi_mask(height, width, "bottom_right")

    # Process both
    _, left_contour, _, left_error = process_lane_detection(frame, left_mask)
    _, right_contour, _, right_error = process_lane_detection(frame, right_mask)

    # Calculate areas
    left_area = cv2.contourArea(left_contour) if left_contour is not None else 0
    right_area = cv2.contourArea(right_contour) if right_contour is not None else 0

    # Minimum area threshold to consider a valid line
    MIN_AREA = 500

    left_valid = left_area > MIN_AREA
    right_valid = right_area > MIN_AREA

    if left_valid and right_valid:
        # Both have lines - pick the one with more area
        if left_area > right_area * 1.2:  # Left significantly larger
            return "left", left_area, right_area
        elif right_area > left_area * 1.2:  # Right significantly larger
            return "right", left_area, right_area
        else:
            # Similar - default to right (or could be forward)
            return "right", left_area, right_area
    elif left_valid:
        return "left", left_area, right_area
    elif right_valid:
        return "right", left_area, right_area
    else:
        return None, left_area, right_area


def pid_follow_line(error):
    """Calculate steering based on error (P-Controller)."""
    if error is None:
        return 0
    Kp = 1.0
    return int(error * Kp)


def visualize_control(frame, error, height, width):
    """Visualize robot control action based on error."""
    if error is None:
        return

    center_x = width // 2
    steering = pid_follow_line(error)

    arrow_start = (center_x, height)
    arrow_end = (center_x + steering, height - 100)
    cv2.arrowedLine(frame, arrow_start, arrow_end, (255, 0, 255), 5, tipLength=0.3)

    if abs(error) < 20:
        text, color = "FORWARD", (0, 255, 0)
    elif error > 0:
        text, color = f"RIGHT ({abs(error)})", (0, 165, 255)
    else:
        text, color = f"LEFT ({abs(error)})", (0, 165, 255)

    cv2.putText(
        frame, f"Action: {text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
    )
    cv2.putText(
        frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
    )


def main():
    cap = init_camera(2)
    # cap = init_camera("rsrc/camera_recording_20251128_113309.mp4")
    model, class_names = load_yolo_model(MODEL_PATH)
    obstacle_model = load_obstacle_model(OBSTACLE_MODEL_ID)

    # State dictionary to keep track of FSM variables
    state = {
        "fsm": "FOLLOW_LANE",
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

    LOST_LINE_TIMEOUT = 0.5  # Seconds before starting search
    SEARCH_TIMEOUT = 2.0  # Seconds to search each direction
    JUNCTION_CONFIRM_TIME = 0.3  # Seconds to confirm junction before acting
    AVOIDANCE_TIMEOUT = 5.0  # Max seconds to stay in avoidance mode
    OBSTACLE_CLEAR_TIME = (
        1.0  # Seconds without seeing obstacle before considering it cleared
    )

    # FPS calculation variables
    prev_time = time.time()
    fps = 0

    # Persistent obstacle detection results (retained between skipped frames)
    obstacle_mask = np.zeros((480, 640), dtype=np.uint8)
    obstacle_info = []

    print("Video Controls:\n  ESC - Exit\n")

    frame_count = 0

    while True:
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
        prev_time = current_time

        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]

        # 1. Detect Signs
        frame, detections = detect_and_annotate_signs(frame, model, class_names)

        # 1.5. Detect Obstacles using FastSAM (only every 3rd frame for performance)
        if frame_count % 3 == 0:
            obstacle_mask, obstacle_info = detect_obstacles(
                frame, obstacle_model, OBSTACLE_CONFIDENCE
            )
            # Filter out green line and road signs from obstacles
            obstacle_info = filter_false_obstacles(frame, obstacle_info, detections)

        # Always annotate with the latest obstacle data (cached from last detection)
        frame = annotate_obstacles(frame, obstacle_mask, obstacle_info)

        # 1.6. Check obstacle proximity and calculate avoidance
        closest_obstacle, obstacles_in_path = check_obstacle_proximity(
            obstacle_info, height, width
        )
        avoidance_error, avoidance_action = calculate_avoidance_steering(
            closest_obstacle, width, lane_error=None  # Will blend with lane error later
        )

        # 2. Update FSM & ROI Logic
        # Draw detection lines
        sign_roi_top = int(height * 0.75)
        sign_roi_bottom = int(height * 0.8)
        cv2.line(frame, (0, sign_roi_top), (width, sign_roi_top), (0, 0, 255), 2)
        cv2.line(frame, (0, sign_roi_bottom), (width, sign_roi_bottom), (0, 0, 255), 2)

        # FSM Logic Block
        if state["fsm"] == "FOLLOW_LANE":
            stop_detected = False
            direction_detected = None

            for det in detections:
                _, y1, _, _ = det["box"]
                if sign_roi_top < y1 < sign_roi_bottom:
                    if det["class_name"] == "stop":
                        if time.time() - state["last_stop_time"] > STOP_COOLDOWN:
                            stop_detected = True
                    elif det["class_name"] in ["left", "right", "forward"]:
                        direction_detected = det["class_name"]

            if stop_detected:
                state["fsm"] = "STOP_WAIT"
                state["stop_start_time"] = time.time()
                # Signs have priority - disable junction auto-detection
                state["junction_deciding"] = False
                state["junction_detected_time"] = None
                if direction_detected == "left":
                    state["pending_roi_mode"] = "bottom_left"
                    state["sign_directed"] = True
                    state["sign_directed_time"] = time.time()
                elif direction_detected == "right":
                    state["pending_roi_mode"] = "bottom_right"
                    state["sign_directed"] = True
                    state["sign_directed_time"] = time.time()
                elif direction_detected == "forward":
                    state["pending_roi_mode"] = "custom_rect"
                    state["sign_directed"] = True
                    state["sign_directed_time"] = time.time()
                else:
                    state["pending_roi_mode"] = None
                print(f"STOP DETECTED. Waiting... Pending: {state['pending_roi_mode']}")

            elif direction_detected:
                # Signs have priority - disable junction auto-detection
                state["sign_directed"] = True
                state["sign_directed_time"] = time.time()
                state["junction_deciding"] = False
                state["junction_detected_time"] = None
                if direction_detected == "left":
                    state["roi_mode"] = "bottom_left"
                elif direction_detected == "right":
                    state["roi_mode"] = "bottom_right"
                elif direction_detected == "forward":
                    state["roi_mode"] = "custom_rect"
                print(
                    f"Sign: {direction_detected} -> ROI: {state['roi_mode']} (sign priority)"
                )

            # Check for obstacle in WARNING zone - start planning avoidance early
            elif closest_obstacle and closest_obstacle["zone"] in ["WARNING", "DANGER"]:
                state["fsm"] = "AVOID_OBSTACLE"
                state["avoiding_obstacle"] = True
                state["avoidance_start_time"] = time.time()
                state["obstacle_last_seen"] = time.time()
                state["original_roi_mode"] = state["roi_mode"]
                state["avoidance_phase"] = "approach"

                # Determine which side to pass: go to the side with more clearance
                # If obstacle is on the right (relative_x > 0), pass on left
                # If obstacle is on the left (relative_x < 0), pass on right
                if closest_obstacle["relative_x"] > 0:
                    state["avoidance_side"] = "left"
                    state["roi_mode"] = (
                        "bottom_left"  # Shift to follow left side of line
                    )
                else:
                    state["avoidance_side"] = "right"
                    state["roi_mode"] = (
                        "bottom_right"  # Shift to follow right side of line
                    )
                print(
                    f"OBSTACLE DETECTED! Planning path around {state['avoidance_side']} side."
                )

        elif state["fsm"] == "STOP_WAIT":
            elapsed = time.time() - state["stop_start_time"]
            remaining = int(WAIT_DURATION - elapsed) + 1
            cv2.putText(
                frame,
                f"STOP: {remaining}s",
                (width // 2 - 100, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 0, 255),
                4,
            )

            if elapsed >= WAIT_DURATION:
                state["fsm"] = "FOLLOW_LANE"
                state["last_stop_time"] = time.time()
                if state["pending_roi_mode"]:
                    state["roi_mode"] = state["pending_roi_mode"]
                    print(f"Wait done. ROI: {state['roi_mode']}")
                state["pending_roi_mode"] = None

        elif state["fsm"] == "TURN_180":
            TURN_DURATION = 3.0  # Adjust based on robot turn speed
            elapsed = time.time() - state["turn_start_time"]
            remaining = int(TURN_DURATION - elapsed) + 1
            cv2.putText(
                frame,
                f"TURN 180: {remaining}s",
                (width // 2 - 150, height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (255, 165, 0),
                4,
            )

            if elapsed >= TURN_DURATION:
                state["fsm"] = "FOLLOW_LANE"
                state["roi_mode"] = "custom_rect"
                state["lost_line_time"] = None
                state["search_direction"] = None
                state["sign_directed"] = False  # Reset sign priority
                state["sign_directed_time"] = None
                # Reset any deciding flags
                state["junction_deciding"] = False
                state["avoiding_obstacle"] = False
                print("Turn complete. Resuming lane follow.")

        elif state["fsm"] == "AVOID_OBSTACLE":
            # Path planning obstacle avoidance:
            # Phase 1 (approach): Shift to side ROI, steer away from center
            # Phase 2 (passing): Stay on offset path while obstacle is beside us
            # Phase 3 (clearing): Obstacle behind us, gradually return to center

            elapsed = time.time() - state["avoidance_start_time"]

            # Update obstacle tracking
            if closest_obstacle and closest_obstacle["zone"] != "FAR":
                state["obstacle_last_seen"] = time.time()

            time_since_obstacle = time.time() - (
                state["obstacle_last_seen"] or time.time()
            )

            # Display current avoidance status
            phase_text = (
                state["avoidance_phase"].upper()
                if state["avoidance_phase"]
                else "AVOIDING"
            )
            side_text = (
                state["avoidance_side"].upper() if state["avoidance_side"] else ""
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

            # Phase transitions based on obstacle position
            if closest_obstacle:
                proximity = closest_obstacle["proximity"]

                if proximity < OBSTACLE_WARNING_ZONE:
                    # Obstacle is far ahead - still approaching
                    state["avoidance_phase"] = "approach"
                elif proximity >= OBSTACLE_DANGER_ZONE:
                    # Obstacle is very close/beside us - passing
                    state["avoidance_phase"] = "passing"
                else:
                    # In warning zone
                    state["avoidance_phase"] = "approach"

            # Check completion conditions
            obstacle_cleared = False

            # Condition 1: Haven't seen obstacle for a while (it's behind us)
            if time_since_obstacle > OBSTACLE_CLEAR_TIME:
                obstacle_cleared = True
                print(f"Obstacle not seen for {OBSTACLE_CLEAR_TIME}s - cleared!")

            # Condition 2: Obstacle is now in FAR zone and we've been avoiding for a bit
            if closest_obstacle and closest_obstacle["zone"] == "FAR" and elapsed > 1.0:
                obstacle_cleared = True
                print("Obstacle now in FAR zone - cleared!")

            # Condition 3: Timeout
            if elapsed > AVOIDANCE_TIMEOUT:
                obstacle_cleared = True
                print("Avoidance timeout - returning to lane.")

            if obstacle_cleared:
                # Return to normal lane following
                state["fsm"] = "FOLLOW_LANE"
                state["avoiding_obstacle"] = False
                state["avoidance_side"] = None
                state["avoidance_phase"] = None
                state["roi_mode"] = "custom_rect"  # Return to center following
                state["obstacle_last_seen"] = None
                state["original_roi_mode"] = None
                print("Obstacle cleared. Resuming center lane follow.")

        # 3. Lane Detection
        roi_mask = create_roi_mask(height, width, state["roi_mode"])
        processed_mask, contour, centroid, error = process_lane_detection(
            frame, roi_mask
        )

        # 4. Junction detection & Auto-switch ROI
        junction_detected, lowest_y = detect_junction(contour, height)

        if error is not None:
            # Line found - reset lost state
            state["lost_line_time"] = None

            # Only do automatic junction detection if signs haven't already set direction
            if (
                junction_detected
                and state["roi_mode"] == "custom_rect"
                and state["fsm"] == "FOLLOW_LANE"
                and not state["sign_directed"]  # Signs have priority
            ):

                if state["junction_detected_time"] is None:
                    state["junction_detected_time"] = time.time()
                    state["junction_deciding"] = True  # Start slowing down

                junction_duration = time.time() - state["junction_detected_time"]
                cv2.putText(
                    frame,
                    "JUNCTION - SLOWING",
                    (width // 2 - 120, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

                if (
                    junction_duration > JUNCTION_CONFIRM_TIME
                    and state["search_direction"] is None
                ):
                    best_dir, left_area, right_area = scan_for_best_direction(
                        frame, height, width
                    )
                    print(
                        f"Junction scan - Left: {left_area}, Right: {right_area}, Best: {best_dir}"
                    )

                    if best_dir == "left":
                        state["search_direction"] = "left"
                        state["roi_mode"] = "bottom_left"
                        print("Junction detected. Going LEFT (more line found)")
                    elif best_dir == "right":
                        state["search_direction"] = "right"
                        state["roi_mode"] = "bottom_right"
                        print("Junction detected. Going RIGHT (more line found)")
                    else:
                        state["search_direction"] = "right"
                        state["roi_mode"] = "bottom_right"
                        state["search_start_time"] = time.time()
                        print("Junction detected. No clear direction, searching...")
            else:
                # No junction detected or not in custom_rect mode
                if state["search_direction"] is None:
                    state["junction_detected_time"] = None
                    # Reset junction_deciding if no junction and no active search
                    if not junction_detected:
                        state["junction_deciding"] = False

                # Check if we've completed the turn and can return to normal following
                if state["roi_mode"] == "bottom_left":
                    if -50 <= error <= 10:  # Wider tolerance for resuming
                        state["roi_mode"] = "custom_rect"
                        state["search_direction"] = None
                        state["junction_detected_time"] = None
                        state["junction_deciding"] = False  # Resume normal speed
                        state["sign_directed"] = False  # Reset sign priority
                        state["sign_directed_time"] = None
                        print("Left turn complete, resuming normal speed")
                elif state["roi_mode"] == "bottom_right":
                    if -10 <= error <= 50:  # Wider tolerance for resuming
                        state["roi_mode"] = "custom_rect"
                        state["search_direction"] = None
                        state["junction_detected_time"] = None
                        state["junction_deciding"] = False  # Resume normal speed
                        state["sign_directed"] = False  # Reset sign priority
                        state["sign_directed_time"] = None
                        print("Right turn complete, resuming normal speed")
        else:
            # Line lost - search logic
            if state["roi_mode"] == "custom_rect" and state["fsm"] == "FOLLOW_LANE":
                if state["lost_line_time"] is None:
                    state["lost_line_time"] = time.time()

                lost_duration = time.time() - state["lost_line_time"]

                if lost_duration > LOST_LINE_TIMEOUT:
                    if state["search_direction"] is None:
                        # Start searching right first
                        state["search_direction"] = "right"
                        state["roi_mode"] = "bottom_right"
                        state["search_start_time"] = time.time()
                        print("Line lost. Searching RIGHT...")

            elif (
                state["roi_mode"] == "bottom_right"
                and state["search_direction"] == "right"
            ):
                search_elapsed = time.time() - state.get(
                    "search_start_time", time.time()
                )
                if search_elapsed > SEARCH_TIMEOUT:
                    # Switch to left search
                    state["search_direction"] = "left"
                    state["roi_mode"] = "bottom_left"
                    state["search_start_time"] = time.time()
                    print("Not found right. Searching LEFT...")

            elif (
                state["roi_mode"] == "bottom_left"
                and state["search_direction"] == "left"
            ):
                search_elapsed = time.time() - state.get(
                    "search_start_time", time.time()
                )
                if search_elapsed > SEARCH_TIMEOUT:
                    # Turn 180 degrees
                    state["fsm"] = "TURN_180"
                    state["turn_start_time"] = time.time()
                    print("Line not found. Turning 180...")

        # 5. Visualization
        # Draw ROI
        roi_contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, roi_contours, -1, (255, 255, 0), 2)

        # Draw junction detection threshold line (cyan)
        junction_threshold = 0.4
        junction_y = int(height * junction_threshold)
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

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)

            # Draw where the line currently ends (if contour exists)
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

            # During obstacle avoidance, we follow the offset ROI, so use lane error
            # The path planning shifts the ROI, steering just follows that shifted path
            visualize_control(frame, error, height, width)

        # Visualize obstacle avoidance zones and closest obstacle
        visualize_obstacle_avoidance(
            frame, closest_obstacle, avoidance_action, height, width
        )

        # Display FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (width - 120, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        # cv2.imshow("Processed Mask", processed_mask)
        # cv2.imshow("Obstacle Mask", obstacle_mask)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
