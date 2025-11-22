import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- Configuration ---
MODEL_PATH = "model/road_signs_yolov8n.pt"
LOWER_HSV = np.array([30, 90, 25])
UPPER_HSV = np.array([85, 250, 255])
STOP_COOLDOWN = 5.0
WAIT_DURATION = 3.0


def init_camera(source=2):
    """Initialize video capture."""
    cap = cv2.VideoCapture(source)
    return cap


def load_yolo_model(path):
    """Load YOLO model and class names."""
    model = YOLO(path)
    return model, model.names


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
    cap = init_camera()
    model, class_names = load_yolo_model(MODEL_PATH)

    # State dictionary to keep track of FSM variables
    state = {
        "fsm": "FOLLOW_LANE",
        "roi_mode": "custom_rect",
        "stop_start_time": 0,
        "last_stop_time": 0,
        "pending_roi_mode": None,
    }

    print("Video Controls:\n  ESC - Exit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        frame = cv2.resize(frame, (640, 480))
        height, width = frame.shape[:2]

        # 1. Detect Signs
        frame, detections = detect_and_annotate_signs(frame, model, class_names)

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
                if direction_detected == "left":
                    state["pending_roi_mode"] = "bottom_left"
                elif direction_detected == "right":
                    state["pending_roi_mode"] = "bottom_right"
                elif direction_detected == "forward":
                    state["pending_roi_mode"] = "custom_rect"
                else:
                    state["pending_roi_mode"] = None
                print(f"STOP DETECTED. Waiting... Pending: {state['pending_roi_mode']}")

            elif direction_detected:
                if direction_detected == "left":
                    state["roi_mode"] = "bottom_left"
                elif direction_detected == "right":
                    state["roi_mode"] = "bottom_right"
                elif direction_detected == "forward":
                    state["roi_mode"] = "custom_rect"
                print(f"Sign: {direction_detected} -> ROI: {state['roi_mode']}")

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

        # 3. Lane Detection
        roi_mask = create_roi_mask(height, width, state["roi_mode"])
        processed_mask, contour, centroid, error = process_lane_detection(
            frame, roi_mask
        )

        # 4. Auto-switch ROI based on error
        if error is not None:
            if state["roi_mode"] == "bottom_left" and -30 <= error <= 0:
                state["roi_mode"] = "custom_rect"
                print("Auto-switch -> custom_rect")
            elif state["roi_mode"] == "bottom_right" and 0 <= error <= 30:
                state["roi_mode"] = "custom_rect"
                print("Auto-switch -> custom_rect")

        # 5. Visualization
        # Draw ROI
        roi_contours, _ = cv2.findContours(
            roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(frame, roi_contours, -1, (255, 255, 0), 2)

        if contour is not None:
            cv2.drawContours(frame, [contour], -1, (0, 255, 0), 2)
            cv2.circle(frame, centroid, 5, (0, 0, 255), -1)
            cv2.line(frame, (width // 2, 0), (width // 2, height), (255, 0, 0), 1)
            visualize_control(frame, error, height, width)

        cv2.imshow("Processed Mask", processed_mask)
        cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
