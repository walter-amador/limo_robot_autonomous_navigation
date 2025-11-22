import cv2
import numpy as np
from ultralytics import YOLO

# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045141.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045332.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045829.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_064927.mp4")
cap = cv2.VideoCapture(2)

# Load the fine-tuned YOLO model
MODEL_PATH = "model/road_signs_yolov8n.pt"
model = YOLO(MODEL_PATH)

# Get class names from the model
class_names = model.names

# Define HSV range
lower_range = np.array([30, 90, 25])
upper_range = np.array([85, 250, 255])

# Video playback controls
paused = False
playback_speed = 100  # milliseconds delay (lower = faster)
frame_skip = 1  # number of frames to skip (for fast forward)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ROI State
nav_roi_mode = "custom_rect"

print("Video Controls:")
print("  A - ROI Bottom Left")
print("  D - ROI Bottom Right")
print("  S - ROI Custom Rect")
print("  W - ROI Bottom (Reset)")
print("  ESC - Exit")
print()


def trackSign_ML(frame, model):
    detections = []
    # Perform inference
    results = model(frame, conf=0.5, iou=0.45, verbose=False)

    # Process results
    for result in results:
        boxes = result.boxes

        # Draw bounding boxes and labels
        for box in boxes:
            # Get box coordinates
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # Get confidence and class
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            detections.append({"class_name": class_name, "box": (x1, y1, x2, y2)})

            # Calculate bounding box size for distance estimation
            box_width = x2 - x1
            box_height = y2 - y1
            box_size = box_width if box_width > box_height else box_height

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Prepare label text
            label = f"{class_name}: {confidence:.2f}"

            # Get text size for background
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )

            # Draw background rectangle for text
            cv2.rectangle(
                frame,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1,
            )

            # Draw label text
            cv2.putText(
                frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2
            )
    return frame, detections


while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
    else:
        ret = True  # Keep the current frame

    frame = cv2.resize(frame, (640, 480))
    frame, detections = trackSign_ML(frame, model)

    # Image processing logic from img_sim.py
    height, width, _ = frame.shape

    # Sign ROI Logic
    sign_roi_y_top = int(height * 0.75)
    cv2.line(frame, (0, sign_roi_y_top), (width, sign_roi_y_top), (0, 0, 255), 2)
    sign_roi_y_bottom = int(height * 0.85)
    cv2.line(frame, (0, sign_roi_y_bottom), (width, sign_roi_y_bottom), (0, 0, 255), 2)

    for det in detections:
        _, y1, _, _ = det["box"]
        if y1 > sign_roi_y_top and y1 < sign_roi_y_bottom:
            if det["class_name"] == "left":
                nav_roi_mode = "bottom_left"
                print(f"Sign Detected: left -> Switching to {nav_roi_mode}")
            elif det["class_name"] == "right":
                nav_roi_mode = "bottom_right"
                print(f"Sign Detected: right -> Switching to {nav_roi_mode}")
            elif det["class_name"] == "forward":
                nav_roi_mode = "custom_rect"
                print(f"Sign Detected: forward -> Switching to {nav_roi_mode}")

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # ROI configuration
    # Modes: 'bottom', 'bottom_left', 'bottom_right', 'custom_rect', 'polygon'
    # nav_roi_mode is controlled by keyboard input
    # For bottom/bottom_* modes: fraction of image height to keep from bottom (0-1)
    roi_bottom_fraction = 0.5
    # For bottom_left/bottom_right: fraction of width to keep on the chosen side (0-1)
    roi_horizontal_fraction = 0.5

    # For custom_rect (coords can be normalized if roi_normalized=True)
    roi_normalized = True
    roi_x = 0.3  # left (normalized 0..1 or pixels)
    roi_y = 0.5  # top
    roi_w = 0.4  # width
    roi_h = 0.5  # height

    # For polygon: list of (x,y) pairs in normalized coords (0..1)
    roi_polygon = [(0.0, 0.6), (0.4, 0.6), (0.4, 1.0), (0.0, 1.0)]

    # Build ROI mask and apply to the color mask
    roi_mask = np.zeros_like(mask)  # single-channel mask same size as `mask`
    if nav_roi_mode == "bottom":
        roi_h_px = int(height * roi_bottom_fraction)
        roi_top = height - roi_h_px
        roi_mask[roi_top:, :] = 255
    elif nav_roi_mode == "bottom_left":
        roi_h_px = int(height * roi_bottom_fraction)
        roi_w_px = int(width * roi_horizontal_fraction)
        roi_top = height - roi_h_px
        roi_mask[roi_top:, :roi_w_px] = 255
    elif nav_roi_mode == "bottom_right":
        roi_h_px = int(height * roi_bottom_fraction)
        roi_w_px = int(width * roi_horizontal_fraction)
        roi_top = height - roi_h_px
        roi_mask[roi_top:, width - roi_w_px :] = 255
    elif nav_roi_mode == "custom_rect":
        if roi_normalized:
            rx = int(roi_x * width)
            ry = int(roi_y * height)
            rw = int(roi_w * width)
            rh = int(roi_h * height)
        else:
            rx, ry, rw, rh = int(roi_x), int(roi_y), int(roi_w), int(roi_h)
        roi_top = ry
        roi_mask[ry : ry + rh, rx : rx + rw] = 255
    elif nav_roi_mode == "polygon":
        pts = np.array(
            [[(int(x * width), int(y * height)) for (x, y) in roi_polygon]],
            dtype=np.int32,
        )
        cv2.fillPoly(roi_mask, pts, 255)
        ys = [int(y * height) for (_, y) in roi_polygon]
        roi_top = min(ys) if ys else int(height / 2)
    else:
        # fallback: bottom half
        roi_top = int(height / 2)
        roi_mask[roi_top:, :] = 255

    # Keep only ROI in the mask
    mask = cv2.bitwise_and(mask, roi_mask)

    # Morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour (assumed to be the line)
        largest_contour = max(contours, key=cv2.contourArea)

        # Calculate moments to find centroid
        M = cv2.moments(largest_contour)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Calculate error (deviation from center of image)
            center_x = width // 2
            error = cx - center_x

            # Auto-switch ROI mode based on error
            if nav_roi_mode == "bottom_left" and -30 <= error <= 0:
                nav_roi_mode = "custom_rect"
                print(f"Auto-switching to ROI Mode: {nav_roi_mode}")
            elif nav_roi_mode == "bottom_right" and 0 <= error <= 30:
                nav_roi_mode = "custom_rect"
                print(f"Auto-switching to ROI Mode: {nav_roi_mode}")

            # Visualization
            # Draw the ROI area
            roi_contours, _ = cv2.findContours(
                roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(frame, roi_contours, -1, (255, 255, 0), 2)
            # Draw the contour
            cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
            # Draw the centroid of the line
            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            # Draw the center of the image (robot's heading)
            cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
            # Display error
            cv2.putText(
                frame,
                f"Error: {error}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            # print(f"Calculated Error: {error}")

    cv2.imshow("Processed Mask", mask)
    cv2.imshow("Frame", frame)

    # Handle keyboard input
    key = cv2.waitKey(playback_speed if not paused else 0) & 0xFF

    if key == 27:  # ESC
        break
    elif key == ord("a") or key == ord("A"):
        nav_roi_mode = "bottom_left"
        print(f"ROI Mode: {nav_roi_mode}")
    elif key == ord("d") or key == ord("D"):
        nav_roi_mode = "bottom_right"
        print(f"ROI Mode: {nav_roi_mode}")
    elif key == ord("s") or key == ord("S"):
        nav_roi_mode = "custom_rect"
        print(f"ROI Mode: {nav_roi_mode}")
    elif key == ord("w") or key == ord("W"):
        nav_roi_mode = "bottom"
        print(f"ROI Mode: {nav_roi_mode}")

cap.release()
cv2.destroyAllWindows()
