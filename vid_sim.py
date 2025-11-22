import cv2
import numpy as np

# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045141.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045332.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045829.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_064927.mp4")
cap = cv2.VideoCapture(2)

# Define HSV range
# lower_range = np.array([30, 60, 25])
# upper_range = np.array([85, 255, 255])
lower_range = np.array([30, 90, 25])
upper_range = np.array([85, 250, 255])

# Video playback controls
paused = False
playback_speed = 100  # milliseconds delay (lower = faster)
frame_skip = 1  # number of frames to skip (for fast forward)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ROI State
roi_mode = 'custom_rect'

print("Video Controls:")
print("  SPACE - Play/Pause")
print("  → (Right Arrow) - Fast Forward")
print("  ← (Left Arrow) - Rewind")
print("  + - Increase Speed")
print("  - - Decrease Speed")
print("  R - Reset Speed")
print("  F - Save Frame")
print("  A - ROI Bottom Left")
print("  D - ROI Bottom Right")
print("  S - ROI Custom Rect")
print("  W - ROI Bottom (Reset)")
print("  ESC - Exit")
print()

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break
    else:
        ret = True  # Keep the current frame

    frame = cv2.resize(frame, (640, 480))

    # Image processing logic from img_sim.py
    height, width, _ = frame.shape

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_range, upper_range)

    # ROI configuration
    # Modes: 'bottom', 'bottom_left', 'bottom_right', 'custom_rect', 'polygon'
    # roi_mode is controlled by keyboard input
    # For bottom/bottom_* modes: fraction of image height to keep from bottom (0-1)
    roi_bottom_fraction = 0.5
    # For bottom_left/bottom_right: fraction of width to keep on the chosen side (0-1)
    roi_horizontal_fraction = 0.5

    # For custom_rect (coords can be normalized if roi_normalized=True)
    roi_normalized = True
    roi_x = 0.3   # left (normalized 0..1 or pixels)
    roi_y = 0.5   # top
    roi_w = 0.4   # width
    roi_h = 0.5   # height

    # For polygon: list of (x,y) pairs in normalized coords (0..1)
    roi_polygon = [(0.0, 0.6), (0.4, 0.6), (0.4, 1.0), (0.0, 1.0)]

    # Build ROI mask and apply to the color mask
    roi_mask = np.zeros_like(mask)  # single-channel mask same size as `mask`
    if roi_mode == 'bottom':
        roi_h_px = int(height * roi_bottom_fraction)
        roi_top = height - roi_h_px
        roi_mask[roi_top:, :] = 255
    elif roi_mode == 'bottom_left':
        roi_h_px = int(height * roi_bottom_fraction)
        roi_w_px = int(width * roi_horizontal_fraction)
        roi_top = height - roi_h_px
        roi_mask[roi_top:, :roi_w_px] = 255
    elif roi_mode == 'bottom_right':
        roi_h_px = int(height * roi_bottom_fraction)
        roi_w_px = int(width * roi_horizontal_fraction)
        roi_top = height - roi_h_px
        roi_mask[roi_top:, width - roi_w_px:] = 255
    elif roi_mode == 'custom_rect':
        if roi_normalized:
            rx = int(roi_x * width)
            ry = int(roi_y * height)
            rw = int(roi_w * width)
            rh = int(roi_h * height)
        else:
            rx, ry, rw, rh = int(roi_x), int(roi_y), int(roi_w), int(roi_h)
        roi_top = ry
        roi_mask[ry:ry + rh, rx:rx + rw] = 255
    elif roi_mode == 'polygon':
        pts = np.array([[(int(x * width), int(y * height)) for (x, y) in roi_polygon]], dtype=np.int32)
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
            if roi_mode == 'bottom_left' and -30 <= error <= 0:
                roi_mode = 'custom_rect'
                print(f"Auto-switching to ROI Mode: {roi_mode}")
            elif roi_mode == 'bottom_right' and 0 <= error <= 30:
                roi_mode = 'custom_rect'
                print(f"Auto-switching to ROI Mode: {roi_mode}")

            # Visualization
            # Draw the ROI area
            roi_contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    elif key == ord(" "):  # Space - Play/Pause
        paused = not paused
        print("PAUSED" if paused else "PLAYING")
    elif key == 83 or key == 3:  # Right arrow - Fast Forward (skip frames)
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = min(current_pos + 30, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        print(f"Fast Forward to frame {int(new_pos)}")
    elif key == 81 or key == 2:  # Left arrow - Rewind (go back frames)
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        new_pos = max(current_pos - 30, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
        print(f"Rewind to frame {int(new_pos)}")
    elif key == ord("+") or key == ord("="):  # Increase speed (decrease delay)
        playback_speed = max(10, playback_speed - 20)
        print(f"Speed increased (delay: {playback_speed}ms)")
    elif key == ord("-") or key == ord("_"):  # Decrease speed (increase delay)
        playback_speed = min(500, playback_speed + 20)
        print(f"Speed decreased (delay: {playback_speed}ms)")
    elif key == ord("r") or key == ord("R"):  # Reset speed
        playback_speed = 100
        print(f"Speed reset to {playback_speed}ms")
    elif key == ord("f") or key == ord("F"):  # Save frame
        filename = f"saved_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")
    elif key == ord("a") or key == ord("A"):
        roi_mode = 'bottom_left'
        print(f"ROI Mode: {roi_mode}")
    elif key == ord("d") or key == ord("D"):
        roi_mode = 'bottom_right'
        print(f"ROI Mode: {roi_mode}")
    elif key == ord("s") or key == ord("S"):
        roi_mode = 'custom_rect'
        print(f"ROI Mode: {roi_mode}")
    elif key == ord("w") or key == ord("W"):
        roi_mode = 'bottom'
        print(f"ROI Mode: {roi_mode}")

cap.release()
cv2.destroyAllWindows()
