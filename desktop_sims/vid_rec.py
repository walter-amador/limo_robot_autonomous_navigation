import cv2
import numpy as np

# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045141.mp4")
cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045332.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_045829.mp4")
# cap = cv2.VideoCapture("rsrc/camera_recording_20251121_064927.mp4")
# cap = cv2.VideoCapture(0)

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

print("Video Controls:")
print("  SPACE - Play/Pause")
print("  → (Right Arrow) - Fast Forward")
print("  ← (Left Arrow) - Rewind")
print("  + - Increase Speed")
print("  - - Decrease Speed")
print("  R - Reset Speed")
print("  S - Save Frame")
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
    elif key == ord("s") or key == ord("S"):  # Save frame
        filename = f"saved_frame_{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Frame saved as {filename}")

cap.release()
cv2.destroyAllWindows()
