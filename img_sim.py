import cv2
import numpy as np

# frame = cv2.imread('imgs/saved_frame_24.jpg')
# frame = cv2.imread('imgs/saved_frame_66.jpg')
# frame = cv2.imread('imgs/saved_frame_92.jpg')
frame = cv2.imread('imgs/saved_frame_350.jpg')

lower_range = np.array([30, 90, 25])
upper_range = np.array([85, 250, 255])

proc_frame = cv2.resize(frame, (640, 480))
frame = proc_frame.copy() # Update frame to resized version for display
height, width, _ = frame.shape

hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_range, upper_range)

# Define ROI (Region of Interest) - Focus on the bottom half of the image
roi_top = int(height / 4) * 3
mask[:roi_top, :] = 0  # Zero out the top half of the mask

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
        
        # Visualization
        # Draw the ROI line
        cv2.line(frame, (0, roi_top), (width, roi_top), (255, 255, 0), 2)
        # Draw the contour
        cv2.drawContours(frame, [largest_contour], -1, (0, 255, 0), 2)
        # Draw the centroid of the line
        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
        # Draw the center of the image (robot's heading)
        cv2.line(frame, (center_x, 0), (center_x, height), (255, 0, 0), 1)
        # Display error
        cv2.putText(frame, f"Error: {error}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        print(f"Calculated Error: {error}")

cv2.imshow("Frame", frame)
cv2.imshow("Processed Mask", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()