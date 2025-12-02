import cv2
import torch
import numpy as np

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
model_type = "DPT_LeViT_224"  # MiDaS v3.1 - LeViT-224 (fastest inference speed)
# model_type = "DPT_SwinV2_T_256"  # MiDaS v3.1 - SwinV2 Tiny (good balance of speed and accuracy)
# print(torch.hub.list('intel-isl/MiDaS'))
# ['DPTDepthModel', 'DPT_BEiT_B_384', 'DPT_BEiT_L_384', 'DPT_BEiT_L_512', 'DPT_Hybrid', 'DPT_Large', 'DPT_LeViT_224', 'DPT_Next_ViT_L_384', 'DPT_SwinV2_B_384', 'DPT_SwinV2_L_384', 'DPT_SwinV2_T_256', 'DPT_Swin_L_384', 'MiDaS', 'MiDaS_small', 'MidasNet', 'MidasNet_small', 'transforms']

midas = torch.hub.load("intel-isl/MiDaS", model_type)

# Select device: CUDA > MPS (Mac) > CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

midas.to(device)
midas.eval()

print(f"Using device: {device}")

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
    transform = midas_transforms.dpt_transform
elif model_type == "DPT_LeViT_224":
    transform = midas_transforms.levit_transform
elif model_type == "DPT_SwinV2_T_256":
    # Custom transform for SwinV2 Tiny 256 - requires exactly 256x256 input
    import torchvision.transforms as T
    transform = T.Compose([
        lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        T.ToTensor(),
        T.Resize((256, 256)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        lambda x: x.unsqueeze(0)
    ])
elif "Swin" in model_type or "BEiT" in model_type or "Next_ViT" in model_type:
    transform = midas_transforms.dpt_transform
else:
    transform = midas_transforms.small_transform


def init_camera(source=0):  # Changed default to 0
    """Initialize video capture."""
    cap = cv2.VideoCapture(source)
    return cap


def compute_expected_floor_depth(height, width, tilt_angle=45):
    """
    Compute expected depth values for floor assuming camera tilt.
    With a 45-degree downward tilt, floor appears closer at bottom of frame.
    Returns a gradient representing expected floor depth at each row.
    """
    # Create a gradient where bottom rows (floor nearby) have higher depth values
    # and upper rows have lower depth values (further away floor)
    expected_depth = np.zeros((height, width), dtype=np.float32)

    # The floor depth increases (closer) as we go down the image
    for y in range(height):
        # Normalize y position (0 at top, 1 at bottom)
        y_norm = y / height
        # Expected relative depth for floor (higher = closer in MiDaS output)
        expected_depth[y, :] = y_norm

    return expected_depth


def detect_obstacles(depth_map, min_obstacle_area=500, depth_threshold_factor=0.3):
    """
    Detect obstacles by finding regions that are closer than expected floor depth.

    Args:
        depth_map: Normalized depth map from MiDaS (higher values = closer)
        min_obstacle_area: Minimum contour area to be considered an obstacle
        depth_threshold_factor: How much closer than expected to trigger detection

    Returns:
        List of bounding boxes (x, y, w, h) for detected obstacles
    """
    height, width = depth_map.shape

    # Normalize depth map to 0-1 range
    depth_norm = cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Compute expected floor depth gradient
    expected_floor = compute_expected_floor_depth(height, width)

    # Scale expected floor to match depth map range
    # Adjust based on typical floor depth values
    floor_scale = np.percentile(
        depth_norm[int(height * 0.8) :, :], 50
    )  # Use bottom 30% as floor reference
    expected_floor_scaled = expected_floor * floor_scale * 1.2

    # Find regions significantly closer than expected (obstacles protrude above floor)
    # Obstacles will have higher depth values (closer) than the expected floor gradient
    obstacle_mask = depth_norm > (expected_floor_scaled + depth_threshold_factor)

    # Also detect objects that are unusually close in the upper portion of frame
    # (obstacles that stick up high)
    upper_region = int(height * 0.5)
    close_threshold = np.percentile(depth_norm, 85)  # Top 15% closest pixels
    obstacle_mask[:upper_region, :] |= depth_norm[:upper_region, :] > close_threshold

    # Convert to uint8 for contour detection
    obstacle_mask_uint8 = (obstacle_mask * 255).astype(np.uint8)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    obstacle_mask_uint8 = cv2.morphologyEx(obstacle_mask_uint8, cv2.MORPH_CLOSE, kernel)
    obstacle_mask_uint8 = cv2.morphologyEx(obstacle_mask_uint8, cv2.MORPH_OPEN, kernel)

    # Find contours
    contours, _ = cv2.findContours(
        obstacle_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter and get bounding boxes
    bounding_boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_obstacle_area:
            x, y, w, h = cv2.boundingRect(contour)
            # Calculate average depth in this region for distance estimation
            roi_depth = depth_norm[y : y + h, x : x + w]
            avg_depth = np.mean(roi_depth)
            bounding_boxes.append((x, y, w, h, avg_depth))

    return bounding_boxes, obstacle_mask_uint8


def draw_obstacle_boxes(frame, bounding_boxes):
    """
    Draw bounding boxes on detected obstacles with depth information.
    """
    for x, y, w, h, avg_depth in bounding_boxes:
        # Color based on proximity (red = very close, yellow = medium, green = further)
        if avg_depth > 0.7:
            color = (0, 0, 255)  # Red - very close
            label = "CLOSE"
        elif avg_depth > 0.5:
            color = (0, 165, 255)  # Orange - medium
            label = "MEDIUM"
        else:
            color = (0, 255, 255)  # Yellow - further
            label = "FAR"

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Draw label with depth info
        depth_text = f"{label} ({avg_depth:.2f})"
        cv2.putText(
            frame, depth_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    return frame


def main():
    # cap = init_camera(2)
    cap = init_camera('rsrc/camera_recording_20251128_113309.mp4')
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video reached")
            break

        # Resize frame for display and processing consistency
        frame = cv2.resize(frame, (640, 480))

        # Transform input for model
        input_batch = transform(frame).to(device)

        with torch.no_grad():
            prediction = midas(input_batch)

            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        output = prediction.cpu().numpy()

        # Detect obstacles based on depth map
        bounding_boxes, obstacle_mask = detect_obstacles(
            # output, min_obstacle_area=800, depth_threshold_factor=0.25
            output, min_obstacle_area=400, depth_threshold_factor=0.2
        )

        # Normalize the output to 0-255 for display
        output_norm = cv2.normalize(
            output, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )

        # Apply a colormap for better visualization (optional, but looks nice)
        output_color = cv2.applyColorMap(output_norm, cv2.COLORMAP_MAGMA)

        # Draw bounding boxes on original frame
        frame_with_boxes = draw_obstacle_boxes(frame.copy(), bounding_boxes)

        # Also draw boxes on depth map for reference
        depth_with_boxes = draw_obstacle_boxes(output_color.copy(), bounding_boxes)

        # Show original with obstacles and depth map
        cv2.imshow("Obstacles Detected", frame_with_boxes)
        cv2.imshow("Depth Map", depth_with_boxes)
        cv2.imshow("Obstacle Mask", obstacle_mask)

        # Press 'q' to exit
        if cv2.waitKey(100) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
